#!/usr/bin/env python3
"""Self-adaptation helper for Isolette GUMBO contracts."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

DEFAULT_CONTRACTS = [
    "Manage_Monitor_Interface_i",
    "Manage_Monitor_Mode_i",
    "Manage_Alarm_i",
]

@dataclass
class ContractInfo:
    name: str
    file_path: Path
    block_text: str
    golden_text: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sysml-dir", default="isolette/sysml", help="Directory containing SysML files")
    parser.add_argument("--golden", default="golden_examples/isolette/sysml", help="Directory containing golden SysML files")
    parser.add_argument("--contracts", nargs="*", default=DEFAULT_CONTRACTS, help="Contracts to adapt")
    parser.add_argument("--plan", default="Gumbo_FSE_agent_Plan.md", help="Plan file to update")
    parser.add_argument("--plan-backup-dir", default="packups_self_adaption_plans", help="Directory for plan backups")
    parser.add_argument("--checkpoints-dir", default="Gumbo_FSE_plan_progress", help="Directory for plan checkpoints")
    parser.add_argument("--mode", choices=["cosine-dist", "git-diff", "strong-diff"], default="cosine-dist", help="Acceptance metric")
    parser.add_argument("--eps", type=float, default=0.001, help="Relative improvement epsilon for cosine mode")
    parser.add_argument("--regen-cmd", default=None, help="Shell command used to regenerate a contract; must contain {contract}")
    parser.add_argument("--eval-only", action="store_true", help="Compute metrics only")
    parser.add_argument("--sourcepath", default="isolette/sysml", help="Sourcepath passed to sireum hamr sysml logika")
    parser.add_argument("--code-logika-cmd", default="isolette/hamr/slang/bin/run-logika.cmd", help="Command for code-level Logika")
    parser.add_argument("--target-metric", type=float, default=None, help="Optional metric threshold for early acceptance")
    parser.add_argument("--max-runtime", type=float, default=None, help="Optional runtime budget in seconds")
    parser.add_argument("--max-contracts", type=int, default=None, help="Maximum contracts to process this run")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def log_stage(msg: str) -> None:
    print(f"[stage] {msg}")


def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def discover_contract_files(sysml_dir: Path, contracts: Sequence[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in sysml_dir.glob("*.sysml"):
        text = path.read_text()
        for contract in contracts:
            if contract in mapping:
                continue
            needle = f"part def {contract}"
            if needle in text:
                mapping[contract] = path
    missing = [c for c in contracts if c not in mapping]
    if missing:
        raise SystemExit(f"Contracts not found in {sysml_dir}: {missing}")
    return mapping


def extract_block(text: str, name: str) -> str:
    needle = f"part def {name}"
    idx = text.find(needle)
    if idx == -1:
        raise ValueError(f"Contract {name} not found")
    sub = text[idx:]
    gumbo = sub.find('language "GUMBO"')
    if gumbo == -1:
        raise ValueError(f"Contract {name} missing GUMBO block")
    sub = sub[gumbo:]
    end = sub.find('}*/')
    if end == -1:
        raise ValueError(f"Contract {name} GUMBO block not closed")
    end += len('}*/')
    return sub[:end]


def load_contracts(sysml_dir: Path, golden_dir: Path, contracts: Sequence[str]) -> Dict[str, ContractInfo]:
    mapping = discover_contract_files(sysml_dir, contracts)
    results: Dict[str, ContractInfo] = {}
    for contract, file_path in mapping.items():
        current_text = file_path.read_text()
        golden_path = golden_dir / file_path.name
        golden_text = golden_path.read_text()
        results[contract] = ContractInfo(
            name=contract,
            file_path=file_path,
            block_text=extract_block(current_text, contract),
            golden_text=extract_block(golden_text, contract),
        )
    return results


def cosine_distance(contract: ContractInfo) -> float:
    if OpenAI is None:
        raise SystemExit("openai package not installed; activate environment or use --mode git-diff")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set; source .env or export it")
    client = OpenAI()
    curr = client.embeddings.create(model="text-embedding-3-small", input=contract.block_text)
    gold = client.embeddings.create(model="text-embedding-3-small", input=contract.golden_text)
    vec_curr = curr.data[0].embedding
    vec_gold = gold.data[0].embedding
    dot = sum(a * b for a, b in zip(vec_curr, vec_gold))
    norm_curr = sum(a * a for a in vec_curr) ** 0.5
    norm_gold = sum(b * b for b in vec_gold) ** 0.5
    if norm_curr == 0 or norm_gold == 0:
        return 1.0
    return 1 - dot / (norm_curr * norm_gold)


def line_diff(contract: ContractInfo) -> float:
    import difflib

    diff = difflib.unified_diff(
        contract.golden_text.splitlines(),
        contract.block_text.splitlines(),
        lineterm="",
    )
    return float(
        sum(
            1
            for line in diff
            if (line.startswith("+") or line.startswith("-"))
            and not line.startswith("+++")
            and not line.startswith("---")
        )
    )


def strong_diff(contract: ContractInfo) -> float:
    import difflib

    def tokenize(text: str) -> List[str]:
        return text.split()

    golden_tokens = tokenize(contract.golden_text)
    current_tokens = tokenize(contract.block_text)
    matcher = difflib.SequenceMatcher(a=golden_tokens, b=current_tokens)
    return float(sum(op[-1] for op in matcher.get_opcodes() if op[0] != "equal"))


def compute_metric(contract: ContractInfo, mode: str) -> float:
    if mode == "cosine-dist":
        return cosine_distance(contract)
    if mode == "git-diff":
        return line_diff(contract)
    if mode == "strong-diff":
        return strong_diff(contract)
    raise ValueError(f"Unknown mode {mode}")


def backup_plan(plan: Path, backup_dir: Path, verbose: bool) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"{plan.stem}_{timestamp}{plan.suffix}"
    shutil.copy2(plan, dest)
    log(f"Backed up plan to {dest}", verbose)
    return dest


def next_plan_version(checkpoints_dir: Path) -> str:
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    max_idx = -1
    for path in checkpoints_dir.glob("plan_v_*.md"):
        stem = path.stem.replace("plan_v_", "")
        try:
            idx = int(stem)
        except ValueError:
            continue
        max_idx = max(max_idx, idx)
    return f"plan_v_{max_idx + 1}.md"


def copy_checkpoint(plan: Path, checkpoints_dir: Path, verbose: bool) -> Path:
    version = next_plan_version(checkpoints_dir)
    dest = checkpoints_dir / version
    shutil.copy2(plan, dest)
    log(f"Wrote checkpoint {dest}", verbose)
    return dest


def run_command(cmd: str, verbose: bool) -> None:
    log(f"Running: {cmd}", verbose)
    subprocess.run(cmd, shell=True, check=True)


def request_agent_help(stage: str, cmd: str, logs: str) -> None:
    payload = {
        "stage": stage,
        "command": cmd,
        "logs": logs[-4000:],
        "request": "Explain the failure and suggest a repair plan for the GUMBO specification.",
    }
    print("[info] Invoking Codex MCP agent for assistance...")
    try:
        result = subprocess.run(
            ["codex", "mcp", "--agent", "gpt-pro", "--input", json.dumps(payload)],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout.strip())
    except FileNotFoundError:
        print("[warn] codex CLI not found. Please consult the plan manually.")
    except subprocess.CalledProcessError as exc:
        print(f"[warn] Codex MCP call failed: {exc}")
        if exc.stdout:
            print(exc.stdout)
        if exc.stderr:
            print(exc.stderr)
    print("[info] Resolve the issue and rerun after reviewing the assistant's guidance.")


def run_with_guard(cmd: str, stage: str, repair_hint: str, verbose: bool) -> None:
    log_stage(stage)
    try:
        completed = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        if completed.stdout.strip():
            log(completed.stdout.strip(), verbose)
        if completed.stderr.strip():
            log(completed.stderr.strip(), verbose)
    except subprocess.CalledProcessError as exc:
        print(f"[error] {stage} failed with exit code {exc.returncode}.")
        if exc.stdout:
            print(exc.stdout)
        if exc.stderr:
            print(exc.stderr)
        print(f"Repair suggestion: {repair_hint}")
        request_agent_help(stage, cmd, (exc.stdout or "") + (exc.stderr or ""))
        raise SystemExit(1)


def run_integration_logika(sourcepath: str, verbose: bool) -> None:
    cmd = f"sireum hamr sysml logika --sourcepath {sourcepath}"
    run_with_guard(cmd, "Running integration Logika", "Inspect the SysML contracts and plan edits", verbose)


def run_code_logika(cmd: str, verbose: bool) -> None:
    run_with_guard(cmd, "Running code-level Logika", "Regenerate the Slang project and validate contracts", verbose)


def should_accept(prev: float, new: float, mode: str, eps: float, target: Optional[float]) -> bool:
    if target is not None and new <= target:
        return True
    if mode == "cosine-dist":
        return new <= prev * (1 - eps)
    return new < prev


def enforce_runtime_budget(start_time: float, max_runtime: Optional[float]) -> None:
    if max_runtime is None:
        return
    if time.time() - start_time > max_runtime:
        log_stage("Runtime budget exceeded; stopping adaptation")
        raise SystemExit(0)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    sysml_dir = Path(args.sysml_dir)
    golden_dir = Path(args.golden)
    plan_path = Path(args.plan)

    if not plan_path.exists():
        raise SystemExit(f"Plan file not found: {plan_path}")

    log_stage("Loading contracts and golden references")
    contracts = load_contracts(sysml_dir, golden_dir, args.contracts)

    log_stage("Computing baseline metrics")
    prev_metrics: Dict[str, float] = {}
    for name, info in contracts.items():
        metric = compute_metric(info, args.mode)
        prev_metrics[name] = metric
        print(f"{name}: {metric:.6f}")

    if args.eval_only:
        print("Eval-only mode: exiting before self-adaptation.")
        return

    if not args.regen_cmd:
        raise SystemExit("--regen-cmd is required unless --eval-only is set")

    log_stage("Backing up plan")
    backup_plan(plan_path, Path(args.plan_backup_dir), args.verbose)

    processed = 0
    improved = False
    start = time.time()
    log_stage("Beginning per-contract adaptation loop")
    for contract in args.contracts:
        enforce_runtime_budget(start, args.max_runtime)
        if args.max_contracts is not None and processed >= args.max_contracts:
            log_stage("Reached max-contracts limit; stopping early")
            break
        processed += 1
        info = contracts[contract]
        log_stage(f"Regenerating {contract}")
        run_command(args.regen_cmd.format(contract=contract), args.verbose)
        updated_text = info.file_path.read_text()
        info.block_text = extract_block(updated_text, contract)
        log_stage(f"Recomputing metric for {contract}")
        new_metric = compute_metric(info, args.mode)
        if info.block_text == info.golden_text:
            print(f"  Warning: {contract} matches the golden contract exactly; rejecting this change.")
            print("  Please ensure the plan-driven regeneration was applied instead of copying the golden text.")
            continue
        print(f"{contract}: prev={prev_metrics[contract]:.6f} new={new_metric:.6f}")
        if should_accept(prev_metrics[contract], new_metric, args.mode, args.eps, args.target_metric):
            print(f"  Accepted improvement for {contract}")
            prev_metrics[contract] = new_metric
            improved = True
        else:
            print(f"  No improvement detected for {contract}; please review plan edits")

    if not improved:
        log_stage("No contract improved; skipping verification and checkpoint")
        return

    run_integration_logika(args.sourcepath, args.verbose)
    run_code_logika(args.code_logika_cmd, args.verbose)

    log_stage("Writing plan checkpoint")
    copy_checkpoint(plan_path, Path(args.checkpoints_dir), args.verbose)
    log_stage("Self-adaptation iteration completed")


if __name__ == "__main__":
    main()
