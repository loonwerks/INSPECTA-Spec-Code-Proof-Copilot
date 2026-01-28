#!/usr/bin/env python3
"""Evaluate plan checkpoints by verifying the model and recording cosine distances."""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit("openai package not installed; activate .venv and install openai") from exc

MODEL = "text-embedding-3-small"


def extract_blocks(text: str) -> Dict[str, str]:
    """Return mapping of part name -> GUMBO block text."""
    pattern = re.compile(r"part def\s+([A-Za-z0-9_]+)")
    blocks: Dict[str, str] = {}
    for match in pattern.finditer(text):
        part = match.group(1)
        sub = text[match.start():]
        gumbo = sub.find('language "GUMBO"')
        if gumbo == -1:
            continue
        sub = sub[gumbo:]
        end = sub.find('}*/')
        if end == -1:
            continue
        end += len('}*/')
        blocks[part] = sub[:end]
    return blocks


def cosine_distance(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 1.0
    return 1 - dot / (norm1 * norm2)


def compute_distances(sysml_file: Path, golden_file: Path, client: OpenAI) -> Dict[str, float]:
    """Compute per-contract cosine distance for a single SysML file."""
    text_curr = sysml_file.read_text()
    text_gold = golden_file.read_text()
    curr_blocks = extract_blocks(text_curr)
    gold_blocks = extract_blocks(text_gold)
    parts: Set[str] = set(curr_blocks.keys()) | set(gold_blocks.keys())
    results: Dict[str, float] = {}
    for part in sorted(parts):
        curr = curr_blocks.get(part)
        gold = gold_blocks.get(part)
        if curr is None or gold is None:
            continue  # skip unmatched blocks
        emb_curr = client.embeddings.create(model=MODEL, input=curr).data[0].embedding
        emb_gold = client.embeddings.create(model=MODEL, input=gold).data[0].embedding
        results[part] = cosine_distance(emb_curr, emb_gold)
    return results


def run_cmd(cmd: List[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", nargs="+", required=True, help="Labels for the plan checkpoints to evaluate")
    parser.add_argument(
        "--sysml-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Directory containing the SysML files (default: isolette/sysml)",
    )
    parser.add_argument(
        "--golden-dir",
        default="golden_examples",
        help="Directory containing golden SysML files (relative to sysml-dir unless absolute)",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root used when running the Sireum verification commands",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip running the Sireum/HAMR verification commands (use only if already validated)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to store evaluation_*.json files (default: current working directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sysml_dir = Path(args.sysml_dir).resolve()
    if not sysml_dir.exists():
        raise SystemExit(f"SysML directory not found: {sysml_dir}")

    golden_dir = Path(args.golden_dir)
    if not golden_dir.is_absolute():
        golden_dir = (sysml_dir / golden_dir).resolve()
    if not golden_dir.exists():
        raise SystemExit(f"Golden directory not found: {golden_dir}")

    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists():
        raise SystemExit(f"Repository root not found: {repo_root}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI()

    for plan_name in args.plans:
        start = time.time()
        if not args.skip_verification:
            run_cmd(["bash", "-lc", "./sireum hamr sysml logika --sourcepath isolette/sysml"], cwd=repo_root)
            run_cmd(["bash", "-lc", "isolette/hamr/slang/bin/run-logika.cmd"], cwd=repo_root)

        distances: Dict[str, Dict[str, float]] = {}
        for sysml_file in sorted(sysml_dir.glob("*.sysml")):
            golden_file = golden_dir / sysml_file.name
            if not golden_file.exists():
                continue
            per_contract = compute_distances(sysml_file, golden_file, client)
            if per_contract:
                distances[sysml_file.name] = per_contract

        duration = time.time() - start
        data = {
            "plan": plan_name,
            "generation_time_sec": duration,
            "distances": distances,
        }
        out_name = output_dir / f"evaluation_{sanitize(plan_name)}.json"
        out_name.write_text(json.dumps(data, indent=2))
        print(f"[evaluation] Wrote {out_name}")


if __name__ == "__main__":  # pragma: no cover
    main()
