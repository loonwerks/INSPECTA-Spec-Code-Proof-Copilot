"""
Visualization script for Gumbo FSE self‑adaptation metrics.

This script reads a progress log in JSON format (`progress.json`) and
generates plots comparing baseline and final metrics for each plan.  It
produces the following figures in the current directory:

1. `arrow_scatter.png` – scatter plot of total tokens vs. run time per contract
   with arrows from baseline to final values.
2. `token_footprint.png` – bar chart comparing average baseline tokens to
   average final tokens for each plan.
3. `savings_per_block.png` – bar chart showing token and runtime savings per
   contract.
4. `efficiency.png` – bar charts for contracts verified per minute and per
   1 k tokens.
5. `human_interventions.png` – bar chart summarising manual interventions per
   plan.

The script assumes `progress.json` contains a list of entries, where each
entry represents a plan version (including a special "baseline" entry).  Each
entry should include a `mode` field identifying the adaptation mode, and a
`contracts` dictionary mapping contract names to metric dictionaries.  Baseline
entries are expected to have `baseline_...` keys, while plan entries are
expected to have `final_...` keys.  See the adaptation plans for details.

Usage:
    python visualize_metrics.py [progress.json]

If no path is given, the script defaults to `progress.json` in the current
directory.  The script prints summary tables to stdout and writes figures to
PNG files.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_progress(path: Path) -> Dict[str, Any]:
    """Load progress.json.  If the file does not exist or is empty, return
    an empty structure."""
    if not path.exists():
        print(f"Warning: {path} does not exist.  No metrics to visualise.")
        return {}
    try:
        with path.open() as f:
            data = json.load(f)
        # progress.json can be either a dict or a list of dicts.  Normalise to
        # a list for ease of processing.
        if isinstance(data, dict):
            return {data.get("mode", "unknown"): data}
        plans: Dict[str, Any] = {}
        for entry in data:
            mode = entry.get("mode", f"version_{entry.get('version', 'unknown')}")
            plans[mode] = entry
        return plans
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}


def extract_metrics(entry: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, float]], int]:
    """Extract per‑contract baseline and final metrics from a progress entry.

    Returns a tuple (metrics, human_interventions) where metrics maps contract
    names to a dictionary with keys:
        baseline_tokens, final_tokens, baseline_runtime, final_runtime,
        tokens_saved, runtime_saved.
    The `human_interventions` count is extracted from the entry if available.
    """
    contracts = entry.get("contracts", {})
    metrics: Dict[str, Dict[str, float]] = {}
    for name, m in contracts.items():
        baseline_tokens = m.get("baseline_tokens") or m.get("prev_tokens")
        final_tokens = m.get("final_tokens") or m.get("final_metric")
        baseline_runtime = m.get("baseline_runtime") or m.get("prev_runtime")
        final_runtime = m.get("final_runtime") or m.get("final_runtime_sec")
        # Skip entries without necessary fields
        if baseline_tokens is None or final_tokens is None:
            continue
        metrics[name] = {
            "baseline_tokens": float(baseline_tokens),
            "final_tokens": float(final_tokens),
            "tokens_saved": float(baseline_tokens) - float(final_tokens),
            "baseline_runtime": float(baseline_runtime) if baseline_runtime is not None else 0.0,
            "final_runtime": float(final_runtime) if final_runtime is not None else 0.0,
            "runtime_saved": (float(baseline_runtime) - float(final_runtime)) if baseline_runtime is not None and final_runtime is not None else 0.0,
        }
    interventions = entry.get("human_interventions", 0)
    return metrics, interventions


def compute_aggregate(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute aggregate statistics (average and median tokens/runtime) and
    efficiency metrics for a set of per‑contract metrics."""
    if not metrics:
        return {}
    baseline_tokens = np.array([m["baseline_tokens"] for m in metrics.values()])
    final_tokens = np.array([m["final_tokens"] for m in metrics.values()])
    baseline_runtime = np.array([m["baseline_runtime"] for m in metrics.values()])
    final_runtime = np.array([m["final_runtime"] for m in metrics.values()])
    n_contracts = len(metrics)
    agg = {
        "avg_baseline_tokens": baseline_tokens.mean(),
        "median_baseline_tokens": float(np.median(baseline_tokens)),
        "avg_final_tokens": final_tokens.mean(),
        "median_final_tokens": float(np.median(final_tokens)),
        "avg_baseline_runtime": baseline_runtime.mean(),
        "median_baseline_runtime": float(np.median(baseline_runtime)),
        "avg_final_runtime": final_runtime.mean(),
        "median_final_runtime": float(np.median(final_runtime)),
        "contracts_per_minute": n_contracts / (final_runtime.sum() / 60.0) if final_runtime.sum() > 0 else 0.0,
        "contracts_per_1k_tokens": n_contracts / (final_tokens.sum() / 1000.0) if final_tokens.sum() > 0 else 0.0,
    }
    return agg


def plot_arrow_scatter(all_metrics: Dict[str, Dict[str, Dict[str, float]]]):
    """Generate an arrow scatter plot of tokens vs. run time for each plan."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(range(len(all_metrics)))
    for idx, (mode, metrics) in enumerate(all_metrics.items()):
        for name, m in metrics.items():
            ax.scatter(m["baseline_tokens"], m["baseline_runtime"], marker="o", color=colors[idx], label=f"{mode} baseline" if name == list(metrics.keys())[0] else "")
            ax.scatter(m["final_tokens"], m["final_runtime"], marker="^", color=colors[idx], label=f"{mode} final" if name == list(metrics.keys())[0] else "")
            ax.arrow(m["baseline_tokens"], m["baseline_runtime"],
                     m["final_tokens"] - m["baseline_tokens"],
                     m["final_runtime"] - m["baseline_runtime"],
                     head_width=0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                     head_length=0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                     length_includes_head=True, color=colors[idx], alpha=0.7)
    ax.set_xlabel("Total tokens per contract (baseline vs. final)")
    ax.set_ylabel("Verification run time (sec)")
    ax.set_title("Token usage vs. runtime: baseline → final")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig("arrow_scatter.png")
    plt.close(fig)


def plot_token_footprint(all_metrics: Dict[str, Dict[str, Dict[str, float]]]):
    """Plot average baseline vs. final tokens per contract for each plan."""
    modes = list(all_metrics.keys())
    avg_baseline = []
    avg_final = []
    for mode in modes:
        metrics = all_metrics[mode]
        avg_baseline.append(np.mean([m["baseline_tokens"] for m in metrics.values()]))
        avg_final.append(np.mean([m["final_tokens"] for m in metrics.values()]))
    x = np.arange(len(modes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, avg_baseline, width, label="Baseline tokens")
    ax.bar(x + width/2, avg_final, width, label="Final tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Average tokens per contract")
    ax.set_title("Average token footprint before and after adaptation")
    ax.legend()
    fig.tight_layout()
    fig.savefig("token_footprint.png")
    plt.close(fig)


def plot_savings_per_block(all_metrics: Dict[str, Dict[str, Dict[str, float]]]):
    """Plot tokens and runtime saved per contract for each plan."""
    fig, axes = plt.subplots(len(all_metrics), 1, figsize=(8, 5 * len(all_metrics)), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    for idx, (mode, metrics) in enumerate(all_metrics.items()):
        names = list(metrics.keys())
        tokens_saved = [metrics[n]["tokens_saved"] for n in names]
        runtime_saved = [metrics[n]["runtime_saved"] for n in names]
        x = np.arange(len(names))
        width = 0.35
        ax = axes[idx]
        ax.bar(x - width/2, tokens_saved, width, label="Tokens saved")
        ax.bar(x + width/2, runtime_saved, width, label="Runtime saved (sec)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel("Savings")
        ax.set_title(f"Per‑contract savings for {mode}")
        ax.legend()
    plt.tight_layout()
    plt.savefig("savings_per_block.png")
    plt.close(fig)


def plot_efficiency_and_interventions(aggregates: Dict[str, Dict[str, float]], interventions: Dict[str, int]):
    """Plot efficiency metrics and human interventions per plan."""
    modes = list(aggregates.keys())
    contracts_per_minute = [aggregates[m]["contracts_per_minute"] for m in modes]
    contracts_per_1k_tokens = [aggregates[m]["contracts_per_1k_tokens"] for m in modes]
    human_ints = [interventions.get(m, 0) for m in modes]
    x = np.arange(len(modes))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(x, contracts_per_minute)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(modes)
    axes[0].set_ylabel("Contracts per minute")
    axes[0].set_title("Verification throughput (time)")

    axes[1].bar(x, contracts_per_1k_tokens)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(modes)
    axes[1].set_ylabel("Contracts per 1k tokens")
    axes[1].set_title("Verification throughput (tokens)")

    axes[2].bar(x, human_ints)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(modes)
    axes[2].set_ylabel("Human interventions")
    axes[2].set_title("Manual interventions per plan")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig("efficiency.png")
    plt.close(fig)


def main(progress_path: Path):
    plans = load_progress(progress_path)
    if not plans:
        return
    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    aggregates: Dict[str, Dict[str, float]] = {}
    interventions: Dict[str, int] = {}
    for mode, entry in plans.items():
        metrics, ints = extract_metrics(entry)
        if not metrics:
            print(f"Warning: no metrics found for plan '{mode}'.")
            continue
        all_metrics[mode] = metrics
        interventions[mode] = ints
        aggregates[mode] = compute_aggregate(metrics)
        # Print summary to stdout
        print(f"\nSummary for plan {mode}:")
        for name, m in metrics.items():
            print(f"  Contract {name}: baseline_tokens={m['baseline_tokens']}, final_tokens={m['final_tokens']},"
                  f" baseline_runtime={m['baseline_runtime']}, final_runtime={m['final_runtime']}")
        print(f"  Avg baseline tokens: {aggregates[mode]['avg_baseline_tokens']:.2f}")
        print(f"  Avg final tokens: {aggregates[mode]['avg_final_tokens']:.2f}")
        print(f"  Contracts per minute: {aggregates[mode]['contracts_per_minute']:.2f}")
        print(f"  Contracts per 1k tokens: {aggregates[mode]['contracts_per_1k_tokens']:.2f}")
        print(f"  Human interventions: {interventions[mode]}")

    # Plot figures
    if all_metrics:
        plot_arrow_scatter(all_metrics)
        plot_token_footprint(all_metrics)
        plot_savings_per_block(all_metrics)
        plot_efficiency_and_interventions(aggregates, interventions)
        print("\nFigures saved: arrow_scatter.png, token_footprint.png, savings_per_block.png, efficiency.png")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("progress.json")
    main(path)