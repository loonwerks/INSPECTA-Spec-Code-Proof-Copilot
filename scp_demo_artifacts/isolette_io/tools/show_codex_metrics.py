#!/usr/bin/env python3
"""
Summarize Codex CLI JSONL sessions with token usage, cache hits, reasoning tokens,
overall elapsed time, and time-to-verification. Prints both a table and a simple chart.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


UsageDict = Dict[str, float]


def parse_timestamp(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def load_events(path: Path) -> List[dict]:
    events: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                events.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_no}: {exc}") from exc
    if not events:
        raise SystemExit("No events found in the provided JSONL file.")
    return events


def accumulate_usage(events: Iterable[dict]) -> UsageDict:
    """Prefer the last reported usage block (Codex usually emits totals there)."""
    latest: UsageDict = {}
    for event in events:
        response = event.get("response")
        usage = None
        if isinstance(response, dict):
            usage = response.get("usage")
        if not isinstance(usage, dict):
            usage = event.get("usage")
        if isinstance(usage, dict):
            latest = {k: float(v) for k, v in usage.items()}
    return latest


def compute_durations(
    events: Iterable[dict], verification_pattern: Optional[str]
) -> Tuple[Optional[float], Optional[float]]:
    start_ts: Optional[datetime] = None
    end_ts: Optional[datetime] = None
    verification_ts: Optional[datetime] = None
    pattern = verification_pattern.lower() if verification_pattern else None

    for event in events:
        ts = parse_timestamp(event.get("timestamp"))
        if ts:
            start_ts = ts if start_ts is None else min(start_ts, ts)
            end_ts = ts if end_ts is None else max(end_ts, ts)

        if pattern and verification_ts is None:
            serialized = json.dumps(event, ensure_ascii=False).lower()
            if pattern in serialized:
                verification_ts = ts

    elapsed = (
        (end_ts - start_ts).total_seconds() if start_ts and end_ts else None
    )
    verification_elapsed = (
        (verification_ts - start_ts).total_seconds()
        if start_ts and verification_ts
        else None
    )
    return elapsed, verification_elapsed


def format_value(value: Optional[float], decimals: int = 2) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.{decimals}f}"


def print_table(rows: List[Tuple[str, str]]) -> None:
    width = max(len(label) for label, _ in rows)
    header = ("Metric".ljust(width), "Value")
    separator = "-" * width
    print(f"{header[0]}  {header[1]}")
    print(f"{separator}  {'-' * len(header[1])}")
    for label, value in rows:
        print(f"{label.ljust(width)}  {value}")


def render_chart(metrics: List[Tuple[str, float]], width: int = 40) -> None:
    numeric_values = [value for _, value in metrics if value > 0]
    if not numeric_values:
        print("\nNo metrics available for chart.")
        return
    max_value = max(numeric_values)
    print("\nMetric Chart")
    for label, value in metrics:
        if value <= 0:
            bar = ""
        else:
            filled = max(1, int((value / max_value) * width))
            bar = "#" * filled
        print(f"{label:>20} | {bar} ({format_value(value, 0)})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display Codex CLI usage metrics from a JSONL transcript."
    )
    parser.add_argument(
        "jsonl",
        type=Path,
        help="Path to the Codex CLI JSONL log produced via `codex exec --json` or `codex resume --json`.",
    )
    parser.add_argument(
        "--verification-pattern",
        default="Verification succeeded",
        help="Substring indicating verification success; used to compute time-to-verification. "
        "Case-insensitive match over serialized events. Set to '' to disable.",
    )
    parser.add_argument(
        "--chart-width",
        type=int,
        default=40,
        help="Width of the ASCII chart bars.",
    )
    args = parser.parse_args()

    events = load_events(args.jsonl)
    usage = accumulate_usage(events)
    elapsed, verification_elapsed = compute_durations(
        events, args.verification_pattern if args.verification_pattern else None
    )

    total_input = usage.get("input_tokens", 0.0)
    total_output = usage.get("output_tokens", 0.0)
    cache_created = usage.get("cache_creation_input_tokens", 0.0)
    cache_read = usage.get("cache_read_input_tokens", 0.0)
    reasoning_in = usage.get("reasoning_input_tokens", 0.0)
    reasoning_out = usage.get("reasoning_output_tokens", 0.0)
    cached_total = cache_created + cache_read
    reasoning_total = reasoning_in + reasoning_out

    table_rows = [
        ("Input tokens", format_value(total_input)),
        ("Output tokens", format_value(total_output)),
        ("Reasoning input tokens", format_value(reasoning_in)),
        ("Reasoning output tokens", format_value(reasoning_out)),
        ("Cached tokens (create)", format_value(cache_created)),
        ("Cached tokens (read)", format_value(cache_read)),
        ("Total cached tokens", format_value(cached_total)),
        ("Total reasoning tokens", format_value(reasoning_total)),
        ("Elapsed time (s)", format_value(elapsed)),
        (
            "Time until verification passed (s)",
            format_value(verification_elapsed),
        ),
    ]

    print_table(table_rows)

    chart_metrics = [
        ("Input", total_input),
        ("Output", total_output),
        ("Reasoning", reasoning_total),
        ("Cached", cached_total),
    ]
    render_chart(chart_metrics, width=args.chart_width)


if __name__ == "__main__":
    main()
