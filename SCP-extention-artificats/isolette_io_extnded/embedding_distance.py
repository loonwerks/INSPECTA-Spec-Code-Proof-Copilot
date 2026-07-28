#!/usr/bin/env python3
"""Compute cosine distances between local GUMBO blocks and golden examples using OpenAI embeddings.

Reads API key from the environment (e.g., via `.env`). Falls back to exiting with an error if blocks
or embeddings cannot be retrieved.
"""
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit("openai package not installed; activate .venv and install openai") from exc

MODEL = "text-embedding-3-small"


def extract_blocks(text: str) -> Dict[str, str]:
    """Return mapping of part name -> GUMBO block text."""
    blocks: Dict[str, str] = {}
    pattern = re.compile(r"part def\s+([A-Za-z0-9_]+)")
    for match in pattern.finditer(text):
        part = match.group(1)
        sub = text[match.start():]
        gumbo_idx = sub.find('language "GUMBO"')
        if gumbo_idx == -1:
            continue
        sub = sub[gumbo_idx:]
        end = sub.find('}*/')
        if end == -1:
            continue
        end += len('}*/')
        blocks[part] = sub[:end]
    return blocks


def embed(client: OpenAI, text: str) -> List[float]:
    resp = client.embeddings.create(model=MODEL, input=text)
    return resp.data[0].embedding


def cosine_distance(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 1.0
    cos = dot / (norm1 * norm2)
    return 1 - cos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current",
        default="isolette/sysml/Monitor.sysml",
        help="Path to the current Monitor.sysml (default: isolette/sysml/Monitor.sysml)",
    )
    parser.add_argument(
        "--golden",
        default="golden_examples/isolette/sysml/Monitor.sysml",
        help="Path to the golden Monitor.sysml (default: golden_examples/isolette/sysml/Monitor.sysml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the JSON results; prints to stdout if omitted",
    )
    parser.add_argument(
        "--contracts",
        nargs="*",
        default=None,
        help="Optional list of contract names to compare; defaults to all GUMBO blocks",
    )
    args = parser.parse_args()

    current_path = Path(args.current)
    golden_path = Path(args.golden)
    if not current_path.exists():
        raise SystemExit(f"Current file not found: {current_path}")
    if not golden_path.exists():
        raise SystemExit(f"Golden file not found: {golden_path}")

    current_blocks = extract_blocks(current_path.read_text())
    golden_blocks = extract_blocks(golden_path.read_text())

    if args.contracts:
        parts: Set[str] = set(args.contracts)
    else:
        parts = set(current_blocks.keys()) | set(golden_blocks.keys())

    client = OpenAI()
    results: Dict[str, Dict[str, Optional[float]]] = {}
    for part in sorted(parts):
        curr_block = current_blocks.get(part)
        gold_block = golden_blocks.get(part)
        if curr_block is None or gold_block is None:
            results[part] = {
                "distance": None,
                "current_found": curr_block is not None,
                "golden_found": gold_block is not None,
            }
            continue
        curr_emb = embed(client, curr_block)
        gold_emb = embed(client, gold_block)
        dist = cosine_distance(curr_emb, gold_emb)
        results[part] = {
            "distance": dist,
            "current_found": True,
            "golden_found": True,
        }

    output = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(output)
    else:
        print(output)


if __name__ == "__main__":  # pragma: no cover
    main()
