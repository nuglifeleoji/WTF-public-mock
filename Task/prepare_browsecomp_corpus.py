#!/usr/bin/env python3
"""Download the BrowseComp-Plus corpus and build a local BM25 index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from .retrieval import build_bm25_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download BrowseComp corpus to JSONL and build a BM25 index."
    )
    parser.add_argument(
        "--output",
        default="data/browsecomp/corpus.jsonl",
        help="Path to the local corpus JSONL file.",
    )
    parser.add_argument(
        "--index",
        default="data/browsecomp/indexes/bm25",
        help="Path to the BrowseComp BM25 index output.",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "pyserini"],
        default="pyserini",
        help="Index backend to build.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Threads to use while building the index.",
    )
    parser.add_argument("--revision", default="main", help="Hugging Face dataset revision or commit.")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Download the corpus JSONL only, without building the BM25 index.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and args.max_records is None:
        print(f"Reusing existing BrowseComp corpus at {output_path}")
    else:
        dataset = load_dataset("Tevatron/browsecomp-plus-corpus", split="train", revision=args.revision)
        total = len(dataset) if args.max_records is None else min(len(dataset), args.max_records)

        with output_path.open("w", encoding="utf-8") as out:
            for idx, row in enumerate(dataset):
                if args.max_records is not None and idx >= args.max_records:
                    break
                payload = {
                    "docid": str(row.get("docid", "")),
                    "text": str(row.get("text", "")),
                    "url": str(row.get("url", "")),
                }
                json.dump(payload, out, ensure_ascii=False)
                out.write("\n")
                if (idx + 1) % max(1, total // 10 or 1) == 0:
                    print(f"Wrote {idx + 1}/{total} corpus records", flush=True)

        print(f"Wrote BrowseComp corpus to {output_path}")
    if not args.skip_index:
        build_bm25_index(
            str(output_path),
            args.index,
            backend=args.backend,
            threads=args.threads,
        )
        print(f"Wrote {args.backend} BM25 index to {args.index}")


if __name__ == "__main__":
    main()
