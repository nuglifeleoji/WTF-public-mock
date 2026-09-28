#!/usr/bin/env python3
"""Generate official-style BrowseComp topics/qrels files from decrypted JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate queries.tsv, qrel_evidence.txt, and qrel_gold.txt for BrowseComp."
    )
    parser.add_argument(
        "--input",
        default="data/browsecomp/browsecomp_plus_decrypted.jsonl",
        help="Path to the decrypted BrowseComp JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/browsecomp/topics-qrels",
        help="Directory where BrowseComp topics/qrels files will be written.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    return parser.parse_args()


def _iter_records(input_path: Path, max_records: int | None):
    with input_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_records is not None and idx >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"BrowseComp data file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries_path = output_dir / "queries.tsv"
    qrel_evidence_path = output_dir / "qrel_evidence.txt"
    qrel_gold_path = output_dir / "qrel_gold.txt"

    with (
        queries_path.open("w", encoding="utf-8") as queries_out,
        qrel_evidence_path.open("w", encoding="utf-8") as evidence_out,
        qrel_gold_path.open("w", encoding="utf-8") as gold_out,
    ):
        n_queries = 0
        n_evidence = 0
        n_gold = 0
        for record in _iter_records(input_path, args.max_records):
            query_id = str(record.get("query_id", ""))
            query = str(record.get("query", "")).replace("\t", " ").replace("\n", " ").replace("\r", " ")
            queries_out.write(f"{query_id}\t{query}\n")
            n_queries += 1

            for doc in record.get("evidence_docs") or []:
                docid = str(doc.get("docid", ""))
                evidence_out.write(f"{query_id} 0 {docid} 1\n")
                n_evidence += 1

            for doc in record.get("gold_docs") or []:
                docid = str(doc.get("docid", ""))
                gold_out.write(f"{query_id} 0 {docid} 1\n")
                n_gold += 1

    print(f"Wrote {n_queries} queries to {queries_path}")
    print(f"Wrote {n_evidence} evidence qrels to {qrel_evidence_path}")
    print(f"Wrote {n_gold} gold qrels to {qrel_gold_path}")


if __name__ == "__main__":
    main()
