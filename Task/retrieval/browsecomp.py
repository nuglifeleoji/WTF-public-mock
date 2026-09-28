"""Local full-corpus retrieval utilities for BrowseComp-Plus."""

from __future__ import annotations

import json
import math
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower())


def _load_corpus_records(corpus_path: Path) -> list[dict]:
    records: list[dict] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append(
                {
                    "docid": str(row.get("docid", "")),
                    "text": str(row.get("text", "")),
                    "url": str(row.get("url", "")),
                }
            )
    return records


def build_local_bm25_index(corpus_path: str, index_path: str) -> dict:
    corpus_file = Path(corpus_path)
    index_file = Path(index_path)
    if not corpus_file.exists():
        raise FileNotFoundError(f"BrowseComp corpus file not found: {corpus_file}")

    records = _load_corpus_records(corpus_file)
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    doc_lengths: list[int] = []
    docids: list[str] = []
    texts: list[str] = []
    urls: list[str] = []

    for idx, record in enumerate(records):
        tokens = tokenize(record["text"])
        freqs = Counter(tokens)
        doc_lengths.append(len(tokens))
        docids.append(record["docid"])
        texts.append(record["text"])
        urls.append(record["url"])
        for term, tf in freqs.items():
            postings[term].append((idx, tf))

    avg_doc_len = (sum(doc_lengths) / len(doc_lengths)) if doc_lengths else 0.0
    payload = {
        "docids": docids,
        "texts": texts,
        "urls": urls,
        "doc_lengths": doc_lengths,
        "avg_doc_len": avg_doc_len,
        "num_docs": len(docids),
        "postings": dict(postings),
    }
    index_file.parent.mkdir(parents=True, exist_ok=True)
    with index_file.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def build_pyserini_bm25_index(
    corpus_path: str,
    index_path: str,
    *,
    threads: int = 8,
) -> None:
    corpus_file = Path(corpus_path)
    index_dir = Path(index_path)
    if not corpus_file.exists():
        raise FileNotFoundError(f"BrowseComp corpus file not found: {corpus_file}")

    if shutil.which("java") is None:
        raise RuntimeError(
            "Java is required to build a Pyserini BM25 index. "
            "Install a JDK such as openjdk-21-jdk first."
        )

    try:
        import pyserini  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Pyserini is required to build the official BM25 index. "
            "Install the `pyserini` package first."
        ) from exc

    index_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="browsecomp_pyserini_") as tmpdir:
        collection_dir = Path(tmpdir) / "json_collection"
        collection_dir.mkdir(parents=True, exist_ok=True)
        collection_path = collection_dir / "docs.jsonl"

        with corpus_file.open("r", encoding="utf-8") as src, collection_path.open(
            "w", encoding="utf-8"
        ) as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                converted = {
                    "id": str(row.get("docid", "")),
                    "contents": str(row.get("text", "")),
                    "url": str(row.get("url", "")),
                }
                json.dump(converted, dst, ensure_ascii=False)
                dst.write("\n")

        command = [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            str(collection_dir),
            "--index",
            str(index_dir),
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            str(threads),
            "--storeRaw",
        ]
        subprocess.run(command, check=True)


def build_bm25_index(
    corpus_path: str,
    index_path: str,
    *,
    backend: str = "local",
    threads: int = 8,
):
    if backend == "local":
        return build_local_bm25_index(corpus_path, index_path)
    if backend == "pyserini":
        return build_pyserini_bm25_index(corpus_path, index_path, threads=threads)
    raise ValueError(f"Unknown BrowseComp BM25 backend: {backend!r}")


class _LocalBM25Retriever:
    def __init__(
        self,
        corpus_path: str,
        index_path: str,
        k1: float = 1.2,
        b: float = 0.75,
        auto_build: bool = True,
    ):
        self.corpus_path = Path(corpus_path)
        self.index_path = Path(index_path)
        self.k1 = k1
        self.b = b

        if self.index_path.exists():
            with self.index_path.open("rb") as f:
                payload = pickle.load(f)
        else:
            if not auto_build:
                raise FileNotFoundError(
                    f"BrowseComp index not found: {self.index_path}. "
                    "Run python -m Task.prepare_browsecomp_corpus first."
                )
            payload = build_local_bm25_index(str(self.corpus_path), str(self.index_path))

        self._docids: list[str] = payload["docids"]
        self._texts: list[str] = payload["texts"]
        self._urls: list[str] = payload["urls"]
        self._doc_lengths: list[int] = payload["doc_lengths"]
        self._avg_doc_len: float = float(payload["avg_doc_len"])
        self._num_docs: int = int(payload["num_docs"])
        self._postings: dict[str, list[tuple[int, int]]] = payload["postings"]
        self._docid_to_idx = {docid: idx for idx, docid in enumerate(self._docids)}

    def search(self, query: str, k: int = 5) -> list[dict]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: dict[int, float] = defaultdict(float)
        unique_terms = set(query_terms)
        for term in unique_terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1.0 + (self._num_docs - df + 0.5) / (df + 0.5))
            for doc_idx, tf in postings:
                doc_len = self._doc_lengths[doc_idx] or 1
                denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / max(self._avg_doc_len, 1.0))
                scores[doc_idx] += idf * ((tf * (self.k1 + 1.0)) / denom)

        top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        return [self._doc_at(doc_idx, score) for doc_idx, score in top]

    def get_document(self, docid: str) -> Optional[dict]:
        idx = self._docid_to_idx.get(str(docid))
        if idx is None:
            return None
        return self._doc_at(idx, score=None)

    def _doc_at(self, doc_idx: int, score: Optional[float]) -> dict:
        payload = {
            "docid": self._docids[doc_idx],
            "text": self._texts[doc_idx],
            "url": self._urls[doc_idx],
        }
        if score is not None:
            payload["score"] = float(score)
        return payload


class _PyseriniBM25Retriever:
    def __init__(
        self,
        corpus_path: str,
        index_path: str,
        auto_build: bool = True,
        build_threads: int = 8,
    ):
        self.corpus_path = Path(corpus_path)
        self.index_path = Path(index_path)

        try:
            from pyserini.search.lucene import LuceneSearcher
        except Exception as exc:
            raise RuntimeError(
                "Pyserini is required for the official BrowseComp BM25 backend."
            ) from exc

        if self.index_path.exists():
            pass
        else:
            if not auto_build:
                raise FileNotFoundError(
                    f"BrowseComp Pyserini index not found: {self.index_path}. "
                    "Run python -m Task.prepare_browsecomp_corpus first."
                )
            build_pyserini_bm25_index(
                str(self.corpus_path),
                str(self.index_path),
                threads=build_threads,
            )

        self._searcher = LuceneSearcher(str(self.index_path))

    def search(self, query: str, k: int = 5) -> list[dict]:
        hits = self._searcher.search(query, k)
        results = []
        for hit in hits:
            raw_payload = self._extract_raw_payload(hit.docid, lucene_doc=getattr(hit, "lucene_document", None))
            results.append(
                {
                    "docid": str(hit.docid),
                    "text": str(raw_payload.get("contents") or raw_payload.get("text") or ""),
                    "url": str(raw_payload.get("url") or ""),
                    "score": float(hit.score),
                }
            )
        return results

    def get_document(self, docid: str) -> Optional[dict]:
        doc = self._searcher.doc(str(docid))
        if doc is None:
            return None
        raw_payload = self._extract_raw_payload(str(docid), lucene_doc=doc.lucene_document)
        return {
            "docid": str(docid),
            "text": str(raw_payload.get("contents") or raw_payload.get("text") or ""),
            "url": str(raw_payload.get("url") or ""),
        }

    def _extract_raw_payload(self, docid: str, lucene_doc=None) -> dict[str, Any]:
        raw_text = None
        if lucene_doc is not None and hasattr(lucene_doc, "get"):
            raw_text = lucene_doc.get("raw")
        if raw_text is None:
            doc = self._searcher.doc(str(docid))
            if doc is not None and hasattr(doc, "raw"):
                raw_text = doc.raw()
        if not raw_text:
            return {}
        try:
            return json.loads(raw_text)
        except Exception:
            return {"contents": str(raw_text)}


class BrowseCompCorpusRetriever:
    """Retriever wrapper over either local BM25 or official Pyserini BM25."""

    def __init__(
        self,
        corpus_path: str,
        index_path: str,
        backend: str = "auto",
        k1: float = 1.2,
        b: float = 0.75,
        auto_build: bool = True,
        build_threads: int = 8,
    ):
        self.corpus_path = Path(corpus_path)
        self.index_path = Path(index_path)
        self.backend = self._resolve_backend(backend, self.index_path)

        if self.backend == "local":
            self._impl = _LocalBM25Retriever(
                corpus_path=corpus_path,
                index_path=index_path,
                k1=k1,
                b=b,
                auto_build=auto_build,
            )
        elif self.backend == "pyserini":
            self._impl = _PyseriniBM25Retriever(
                corpus_path=corpus_path,
                index_path=index_path,
                auto_build=auto_build,
                build_threads=build_threads,
            )
        else:
            raise ValueError(f"Unknown BrowseComp retriever backend: {self.backend!r}")

    def search(self, query: str, k: int = 5) -> list[dict]:
        return self._impl.search(query, k)

    def get_document(self, docid: str) -> Optional[dict]:
        return self._impl.get_document(docid)

    @staticmethod
    def _resolve_backend(backend: str, index_path: Path) -> str:
        if backend != "auto":
            if backend not in {"local", "pyserini"}:
                raise ValueError(
                    f"Unsupported BrowseComp retriever backend: {backend!r}. "
                    "Expected 'auto', 'local', or 'pyserini'."
                )
            return backend

        if index_path.suffix == ".pkl":
            return "local"
        return "pyserini"
