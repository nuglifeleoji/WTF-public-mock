"""Prepare local benchmark resources without adding datasets to the repository."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


TAU2_REVISION = "0ed2fd8d830a20657d89ae9c2efcc94838aa7129"
TERMINALBENCH_REVISION = "53ff2b87d621bdb97b455671f2bd9728b7d86c11"
SWEBENCH_REVISION = "6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2"


def checkout(url: str, revision: str, destination: Path) -> None:
    """Clone a fixed revision; never reset or overwrite an existing checkout."""
    if destination.exists():
        current = subprocess.check_output(
            ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
        ).strip()
        if current != revision:
            raise ValueError(f"{destination} is at {current}, expected {revision}; use a new data root")
        print(f"Reusing {destination} at {revision}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--no-checkout", url, str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", revision], check=True)


def apply_tau2_overlay(data_root: Path, overlay: Path | None = None) -> None:
    """Restore experiment task definitions after verifying all source hashes."""
    overlay = overlay or Path(__file__).parent / "assets" / "tau2_overlay"
    manifest = json.loads((overlay / "MANIFEST.json").read_text())
    if manifest["upstream_commit"] != TAU2_REVISION:
        raise ValueError("tau2 overlay does not match the pinned checkout")
    writes = []
    for entry in manifest["files"]:
        path = Path(entry["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Invalid tau2 overlay path")
        source = (overlay / path).read_bytes()
        destination = data_root / path
        if hashlib.sha256(source).hexdigest() != entry["sha256"]:
            raise ValueError(f"Corrupt tau2 overlay: {path}")
        current = hashlib.sha256(destination.read_bytes()).hexdigest()
        if current not in {entry["upstream_sha256"], entry["sha256"]}:
            raise ValueError(f"Local tau2 data has unexpected changes: {path}; use a fresh data root")
        writes.append((destination, source))
    for destination, source in writes:
        destination.write_bytes(source)
    print(f"Applied {len(writes)} verified tau2 experiment data files")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=["appworld", "tau2", "swebench", "terminalbench", "browsecomp"])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    root = args.data_root.resolve()
    if args.benchmark == "appworld":
        subprocess.run([sys.executable, "-m", "appworld.cli", "install"], check=True)
        subprocess.run([sys.executable, "-m", "appworld.cli", "download", "data",
                        "--root", str(root / "appworld")], check=True)
    elif args.benchmark == "tau2":
        checkout("https://github.com/sierra-research/tau2-bench.git", TAU2_REVISION, root / "tau2-bench")
        apply_tau2_overlay(root / "tau2-bench" / "data")
    elif args.benchmark == "terminalbench":
        from .patch_runtime import apply_runtime_patches
        apply_runtime_patches("terminalbench")
        checkout("https://github.com/harbor-framework/terminal-bench-2.git", TERMINALBENCH_REVISION,
                 root / "terminalbench" / "terminal-bench-2")
    elif args.benchmark == "swebench":
        from .patch_runtime import apply_runtime_patches
        apply_runtime_patches("swebench")
        from datasets import load_dataset
        data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test", revision=SWEBENCH_REVISION)
        print(f"Cached {len(data)} SWE-bench Lite test tasks at {SWEBENCH_REVISION}")
    else:
        subprocess.run([sys.executable, "-m", "Task.prepare_browsecomp_data", "--output",
                        str(root / "browsecomp" / "browsecomp_plus_decrypted.jsonl"),
                        "--queries-tsv", str(root / "browsecomp" / "queries.tsv")], check=True)
        subprocess.run([sys.executable, "-m", "Task.prepare_browsecomp_corpus", "--output",
                        str(root / "browsecomp" / "corpus.jsonl"), "--index",
                        str(root / "browsecomp" / "indexes" / "bm25")], check=True)


if __name__ == "__main__":
    main()
