"""Restore the experiment's small compatibility fixes in pinned dependencies.

Every target must match either the published wheel or the recorded patched
file. All files are checked before any write; no third-party module is imported.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path


ASSETS = Path(__file__).parent / "assets" / "runtime_patches"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def apply_runtime_patches(benchmark: str) -> dict[str, int]:
    """Apply recorded fixes to the installed pinned SWE-bench or Harbor package."""
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported runtime patch manifest")
    try:
        spec = manifest["packages"][benchmark]
    except KeyError as exc:
        raise ValueError(f"No runtime patches for benchmark {benchmark!r}") from exc

    distribution = metadata.distribution(spec["distribution"])
    if distribution.version != spec["version"]:
        raise ValueError(
            f"Runtime patches require {spec['distribution']}=={spec['version']}; "
            f"found {distribution.version}. Install the pinned benchmark extra."
        )

    writes: list[tuple[Path, bytes]] = []
    unchanged = 0
    for entry in spec["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid runtime patch path: {relative}")
        path = Path(distribution.locate_file(relative))
        content = path.read_bytes()
        digest = _sha256(content)
        if digest == entry["patched_sha256"]:
            unchanged += 1
            continue
        if digest != entry["upstream_sha256"]:
            raise ValueError(
                f"Refusing to patch {path}: unrecognized local changes. "
                "Use a fresh environment with the pinned benchmark extra."
            )
        patched = content.decode("utf-8")
        for replacement in entry["replacements"]:
            before = replacement["before"]
            if not before or patched.count(before) != 1:
                raise ValueError(f"Runtime patch does not uniquely match {path}")
            patched = patched.replace(before, replacement["after"], 1)
        output = patched.encode("utf-8")
        if _sha256(output) != entry["patched_sha256"]:
            raise ValueError(f"Runtime patch output checksum mismatch for {path}")
        writes.append((path, output))

    # Validation above must finish for every file before modifying the install.
    for path, content in writes:
        path.write_bytes(content)
    print(
        f"{spec['distribution']}=={spec['version']}: "
        f"applied {len(writes)} compatibility patches; {unchanged} already present"
    )
    return {"applied": len(writes), "already_patched": unchanged}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=["swebench", "terminalbench"])
    args = parser.parse_args(argv)
    apply_runtime_patches(args.benchmark)


if __name__ == "__main__":
    main()
