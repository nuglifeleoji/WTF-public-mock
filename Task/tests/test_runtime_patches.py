"""Compatibility patch installation only touches synthetic package fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from Task import patch_runtime


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class RuntimePatchTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.assets = self.root / "assets"
        self.assets.mkdir()
        self.original = [b"setting = 'old'\n", b"mode = 'legacy'\n"]
        self.modified = [b"setting = 'new'\n", b"mode = 'compatible'\n"]
        self.paths = [self.root / "first.py", self.root / "second.py"]
        entries = []
        for path, before, after in zip(self.paths, self.original, self.modified):
            path.write_bytes(before)
            entries.append({
                "path": path.name,
                "upstream_sha256": digest(before),
                "patched_sha256": digest(after),
                "replacements": [{"before": before.decode(), "after": after.decode()}],
            })
        self.manifest = {
            "schema_version": 1,
            "packages": {"swebench": {
                "distribution": "fixture", "version": "1.0", "files": entries,
            }},
        }
        self.save_manifest()
        self.distribution = SimpleNamespace(version="1.0", locate_file=lambda path: self.root / path)
        for patcher in (
            mock.patch.object(patch_runtime, "ASSETS", self.assets),
            mock.patch.object(patch_runtime.metadata, "distribution", return_value=self.distribution),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def save_manifest(self) -> None:
        (self.assets / "manifest.json").write_text(json.dumps(self.manifest))

    def test_apply_is_idempotent(self) -> None:
        self.assertEqual(patch_runtime.apply_runtime_patches("swebench"), {
            "applied": 2, "already_patched": 0,
        })
        self.assertEqual([p.read_bytes() for p in self.paths], self.modified)
        self.assertEqual(patch_runtime.apply_runtime_patches("swebench"), {
            "applied": 0, "already_patched": 2,
        })

    def test_mixed_original_and_patched_files_are_supported(self) -> None:
        self.paths[0].write_bytes(self.modified[0])
        self.assertEqual(patch_runtime.apply_runtime_patches("swebench"), {
            "applied": 1, "already_patched": 1,
        })
        self.assertEqual([p.read_bytes() for p in self.paths], self.modified)

    def test_version_mismatch_does_not_modify_files(self) -> None:
        self.distribution.version = "2.0"
        with self.assertRaisesRegex(ValueError, "require fixture==1.0"):
            patch_runtime.apply_runtime_patches("swebench")
        self.assertEqual([p.read_bytes() for p in self.paths], self.original)

    def test_unrecognized_later_file_prevents_all_writes(self) -> None:
        edited = b"user changes\n"
        self.paths[1].write_bytes(edited)
        with self.assertRaisesRegex(ValueError, "unrecognized local changes"):
            patch_runtime.apply_runtime_patches("swebench")
        self.assertEqual(self.paths[0].read_bytes(), self.original[0])
        self.assertEqual(self.paths[1].read_bytes(), edited)

    def test_bad_output_checksum_prevents_all_writes(self) -> None:
        entry = self.manifest["packages"]["swebench"]["files"][1]
        entry["replacements"][0]["after"] = "unexpected replacement\n"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "output checksum mismatch"):
            patch_runtime.apply_runtime_patches("swebench")
        self.assertEqual([p.read_bytes() for p in self.paths], self.original)

    def test_missing_replacement_is_rejected(self) -> None:
        entry = self.manifest["packages"]["swebench"]["files"][1]
        entry["replacements"][0]["before"] = "absent text"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "uniquely match"):
            patch_runtime.apply_runtime_patches("swebench")
        self.assertEqual([p.read_bytes() for p in self.paths], self.original)

    def test_relative_path_escape_is_rejected_before_writes(self) -> None:
        self.manifest["packages"]["swebench"]["files"][1]["path"] = "../other.py"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "Invalid runtime patch path"):
            patch_runtime.apply_runtime_patches("swebench")
        self.assertEqual([p.read_bytes() for p in self.paths], self.original)


if __name__ == "__main__":
    unittest.main()
