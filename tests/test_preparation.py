"""Preparation commands are explicit and existing task checkouts are preserved."""

from pathlib import Path
from unittest.mock import Mock
import hashlib
import json

import pytest

from Task import prepare


def test_new_checkout_uses_pinned_revision(tmp_path, monkeypatch):
    run = Mock()
    monkeypatch.setattr(prepare.subprocess, "run", run)
    destination = tmp_path / "tasks"
    prepare.checkout("https://example.invalid/tasks.git", "fixed-revision", destination)
    assert run.call_args_list[0].args[0] == ["git", "clone", "--no-checkout", "https://example.invalid/tasks.git", str(destination)]
    assert run.call_args_list[1].args[0][-3:] == ["checkout", "--detach", "fixed-revision"]


def test_existing_checkout_at_different_revision_is_not_modified(tmp_path, monkeypatch):
    run = Mock()
    monkeypatch.setattr(prepare.subprocess, "run", run)
    monkeypatch.setattr(prepare.subprocess, "check_output", lambda *a, **kw: "other-revision\n")
    with pytest.raises(ValueError, match="expected fixed-revision"):
        prepare.checkout("https://example.invalid/tasks.git", "fixed-revision", tmp_path)
    run.assert_not_called()


def test_overlay_verifies_and_restores_experiment_data_idempotently(tmp_path):
    overlay = tmp_path / "overlay"
    data = tmp_path / "data"
    overlay.mkdir()
    data.mkdir()
    original, patched = b"upstream", b"experiment data"
    manifest = {"upstream_commit": prepare.TAU2_REVISION, "files": [{
        "path": "tasks.json", "sha256": hashlib.sha256(patched).hexdigest(),
        "upstream_sha256": hashlib.sha256(original).hexdigest(),
    }]}
    (overlay / "MANIFEST.json").write_text(json.dumps(manifest))
    (overlay / "tasks.json").write_bytes(patched)
    (data / "tasks.json").write_bytes(original)
    prepare.apply_tau2_overlay(data, overlay)
    assert (data / "tasks.json").read_bytes() == patched
    prepare.apply_tau2_overlay(data, overlay)
    assert (data / "tasks.json").read_bytes() == patched
    (data / "tasks.json").write_bytes(b"local edits")
    with pytest.raises(ValueError, match="unexpected changes"):
        prepare.apply_tau2_overlay(data, overlay)
    assert (data / "tasks.json").read_bytes() == b"local edits"
