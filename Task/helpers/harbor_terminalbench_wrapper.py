#!/usr/bin/env python3
"""Wrapper around Harbor CLI for Terminal-Bench runs.

This keeps the Harbor command-line interface unchanged while allowing targeted
runtime patches for known remote-agent issues. The current patch trims the
Codex agent log upload on non-mounted environments so Harbor does not spend a
long time re-uploading Codex cache and temp files before verification.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from harbor.cli.main import app
from harbor.models.trial.paths import EnvironmentPaths
from harbor.trial.trial import Trial

_FILTER_ENV_VAR = "WTF_FILTER_CODEX_AGENT_UPLOADS"
_KEEP_TOP_LEVEL_FILES = {
    "codex.txt",
    "config.toml",
    "installation_id",
    "trajectory.json",
}
_KEEP_DIRECTORIES = {"sessions"}
_ORIGINAL_MAYBE_DOWNLOAD_LOGS = Trial._maybe_download_logs
_ORIGINAL_MAYBE_UPLOAD_AGENT_LOGS = Trial._maybe_upload_agent_logs


def _should_filter_codex_agent_logs(trial: Trial) -> bool:
    if os.environ.get(_FILTER_ENV_VAR) != "1":
        return False
    if trial._environment.is_mounted:
        return False
    return trial._agent.__class__.__name__.lower() == "codex"


def _copy_filtered_agent_logs(source_dir: Path, target_dir: Path) -> bool:
    copied_any = False

    for file_name in _KEEP_TOP_LEVEL_FILES:
        source_path = source_dir / file_name
        if not source_path.is_file():
            continue
        destination = target_dir / file_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        copied_any = True

    for directory_name in _KEEP_DIRECTORIES:
        source_path = source_dir / directory_name
        if not source_path.exists():
            continue
        destination = target_dir / directory_name
        shutil.copytree(source_path, destination, dirs_exist_ok=True)
        copied_any = True

    return copied_any


async def _download_filtered_agent_logs(trial: Trial, target_dir: Path) -> bool:
    copied_any = False

    for file_name in _KEEP_TOP_LEVEL_FILES:
        source_path = (EnvironmentPaths.agent_dir / file_name).as_posix()
        target_path = target_dir / file_name
        try:
            await trial._environment.download_file(source_path=source_path, target_path=target_path)
        except Exception:
            continue
        copied_any = True

    for directory_name in _KEEP_DIRECTORIES:
        source_dir = (EnvironmentPaths.agent_dir / directory_name).as_posix()
        target_path = target_dir / directory_name
        try:
            await trial._environment.download_dir(source_dir=source_dir, target_dir=target_path)
        except Exception:
            continue
        copied_any = True

    return copied_any


async def _patched_maybe_download_logs(
    self: Trial,
    source_dir: str,
    target_dir: Path,
) -> None:
    should_filter = (
        _should_filter_codex_agent_logs(self)
        and source_dir == EnvironmentPaths.agent_dir.as_posix()
        and target_dir == self._trial_paths.agent_dir
    )
    if not should_filter:
        await _ORIGINAL_MAYBE_DOWNLOAD_LOGS(self, source_dir, target_dir)
        return

    if self._environment.is_mounted or self._are_agent_logs_downloaded:
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    copied_any = await _download_filtered_agent_logs(self, target_dir)
    if not copied_any:
        self._logger.error(f"Failed to download filtered logs to {target_dir}")
        return

    self._are_agent_logs_downloaded = True


async def _patched_maybe_upload_agent_logs(self: Trial) -> None:
    if not _should_filter_codex_agent_logs(self):
        await _ORIGINAL_MAYBE_UPLOAD_AGENT_LOGS(self)
        return

    with tempfile.TemporaryDirectory(prefix="terminalbench_codex_logs_") as tmpdir:
        staged_logs_dir = Path(tmpdir)
        copied_any = _copy_filtered_agent_logs(self._trial_paths.agent_dir, staged_logs_dir)
        if not copied_any:
            return

        try:
            await self._environment.upload_dir(
                source_dir=staged_logs_dir,
                target_dir=EnvironmentPaths.agent_dir.as_posix(),
            )
        except Exception:
            self._logger.error("Failed to upload filtered agent logs back to environment")


def main() -> None:
    Trial._maybe_download_logs = _patched_maybe_download_logs
    Trial._maybe_upload_agent_logs = _patched_maybe_upload_agent_logs
    sys.argv[0] = "harbor"
    sys.exit(app())


if __name__ == "__main__":
    main()
