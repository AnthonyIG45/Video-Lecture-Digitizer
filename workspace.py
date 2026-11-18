"""
Throwaway working storage for a prototype: everything a Streamlit instance writes (downloaded
video, frames, audio, transcripts) lives under one temporary folder that is deleted when the
process exits. If a previous instance was killed before it could clean up (e.g. a closed
terminal), the next start sweeps up its leftovers.
"""
import atexit
import os
import shutil
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path

import psutil

SESSIONS_ROOT = Path(tempfile.gettempdir()) / "lecture_pdf_sessions"


def _sweep_stale() -> None:
    """Removes workspaces left by instances that are no longer running. Folders are named by PID."""
    if not SESSIONS_ROOT.exists():
        return
    for entry in SESSIONS_ROOT.iterdir():
        if entry.name.isdigit() and int(entry.name) != os.getpid() and not psutil.pid_exists(int(entry.name)):
            shutil.rmtree(entry, ignore_errors=True)


@lru_cache(maxsize=1)
def session_workspace() -> Path:
    """This process's temporary workspace; created on first use and removed at interpreter exit."""
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    _sweep_stale()

    workspace = SESSIONS_ROOT / str(os.getpid())
    shutil.rmtree(workspace, ignore_errors=True)  # a previous process may have had this PID
    workspace.mkdir()
    atexit.register(shutil.rmtree, workspace, ignore_errors=True)
    return workspace


def new_run_dir() -> Path:
    """A fresh sub-folder of the workspace for one processing run."""
    run_dir = session_workspace() / uuid.uuid4().hex[:8]
    run_dir.mkdir()
    return run_dir
