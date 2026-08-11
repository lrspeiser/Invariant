from __future__ import annotations

import os


def pid_alive(pid: int | None) -> bool:
    """Check process liveness without using destructive signal-zero semantics on Windows."""

    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        try:
            import psutil
        except ImportError:
            return False
        try:
            process = psutil.Process(pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
