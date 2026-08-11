from __future__ import annotations

import os

from sigma_theory_compiler.process_health import pid_alive


def test_pid_alive_is_nondestructive_for_current_process() -> None:
    assert pid_alive(os.getpid()) is True
    assert pid_alive(None) is False
    assert pid_alive(0) is False
    assert pid_alive(-1) is False
    assert pid_alive(2**31 - 1) is False
