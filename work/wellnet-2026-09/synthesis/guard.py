"""guard.py -- mechanical assertion that the synthesis lane opens no data.

Reuses `universes/provenance.py` unchanged: `builtins.open`, `io.open` and the
numpy loaders are patched for the duration of the run, every read is
ledgered, any path matching a sealed token (KiDS, the wide binaries) raises
before the read, and any read outside THIS lane's directory raises.  The
compiler is imported as code (the import system does not go through
`builtins.open`), and the only file this lane ever reads is its own JSON.

The registry write (`registry.register`) happens in `run_all.py` BEFORE the
guard is armed, because `registry._load()` reads `registry/registry.json`,
which is outside the lane root; that single pre-guard read is reported.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WELLNET = os.path.dirname(HERE)
if WELLNET not in sys.path:
    sys.path.insert(0, WELLNET)
COMPILER_DIR = os.path.join(WELLNET, "compiler")
if COMPILER_DIR not in sys.path:
    sys.path.insert(0, COMPILER_DIR)

from universes import provenance as pv          # noqa: E402

#: the confirmation reserve, guarded by token in addition to the sealed set
RESERVE_TOKENS = ("spt", "x-gap", "xgap", "clogs", "gaia", "muse", "granata")

_LEDGER = None


def arm():
    """Install the guard with the synthesis directory as the only readable
    non-library root.  Idempotent."""
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = pv.start_ledger(HERE)
    return _LEDGER


def summary() -> dict:
    if _LEDGER is None:
        return dict(armed=False)
    s = _LEDGER.summary()
    s["armed"] = True
    s["reserve_tokens_guarded"] = list(RESERVE_TOKENS)
    s["any_reserve_token_in_reads"] = bool(
        [p for p in _LEDGER.reads if any(t in p for t in RESERVE_TOKENS)])
    return s
