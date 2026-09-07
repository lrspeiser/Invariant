"""guard.py -- mechanical assertion that the bridge lane opens no data.

Reuses `universes/provenance.py` unchanged (as the synthesis and cdm-separation
lanes did): `builtins.open`, `io.open` and the numpy loaders are patched for the
duration of the run, every read is ledgered, any path matching a SEALED token
(KiDS, the wide binaries) or a CONFIRMATION-RESERVE token (SPT, X-GAP, CLoGS,
Gaia dynamical products, MUSE/Granata) raises BEFORE the read, and any read
outside THIS lane's directory raises.  The compiler, the synthesis modules and
the universes package are imported as code (imports do not go through
`builtins.open`); the only files this lane ever reads are its own JSONs.

The registry write (`registry.register`) happens in `run_all.py` BEFORE the
guard is armed, because `registry._load()` reads `registry/registry.json`,
which is outside the lane root; that single pre-guard read is reported.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WELLNET = os.path.dirname(HERE)
for _p in (WELLNET, os.path.join(WELLNET, "compiler"),
           os.path.join(WELLNET, "synthesis"), os.path.join(WELLNET, "stage4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from universes import provenance as pv          # noqa: E402

#: the confirmation reserve, guarded by token in addition to the sealed set
#: (union of the synthesis and cdm-separation lanes' lists)
RESERVE_TOKENS = (
    "spt_", "spt-", "sptcl", "south_pole", "southpole",
    "x-gap", "xgap", "x_gap", "clogs", "granata", "muse_", "muse-",
    "gaia_dr", "gaiadr", "gaia_edr",
)

_LEDGER = None
_ORIG_SEALED = tuple(pv.SEALED_TOKENS)


def arm():
    """Install the guard with the bridge directory as the only readable
    non-library root, with the reserve tokens added to the sealed list.
    Idempotent."""
    global _LEDGER
    if _LEDGER is None:
        pv.SEALED_TOKENS = _ORIG_SEALED + tuple(
            t for t in RESERVE_TOKENS if t not in _ORIG_SEALED)
        _LEDGER = pv.start_ledger(HERE)
    return _LEDGER


def summary() -> dict:
    if _LEDGER is None:
        return dict(armed=False)
    s = _LEDGER.summary()
    s["armed"] = True
    s["lane"] = "bridge"
    s["reserve_tokens_guarded"] = list(RESERVE_TOKENS)
    s["any_reserve_token_in_reads"] = bool(
        [p for p in _LEDGER.reads if any(t in p for t in RESERVE_TOKENS)])
    return s
