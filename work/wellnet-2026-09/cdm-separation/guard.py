"""guard.py -- mechanical assertion of what this lane opened.

Standing constraints 1 and 2 of the task brief:

  1. KiDS and the wide binaries are SEALED.
  2. SPT, X-GAP, CLoGS, Gaia dynamical products and MUSE/Granata dispersions
     are the CONFIRMATION RESERVE.  This lane is entirely synthetic; assert
     mechanically what you open.

So we do not promise it in prose.  ``universes.provenance`` already patches
``open``/``io.open``/``numpy.load`` and raises on a sealed token or a read
outside the lane root.  This module extends its token list with the
confirmation reserve and pins the lane root at wellnet-2026-09 (this lane must
read the BF universes package and its cached scene library, and nothing else).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LANE_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if LANE_ROOT not in sys.path:
    sys.path.insert(0, LANE_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from universes import provenance as pv        # noqa: E402

# the confirmation reserve (work/wellnet-2026-09/confirmation/reserve.py).
# Touching any of these would spend a one-shot evaluation.
RESERVE_TOKENS = (
    "spt_", "spt-", "sptcl", "south_pole", "southpole",
    "x-gap", "xgap", "x_gap", "clogs", "granata", "muse_", "muse-",
    "gaia_dr", "gaiadr", "gaia_edr",
)


def start(extra_tokens=RESERVE_TOKENS):
    """Install the ledger with the reserve tokens added to the sealed list."""
    pv.SEALED_TOKENS = tuple(pv.SEALED_TOKENS) + tuple(extra_tokens)
    return pv.start_ledger(LANE_ROOT)


def stop():
    s = pv.stop_ledger()
    s["confirmation_reserve_tokens_guarded"] = list(RESERVE_TOKENS)
    s["lane"] = "cdm-separation"
    return s
