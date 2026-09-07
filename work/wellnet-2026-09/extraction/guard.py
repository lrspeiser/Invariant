"""guard.py -- mechanical assertion of what this lane opened.

Standing constraints of the Extraction Lane brief:

  1. KiDS and the wide binaries are SEALED.
  2. SPT, X-GAP, CLoGS, Gaia dynamical products and MUSE/Granata dispersions
     are the CONFIRMATION RESERVE.
  3. The lane is entirely synthetic: assert mechanically what you open.

``universes.provenance`` patches ``open``/``io.open``/``numpy.load`` and
raises on a sealed token or on a read outside the lane root.  This module
extends the token list with the confirmation reserve and pins the lane root at
``wellnet-2026-09`` (this lane reads the BF universes package, its cached
scene library, and BK's ``cdm-separation/forward.py`` -- and nothing else).

The guard is installed in EVERY worker process, not only the parent: Run BF's
ledger covered the parent only, and a read inside a worker would not have
raised.  Here it raises in the worker and the batch fails loudly.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LANE_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (LANE_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
for p in (os.path.join(LANE_ROOT, "cdm-separation"), os.path.join(LANE_ROOT, "stage4")):
    if p not in sys.path:
        sys.path.append(p)        # LAST: their worker.py / guard.py must not shadow this lane's

from universes import provenance as pv        # noqa: E402

RESERVE_TOKENS = (
    "spt_", "spt-", "sptcl", "south_pole", "southpole",
    "x-gap", "xgap", "x_gap", "clogs", "granata", "muse_", "muse-",
    "gaia_dr", "gaiadr", "gaia_edr",
)


def start(extra_tokens=RESERVE_TOKENS):
    """Install the ledger with the reserve tokens added to the sealed list."""
    if not any(t in pv.SEALED_TOKENS for t in extra_tokens):
        pv.SEALED_TOKENS = tuple(pv.SEALED_TOKENS) + tuple(extra_tokens)
    return pv.start_ledger(LANE_ROOT)


def stop():
    s = pv.stop_ledger()
    s["confirmation_reserve_tokens_guarded"] = list(RESERVE_TOKENS)
    s["lane"] = "extraction"
    return s
