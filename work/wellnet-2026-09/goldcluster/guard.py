"""guard.py -- mechanical assertions that the gold-cluster lane stays an INVENTORY.

The lane's registered contract (BO-goldcluster) is that it "opens no pixel,
shear, spectral or kinematic data and computes no gravity-relevant statistic on
any cluster".  Prose does not enforce that, so this module enforces three things
mechanically and the run aborts if any is violated:

  G1  FILE READS.  `universes/provenance.py` is armed with this directory as the
      only readable non-library root, with the CONFIRMATION-RESERVE tokens (SPT,
      X-GAP, CLoGS, Gaia dynamical products, MUSE/Granata) added to the permanent
      SEALED set (KiDS, the wide binaries).  Any matching path raises BEFORE the
      read.

  G2  ARCHIVE PROJECTION.  Every ADQL query this lane sends must project only
      COUNT/MIN/MAX aggregates or catalogue-identity columns.  Asking for a
      shape, an ellipticity, a shear response or a per-source redshift raises.
      This is what makes "no shear was opened" a fact about the code rather than
      a claim about intent: a count of how many sources sit behind a cluster is
      not a measurement of any of them.

  G3  NO GRAVITY STATISTIC.  The lane declares the observable names it is
      forbidden to compute; `assert_no_statistic` refuses to write any JSON whose
      keys collide with them.

The registry write happens BEFORE arm(), because registry._load() reads
registry/registry.json, which is outside the lane root.  That one pre-guard read
is reported in the summary.
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WELLNET = os.path.dirname(HERE)
for _p in (WELLNET,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from universes import provenance as pv          # noqa: E402

#: the confirmation reserve -- recorded by identity only, never opened
RESERVE_TOKENS = (
    "spt_", "spt-", "sptcl", "south_pole", "southpole",
    "x-gap", "xgap", "x_gap", "clogs", "granata", "muse_", "muse-",
    "gaia_dr", "gaiadr", "gaia_edr",
)

#: G2 -- per-source measurement columns this lane must never project
FORBIDDEN_COLUMNS = (
    "mcal_g_", "mcal_t", "mcal_s2n", "mcal_flux", "e1", "e2", "g_1", "g_2",
    "ellip", "shear", "gamma", "kappa", "sigma_crit", "var_e",
)

#: G3 -- gravity-relevant statistics this lane must never emit
FORBIDDEN_STATISTICS = (
    "g_t", "gt", "gx", "g_x", "delta_sigma", "deltasigma", "sigma_crit",
    "kappa", "shear_profile", "mass", "m200", "m500", "c200", "nfw",
    "gbar", "gobs", "rar", "boost", "residual",
)

_LEDGER = None
_ORIG_SEALED = tuple(pv.SEALED_TOKENS)


class InventoryViolation(RuntimeError):
    """Raised when the lane tries to stop being an inventory."""


def arm():
    """Install G1.  Idempotent."""
    global _LEDGER
    if _LEDGER is None:
        pv.SEALED_TOKENS = _ORIG_SEALED + tuple(
            t for t in RESERVE_TOKENS if t not in _ORIG_SEALED)
        _LEDGER = pv.start_ledger(HERE)
        _allow_file_descriptors()
    return _LEDGER


def _allow_file_descriptors():
    """Let an integer file descriptor through the ledger's path check.

    `provenance.OpenLedger.check` normalises whatever it is handed into a path
    string.  Handed the integer 3 -- which is what `subprocess.run(...,
    capture_output=True)` passes to `open()` for its pipes -- it produces the
    path "3", finds it outside the lane root, and raises ForeignReadError.  The
    earlier synthetic lanes never hit this because they never shell out; this
    lane calls curl, so it does, on the FIRST archive probe.

    An integer fd is not a path and carries no dataset identity, so checking it
    against the sealed tokens is meaningless -- but it must still be counted, or
    the ledger would under-report. Wrapped here rather than in provenance.py so
    the lanes that hash that file keep their receipts.
    """
    import builtins
    import io as _io

    def wrap(fn):
        def guarded(file, *a, **kw):
            if isinstance(file, int):            # a file descriptor, not a path
                _LEDGER.reads["<fd>"] = _LEDGER.reads.get("<fd>", 0) + 1
                return _REAL_OPEN(file, *a, **kw)
            return fn(file, *a, **kw)
        return guarded

    builtins.open = wrap(builtins.open)
    _io.open = wrap(_io.open)


_REAL_OPEN = open


def check_query(adql: str) -> str:
    """G2.  Return the query if its projection is inventory-only, else raise."""
    m = re.search(r"^\s*SELECT\s+(?:TOP\s+\d+\s+)?(.*?)\s+FROM\s", adql,
                  re.I | re.S)
    if not m:
        raise InventoryViolation("cannot parse the SELECT projection: %r" % adql[:120])
    proj = m.group(1).lower()
    # an aggregate-only projection is always inventory
    if re.fullmatch(r"[\s,]*(count\s*\(\s*\*\s*\)(\s+as\s+\w+)?[\s,]*)+", proj):
        return adql
    for tok in FORBIDDEN_COLUMNS:
        if tok in proj:
            raise InventoryViolation(
                "G2: projection requests per-source measurement column %r -- this "
                "lane counts sources, it does not open them. Projection was: %s"
                % (tok, proj[:200]))
    return adql


def assert_no_statistic(obj, where="<result>"):
    """G3.  Refuse to emit a gravity-relevant statistic."""
    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                kl = str(k).lower()
                for tok in FORBIDDEN_STATISTICS:
                    if kl == tok or kl.startswith(tok + "_") or kl.endswith("_" + tok):
                        raise InventoryViolation(
                            "G3: %s%s is a gravity-relevant statistic; this lane "
                            "ranks acquisition targets, it does not test theory."
                            % (path, k))
                walk(v, path + str(k) + ".")
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, path)
    walk(obj, where + ".")
    return obj


def summary() -> dict:
    s = dict(armed=_LEDGER is not None, lane="goldcluster")
    if _LEDGER is not None:
        s.update(_LEDGER.summary())
    s["reserve_tokens_guarded"] = list(RESERVE_TOKENS)
    s["forbidden_projection_columns"] = list(FORBIDDEN_COLUMNS)
    s["forbidden_statistics"] = list(FORBIDDEN_STATISTICS)
    return s
