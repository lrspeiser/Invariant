"""SPARC ingest, quality cuts, and the frozen train/validation/blind split.

Implements steps 1-3 of Run A in the anisotropic-void test program.

The quality cuts and the split are defined HERE, in code, and are fixed before
any model residual is computed. That ordering is the whole point of the step:
a cut chosen after seeing residuals is a fitted parameter wearing a disguise.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

ROOT = "C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
TABLE1 = ROOT + "runs/gravity/roadmap/item-02-shape-anisotropy-v1-source/sparc_table1.tsv"
CURVES = ROOT + "configs/sparc_rotation_curves_full_v1.json"

G = 6.674e-11
KPC = 3.0856775814913673e19
KMS = 1e3
MSUN = 1.98892e30
LSUN = 1.0

#: Quality cuts, declared before any residual is examined.
CUTS = dict(
    max_qual=2,        # SPARC flag 3 is "low"; the program's step 2
    min_inclination=30.0,   # below this, sin(i) deprojection is unstable
    min_points=5,      # need enough radii to constrain anything
    require_vflat=True,     # BTFR and h_eff both need the flat velocity
)

#: Split fractions and the stratification variables, from the program's
#: "Train, validation, and blind split" section. Split is by WHOLE GALAXY.
SPLIT = dict(train=0.60, validation=0.20, blind=0.20, seed=20260903)
STRATIFY = ("Mb", "SBeff", "fgas", "Vflat", "Qual")


@dataclass
class Galaxy:
    name: str
    # per-point, as tabulated
    R0: np.ndarray            # kpc, at the catalogue distance
    Vobs0: np.ndarray         # km/s, at the catalogue inclination
    eV: np.ndarray            # km/s
    Vgas: np.ndarray          # km/s, signed
    Vdisk: np.ndarray         # km/s, at M/L = 1
    Vbul: np.ndarray          # km/s, at M/L = 1
    # per-galaxy
    D0: float                 # Mpc
    eD: float
    i0: float                 # deg
    ei: float
    L36: float                # 1e9 Lsun
    Reff: float               # kpc
    SBeff: float              # Lsun/pc^2
    Rdisk: float              # kpc
    MHI: float                # 1e9 Msun
    Vflat: float              # km/s
    eVflat: float
    Qual: int
    Type: int
    split: str = ""
    Mb: float = field(default=0.0)
    fgas: float = field(default=0.0)

    def __len__(self):
        return len(self.R0)


def _f(tok, default=float("nan")):
    tok = tok.strip()
    if not tok:
        return default
    try:
        return float(tok)
    except ValueError:
        return default


def load_table1() -> dict[str, dict]:
    """Galaxy-level properties from SPARC Table 1 (Lelli+ 2016)."""
    out: dict[str, dict] = {}
    with open(TABLE1, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("-"):
                continue
            q = line.rstrip("\n").split("\t")
            if len(q) < 20 or not q[0].strip().isdigit():
                continue
            name = q[1].strip()
            out[name] = dict(
                Type=int(_f(q[3], -1)), D0=_f(q[4]), eD=_f(q[5]),
                i0=_f(q[7]), ei=_f(q[8]), L36=_f(q[9]),
                Reff=_f(q[11]), SBeff=_f(q[12]), Rdisk=_f(q[13]),
                MHI=_f(q[15]), Vflat=_f(q[17]), eVflat=_f(q[18]),
                Qual=int(_f(q[19], 3)),
            )
    return out


def load_curves() -> dict[str, list]:
    cfg = json.load(open(CURVES, encoding="utf-8"))
    return {g["name"]: g["rows"] for g in cfg["galaxies"]}


def ingest(verbose: bool = True) -> list[Galaxy]:
    """Join the two tables, apply the declared cuts, and report attrition."""
    t1, cur = load_table1(), load_curves()
    kept: list[Galaxy] = []
    drop = dict(no_table1=0, qual=0, inclination=0, points=0, vflat=0)
    for name, rows in cur.items():
        meta = t1.get(name)
        if meta is None:
            drop["no_table1"] += 1
            continue
        if meta["Qual"] > CUTS["max_qual"]:
            drop["qual"] += 1
            continue
        if not (meta["i0"] >= CUTS["min_inclination"]):
            drop["inclination"] += 1
            continue
        if CUTS["require_vflat"] and not (meta["Vflat"] > 0):
            drop["vflat"] += 1
            continue
        R, V, E, Vg, Vd, Vb = [], [], [], [], [], []
        for row in rows:
            try:
                r, vo, ev, vg, vd, vb = (float(x) for x in row)
            except (ValueError, TypeError):
                continue
            if r <= 0 or vo <= 0 or ev <= 0:
                continue
            R.append(r); V.append(vo); E.append(ev)
            Vg.append(vg); Vd.append(vd); Vb.append(vb)
        if len(R) < CUTS["min_points"]:
            drop["points"] += 1
            continue
        g = Galaxy(name=name, R0=np.array(R), Vobs0=np.array(V), eV=np.array(E),
                   Vgas=np.array(Vg), Vdisk=np.array(Vd), Vbul=np.array(Vb),
                   **meta)
        # baryonic mass at the catalogue M/L, for stratification only
        Mstar = 0.5 * g.L36 * 1e9
        Mgas = 1.33 * g.MHI * 1e9
        g.Mb = Mstar + Mgas
        g.fgas = Mgas / g.Mb if g.Mb > 0 else float("nan")
        kept.append(g)
    if verbose:
        print(f"   SPARC galaxies with curves : {len(cur)}")
        print(f"   dropped, no Table 1 entry  : {drop['no_table1']}")
        print(f"   dropped, Qual > {CUTS['max_qual']}          : {drop['qual']}")
        print(f"   dropped, i < {CUTS['min_inclination']:.0f} deg        : {drop['inclination']}")
        print(f"   dropped, no V_flat         : {drop['vflat']}")
        print(f"   dropped, < {CUTS['min_points']} points        : {drop['points']}")
        print(f"   RETAINED                   : {len(kept)}")
        print(f"   radial points              : {sum(len(g) for g in kept)}")
    return kept


def stratified_split(gals: list[Galaxy], verbose: bool = True) -> None:
    """Assign train/validation/blind by whole galaxy, stratified.

    Galaxies are ordered inside each stratum by a hash of the NAME, not by any
    measured quantity, so the assignment cannot correlate with anything the
    models see.
    """
    rng = np.random.default_rng(SPLIT["seed"])
    lm = np.array([math.log10(max(g.Mb, 1.0)) for g in gals])
    lv = np.array([math.log10(max(g.Vflat, 1.0)) for g in gals])
    fg = np.array([g.fgas for g in gals])
    # 2x2x2 strata on mass, flat velocity, gas fraction, times the quality flag
    def terc(a):
        return np.digitize(a, [np.median(a)])
    key = [f"{terc(lm)[i]}{terc(lv)[i]}{terc(fg)[i]}{gals[i].Qual}"
           for i in range(len(gals))]
    strata: dict[str, list[int]] = {}
    for i, k in enumerate(key):
        strata.setdefault(k, []).append(i)
    counts = dict(train=0, validation=0, blind=0)
    for k, idx in sorted(strata.items()):
        idx = sorted(idx, key=lambda j: hashlib.sha256(
            gals[j].name.encode()).hexdigest())
        n = len(idx)
        n_tr = int(round(SPLIT["train"] * n))
        n_va = int(round(SPLIT["validation"] * n))
        for j, gi in enumerate(idx):
            lab = ("train" if j < n_tr else
                   "validation" if j < n_tr + n_va else "blind")
            gals[gi].split = lab
            counts[lab] += 1
    if verbose:
        tot = sum(counts.values())
        print(f"\n   strata                     : {len(strata)}")
        for k, v in counts.items():
            print(f"   {k:<11}                : {v:>4}  ({100*v/tot:.1f}%)")
    return None


def freeze_split(gals: list[Galaxy], path: str) -> str:
    """Write the split to disk and return its sha256, so it is auditable."""
    payload = {g.name: g.split for g in sorted(gals, key=lambda x: x.name)}
    blob = json.dumps(dict(cuts=CUTS, split=SPLIT, stratify=list(STRATIFY),
                           assignment=payload), indent=1, sort_keys=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(blob)
    return hashlib.sha256(blob.encode()).hexdigest()


if __name__ == "__main__":
    print("=" * 74)
    print("Run A, steps 1-3: ingest, declared cuts, frozen split")
    print("=" * 74)
    gals = ingest()
    stratified_split(gals)
    h = freeze_split(gals, ROOT + "work/gravitylab/configs/splits/runA_split.json")
    print(f"\n   split sha256               : {h[:16]}")
    print(f"   blind galaxies are sealed from here on.")
