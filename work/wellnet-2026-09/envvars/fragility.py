"""How many objects is each within-object beta actually made of?

A variable that is nearly constant inside most objects can still show Fisher
information for beta if a handful of objects happen to have a close neighbour.
The estimate is then a measurement of those few objects, not of the sample.
This measures that directly: rank objects by their contribution to the Fisher
information for beta, then refit with the top contributors removed.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import envvars as EV
import fixedeffects as F
from envvars import MPC, A0, EPS_PRIMARY, PRIMARY_RULE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}


def per_object_fisher(a0, d1, e, edge, per_system):
    """Fisher information for beta contributed by each object separately."""
    w = 1.0 / e ** 2
    if per_system:
        num = np.add.reduceat(a0 * d1 * w, edge[:-1])
        den = np.add.reduceat(a0 * a0 * w, edge[:-1])
        k = np.repeat(num / np.maximum(den, 1e-300), np.diff(edge))
    else:
        k = np.sum(a0 * d1 * w) / max(np.sum(a0 * a0 * w), 1e-300)
    dt = d1 - k * a0
    return np.add.reduceat(dt ** 2 * w, edge[:-1])


def main():
    recs, obs, systems = EV.load_all(0.0)
    wells = EV.Wells(recs, 0.0)
    V = EV.build_variables(systems, wells, EPS_PRIMARY, PRIMARY_RULE)
    rr = V["lr"]
    mask = (rr >= math.log10(F.PROJ_RANGE_MPC[0])) & \
           (rr <= math.log10(F.PROJ_RANGE_MPC[1]))
    V["xr"] = V["lr"].copy()
    V["xgb"] = V["lgb"] - math.log10(A0)
    rgrid = systems[0].r
    for k in F.VARS:
        V[k] = F.freeze_outside(V[k], rgrid)
    _, coefs = EV.collinearity(V, F.ENVVARS, mask)
    Vp = EV.residualise(V, F.ENVVARS, coefs)
    for k in F.ENVVARS:
        Vp[k] = F.freeze_outside(Vp[k], rgrid)
    for k in F.COMPET:
        Vp[k] = V[k].copy()
    for k in F.VARS:
        V[k] = F.standardise(V[k], mask)[0]
        Vp[k] = F.standardise(Vp[k], mask)[0]

    train, _ = F.declared_split(obs)
    A0t = F.unit_shear(systems, obs, train)
    a0, y, e, edge = F._pack(obs, train, A0t)

    print(f"{'var':10s} {'est':14s} {'beta all':>9s} {'beta -1%':>9s} "
          f"{'beta -5%':>9s} {'beta -10%':>9s} {'top1 share':>10s} "
          f"{'top5 share':>10s}")
    for key in F.VARS:
        for tag, XX in (("raw", V[key]), ("perp", Vp[key])):
            D1, D2 = F.derivs(systems, obs, train, XX, A0t)
            d1 = np.concatenate(D1)
            d2 = np.concatenate(D2)
            for ps, nm in ((True, "within_object"), (False, "within_class")):
                Fi = per_object_fisher(a0, d1, e, edge, ps)
                order = np.argsort(-Fi)
                share1 = float(Fi[order[:max(1, len(Fi)//100)]].sum()
                               / max(Fi.sum(), 1e-300))
                share5 = float(Fi[order[:max(1, len(Fi)//20)]].sum()
                               / max(Fi.sum(), 1e-300))
                row = {}
                for frac, lab in ((0.0, "all"), (0.01, "-1%"), (0.05, "-5%"),
                                  (0.10, "-10%")):
                    drop = set(order[:int(round(frac * len(Fi)))].tolist())
                    keep = np.array([i for i in range(len(Fi))
                                     if i not in drop])
                    m = np.zeros(len(a0), bool)
                    for i in keep:
                        m[edge[i]:edge[i + 1]] = True
                    ed2 = np.concatenate(
                        [[0], np.cumsum([edge[i + 1] - edge[i] for i in keep])])
                    c = np.array([F.chi2_profiled(
                        a0[m] + b * d1[m] + 0.5 * b * b * d2[m],
                        y[m], e[m], ed2, ps)[0] for b in F.BETA_GRID])
                    row[lab] = F._beta_from_grid(c)
                OUT[f"{key}_{tag}_{nm}"] = dict(
                    beta=row, top1pct_fisher_share=share1,
                    top5pct_fisher_share=share5, n_objects=int(len(Fi)))
                print(f"{key+'_'+tag:10s} {nm:14s} {row['all']:+9.4f} "
                      f"{row['-1%']:+9.4f} {row['-5%']:+9.4f} "
                      f"{row['-10%']:+9.4f} {share1:10.3f} {share5:10.3f}")

    with open(os.path.join(HERE, "envvars_fragility.json"), "w",
              encoding="utf-8") as f:
        json.dump(OUT, f, indent=1)
    print("\nwrote envvars_fragility.json")


if __name__ == "__main__":
    main()
