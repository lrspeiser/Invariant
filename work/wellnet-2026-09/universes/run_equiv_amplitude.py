"""run_equiv_amplitude.py -- the equivalence map where it is actually interesting.

At the fiducial knob settings every pair of the ten universes separates with
AUC = 1.0 on a single corpus.  That is a real result -- it says the fiducial
amplitudes are far above what this corpus can already see -- but it makes the
equivalence-class map vacuous, and the charter's question is precisely about
the pairs that CANNOT be told apart.

So the map is recomputed at two amplitude sets taken from the E3 scans:

  THRESHOLD  each deformation's knob set to the amplitude at which it just
             reaches the family-wise critical value against the base U3
  HALF       half that amplitude, where each deformation is by construction
             NOT separable from the base

The question then becomes the interesting one: at a common observable
amplitude, are two fundamentally different modifications distinguishable from
EACH OTHER?  U2 (dark matter) and U10 (systematics only) are carried along
unchanged, since they have no amplitude knob.

This pass does its own sizing: A-vs-A nulls on its own arms, its own
family-wise critical value for its own number of pairs.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from universes import generate as gn          # noqa: E402
from universes import physics as ph           # noqa: E402
from universes import provenance as pv        # noqa: E402
from universes import stats as st             # noqa: E402
from universes import run_stage5 as R5        # noqa: E402

RES = os.path.join(HERE, "results")
N = int(os.environ.get("N_EQ", 480))
SEED = 3100000


def main():
    gn.get_lib(); gn.get_pool()
    pv.start_ledger(os.path.dirname(HERE))
    with open(os.path.join(RES, "E3_amplitude_scans.json")) as f:
        E3 = json.load(f)
    with open(os.path.join(RES, "channel_map.json")) as f:
        CM = json.load(f)
    R5.KEYS = CM["feature_order"]
    R5.CHAN_IDX = {c: [R5.KEYS.index(k) for k in v] for c, v in CM["channels"].items()}

    # ---- the two amplitude sets, taken from the scans, not chosen by hand ----
    thr = {}
    for uid, v in E3.items():
        a = v["z_crit"].get("amp") or v["z3"].get("amp")
        if a is None:
            a = max(r["amp"] for r in v["rows"])
            note = "family-wise threshold not reached in the scan; using the largest scanned amplitude"
        else:
            note = "amplitude at which the scan reaches the family-wise critical value against U3"
        thr[uid] = {"amp": float(a), "note": note, "knob": v["knob"],
                    "fiducial": v["fiducial"]}

    out = {"threshold_amplitudes": thr, "sets": {}}
    for setname, scale in (("THRESHOLD", 1.0), ("HALF", 0.5)):
        arms = {"U03_mond_scalar": ("U03_mond_scalar", None, 1.0, 1.0),
                "U02_cdm": ("U02_cdm", None, 1.0, 1.0),
                "U10_systematics": ("U10_systematics", None, 3.0, 1.0)}
        for uid, t in thr.items():
            arms[uid] = (uid, t["amp"] * scale, 1.0, 1.0)
        names = list(arms)
        print(f"[{setname}] {len(names)} arms", flush=True)
        pools = {}
        for i, (nm, arm) in enumerate(arms.items()):
            tag = f"eq{setname}_{nm}"
            pools[nm] = R5.pool(tag, arm, N, SEED + 1000 * i + int(scale * 7))
            print(f"  {nm} amp={arm[1]}", flush=True)

        # ---- size this pass on its own A-vs-A nulls -----------------------
        nullz = []
        for nm, recs in pools.items():
            X, _ = gn.to_matrix(recs, R5.KEYS)
            q = len(X) // 4
            for rep in range(20):
                idx = np.random.default_rng(500 + rep).permutation(len(X))
                r = R5.sepmax(X[idx[:q]], X[idx[q:2 * q]],
                              X[idx[2 * q:3 * q]], X[idx[3 * q:4 * q]], seed=rep)
                nullz.append(r["z_max"])
        nullz = np.array(nullz)
        npair = len(names) * (len(names) - 1) // 2
        zc_single = float(np.quantile(nullz, 0.95))
        zc_family = float(np.quantile(
            [np.max(np.random.default_rng(k).choice(nullz, size=npair))
             for k in range(6000)], 0.95))
        print(f"  sizing: null median {np.median(nullz):.2f}  "
              f"single {zc_single:.2f}  family({npair} pairs) {zc_family:.2f}", flush=True)

        rows = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                Ac, Aa = R5.quarters(pools[a]); Bc, Ba = R5.quarters(pools[b])
                r = R5.sepmax(Ac, Bc, Aa, Ba, seed=i * 100 + j)
                rows.append({"a": a, "b": b, "z_max": r["z_max"],
                             "best_test": r["best_test"], "auc_full": r["auc_full"],
                             "z_full": r["z_full"],
                             "test_z": {k: v["z"] for k, v in r["per_test"].items()}})
        parent = {n: n for n in names}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x

        for r in rows:
            if r["z_max"] < zc_family:
                ra, rb = find(r["a"]), find(r["b"])
                if ra != rb:
                    parent[ra] = rb
        classes = {}
        for n in names:
            classes.setdefault(find(n), []).append(n)
        out["sets"][setname] = {
            "scale_of_threshold_amplitude": scale,
            "arms": {n: arms[n][1] for n in names},
            "n_pairs": npair,
            "null_zmax_median": float(np.median(nullz)),
            "z_crit_single": zc_single, "z_crit_family": zc_family,
            "pairs": rows,
            "equivalence_classes": [sorted(v) for v in classes.values()],
            "indistinguishable_pairs": [
                {"a": r["a"], "b": r["b"], "z_max": r["z_max"],
                 "auc_full": r["auc_full"], "best_test": r["best_test"],
                 "test_z": r["test_z"]}
                for r in rows if r["z_max"] < zc_family]}
        # ---- the missing-observation oracle for every indistinguishable pair
        ORACLES = (("noise x0.25: 16x the effective source density and 4x the "
                    "spectroscopic / temperature precision", 1.0, 0.25, 30, 12),
                   ("systematics x0.25: shear calibration, photo-z, M/L, inclination, "
                    "miscentring and non-thermal pressure all 4x better controlled",
                    0.25, 1.0, 30, 12),
                   ("survey x1.5: 45 galaxies and 18 clusters per corpus",
                    1.0, 1.0, 45, 18))
        for k, r in enumerate(out["sets"][setname]["indistinguishable_pairs"]):
            print(f"    SAME CLASS: {r['a']} ~ {r['b']}  z={r['z_max']:.2f} "
                  f"auc={r['auc_full']:.3f} best {r['best_test']}", flush=True)
            r["oracles"] = {}
            for tag, ss, ns, ng_, nc_ in ORACLES:
                got = {}
                for u in (r["a"], r["b"]):
                    base = arms[u]
                    s2 = base[2] * ss
                    gn.N_GAL, gn.N_CLU = ng_, nc_
                    got[u] = R5.pool(
                        f"eqorc{setname}_{u}_{ss}_{ns}_{ng_}_{nc_}",
                        (base[0], base[1], s2, ns), N,
                        SEED + 500000 + 991 * names.index(u) + int(scale * 13))
                    gn.N_GAL, gn.N_CLU = 30, 12
                Ac, Aa = R5.quarters(got[r["a"]]); Bc, Ba = R5.quarters(got[r["b"]])
                rr = R5.sepmax(Ac, Bc, Aa, Ba, seed=77 + k)
                r["oracles"][tag] = {"z_max": rr["z_max"], "best_test": rr["best_test"],
                                     "separated": bool(rr["z_max"] >= zc_family)}
                print(f"       {tag[:28]}: z={rr['z_max']:.2f} ({rr['best_test']})",
                      flush=True)
            r["corpus_multiplier_for_z5_sqrtN"] = (
                float((5.0 / max(r["z_max"], 1e-3)) ** 2) if r["z_max"] > 0.05 else None)

    gn.close_pool()
    out["file_access"] = pv.stop_ledger()
    with open(os.path.join(RES, "E9_equivalence_at_threshold.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote E9_equivalence_at_threshold.json")


if __name__ == "__main__":
    main()
