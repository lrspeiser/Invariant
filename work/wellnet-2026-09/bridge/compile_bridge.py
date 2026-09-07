"""compile_bridge.py -- Job 5: the path family re-compiled through the
extended bench.

Each (eps, rho_*) setting is compiled TWICE: exactly as BL declared it
(`compile_families.path_candidates`: pair kernel, shell Green's function,
single-probe force factors), and again with the two-body probe field
declared (`Candidate.bridge_field`: |g| at the bridge probe's 15 points,
computed here from Phi = Phi_N + eps Phi_v + Phi_3 with central differences).
The bin each lands in, and whether `non_identifiable_on_this_bench` moves,
is the deliverable.  A fine eps scan through GATE 1 alone locates the bench's
new identifiability threshold.

BL's single-probe force-factor tables are RECOMPUTED here through BL's own
functions (path_family.force_factor_table) rather than read from the
synthesis lane's JSON, so this lane's guard stays pinned to its own
directory.

    python compile_bridge.py
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict

import numpy as np

import guard
import compiler as C
import path_family as PF
import scene as S

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
KPC, MPC, MSUN, G = S.KPC, S.MPC, S.MSUN, S.G
EPS_SCAN = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001)


# --------------------------------------------- the P field at the probe points
class BridgeField:
    """|g| of the path law at arbitrary points of BL's bridge scene."""

    def __init__(self, eps: float, rho_star: float, n_dir: int = 4000,
                 h: float = 10.0 * KPC):
        self.eps, self.rs, self.n_dir, self.h = eps, rho_star, n_dir, h
        self.sc = S.TwoBody(MA=C.BRIDGE_M, MB=C.BRIDGE_M, a=C.BRIDGE_A, D=C.BRIDGE_D)
        comp = S.Plummer(C.BRIDGE_M, C.BRIDGE_A, (0.0, 0.0, 0.0))
        self.rg = np.geomspace(2e-2 * comp.a, 60.0 * comp.a, 70)
        self.phiv = np.array([S._phi_v_single(comp, r, rho_star) for r in self.rg])

    def phi(self, pts: np.ndarray) -> np.ndarray:
        sc = self.sc
        pts = np.asarray(pts, float)
        phiN = sum(c.phi(pts) for c in sc.comps)
        out = phiN.copy()
        rho_all = lambda p: sc.rho(p)                        # noqa: E731
        for g in "AB":
            comp = sc.groups[g][0]
            d = np.linalg.norm(pts - comp.c, axis=-1)
            out += self.eps * np.interp(d, self.rg, self.phiv,
                                        left=self.phiv[0], right=0.0)
            cc = np.repeat(comp.c[None, :], len(pts), 0)
            v_all = S.phi_vac_mean(rho_all, pts, cc, self.rs)
            v_own = S.phi_vac_mean(comp.rho, pts, cc, self.rs)
            out += -G * self.eps * comp.M * (v_all - v_own) / np.maximum(d, 0.05 * comp.a)
        P = S.P_total(S.pairwise_P(sc, pts, self.n_dir))
        out += -0.5 * G * self.eps * S.dphi_vac(sc.rho(pts), self.rs) * P
        return out

    def parts(self, pts: np.ndarray) -> Dict[str, np.ndarray]:
        """|g_N|, |g_dir|, |g_3| and |g| at the points (central differences)."""
        pts = np.asarray(pts, float)
        n = len(pts)
        E = np.eye(3)
        disp = np.concatenate([pts[:, None, :] + self.h * E[None, :, :],
                               pts[:, None, :] - self.h * E[None, :, :]], 1).reshape(-1, 3)
        sc = self.sc
        phiN = sum(c.phi(disp) for c in sc.comps).reshape(n, 6)
        P = S.P_total(S.pairwise_P(sc, disp, self.n_dir))
        phi3 = (-0.5 * G * self.eps * S.dphi_vac(sc.rho(disp), self.rs) * P).reshape(n, 6)
        phi_all = self.phi(disp).reshape(n, 6)

        def grad(f):
            return (f[:, :3] - f[:, 3:]) / (2.0 * self.h)
        gN, g3, g = grad(phiN), grad(phi3), grad(phi_all)
        gdir = g - gN - g3
        return dict(gN=np.linalg.norm(gN, axis=1), g3=np.linalg.norm(g3, axis=1),
                    gdir=np.linalg.norm(gdir, axis=1), g=np.linalg.norm(g, axis=1),
                    g_vec=g, gN_vec=gN)

    def __call__(self, pts):
        return self.parts(pts)["g"]


# --------------------------------------------------------- BL's candidates
def rescale_tables(tables: Dict, eps_new: float, eps_ref: float) -> Dict:
    """Both the endpoint term and the carrier term are linear in eps at fixed
    rho_*, so g/g_N - 1 scales exactly with eps (compile_families._rescale_tables)."""
    out = {}
    for nm, t in tables.items():
        s_ = eps_new / eps_ref
        out[nm] = dict(t)
        for k in ("factor", "factor_direct_only", "factor_carrier_only"):
            out[nm][k] = (1.0 + s_ * (np.array(t[k]) - 1.0)).tolist()
        out[nm]["eps"] = eps_new
    return out


def ff_tables(rho_star: float, cache: Dict) -> Dict:
    key = f"{rho_star:.3e}"
    if key not in cache:
        cache[key] = {nm: PF.force_factor_table(nm, PF.EPS_FID, rho_star)
                      for nm in ("galaxy_field", "cluster_shell", "galaxy_member")}
    return cache[key]


def candidate(eps: float, rho_star: float, tables: Dict, bf=None) -> C.Candidate:
    bg = PF.Scene([C.Plummer(1.0e12 * C.MSUN, 20.0 * C.KPC)], rho_star)
    tabs = rescale_tables(tables, eps, PF.EPS_FID)
    tag = f"P_eps{eps:g}_rho{rho_star:.0e}" + ("_bridge" if bf is not None else "_BL")
    return C.Candidate(
        tag, base="newton", struct="path_kernel", inv="one", form="off", A=eps,
        pair_kernel=lambda x, y, e=eps: float(bg.W(np.asarray(x), np.asarray(y), e)),
        green=PF.make_green(eps, rho_star), force_factor=PF.make_force_factor(tabs),
        momentum_carrier="matter on the connecting segment (three-body term "
                         "-grad Phi_3); verified in path_results.json and in "
                         "bridge/results/crosscheck.json",
        bridge_field=bf,
        note=(f"the same action at eps = {eps}, rho_* = {rho_star:.1e}"
              + (" with the two-body probe field declared" if bf is not None
                 else " exactly as BL declared it")))


def summ(r: Dict) -> Dict:
    g1 = r["gate1_constant_K"][1]
    return dict(verdict=r["_verdict"], failed=r["_failed"], labels=r["_labels"],
                flags=r["_flags"], primary_bin=r["_taxonomy"]["primary"],
                defects=[d["code"] for d in r["_taxonomy"]["defects"]],
                gate1=dict(passed=r["gate1_constant_K"][0],
                           escapes=g1.get("escapes"),
                           max_single_probe_resid_dex=g1.get("max_single_probe_resid_dex"),
                           joint_resid_dex=g1.get("joint_resid_dex"),
                           two_body_probe_consulted=g1.get("two_body_probe_consulted"),
                           two_body_probe_resid_dex=g1.get("two_body_probe_resid_dex"),
                           per_probe={k: v["resid_dex"] for k, v in g1.get("per_probe", {}).items()},
                           reason=r["gate1_constant_K"][2]),
                gate4=dict(passed=r[C.GATE4][0], reciprocity=r[C.GATE4][1].get("reciprocity"),
                           reason=r[C.GATE4][2][:300]))


def main():
    guard.arm()
    t0 = time.perf_counter()
    cache: Dict = {}
    out = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               compiler="work/wellnet-2026-09/compiler/compiler.py (v2 + BL elements + "
                        "Run BM two-body probe `bridge_pair`)",
               probe=dict(points_Mpc=(C.bridge_probe_points() / MPC).tolist(),
                          M_Msun=C.BRIDGE_M / MSUN, a_kpc=C.BRIDGE_A / KPC, D_Mpc=C.BRIDGE_D / MPC,
                          gN_over_a0=(C.probes()["bridge_pair"].inv["gn"]).tolist()),
               settings={}, fields={}, threshold_scan={})
    pts = C.bridge_probe_points()
    for rtag, rs in S.RHO_STAR_LADDER.items():
        t1 = time.perf_counter()
        tabs = ff_tables(rs, cache)
        print(f"force-factor tables rho_*={rtag}: {time.perf_counter() - t1:.0f}s")
        for eps in S.EPS_LADDER:
            bf = BridgeField(eps, rs)
            parts = bf.parts(pts)
            out["fields"][f"{rtag}|eps={eps}"] = dict(
                eps=eps, rho_star=rs,
                log10_g_over_gN=np.log10(parts["g"] / parts["gN"]).tolist(),
                g3_over_gN=(parts["g3"] / parts["gN"]).tolist(),
                gdir_over_gN=(parts["gdir"] / parts["gN"]).tolist(),
                gN=parts["gN"].tolist())
            for label, bfield in (("BL", None), ("bridge", bf)):
                c = candidate(eps, rs, tabs, bfield)
                r = C.check(c, cheap=True)
                out["settings"][f"{rtag}|eps={eps}|{label}"] = dict(
                    eps=eps, rho_star=rs, declared=label, **summ(r))
                s = out["settings"][f"{rtag}|eps={eps}|{label}"]
                print(f"  {rtag:<8} eps={eps:<6} {label:<7} {s['verdict']:<8} {s['primary_bin']:<32} "
                      f"escapes={s['gate1']['escapes']} single={s['gate1']['max_single_probe_resid_dex']:.4f} "
                      f"joint={s['gate1']['joint_resid_dex']:.4f} "
                      f"bridge={s['gate1']['two_body_probe_resid_dex']}")
        # the threshold scan through GATE 1 alone, with the bridge probe
        scan = {}
        for eps in EPS_SCAN:
            bf = BridgeField(eps, rs)
            c = candidate(eps, rs, tabs, bf)
            ok, v, _ = C.gate1(c)
            scan[str(eps)] = dict(escapes=v["escapes"],
                                  bridge_resid_dex=v["two_body_probe_resid_dex"],
                                  max_single=v["max_single_probe_resid_dex"],
                                  joint=v["joint_resid_dex"], passed=bool(ok))
            print(f"    scan eps={eps:<7} bridge {v['two_body_probe_resid_dex']:.4f} dex "
                  f"single {v['max_single_probe_resid_dex']:.4f} joint {v['joint_resid_dex']:.4f} "
                  f"-> {v['escapes']}")
        ident = [float(e) for e, d in scan.items() if d["passed"]]
        out["threshold_scan"][rtag] = dict(
            rows=scan, smallest_identifiable_eps=min(ident) if ident else None,
            tol_dex=C.TOL_DEX,
            note="GATE 1 alone; escapes (a)-(d); the bench's identifiability threshold "
                 "for the path family with the two-body probe present")
    out["force_factor_tables"] = {k: {nm: dict(r_kpc=[x / KPC for x in t["r_m"]], factor=t["factor"])
                                      for nm, t in v.items()} for k, v in cache.items()}
    out["provenance"] = guard.summary()
    out["wall_seconds"] = time.perf_counter() - t0
    with open(os.path.join(RES, "compile_bridge.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote compile_bridge.json in {out['wall_seconds']:.0f}s; provenance:",
          out["provenance"]["assertion"])


if __name__ == "__main__":
    main()
