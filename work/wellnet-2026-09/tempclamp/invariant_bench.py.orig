"""
INVARIANT GRAVITY BENCH
=======================

A reusable harness: every dataset acquired in this work, a single scoring
interface for any candidate law, and -- most importantly -- the confound checks
that caught six false positives.

    from invariant_bench import Bench
    b = Bench()
    b.summary()
    b.score(lambda d: 1/(1-np.exp(-np.sqrt(d.x))))       # any nu(data) -> array
    b.confound("my_variable", values)                     # is it a dataset label?

WHY THE CONFOUND CHECK IS BUILT IN
Six separate "discoveries" in this project turned out to be a variable that
correlated with which dataset a point came from:
    shape (binary), compact=r/extent, mass at n=120, a0-per-population,
    the pooled cluster correlations, and PySR's sphericity.
Each looked significant and each was reproduced exactly by a bare 0/1 dataset
indicator. The check costs one line and is therefore mandatory here, not
optional.

REGIME NOTES CARRIED WITH THE DATA
  X-ray hydrostatic masses give reliable AMPLITUDES (agree with lensing to 5%)
  but biased RADIAL SHAPES -- non-thermal pressure rises outward. Never draw a
  profile-shape conclusion from X-ray alone.
  Solar System rows are a BOUND (nu = 1 by construction), never a fit target.
  Wide binaries sit in the Milky Way's ~1.8 a0 external field.
"""
import json, glob, os, math, csv
import numpy as np
from astropy.io import fits

ROOT = "C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
DD = os.path.dirname(os.path.abspath(__file__)) + "/"
G, MSUN, KPC, AU = 6.67430e-11, 1.98892e30, 3.0856775814913673e19, 1.495978707e11
A0, KMS = 1.2e-10, 1e3
MP, MU, MU_E = 1.67262192e-27, 0.6, 1.14
GM_SUN = 1.32712440018e20
G_PC, PC_M = 4.52e-30, 3.086e16
H0 = 70.0*1e3/3.0856775814913673e22
RHOC = 3*H0**2/(8*math.pi*G)


class D:
    """one probe's data, with everything a candidate law might need"""
    def __init__(self, name, r, gb, go, **kw):
        self.name = name
        self.r = np.asarray(r, float)          # metres
        self.gb = np.asarray(gb, float)        # baryonic acceleration, m/s^2
        self.go = np.asarray(go, float)        # observed acceleration, m/s^2
        self.x = self.gb/A0                    # dimensionless acceleration
        self.nu = self.go/self.gb
        self.M = self.gb*self.r**2/G           # enclosed baryonic mass
        self.phi = self.gb*self.r              # |potential|
        self.rho = self.gb/(G*self.r)          # mean enclosed density scale
        for k, v in kw.items():
            setattr(self, k, np.asarray(v, float) if np.ndim(v) else v)
    def __len__(self):
        return len(self.r)


class Bench:
    PROBE_KIND = {
        "sparc": ("matter", "disk", "fit"),
        "xcop": ("matter", "spheroid", "fit"),
        "clash": ("photon", "spheroid", "fit"),
        "kids": ("photon", "disk", "holdout"),
        "wicker": ("matter", "spheroid", "fit"),
        "solar": ("matter", "point", "bound"),
        "widebin": ("matter", "two_body", "holdout"),
    }
    CAVEATS = {
        "xcop": "X-ray: amplitudes reliable to 5%, RADIAL SHAPES biased by "
                "outward-rising non-thermal pressure",
        "wicker": "X-ray at R500 only; SZ flux-limited so mass and z correlate at 0.69",
        "clash": "lensing-selected, so prolate systems aligned to the line of sight "
                 "are over-represented",
        "kids": "beyond ~1 Mpc the signal includes neighbouring mass, so g_bar is "
                "underestimated and nu inflated",
        "solar": "nu = 1 by CONSTRUCTION -- a bound, never a fit target",
        "widebin": "sits in the Milky Way's ~1.8 a0 external field, which suppresses "
                   "any internal-acceleration effect",
        "sparc": "fixed mass-to-light 0.5 disk / 0.7 bulge; IMF is the dominant "
                 "systematic at ~0.25 dex",
    }

    def __init__(self, verbose=True):
        self.d = {}
        self._load(verbose)

    # ---------------------------------------------------------------- loaders
    def _load(self, verbose):
        L = self._sparc()
        if L: self.d["sparc"] = L
        for nm, fn in (("xcop", self._xcop), ("clash", self._clash),
                       ("kids", self._kids), ("wicker", self._wicker),
                       ("solar", self._solar), ("widebin", self._widebin)):
            try:
                v = fn()
                if v is not None and len(v):
                    self.d[nm] = v
            except Exception as e:
                if verbose:
                    print(f"   [{nm}] unavailable: {type(e).__name__}")

    def _sparc(self):
        p = ROOT+"configs/sparc_rotation_curves_full_v1.json"
        if not os.path.exists(p):
            return None
        cfg = json.load(open(p, encoding="utf-8"))
        r, gb, go, sph, ext = [], [], [], [], []
        for gal in cfg["galaxies"]:
            rows = [[float(q) for q in w] for w in gal["rows"]]
            rad = np.array([q[0] for q in rows])*KPC
            if len(rad) < 4:
                continue
            rmax = rad.max()
            for q in rows:
                r0 = q[0]*KPC; vo = q[1]*KMS; ev = q[2]*KMS
                cg = (q[3]*KMS)*abs(q[3]*KMS); cd = 0.5*(q[4]*KMS)**2; cb = 0.7*(q[5]*KMS)**2
                v2 = cg+cd+cb
                if r0 <= 0 or vo <= 0 or ev <= 0 or v2 <= 0:
                    continue
                r.append(r0); gb.append(v2/r0); go.append(vo**2/r0)
                sph.append(cb/v2); ext.append(rmax)
        return D("sparc", r, gb, go, sphericity=sph, extent=ext)

    def _cluster_profile(self, d):
        fd = glob.glob(os.path.join(d, "*density*.fits"))
        ft = glob.glob(os.path.join(d, "*temperature*.fits"))
        if not fd or not ft:
            return None
        hd, ht = fits.open(fd[0]), fits.open(ft[0]); H = hd[1].header
        M500 = float(H["M500"])*1e14*MSUN; R500 = float(H["R500"])*KPC
        kT500 = G*M500*MU*MP/(2*R500)
        da = hd[1].data
        r = (0.5*(da["R_IN"].astype(np.float64)+da["R_OUT"].astype(np.float64)))*KPC
        ne = da["NE"].astype(np.float64)*1e6
        assert r.dtype == np.float64, "float32 overflow guard"
        td = ht[1].data
        kT = np.interp(r, td["RW_X"].astype(np.float64)*R500,
                       td["T_X"].astype(np.float64)*kT500)
        lr = np.log(r)
        go = -(kT/(MU*MP))*(np.gradient(np.log(ne), lr)+np.gradient(np.log(kT), lr))/r
        rho = MU_E*ne*MP
        Mg = (4/3*np.pi*r[0]**3*rho[0]
              + np.concatenate([[0.], np.cumsum(4*np.pi*rho[:-1]*r[:-1]**2*np.diff(r))]))
        fs = glob.glob(os.path.join(d, "*mstar*.fits"))
        Mst = (np.interp(r, fits.open(fs[0])[2].data["RADIUS"].astype(np.float64)*KPC,
                         fits.open(fs[0])[2].data["MSTAR"].astype(np.float64)*MSUN)
               if fs else Mg*0.10)
        Mb = Mg+Mst
        return r, G*Mb/r**2, go, R500

    def _xcop(self):
        XC = ROOT+"runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/"
        r_, b_, o_, e_ = [], [], [], []
        for nm in sorted(os.listdir(XC)):
            d = os.path.join(XC, nm)
            if not os.path.isdir(d):
                continue
            v = self._cluster_profile(d)
            if v is None:
                continue
            r, gb, go, R500 = v
            m = (r > 120*KPC) & (r < 1650*KPC) & (go > 0) & (gb > 0)
            r_ += list(r[m]); b_ += list(gb[m]); o_ += list(go[m])
            e_ += [R500]*int(m.sum())
        return D("xcop", r_, b_, o_, sphericity=np.ones(len(r_)), extent=e_)

    def _clash(self):
        p = ROOT+"runs/gravity/g4/cluster-lensing-exploration-v7-source/fig2.tsv"
        L = [l.rstrip("\n") for l in open(p, encoding="utf-8")]
        i0 = next(i for i, l in enumerate(L) if l.startswith("recno\t"))
        r, gb, go = [], [], []
        for l in L[i0+3:]:
            q = l.split("\t")
            if len(q) < 7 or not q[0].strip():
                continue
            try:
                r.append(float(q[2])*KPC); gb.append(10**float(q[3])); go.append(10**float(q[4]))
            except ValueError:
                pass
        return D("clash", r, gb, go, sphericity=np.ones(len(r)),
                 extent=np.full(len(r), 1500*KPC))

    def _kids(self):
        EDG = [8.5, 10.3, 10.6, 10.8, 11.0]
        r, gb, go = [], [], []
        for b in (1, 2, 3, 4):
            f = DD+f"g2/Fig-3_Lensing-rotation-curves_Massbin-{b}.txt"
            if not os.path.exists(f):
                continue
            Mb = 1.4*10**(0.5*(EDG[b-1]+EDG[b]))*MSUN
            for l in open(f, encoding="utf-8"):
                l = l.strip()
                if not l or l.startswith("#"):
                    continue
                v = [float(z) for z in l.split()]
                if len(v) < 5:
                    continue
                R = v[0]*1000*KPC; g = 4*G_PC*(v[1]/v[4])*PC_M
                if g > 0:
                    r.append(R); gb.append(G*Mb/R**2); go.append(g)
        return D("kids", r, gb, go, sphericity=np.zeros(len(r)),
                 extent=np.full(len(r), 300*KPC)) if r else None

    def _wicker(self):
        p = DD+"wl/wicker.tsv"
        if not os.path.exists(p):
            return None
        L = [l.rstrip("\n") for l in open(p, encoding="utf-8", errors="replace")]
        h = next(i for i, l in enumerate(L) if l.startswith("recno\t")); c = L[h].split("\t")
        s = next(i for i in range(h, len(L)) if L[i].startswith("-----"))
        r, gb, go = [], [], []
        for l in L[s+1:]:
            q = l.split("\t")
            if len(q) != len(c) or not q[0].strip():
                continue
            d = dict(zip(c, [w.strip() for w in q]))
            try:
                z = float(d["z"]); mg = float(d["Mgas"])*1e14*MSUN
                mt = float(d["Mtot"])*1e14*MSUN
            except ValueError:
                continue
            R5 = (mt/((4/3)*math.pi*500*RHOC*(0.3*(1+z)**3+0.7)))**(1/3)
            r.append(R5); gb.append(G*mg*1.15/R5**2); go.append(G*mt/R5**2)
        return D("wicker", r, gb, go, sphericity=np.ones(len(r)), extent=r)

    def _solar(self):
        a = np.array([0.3871, 0.7233, 1.0, 1.5237, 5.2026, 9.5549, 19.2184, 30.11])*AU
        g = GM_SUN/a**2
        return D("solar", a, g, g, sphericity=np.zeros(8), extent=np.full(8, 1e-6*KPC))

    def _widebin(self):
        # separations in AU, calibrated boosts from the El-Badry analysis
        sep = np.array([220., 557., 1565., 4568., 14040., 37223.])*AU
        boost = np.array([0.971, 1.029, 1.038, 1.067, 1.188, 1.203])
        Mt = 0.9*MSUN
        gb = G*Mt/sep**2
        return D("widebin", sep, gb, gb*boost, sphericity=np.zeros(6), extent=sep)

    # ---------------------------------------------------------------- scoring
    def summary(self):
        print(f"{'probe':<10}{'n':>7}{'kind':>9}{'shape':>10}{'role':>9}"
              f"{'g_bar/a0 range':>26}")
        print("-"*72)
        for k, d in self.d.items():
            kind, shape, role = self.PROBE_KIND[k]
            print(f"{k:<10}{len(d):>7}{kind:>9}{shape:>10}{role:>9}"
                  f"{f'{d.x.min():.2e} - {d.x.max():.2e}':>26}")
        print("-"*72)
        print(f"total measurements: {sum(len(d) for d in self.d.values())}")

    def score(self, law, probes=None, verbose=True):
        """law(d) -> predicted nu array. Returns per-probe median |log10| error."""
        out = {}
        for k, d in self.d.items():
            if probes and k not in probes:
                continue
            try:
                p = np.asarray(law(d), float)
            except Exception as e:
                out[k] = float("nan")
                continue
            good = np.isfinite(p) & (p > 0)
            if good.sum() < 3:
                out[k] = float("nan"); continue
            e = np.abs(np.log10(p[good]) - np.log10(d.nu[good]))
            out[k] = float(np.median(e))
        if verbose:
            print(f"   {'probe':<10}{'role':>9}{'median |err| dex':>20}{'factor':>10}")
            print("   "+"-"*50)
            for k, v in out.items():
                role = self.PROBE_KIND[k][2]
                print(f"   {k:<10}{role:>9}{v:>20.4f}{10**v:>10.3f}")
            print("   "+"-"*50)
            fit = [v for k, v in out.items() if self.PROBE_KIND[k][2] == "fit"
                   and np.isfinite(v)]
            hold = [v for k, v in out.items() if self.PROBE_KIND[k][2] == "holdout"
                    and np.isfinite(v)]
            if fit:
                print(f"   worst fitted probe   {max(fit):.4f} dex ({100*(10**max(fit)-1):.0f}%)")
            if hold:
                print(f"   worst held-out probe {max(hold):.4f} dex ({100*(10**max(hold)-1):.0f}%)")
        return out

    @staticmethod
    def _rank(a):
        """Average-rank. argsort(argsort(x)) breaks ties by ARRAY POSITION,
        and this bench concatenates probe by probe -- so under that scheme a
        global constant scored corr = +0.948 with the dataset label, an entirely
        manufactured number. Ties must share the average rank."""
        import numpy as _np
        o = _np.argsort(a, kind="mergesort")
        r = _np.empty(len(a), float); r[o] = _np.arange(len(a), dtype=float)
        _, inv, cnt = _np.unique(a, return_inverse=True, return_counts=True)
        sm = _np.zeros(len(cnt)); _np.add.at(sm, inv, r)
        return (sm / cnt)[inv]

    def confound(self, name, getter, verbose=True, nperm=4000, seed=0):
        """THE MANDATORY CHECK -- does this variable explain the RAR residual
        BEYOND what a bare dataset label already explains?

        Three defects were found in the earlier version and are fixed here:

          A. It fired when |r_vy| and |r_ly| were within 0.08 of each other, so
             it could not tell "beats the label" from "far worse than the
             label". Pure random noise was reported as carrying information
             beyond the label. Replaced by a partial correlation.
          B. Ranks broke ties by array position -- see _rank above.
          C. The probe filter excluded only role == "bound" (solar), so BOTH
             blind holdouts, kids and widebin, were being consumed. Holdout
             data was influencing variable selection. Now excluded.

        Returns three verdicts, not two. The old binary rule collapsed
        CARRIES NIL into the pass bucket, which is how noise got through.

        NOTE ON EFFECT SIZE: with n ~ 4000 a partial correlation of 0.05 is
        significant at p < 0.005 and physically negligible. Read `partial`,
        not `p`.

        AND A HARD LIMIT ON WHAT THIS CHECK CAN DO. Calibrated against 600
        pure-noise draws, the floor is |partial| = 0.031. The bare dataset
        label reaches 0.563 -- 18x the floor. Every physical variable tested
        lands between 0.046 and 0.147, i.e. 1.5x to 4.7x the floor and no more
        than 26% of the label. The check therefore separates a real variable
        from noise but CANNOT rank real variables against each other and
        CANNOT justify killing one by itself. Kills require an independent
        control: a synthetic twin, a within-survey split, a placebo pair, or a
        blind holdout.
        """
        rng = np.random.default_rng(seed)
        vals, labs, nus, xs = [], [], [], []
        for k, d in self.d.items():
            if self.PROBE_KIND[k][2] in ("bound", "holdout"):
                continue
            try:
                v = np.asarray(getter(d), float)
            except Exception:
                continue
            if np.ndim(v) == 0:
                v = np.full(len(d), float(v))
            if len(v) != len(d):
                continue
            vals.append(v); nus.append(d.nu); xs.append(d.x)
            labs.append(np.full(len(d),
                        1.0 if self.PROBE_KIND[k][1] == "spheroid" else 0.0))
        V, LB = np.concatenate(vals), np.concatenate(labs)
        NU, XX = np.concatenate(nus), np.concatenate(xs)
        m = np.isfinite(V) & np.isfinite(NU) & (NU > 0) & (XX > 0)
        V, LB, NU, XX = V[m], LB[m], NU[m], XX[m]

        def co(u, w):
            u = u - u.mean(); w = w - w.mean()
            dn = math.sqrt((u @ u) * (w @ w))
            return float(u @ w / dn) if dn > 0 else 0.0

        if len(V) < 50 or np.ptp(V) == 0:
            out = dict(n=len(V), var_label=0.0, var_resid=0.0, label_resid=0.0,
                       partial=0.0, p=1.0, verdict="CARRIES NIL")
            if verbose:
                print(f"   confound check on '{name}': constant or too few "
                      f"points -> CARRIES NIL")
            return out
        y = np.log10(NU) - np.log10(1 / (1 - np.exp(-np.sqrt(XX))))
        rv, rl, ry = self._rank(V), self._rank(LB), self._rank(y)
        r_vl, r_vy, r_ly = co(rv, rl), co(rv, ry), co(rl, ry)
        den = math.sqrt(max(1e-12, (1 - r_vl ** 2) * (1 - r_ly ** 2)))
        part = (r_vy - r_vl * r_ly) / den
        i0, i1 = np.where(LB == 0)[0], np.where(LB == 1)[0]
        null = np.empty(nperm)
        for i in range(nperm):
            vp = rv.copy()
            vp[i0] = rng.permutation(vp[i0]); vp[i1] = rng.permutation(vp[i1])
            a_, b_ = co(vp, rl), co(vp, ry)
            null[i] = (b_ - a_ * r_ly) / math.sqrt(
                max(1e-12, (1 - a_ ** 2) * (1 - r_ly ** 2)))
        p = float(np.mean(np.abs(null) >= abs(part)))
        # EFFECT SIZE, not significance. At n ~ 4000 pure noise reaches
        # p = 0.04 with |partial| = 0.031, so a p-value cannot decide this.
        # FLOOR is the 95th percentile of |partial| over 600 pure-noise draws
        # (p05_floor.py). The bare label itself reaches |r| ~ 0.56, i.e. 18x
        # the floor, while the best physical variable tested reaches 4.7x.
        FLOOR = 0.031
        ratio = abs(part) / FLOOR
        if ratio <= 1.0:
            verdict = "CARRIES NIL -- indistinguishable from noise"
        elif ratio <= 3.0:
            verdict = ("INDETERMINATE -- above noise but too weak for this "
                       "check to decide; use an independent control")
        else:
            verdict = (f"clears the noise floor by {ratio:.1f}x, but is still "
                       f"only {100*abs(part)/max(abs(r_ly),1e-9):.0f}% of the "
                       f"bare label -- NOT a kill or a pass on its own")
        if verbose:
            print(f"   confound check on '{name}'  (n = {len(V)}, "
                  f"holdouts excluded)")
            print(f"      corr(variable, spheroid label)   {r_vl:+.3f}")
            print(f"      corr(variable, RAR residual)     {r_vy:+.3f}")
            print(f"      corr(label,    RAR residual)     {r_ly:+.3f}")
            print(f"      PARTIAL (variable | label)       {part:+.3f}  "
                  f"(p = {p:.4f})")
            print(f"      -> {verdict}")
        return dict(n=len(V), var_label=r_vl, var_resid=r_vy, label_resid=r_ly,
                    partial=part, p=p, verdict=verdict)

    def caveats(self):
        for k in self.d:
            print(f"   {k:<10}{self.CAVEATS.get(k, '')}")


if __name__ == "__main__":
    BAR = "="*78
    print(BAR); print("INVARIANT GRAVITY BENCH"); print(BAR)
    b = Bench()
    print()
    b.summary()
    print("\ncaveats carried with each probe:")
    b.caveats()

    print("\n" + BAR); print("baseline: the acceleration law, no free parameters"); print(BAR)
    b.score(lambda d: 1.0/(1-np.exp(-np.sqrt(d.x))))

    print("\n" + BAR); print("mandatory confound checks"); print(BAR)
    b.confound("sphericity", lambda d: d.sphericity)
    print()
    b.confound("compact = r/extent", lambda d: d.r/d.extent)
    print()
    b.confound("log10 enclosed mass", lambda d: np.log10(d.M/MSUN))
