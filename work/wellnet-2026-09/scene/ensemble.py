"""ensemble.py -- Job 2: probabilistic scenes, not best-fit scenes.

    "Do not pretend projected galaxy positions determine exact depth.  Generate
     an ensemble consistent with redshifts, cluster phase space, morphology,
     substructure, spatial selection."   -- the charter, Stage 1 / step 2

WHAT IS AND IS NOT KNOWN ABOUT A CLUSTER MEMBER
-----------------------------------------------
Measured : sky position (0.026 arcsec agreement between independent solutions),
           one scalar line-of-sight velocity, a flux, a morphology.
Not measured : the line-of-sight DEPTH z, and both transverse velocity
           components.  cz_obs = H(z) d + v_pec is one equation in two
           unknowns.  Reading the dispersion as Hubble flow implies a depth
           3.2-4.7x the cluster's own diameter and, because of the
           Finger-of-God distortion, makes inferred depth ANTI-correlate with
           true 3-D radius.

So the depth is not noisy -- it is *absent*, and any single-value substitute is
a fabrication.  What CAN be written down is a posterior.

THE POSTERIOR
-------------
For member i with projected radius R_i and line-of-sight velocity v_i:

    p(z_i | R_i, v_i, morph_i, theta)
        proportional to
            n_3d(sqrt(R_i^2 + z_i^2) ; theta)          <- where galaxies ARE
          x N(v_i ; 0, sigma_los^2(r_i ; theta))       <- cluster phase space
          x p(morph_i | r_i ; theta)                   <- morphology-density
          x S(R_i, z_i)                                <- spatial selection

CRITICAL: no term in this posterior is a MASS MODEL.  `n_3d` is the Abel
deprojection of the OBSERVED projected member counts, and `sigma_los(r)` is the
OBSERVED projected dispersion profile.  Both describe where the galaxies are and
how fast they move.  Neither assumes a halo, an NFW profile, or dark matter, and
neither assumes a gravity law -- which is mandatory here, because the gravity
law is the thing under test.  Using an NFW-based phase-space model to place the
sources would smuggle the answer into the scene.

CORRELATION ACROSS MEMBERS
--------------------------
Depths are NOT independent.  Members of a substructure share a bulk depth, so
the sampler draws a substructure-level offset first and conditions member depths
on it.  Drawing each member's depth independently would destroy exactly the
lumpy, correlated geometry a well-network law is supposed to be sensitive to.

WEIGHTS AND ESS
---------------
Every factor above is a ONE-DIMENSIONAL function of z for a given member, so all
of them go into an exact grid inverse-CDF proposal and the importance weights
are identically zero: ESS equals the draw count exactly.

That is not how this module was first written, and the difference matters.  The
first version put the morphology term in by importance REWEIGHTING, which is
formally correct and numerically hopeless: the log weight is a SUM over members,
so its variance grows with N.  Measured ESS was 17.2 out of 64 draws at only 120
members, and it would be far worse at the 300 a real cluster has.  An ensemble
whose ESS has collapsed is a point estimate wearing a posterior's clothes, which
is the exact failure this module exists to prevent.  `exact_morphology=False`
keeps the old path available so the contrast can be measured rather than
asserted -- `SceneEnsemble.ess()` is a live diagnostic either way.

VALIDATION
----------
`coverage_test()` runs the sampler on a SYNTHETIC cluster whose true depths are
known and measures the frequentist coverage of its credible intervals.  A
sampler that has not been coverage-tested is an assumption, not a posterior.
No real cluster data is opened anywhere in this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from metadata import Registry
from schema import (Fixed, Uncertain, Node, Edge, SceneGraph, SceneEnsemble,
                    SceneRealisation)

KPC = 3.0856775814913673e19
MPC = 1000.0 * KPC
MSUN = 1.98892e30


# ====================================================== observed descriptions

@dataclass
class ProjectedProfile:
    """A description of WHERE the member galaxies are, fitted to counts.

    A projected King-like profile,  N(R) = N0 / (1 + (R/rc)^2)^alpha,  whose
    3-D deprojection is analytic.  This is a curve through observed galaxy
    counts.  It is not a mass profile and it carries no gravity assumption.
    """
    rc: float                      # core radius, metres
    alpha: float = 1.0             # projected slope index
    n0: float = 1.0

    def sigma_2d(self, R: np.ndarray) -> np.ndarray:
        return self.n0 / (1.0 + (R / self.rc) ** 2) ** self.alpha

    def n_3d(self, r: np.ndarray) -> np.ndarray:
        """Abel deprojection of sigma_2d, up to a constant.

        For Sigma(R) = (1 + (R/rc)^2)^-alpha the deprojection is
        n(r) propto (1 + (r/rc)^2)^-(alpha + 1/2).  Exact, so the sampler's
        radial prior is analytic rather than numerical.
        """
        return self.n0 / (1.0 + (r / self.rc) ** 2) ** (self.alpha + 0.5)


@dataclass
class DispersionProfile:
    """A description of HOW FAST members move, fitted to observed velocities.

    sigma_los(r) = sigma0 * (1 + (r/rs)^2)^(-beta/2).  Again: a fit to observed
    velocities, not a mass model.
    """
    sigma0: float                  # m/s at the centre
    rs: float                      # metres
    beta: float = 0.35

    def sigma_los(self, r: np.ndarray) -> np.ndarray:
        return self.sigma0 / (1.0 + (r / self.rs) ** 2) ** (0.5 * self.beta)


@dataclass
class MorphologyPrior:
    """The morphology-density relation, as a likelihood on 3-D radius.

    p(early | r) = f_inf + (f_0 - f_inf) / (1 + (r/r_m)^2).  Early types are
    centrally concentrated, so an early-type member is mildly informative about
    its 3-D radius and therefore about its depth.  The effect is weak, and the
    sampler must show it is weak rather than assume it.
    """
    f0: float = 0.85
    f_inf: float = 0.25
    r_m: float = 400.0 * KPC

    def p_early(self, r: np.ndarray) -> np.ndarray:
        return self.f_inf + (self.f0 - self.f_inf) / (1.0 + (r / self.r_m) ** 2)

    def loglike(self, r: np.ndarray, is_early: np.ndarray) -> np.ndarray:
        p = np.clip(self.p_early(r), 1e-6, 1 - 1e-6)
        return np.where(is_early, np.log(p), np.log1p(-p))


@dataclass
class Selection:
    """Spatial selection AND the declared scene volume.

    Two different bounds, and BUG 1 of this lane was conflating them:

      `r_max`      the projected survey footprint -- a selection on the sky.
      `r_max_3d`   the radius of the volume the scene is declared to describe.

    The magnitude selection itself is depth-independent at a fixed cluster
    redshift (1 Mpc of depth is <0.01 mag at z~0.3, far below the catalogue
    limit's own uncertainty), so it constrains R and not z.  But the SCENE is
    still a statement about a finite volume, and the depth prior has to be
    truncated at the same boundary the scene claims, or the posterior is
    over-dispersed: it spreads members through a volume the scene does not
    claim to describe.  That sounds conservative and is not -- an
    over-dispersed depth ensemble washes out exactly the lumpy correlated
    geometry a well-network law is meant to be sensitive to.
    """
    r_max: float = 3.0 * MPC              # projected footprint
    r_max_3d: float = 3.0 * MPC           # declared scene volume
    note: str = ("magnitude selection is depth-independent at fixed cluster "
                 "redshift; the depth bound is the DECLARED SCENE VOLUME, "
                 "which is a different statement and must match whatever the "
                 "scene claims to describe")

    def logS(self, R: np.ndarray, z: np.ndarray) -> np.ndarray:
        r3 = np.sqrt(np.asarray(R, float) ** 2 + np.asarray(z, float) ** 2)
        ok = (np.asarray(R, float) <= self.r_max) & (r3 <= self.r_max_3d)
        return np.where(ok, 0.0, -np.inf)


@dataclass
class ClusterPhaseSpace:
    """The full description a depth posterior needs.  Hyperparameters carry
    their own uncertainty and are re-drawn for each realisation, so the
    ensemble marginalises over the description as well as over the depths."""
    profile: ProjectedProfile
    dispersion: DispersionProfile
    morphology: MorphologyPrior = field(default_factory=MorphologyPrior)
    selection: Selection = field(default_factory=Selection)
    #: fractional 1-sigma uncertainty on rc and sigma0
    rc_frac_err: float = 0.15
    sigma0_frac_err: float = 0.08
    #: substructure bulk-depth scale
    substructure_depth_sigma: float = 0.6 * MPC

    def perturb(self, rng: np.random.Generator) -> "ClusterPhaseSpace":
        p = ProjectedProfile(
            rc=self.profile.rc * float(np.exp(rng.normal(0, self.rc_frac_err))),
            alpha=self.profile.alpha, n0=self.profile.n0)
        d = DispersionProfile(
            sigma0=self.dispersion.sigma0
            * float(np.exp(rng.normal(0, self.sigma0_frac_err))),
            rs=self.dispersion.rs, beta=self.dispersion.beta)
        return ClusterPhaseSpace(p, d, self.morphology, self.selection,
                                 self.rc_frac_err, self.sigma0_frac_err,
                                 self.substructure_depth_sigma)


# ================================================================== members

@dataclass
class MemberObservation:
    """What is actually measured for one member.  Note what is NOT here."""
    mid: str
    x: float                      # metres, measured (sky)
    y: float                      # metres, measured (sky)
    v_los: float                  # m/s, measured
    m_star: float                 # kg
    is_early: bool
    p_member: float = 1.0
    substructure: int = -1        # -1 = not assigned to a substructure

    @property
    def R(self) -> float:
        return math.hypot(self.x, self.y)


# ================================================================== sampler

class DepthSampler:
    """Draws line-of-sight depths from the exact 1-D conditional.

    Method: build p(z) on a grid, normalise, invert the CDF.  Exact to the grid
    resolution and, unlike MCMC, gives INDEPENDENT draws -- which matters here
    because the downstream commutation gate compares scene functionals across
    realisations and autocorrelated draws would understate their spread.
    """

    def __init__(self, phase: ClusterPhaseSpace, z_max: float = 5.0 * MPC,
                 n_grid: int = 513, exact_morphology: bool = True):
        self.phase = phase
        self.z_max = z_max
        self.n_grid = int(n_grid)
        self.zgrid = np.linspace(-z_max, z_max, self.n_grid)
        #: BUG 8 of this lane.  The first version applied the
        #: morphology-density term by IMPORTANCE REWEIGHTING, which is correct
        #: but scales badly: the log weight is a SUM over members, so its
        #: variance grows with N and the effective sample size collapses --
        #: measured ESS 17.2 out of 64 draws at only 120 members, and it would
        #: be far worse at the 300 a real cluster has.  An ensemble whose ESS
        #: has collapsed is a point estimate wearing a posterior's clothes,
        #: which is the exact failure this module exists to prevent.
        #:
        #: The fix is structural rather than numerical: every factor in the
        #: depth posterior is a one-dimensional function of z for a given
        #: member, so ALL of them can go into the exact grid proposal and none
        #: needs a weight at all.  With this True the weights are identically
        #: zero and ESS equals the draw count exactly.  It is kept switchable
        #: so the contrast can be measured rather than asserted.
        self.exact_morphology = bool(exact_morphology)

    def log_conditional(self, R: float, v_los: float, is_early: bool,
                        phase: ClusterPhaseSpace,
                        z: Optional[np.ndarray] = None,
                        with_morphology: Optional[bool] = None) -> np.ndarray:
        z = self.zgrid if z is None else np.asarray(z, float)
        r = np.sqrt(R * R + z * z)
        lp = np.log(np.maximum(phase.profile.n_3d(r), 1e-300))
        s = np.maximum(phase.dispersion.sigma_los(r), 1.0)
        lp = lp - 0.5 * (v_los / s) ** 2 - np.log(s)
        lp = lp + phase.selection.logS(np.full_like(z, R), z)
        use = self.exact_morphology if with_morphology is None else with_morphology
        if use:
            lp = lp + phase.morphology.loglike(r, np.full(z.shape, is_early))
        return lp

    def draw(self, R: float, v_los: float, is_early: bool,
             phase: ClusterPhaseSpace, rng: np.random.Generator,
             n: int = 1, offset: float = 0.0) -> Tuple[np.ndarray, float]:
        """Return (n draws of z, log importance weight for the morphology term).

        The density x velocity factor is sampled exactly, so it adds no weight
        variance.  The morphology term is applied as a weight: it is a weak,
        genuinely uncertain constraint and forcing it into the proposal would
        overstate its strength.
        """
        lp = self.log_conditional(R, v_los, is_early, phase)
        lp = lp - lp.max()
        p = np.exp(lp)
        tot = p.sum()
        if not np.isfinite(tot) or tot <= 0:
            return np.zeros(n), -np.inf
        cdf = np.cumsum(p)
        cdf /= cdf[-1]
        u = rng.random(n)
        zz = np.interp(u, cdf, self.zgrid) + offset
        if self.exact_morphology:
            return zz, 0.0          # the term is already in the proposal
        r = np.sqrt(R * R + zz * zz)
        lw = float(np.mean(phase.morphology.loglike(r, np.full(n, is_early))))
        return zz, lw

    def credible_interval(self, R: float, v_los: float, is_early: bool,
                          phase: ClusterPhaseSpace,
                          level: float = 0.68) -> Tuple[float, float]:
        lp = self.log_conditional(R, v_los, is_early, phase)
        p = np.exp(lp - lp.max())
        c = np.cumsum(p)
        c /= c[-1]
        lo = 0.5 * (1 - level)
        return (float(np.interp(lo, c, self.zgrid)),
                float(np.interp(1 - lo, c, self.zgrid)))


class SceneEnsembleBuilder:
    """Turn measured member observations into a probabilistic scene graph.

    The output is a `SceneGraph` in which x and y are `Fixed` and z is
    `Uncertain`, plus a joint sampler that honours the substructure correlation.
    """

    def __init__(self, scene_id: str, registry: Registry,
                 phase: ClusterPhaseSpace,
                 members: Sequence[MemberObservation],
                 unresolved_frac: float = 0.0,
                 unresolved_mass_frac: float = 0.0,
                 seed: int = 20260904, exact_morphology: bool = True):
        self.scene_id = scene_id
        self.registry = registry
        self.phase = phase
        self.members = list(members)
        self.unresolved_frac = float(unresolved_frac)
        self.unresolved_mass_frac = float(unresolved_mass_frac)
        self.seed = int(seed)
        self.exact_morphology = bool(exact_morphology)
        self.sampler = DepthSampler(phase, exact_morphology=exact_morphology)

    # ------------------------------------------------------------------ build
    def build(self) -> SceneGraph:
        g = SceneGraph(self.scene_id, self.registry,
                       notes="depths are sampled, never assigned")
        for m in self.members:
            g.add_node(Node(
                id=m.mid, node_type="galaxy",
                attrs={
                    "x": Fixed(m.x, "astrometric centroid, 0.026 arcsec"),
                    "y": Fixed(m.y, "astrometric centroid, 0.026 arcsec"),
                    # placeholder marginal; the joint sampler overrides it
                    "z": Uncertain(
                        draw=self._marginal_z(m),
                        support=(-self.sampler.z_max, self.sampler.z_max),
                        label=f"depth posterior for {m.mid}",
                        correlated_with=("substructure_offset",)),
                    "v_los": Fixed(m.v_los, "spectroscopic line centroid"),
                    "m_star": Fixed(m.m_star, "population fit at a global IMF"),
                    "p_member": Fixed(m.p_member, "membership probability"),
                },
                source="member catalogue (positions, redshifts, photometry)",
                presupposes_dm=False,
                meta={"substructure": m.substructure,
                      "is_early": m.is_early}))
        g.add_joint_sampler(("z",), self._joint_depths)
        return g

    def _marginal_z(self, m: MemberObservation):
        def draw(rng, n):
            z, _ = self.sampler.draw(m.R, m.v_los, m.is_early, self.phase,
                                     rng, n)
            return z
        return draw

    def _joint_depths(self, rng: np.random.Generator, graph: SceneGraph):
        """Draw ALL member depths together, honouring substructure and
        re-drawing the phase-space hyperparameters for this realisation."""
        phase = self.phase.perturb(rng)
        self.sampler.exact_morphology = self.exact_morphology
        subs = sorted({m.substructure for m in self.members
                       if m.substructure >= 0})
        offs = {s: float(rng.normal(0.0, phase.substructure_depth_sigma))
                for s in subs}
        out: Dict[str, Dict[str, float]] = {}
        lw = 0.0
        for m in self.members:
            off = offs.get(m.substructure, 0.0)
            z, w = self.sampler.draw(m.R, m.v_los, m.is_early, phase, rng, 1,
                                     offset=off)
            out[m.mid] = {"z": float(z[0])}
            lw += w
        return out, lw

    # ------------------------------------------------------------ diagnostics
    def diagnostics(self, ens: SceneEnsemble) -> Dict[str, Any]:
        """Everything needed to tell a posterior from a point estimate."""
        ids = [m.mid for m in self.members]
        Z = np.array([[d.node_attrs[i]["z"] for i in ids] for d in ens.draws])
        w = ens.weights()
        spread = Z.std(axis=0, ddof=1)
        # the correlation the substructure term is supposed to create
        rho = np.corrcoef(Z.T) if len(ids) > 1 else np.array([[1.0]])
        off = rho[np.triu_indices_from(rho, k=1)]
        R = np.array([m.R for m in self.members])
        # how much of the 3-D radius is actually determined?
        r3 = np.sqrt(R[None, :] ** 2 + Z ** 2)
        return {
            "n_draws": len(ens),
            "n_members": len(ids),
            "ess": ens.ess(),
            "ess_frac": ens.ess() / max(len(ens), 1),
            "depth_sd_median_Mpc": float(np.median(spread) / MPC),
            "depth_sd_min_Mpc": float(spread.min() / MPC),
            "depth_sd_max_Mpc": float(spread.max() / MPC),
            "mean_pairwise_depth_corr": float(off.mean()) if off.size else 0.0,
            "r3d_sd_over_r3d_median": float(
                np.median(r3.std(axis=0, ddof=1) / np.maximum(r3.mean(axis=0), 1))),
            "r_proj_median_Mpc": float(np.median(R) / MPC),
        }


# ============================================================== validation

def synthetic_cluster(n: int = 200, seed: int = 20260904,
                      n_sub: int = 3) -> Tuple[List[MemberObservation],
                                               ClusterPhaseSpace, np.ndarray]:
    """A synthetic cluster with KNOWN depths, for coverage testing.

    Purely generated numbers -- no catalogue is read.  Its parameters are of the
    right ORDER for a rich cluster (rc ~ 300 kpc, sigma ~ 1000 km/s) so the
    coverage test is run in the regime it will be used in, but nothing here is
    a measurement of any real cluster.
    """
    rng = np.random.default_rng(seed)
    prof = ProjectedProfile(rc=300.0 * KPC, alpha=1.0)
    disp = DispersionProfile(sigma0=1000.0e3, rs=700.0 * KPC, beta=0.35)
    phase = ClusterPhaseSpace(prof, disp)

    # draw true 3-D positions from n_3d by rejection, inside EXACTLY the
    # volume the sampler's selection declares -- see Selection's docstring.
    rmax = phase.selection.r_max_3d
    xs = []
    while len(xs) < n:
        p = rng.uniform(-rmax, rmax, size=(4 * n, 3))
        r = np.linalg.norm(p, axis=1)
        keep = r < rmax
        p, r = p[keep], r[keep]
        acc = rng.random(len(r)) < prof.n_3d(r) / prof.n_3d(np.array([0.0]))[0]
        xs.extend(p[acc].tolist())
    P = np.array(xs[:n])
    r = np.linalg.norm(P, axis=1)
    sub = rng.integers(-1, n_sub, size=n)
    v = rng.normal(0.0, disp.sigma_los(r))
    early = rng.random(n) < phase.morphology.p_early(r)
    ms = 10.0 ** rng.uniform(9.5, 11.5) * MSUN * np.ones(n)
    ms = 10.0 ** rng.uniform(9.5, 11.5, size=n) * MSUN
    mem = [MemberObservation(mid=f"g{i:04d}", x=float(P[i, 0]),
                             y=float(P[i, 1]), v_los=float(v[i]),
                             m_star=float(ms[i]), is_early=bool(early[i]),
                             substructure=int(sub[i]))
           for i in range(n)]
    return mem, phase, P[:, 2].copy()


def coverage_test(n_members: int = 200, seed: int = 20260904,
                  levels: Sequence[float] = (0.5, 0.68, 0.90, 0.95)
                  ) -> Dict[str, Any]:
    """Frequentist coverage of the depth credible intervals.

    Generates a synthetic cluster, hides the true depths, runs the sampler, and
    asks how often the true depth falls inside each interval.  A calibrated
    sampler returns coverage equal to the nominal level; anything materially
    below it means the ensemble is overconfident and the scene is a disguised
    point estimate.
    """
    mem, phase, z_true = synthetic_cluster(n_members, seed)
    s = DepthSampler(phase, exact_morphology=True)
    out: Dict[str, Any] = {"n_members": len(mem), "seed": seed, "levels": {}}
    for lv in levels:
        inside = 0
        for m, zt in zip(mem, z_true):
            lo, hi = s.credible_interval(m.R, m.v_los, m.is_early, phase, lv)
            inside += int(lo <= zt <= hi)
        cov = inside / len(mem)
        se = math.sqrt(max(cov * (1 - cov), 1e-9) / len(mem))
        out["levels"][f"{lv:.2f}"] = {
            "nominal": lv, "empirical": cov, "se": se,
            "z_score": (cov - lv) / se if se > 0 else 0.0,
            "calibrated": abs(cov - lv) < 3.0 * se}
    # how much information IS there?  Compare the posterior width to the prior.
    prior_sd = float(np.std(z_true))
    post_sd = float(np.mean([
        0.5 * (s.credible_interval(m.R, m.v_los, m.is_early, phase, 0.68)[1]
               - s.credible_interval(m.R, m.v_los, m.is_early, phase, 0.68)[0])
        for m in mem]))
    out["prior_sd_Mpc"] = prior_sd / MPC
    out["posterior_halfwidth_68_Mpc"] = post_sd / MPC
    out["information_gain_ratio"] = prior_sd / max(post_sd, 1e-9)
    out["all_calibrated"] = all(v["calibrated"] for v in out["levels"].values())
    return out
