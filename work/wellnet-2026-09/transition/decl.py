"""DECLARATION BLOCK -- frozen before any residual is examined.

This module contains NOTHING but declarations: the frozen dynamics law, the
definition of the common observable, the prespecified model hierarchy, the
survey-offset priors and their external sources, and the blind plan.  It opens
no data file.  Its SHA-256 is printed and recorded in results.json so that the
hierarchy can be shown to predate every number in this lane.

WHY A SEPARATE FILE
-------------------
Run AI's checklist item "refitting on the held-out set" and Run J's
"training gain rises with complexity while blind gain falls" both come down to
the same discipline: the model space must be written down before the residual
is seen.  Putting it in its own file with its own hash is the cheapest way to
make that auditable.

THE COMMON OBSERVABLE
---------------------
    S  =  observed lensing response / response predicted from
          (baryons + frozen dynamics law + NO SLIP)

S is exactly the lensing response Sigma_s = (Phi + Psi) / (2 Psi) of the
closure lane, measured in three regimes that have never been put on one axis:

    eFEDS  raw DECADE tangential reduced shear around 496 X-ray systems,
           0.29 - 5.0 Mpc.  S enters the forward model as Sigma_s(r) applied
           to the 3-D mass and re-projected; the likelihood is chi2 on g_+.
    LoCuSS Subaru weak-lensing aperture mass M_WL(<r500) for massive clusters.
           S_i = M_WL,i / M_dyn(r500,i).
    SL     Hubble Frontier Field multiple-image systems.  For each image
           system, S = 1 / kappa_bar_dyn(<theta>) with that system's own
           Sigma_cr: the factor by which the frozen law's convergence must be
           multiplied to make the observed arcs critical.

IDENTIFIABILITY.  Within lensing alone S is exactly degenerate with the lens
mass (closure lane section 0, relative chi2 difference 0.0e+00).  S is
identifiable ONLY because the dynamics law is frozen first, from SPARC
rotation curves, and is never refitted here.

THE ALGEBRAIC TRAP THIS LANE MUST AVOID
---------------------------------------
    (r/Mpc)^s * (r/Mpc)^-1 = (r/Mpc)^(s-1)

A radial closure and a radial modification of the force law are the same
object.  Therefore beta below is NOT a nuisance parameter: it is the
measurement.  Consequently the survey offsets are pure amplitudes, constant in
r, with informative external priors -- never radial, never free.
"""
from __future__ import annotations

import math

# --------------------------------------------------------------- frozen law
# a0 from the tournament BASE_rar, fitted to SPARC TRAIN rotation curves.
# Dynamics only.  Not refitted anywhere in this lane.
A0_RAR = 1.0844e-10          # m s^-2
LAW_NAME = "RAR (Lelli+2017 interpolation), a0 = 1.0844e-10 m/s^2, SPARC train"

# --------------------------------------------------- prespecified hierarchy
# u = ln(M_gas,500 / M0),  v = ln(r / R500)
M0_MSUN = 1.0e14             # pivot, declared, near the geometric mean of the
                             # pooled sample -- fixed, never fitted
A0_PIVOT = 1.0               # g_b / a0 pivot for the acceleration model

HIERARCHY = {
    "H0":   dict(pars=[],                     k=0,
                 desc="ln S = 0.  No excess anywhere."),
    "H_P":  dict(pars=[],                     k=0,
                 desc="ln S = survey offsets only.  PURE PIPELINE: each survey "
                      "carries a constant, no dependence on M, r or g."),
    "H_M":  dict(pars=["c", "alpha"],         k=2,
                 desc="ln S = c + alpha ln(M/M0).  MASS."),
    "H_R":  dict(pars=["c", "beta"],          k=2,
                 desc="ln S = c + beta ln(r/R500).  CLUSTERCENTRIC RADIUS."),
    "H_G":  dict(pars=["c", "gamma"],         k=2,
                 desc="ln S = c + gamma ln(g_b/a0).  ACCELERATION REGIME."),
    "H_MR": dict(pars=["c", "alpha", "beta"], k=3,
                 desc="ln S = c + alpha ln(M/M0) + beta ln(r/R500)."),
    "H_T":  dict(pars=["c", "A", "lnxt"],     k=3,
                 desc="TRANSITION, form declared in advance: "
                      "ln S = c + A / (1 + (r/(x_t R500))^p) with p = 2 FIXED. "
                      "A smooth inner-excess plateau of height A that switches "
                      "off outside x_t R500.  Admitted ONLY if H_MR fails to "
                      "describe the pooled data."),
}
TRANSITION_P = 2.0           # declared in advance, NOT fitted

# The pure-pipeline model H_P always carries the survey offsets, as do all the
# others; H_P is the model in which the offsets are the ONLY structure.  For a
# like-for-like comparison the offsets are counted as data-constrained
# nuisances in every model (they carry priors), so k above counts only the
# gravity parameters.

# ------------------------------------------- survey offsets and their priors
# Each dataset gets ONE constant multiplicative offset with an EXTERNALLY
# CONSTRAINED prior.  Declared as a Gaussian on the natural log of the offset.
# These are calibration offsets, not free amplitudes: a survey whose whole
# signal could be absorbed by its offset would be uninformative by
# construction, which is exactly what the priors prevent.
OFFSET_PRIORS = {
    "efeds": dict(
        mean=0.0, sd=0.05 * math.log(10.0),
        source="DECADE-vs-HSC shear amplitude, MASS-MATCHED, closure lane "
               "section 3: DECADE/HSC = 0.914 (top 50% by M_gas,500) and "
               "0.875 (top 20%), i.e. -0.039 and -0.058 dex.  Prior sd taken "
               "as 0.05 dex.  NOTE this is the SHEAR CALIBRATION only; the "
               "X-ray density-fit bias is a separate bracket, applied in "
               "PART 3, not folded into this prior."),
    "locuss": dict(
        mean=0.0, sd=0.05 * math.log(10.0),
        source="Okabe & Smith (2016) Subaru weak-lensing mass calibration: "
               "the quoted systematic mass-calibration budget is ~5-10%.  "
               "Prior sd taken as 0.05 dex (12%), the conservative end."),
    "sl": dict(
        mean=0.0, sd=0.15 * math.log(10.0),
        source="Monopole/lens-model systematic.  The Refsdal lane measured "
               "source-plane rms 0.40-0.61 arcsec against theta_E = 10.6 "
               "arcsec for a merging cluster, and the two independent "
               "estimators (images, delay) disagreed by 11-14%.  Prior sd "
               "taken as 0.15 dex, deliberately wide."),
}

# ------------------------------------------------------------- the blind plan
# DECLARED BEFORE ANY FIT.  Fit on eFEDS + SL, freeze, predict LoCuSS once.
#
# Rationale: eFEDS supplies the radial axis at low mass; the strong-lensing
# cores supply small r/R500 at high mass.  LoCuSS occupies the corner NEITHER
# covers -- high mass at r ~ R500 -- and it is exactly the corner where H_M
# and H_R make OPPOSITE predictions:
#     H_M  (mass)   -> LoCuSS is the second-most-massive set, so S >> 1
#     H_R  (radius) -> LoCuSS sits at r/R500 ~ 1, so S ~ e^c ~ 1
# HONESTY NOTE, recorded here rather than discovered later: the LoCuSS excess
# (E median 1.62) is already in the programme record and has been read by the
# author of this lane.  The freeze is therefore PROCEDURAL, not epistemic:
# the code and the predicted intervals are written and hashed before this
# lane's own LoCuSS forward chain is run, and the held-out set is touched
# once.  It is not a blind test in the strong sense and is not reported as one.
BLIND = dict(train=("efeds", "sl"), held="locuss",
             note="declared in decl.py before any fit; see docstring")

# ------------------------------------------------- variable definitions
# PRIMARY definitions, declared in advance.
#
# M   = M_gas,500, the gas mass inside R500, from the same profile that sets
#       g_b.  Shared with the denominator of S by construction -- audited and
#       simulated in PART 3, never assumed harmless.
# ALT = k T_X (core-excised where published).  Breaks the shared-input path
#       with the density fit, at the cost of being a proxy rather than a mass.
#
# R500 = DYNAMICAL, defined under the FROZEN LAW from the baryons alone:
#        the radius where  M_dyn(R) = 500 rho_c(z) (4 pi/3) R^3,
#        M_dyn(r) = g_law(r) r^2 / G.
#        This uses NO lensing mass, NO NFW fit and NO hydrostatic mass, so it
#        is available identically in all three surveys and cannot import the
#        answer.  It is also the definition that removes LoCuSS's
#        r500 = f(M_WL) shared-input problem from the radius axis.
# ALT  = catalogue R500 (Bahar+2022 for eFEDS, r500(M_WL) for LoCuSS).  Not
#        available for the strong-lensing cores without a mass model, so the
#        alternative can only be run on eFEDS + LoCuSS.
PRIMARY_MASS = "M_gas500_dyn"
PRIMARY_R500 = "R500_dyn"

# --------------------------------------------------------------- the question
QUESTION = ("Is the cluster lensing residual organised by MASS, by "
            "CLUSTERCENTRIC RADIUS, by ACCELERATION REGIME, or by SURVEY "
            "PIPELINE?  H_M, H_R, H_G and H_P are those four stories, each "
            "with the same number of gravity parameters where possible, run "
            "through one forward framework.")

DECL_VERSION = "transition-decl-1"
