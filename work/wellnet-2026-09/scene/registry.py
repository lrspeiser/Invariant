"""registry.py -- the populated parameter ontology for a cluster scene.

Every quantity a cluster scene graph can carry, each with the full seventeen-item
metadata contract from `metadata.py`.  This is the ontology INSTANCE; the
charter's section "The complete parameter ontology" is the specification, and
`ONTOLOGY_SECTION` tags each quantity with the charter section it comes from so
coverage is auditable.

Three things in here are load-bearing rather than descriptive:

  * `phi_baryon` is declared `translation=SHIFTS_BY_CONSTANT` and therefore
    CANNOT be constructed without a gauge rule -- `metadata.Quantity` raises.
    The four gauge-fixed variants below each name their boundary rule.  Run AH
    measured a 0.87 dex spread between two defensible global rules against a
    0.9 dex gate margin, so this is the difference between a result and an
    artefact.

  * The five EXACT IDENTITIES at the bottom are recorded symbolically.  The
    programme's `variable-lists-collapse` finding is that rich-looking variable
    sets shrink under identities; recording them here lets the compiler take
    the rank of a candidate's variable set BEFORE fitting.

  * `coarse_grain` is set from physics, not convenience.  Anything marked
    NONLINEAR or CATALOGUE_DEPENDENT may not be read off an averaged scene
    without passing the commutation gate in `commutation.py`.

NO OBSERVATIONAL DATA IS OPENED BY THIS MODULE.
"""
from __future__ import annotations

from typing import Dict, List

from metadata import (
    Dim, Quantity, Registry, DIMLESS, DIM_M, DIM_L, DIM_T, DIM_V, DIM_ACC,
    DIM_RHO, DIM_SIGMA, DIM_PHI, DIM_TIDAL, DIM_PRESSURE, DIM_TEMP,
    DIM_ENERGY, DIM_FLUX, DIM_G, DIM_C, DIM_ANGMOM_SPEC,
)

KPC = 3.0856775814913673e19          # m
MPC = 1000.0 * KPC
MSUN = 1.98892e30                    # kg

#: charter ontology sections, used to tag coverage
ONTOLOGY_SECTION = {
    1: "Location, time, scale, and reference",
    2: "Amount and composition of matter",
    3: "Thermodynamic and material state",
    4: "Source geometry",
    5: "Motion of matter",
    6: "Local gravitational descriptors",
    7: "Spacetime and curvature descriptors",
    8: "Directional structure",
    9: "Environment and cosmic-web state",
    10: "Network-of-wells parameters",
    11: "Nonlocal and path variables",
    12: "History and memory",
    13: "Light and radiation observables",
    14: "Cosmological and large-scale parameters",
    15: "Speculative latent-state parameters",
    16: "Universal constants and transition scales",
    17: "Measurement and astrophysical nuisances",
}

_SECTION_OF: Dict[str, int] = {}


def _q(section: int, **kw) -> Quantity:
    q = Quantity(**kw)
    _SECTION_OF[q.name] = section
    return q


#: Quantities the resolved scene DETERMINES through a declared procedure.
#: Not directly observed, but carrying no freedom of their own.
CONSTRUCTIBLE = frozenset({
    "g_N", "g_total", "g_vec", "M_enc", "tidal_tensor", "tidal_anisotropy",
    "phi_depth_saddle", "phi_depth_r500", "phi_depth_scaleradius",
    "phi_depth_volume", "rho_env", "n_wells", "graph_degree", "path_density",
    "path_void_fraction", "v_circ", "R500", "alignment_angle", "kappa",
    "P_e", "t", "mass",
})

#: Quantities the SCENE ENSEMBLE integrates over against a declared prior.
#: This class is the entire reason Stage 1 exists.
MARGINALISABLE = frozenset({
    "z", "r_3d", "v_x", "v_y", "m_bh", "m_icl", "m_h2", "m_hi", "sigma_turb",
    "t_since_merger", "upsilon_star", "distance", "psf_fwhm", "shear_m",
})

#: Free latent fields with no observational handle.  Admissible ONLY through a
#: generative law that says what creates them and how they evolve.
NON_IDENTIFIABLE = frozenset({
    "vacuum_order", "vacuum_axis", "field_memory", "phi_lensing", "phi_slip",
})


def _assign_identifiability(R: Registry) -> Registry:
    for q in R.all():
        if q.name in NON_IDENTIFIABLE:
            q.identifiability = "non_identifiable"
        elif q.name in MARGINALISABLE:
            q.identifiability = "marginalisable"
        elif q.name in CONSTRUCTIBLE:
            q.identifiability = "constructible"
        else:
            q.identifiability = "measured"
    return R


def build_registry() -> Registry:
    R = Registry()
    add = R.add

    # ============================================ 1. location, time, scale
    for ax in "xyz":
        add(_q(1, name=ax,
               definition=f"Cartesian {ax}-coordinate of the node in the "
                          f"cluster rest frame, origin at the declared centre. "
                          f"{'z is along the line of sight and is NOT measured'
                             if ax == 'z' else
                             'from the astrometric image centroid'}.",
               kind="scalar", dim=DIM_L, frame="cluster_rest",
               translation="COVARIANT", rotation="VECTOR", boost="COVARIANT",
               parity="ODD", time_reversal="EVEN", support="point",
               source=("sampled: only a scalar redshift is observed along the "
                       "line of sight" if ax == "z"
                       else "measured: astrometric centroid on a calibrated frame"),
               status="direct" if ax != "z" else "nuisance",
               resolution_m=0.15 * KPC if ax != "z" else None,
               uncertainty=("0.10 arcsec centroid -> 0.15-0.64 kpc "
                            "depending on cluster redshift" if ax != "z"
                            else "posterior; see ensemble.py -- 1 Mpc of depth "
                                 "makes only 5-9% of the measured dispersion"),
               covariance_group="astrometry" if ax != "z" else "los_depth",
               coarse_grain="INTENSIVE_LINEAR", causal="LOCAL_NOW",
               completeness="members above the catalogue magnitude limit",
               selection="magnitude-limited spectroscopic membership",
               allowed_ops=("add_same_dim", "multiply", "divide",
                            "power_rational", "gradient", "project_on_axis"),
               independently_measurable=(ax != "z"),
               measurability_note=(
                   "NOT independently measurable: cz = H(z) d + v_pec is one "
                   "equation in two unknowns, and the Finger-of-God distortion "
                   "makes inferred depth ANTI-correlate with true 3-D radius"
                   if ax == "z" else
                   "two independent astrometric solutions agree to 0.026 arcsec")))

    add(_q(1, name="t", definition="Epoch to which the node's state refers, "
                                   "as proper time in the cluster frame.",
           kind="scalar", dim=DIM_T, frame="cluster_rest",
           translation="SHIFTS_BY_CONSTANT",
           gauge="zero at the observed cluster epoch (t=0 at the lookback time "
                 "of the cluster redshift); differences only",
           rotation="SCALAR", boost="COVARIANT", parity="EVEN",
           time_reversal="ODD", support="point",
           source="derived from redshift and an assumed distance-redshift law",
           status="derived", uncertainty="cosmology-dependent",
           covariance_group="cosmology", coarse_grain="INTENSIVE_LINEAR",
           causal="LOCAL_NOW", completeness="complete", selection="none",
           allowed_ops=("add_same_dim", "multiply", "divide",
                        "power_rational", "time_derivative"),
           independently_measurable=False,
           measurability_note="requires a distance-redshift relation, which is "
                              "one of the things under test"))

    add(_q(1, name="r_proj", definition="Projected separation from the declared "
                                        "cluster centre on the sky.",
           kind="scalar", dim=DIM_L, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="point",
           source="measured: angular separation times an angular-diameter distance",
           status="direct", resolution_m=0.15 * KPC,
           uncertainty="centroid plus centre definition; the centre choice "
                       "dominates (BCG vs X-ray peak vs light centroid)",
           covariance_group="astrometry+centre", coarse_grain="INTENSIVE_LINEAR",
           causal="LOCAL_NOW", completeness="complete within the field",
           selection="field footprint", independently_measurable=True,
           measurability_note="yes, once a centre convention is declared",
           allowed_ops=("add_same_dim", "multiply", "divide", "power_rational",
                        "compare_to_scale")))

    add(_q(1, name="r_3d", definition="True three-dimensional separation from "
                                      "the cluster centre.",
           kind="scalar", dim=DIM_L, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="point",
           source="LATENT: sqrt(r_proj^2 + z^2) with z sampled, not measured",
           status="latent", uncertainty="inherits the whole line-of-sight "
                                        "depth posterior",
           covariance_group="los_depth", coarse_grain="INTENSIVE_LINEAR",
           causal="LOCAL_NOW", completeness="as r_proj", selection="as r_proj",
           depends_on=("r_proj", "z"),
           exact_identities=("r_3d^2 = r_proj^2 + z^2",),
           independently_measurable=False,
           measurability_note="no external galaxy cluster has a measured "
                              "member depth; this is the ensemble's job",
           allowed_ops=("add_same_dim", "multiply", "divide", "power_rational",
                        "compare_to_scale")))

    add(_q(1, name="smoothing_scale",
           definition="Declared Gaussian smoothing scale at which a field-type "
                      "quantity is evaluated. A PHYSICAL candidate variable, "
                      "not only a technical choice (charter section 1).",
           kind="scalar", dim=DIM_L, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="region", source="declared by the analysis",
           status="nuisance", smoothing_m=0.0, resolution_m=0.0,
           uncertainty="exact by declaration", covariance_group="analysis_choice",
           coarse_grain="SCALE_DEFINED", causal="LOCAL_NOW",
           completeness="n/a", selection="n/a", independently_measurable=True,
           measurability_note="it is a declaration, so it is exactly known; "
                              "the question is whether the RESULT depends on it",
           allowed_ops=("multiply", "divide", "power_rational",
                        "compare_to_scale")))

    add(_q(1, name="R500",
           definition="Radius enclosing 500 times the critical density. "
                      "RETAINED WITH A WARNING: it is defined through a mass, "
                      "so any statistic binned in r/R500 that also involves "
                      "mass shares a denominator with its own axis.",
           kind="scalar", dim=DIM_L, translation="INVARIANT", rotation="SCALAR",
           boost="INVARIANT", parity="EVEN", time_reversal="EVEN",
           support="global", source="DERIVED from a hydrostatic or lensing mass",
           status="derived", uncertainty="inherits the mass estimator's error "
                                         "and its equilibrium assumption",
           covariance_group="mass_estimator", coarse_grain="NONLINEAR",
           causal="LOCAL_NOW", completeness="n/a", selection="n/a",
           depends_on=("M_enc",), derived_under_theory=True,
           exact_identities=("M(<R500) = 500 rho_crit (4/3) pi R500^3",),
           independently_measurable=False,
           measurability_note="NOT a raw observable. Flagged by this "
                              "programme's R500 tautology audit (Run AT).",
           allowed_ops=("multiply", "divide", "compare_to_scale"),
           notes="shared-denominator hazard: see shared-denominator-artefacts"))

    # ============================================ 2. amount of matter
    _matter = [
        ("mass", "Total gravitating rest mass assigned to the node.", "derived",
         "sum of the component masses below"),
        ("m_star", "Stellar mass from a population-synthesis fit to multi-band "
                   "photometry at a declared IMF.", "derived",
         "photometry + stellar population model"),
        ("m_hi", "Atomic hydrogen mass from the 21 cm line flux.", "derived",
         "21 cm integrated flux and a distance"),
        ("m_h2", "Molecular gas mass from a CO line and a conversion factor.",
         "derived", "CO flux and alpha_CO"),
        ("m_gas_hot", "Hot ionised intracluster gas mass.", "derived",
         "X-ray surface brightness deprojection"),
        ("m_bh", "Central black-hole mass.", "derived",
         "M-sigma or a resolved kinematic measurement"),
        ("m_icl", "Diffuse intracluster stellar mass.", "derived",
         "deep surface photometry below a stated surface-brightness cut"),
    ]
    for nm, dfn, st, src in _matter:
        add(_q(2, name=nm, definition=dfn, kind="scalar", dim=DIM_M,
               translation="INVARIANT", rotation="SCALAR", boost="INVARIANT",
               parity="EVEN", time_reversal="EVEN", support="point",
               source=src, status=st,
               uncertainty=("0.06 dex global M/L offset with 0.045 dex "
                            "galaxy-to-galaxy scatter (mid-IR route)"
                            if nm == "m_star" else "component-specific"),
               covariance_group=("stellar_ML" if nm == "m_star"
                                 else f"{nm}_calibration"),
               coarse_grain="EXTENSIVE", causal="LOCAL_NOW",
               completeness="above the catalogue limit; an unresolved "
                            "population is carried statistically",
               selection="magnitude/flux limited",
               allowed_ops=("add_same_dim", "multiply", "divide",
                            "power_rational"),
               independently_measurable=(nm in ("m_star", "m_hi", "m_gas_hot")),
               measurability_note=("yes -- but only up to one GLOBAL nuisance "
                                   "(Upsilon* = 0.5 to 0.06 dex)"
                                   if nm == "m_star" else "component-specific")))

    add(_q(2, name="rho_star",
           definition="Stellar mass volume density at a point, from a "
                      "deprojected light profile times a mass-to-light ratio.",
           kind="scalar", dim=DIM_RHO, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="region",
           source="deprojected surface photometry", status="derived",
           resolution_m=0.2 * KPC, smoothing_m=0.2 * KPC,
           uncertainty="deprojection is not unique for a triaxial source",
           covariance_group="stellar_ML+deprojection",
           coarse_grain="INTENSIVE_LINEAR", causal="LOCAL_NOW",
           completeness="above the surface-brightness limit",
           selection="surface-brightness limited",
           independently_measurable=True,
           measurability_note="up to the global M/L and the deprojection",
           allowed_ops=("add_same_dim", "multiply", "divide", "power_rational",
                        "gradient", "smooth")))

    add(_q(2, name="n_e",
           definition="Electron number density of the hot gas at a point.",
           kind="scalar", dim=Dim.of(L=-3), translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="region",
           source="X-ray surface-brightness deprojection with an emissivity model",
           status="derived", resolution_m=10.0 * KPC, smoothing_m=10.0 * KPC,
           uncertainty="clumping-corrected; the correction is model dependent",
           covariance_group="xray_deprojection", coarse_grain="INTENSIVE_LINEAR",
           causal="LOCAL_NOW", completeness="inside the X-ray detection radius",
           selection="surface-brightness limited; stops at 0.7-1.1 R500 for "
                     "the Frontier Fields targets",
           independently_measurable=True,
           measurability_note="yes, and it does NOT presuppose dark matter -- "
                              "emissivity depends on n_e^2 and T, not on mass",
           allowed_ops=("add_same_dim", "multiply", "divide", "power_rational",
                        "gradient", "smooth")))

    # ============================================ 3. thermodynamic state
    add(_q(3, name="T_x", definition="X-ray spectroscopic gas temperature.",
           kind="scalar", dim=DIM_TEMP, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="region",
           source="spectral fit to the X-ray continuum and line ratios",
           status="direct", resolution_m=30.0 * KPC, smoothing_m=30.0 * KPC,
           uncertainty="counts-limited; unusable below ~150 counts in R500",
           covariance_group="xray_spectral", coarse_grain="NONLINEAR",
           causal="LOCAL_NOW", completeness="inside the X-ray detection radius",
           selection="requires enough counts for a spectral fit",
           independently_measurable=True,
           measurability_note="yes -- a spectroscopic measurement, "
                              "theory-independent",
           allowed_ops=("add_same_dim", "multiply", "divide", "power_rational",
                        "gradient"),
           notes="spectroscopic-like weighting is nonlinear in T, so a "
                 "temperature read off an averaged spectrum is NOT the average "
                 "temperature"))

    add(_q(3, name="P_e", definition="Electron thermal pressure, n_e k T.",
           kind="scalar", dim=DIM_PRESSURE, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="region",
           source="n_e times T_x, or SZ y deprojection", status="derived",
           resolution_m=30.0 * KPC, smoothing_m=30.0 * KPC,
           uncertainty="propagated from n_e and T", covariance_group="gas_thermo",
           coarse_grain="NONLINEAR", causal="LOCAL_NOW",
           completeness="as n_e", selection="as n_e",
           depends_on=("n_e", "T_x"),
           exact_identities=("P_e = n_e k_B T_x",),
           independently_measurable=True,
           measurability_note="measurable two independent ways (X-ray n_e*T "
                              "and SZ y), which is what makes it useful",
           allowed_ops=("add_same_dim", "multiply", "divide", "power_rational",
                        "gradient")))

    add(_q(3, name="sigma_turb",
           definition="One-dimensional turbulent velocity of the hot gas.",
           kind="scalar", dim=DIM_V, translation="INVARIANT",
           rotation="SCALAR", boost="FRAME_FIXED", parity="EVEN",
           time_reversal="EVEN", support="region",
           source="X-ray line broadening (microcalorimeter) or a surface-"
                  "brightness fluctuation argument",
           status="derived", resolution_m=50.0 * KPC, smoothing_m=50.0 * KPC,
           uncertainty="large; measured directly for very few clusters",
           covariance_group="gas_kinematics", coarse_grain="NONLINEAR",
           causal="LOCAL_NOW", completeness="rare", selection="bright cores only",
           independently_measurable=False,
           measurability_note="not available for the Frontier Fields targets",
           allowed_ops=("multiply", "divide", "power_rational")))

    # ============================================ 4. source geometry
    add(_q(4, name="r_e", definition="Half-light radius from a Sersic fit.",
           kind="scalar", dim=DIM_L, translation="INVARIANT", rotation="SCALAR",
           boost="INVARIANT", parity="EVEN", time_reversal="EVEN",
           support="region", source="Sersic fit to a calibrated image",
           status="derived", resolution_m=0.1 * KPC, smoothing_m=0.1 * KPC,
           uncertainty="PSF-model dependent", covariance_group="morphology_fit",
           coarse_grain="NONLINEAR", causal="LOCAL_NOW",
           completeness="only where a resolved fit exists",
           selection="HST-resolved members only; absent for A370 and MACS J0717",
           independently_measurable=True,
           measurability_note="yes where fitted; MISSING for 2 of 7 target "
                              "clusters, which is a real inventory hole",
           allowed_ops=("multiply", "divide", "power_rational",
                        "compare_to_scale")))

    add(_q(4, name="sersic_n", definition="Sersic index of the light profile.",
           kind="scalar", dim=DIMLESS, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="region",
           source="Sersic fit to a calibrated image", status="derived",
           resolution_m=0.1 * KPC, smoothing_m=0.1 * KPC,
           uncertainty="degenerate with sky subtraction at large n",
           covariance_group="morphology_fit", coarse_grain="NONLINEAR",
           causal="LOCAL_NOW", completeness="as r_e", selection="as r_e",
           independently_measurable=True,
           measurability_note="yes where fitted",
           allowed_ops=("multiply", "divide", "power_rational",
                        "log_dimensionless")))

    add(_q(4, name="axis_ratio_q",
           definition="Projected minor-to-major axis ratio of the light.",
           kind="scalar", dim=DIMLESS, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="region",
           source="second moments or a Sersic fit", status="direct",
           resolution_m=0.1 * KPC, smoothing_m=0.1 * KPC,
           uncertainty="PSF-dependent at small size",
           covariance_group="morphology_fit", coarse_grain="NONLINEAR",
           causal="LOCAL_NOW", completeness="all detected members",
           selection="detection limited", independently_measurable=True,
           measurability_note="available for all seven target clusters",
           allowed_ops=("multiply", "divide", "power_rational",
                        "log_dimensionless")))

    add(_q(4, name="position_angle",
           definition="Position angle of the light's major axis, east of north.",
           kind="pseudoscalar", dim=DIMLESS, translation="INVARIANT",
           rotation="FRAME_DEPENDENT", boost="INVARIANT", parity="ODD",
           time_reversal="EVEN", support="region",
           source="second moments or a Sersic fit", status="direct",
           resolution_m=0.1 * KPC, smoothing_m=0.1 * KPC,
           uncertainty="degenerate as q -> 1", covariance_group="morphology_fit",
           coarse_grain="NONLINEAR", causal="LOCAL_NOW",
           completeness="all detected members", selection="detection limited",
           independently_measurable=True,
           measurability_note="yes, but it is a SKY-FRAME angle: an alignment "
                              "statistic built from it must state its frame",
           allowed_ops=("project_on_axis", "compare_to_scale")))

    add(_q(4, name="M_enc",
           definition="Baryonic mass enclosed inside a stated radius of the "
                      "declared centre, summed over resolved components.",
           kind="scalar", dim=DIM_M, translation="INVARIANT", rotation="SCALAR",
           boost="INVARIANT", parity="EVEN", time_reversal="EVEN",
           support="region", source="integral of the resolved source scene",
           status="derived", resolution_m=1.0 * KPC, smoothing_m=1.0 * KPC,
           uncertainty="component-wise", covariance_group="baryon_budget",
           coarse_grain="EXTENSIVE", causal="LOCAL_NOW",
           completeness="above the completeness threshold plus a statistical "
                        "unresolved population",
           selection="as the component catalogues",
           independently_measurable=True,
           measurability_note="yes for BARYONS. A dynamical or lensing 'enclosed "
                              "mass' is a different object and is NOT this one.",
           allowed_ops=("add_same_dim", "multiply", "divide", "power_rational")))

    # ============================================ 5. motion of matter
    add(_q(5, name="v_los",
           definition="Line-of-sight velocity relative to the cluster systemic "
                      "redshift, from a spectroscopic line centroid.",
           kind="scalar", dim=DIM_V, frame="cluster_rest",
           translation="INVARIANT", rotation="SCALAR", boost="COVARIANT",
           parity="EVEN", time_reversal="ODD", support="point",
           source="measured: spectroscopic redshift minus the systemic value",
           status="direct", resolution_m=None,
           uncertainty="typically 20-150 km/s depending on the instrument",
           covariance_group="spectroscopy", coarse_grain="INTENSIVE_LINEAR",
           causal="LOCAL_NOW", completeness="spectroscopic sample only",
           selection="spectroscopic targeting -- NOT the same selection as the "
                     "photometric member catalogue",
           independently_measurable=True,
           measurability_note="yes -- one of the cleanest direct observables "
                              "in the whole scene",
           allowed_ops=("add_same_dim", "multiply", "divide", "power_rational",
                        "project_on_axis")))

    for ax in "xyz":
        add(_q(5, name=f"v_{ax}",
               definition=f"{ax}-component of the node's velocity in the "
                          f"cluster rest frame.",
               kind="scalar", dim=DIM_V, frame="cluster_rest",
               translation="INVARIANT", rotation="VECTOR", boost="COVARIANT",
               parity="ODD", time_reversal="ODD", support="point",
               source=("measured (= v_los)" if ax == "z"
                       else "LATENT: transverse velocity is not measured at "
                            "cluster distances"),
               status="direct" if ax == "z" else "latent",
               uncertainty="spectroscopic" if ax == "z" else "prior only",
               covariance_group="spectroscopy" if ax == "z" else "orbit_prior",
               coarse_grain="INTENSIVE_LINEAR", causal="LOCAL_NOW",
               completeness="spectroscopic sample", selection="targeting",
               independently_measurable=(ax == "z"),
               measurability_note=("yes" if ax == "z" else
                                   "no: proper motions at z~0.3 are far below "
                                   "any current astrometric capability"),
               allowed_ops=("add_same_dim", "multiply", "divide",
                            "power_rational", "project_on_axis")))

    add(_q(5, name="sigma_star",
           definition="Aperture stellar velocity dispersion of a member galaxy "
                      "from a pPXF fit, inside a stated aperture.",
           kind="scalar", dim=DIM_V, translation="INVARIANT", rotation="SCALAR",
           boost="INVARIANT", parity="EVEN", time_reversal="EVEN",
           support="region", source="pPXF fit to an integral-field spectrum",
           status="direct", resolution_m=1.0 * KPC, smoothing_m=1.0 * KPC,
           uncertainty="template and aperture dependent",
           covariance_group="ifu_kinematics", coarse_grain="NONLINEAR",
           causal="LOCAL_NOW",
           completeness="the brightest members only",
           selection="IFU coverage and S/N; ~213 members across four Frontier "
                     "Fields clusters",
           independently_measurable=True,
           measurability_note="yes, and it is a MEASURED alternative to an "
                              "assumed sigma-luminosity scaling",
           allowed_ops=("multiply", "divide", "power_rational"),
           notes="an APERTURE quantity, not a resolved map: it is already an "
                 "average, so it must pass the commutation gate before being "
                 "used to stand in for resolved internal kinematics"))

    add(_q(5, name="v_circ",
           definition="Circular speed implied by the gravitational field at a "
                      "radius, for a tracer on a circular orbit.",
           kind="scalar", dim=DIM_V, translation="INVARIANT", rotation="SCALAR",
           boost="INVARIANT", parity="EVEN", time_reversal="EVEN",
           support="point", source="derived from a candidate law",
           status="derived", uncertainty="inherits the law and the scene",
           covariance_group="prediction", coarse_grain="NONLINEAR",
           causal="LOCAL_NOW", completeness="n/a", selection="n/a",
           depends_on=("g_total", "r_3d"),
           exact_identities=("v_circ^2 = g_total * r_3d",),
           independently_measurable=False,
           measurability_note="a PREDICTION, not an observation",
           allowed_ops=("multiply", "divide", "power_rational")))

    # ============================================ 6. local gravitational
    add(_q(6, name="g_N",
           definition="Magnitude of the Newtonian acceleration generated by the "
                      "RESOLVED BARYONIC scene at the evaluation point.",
           kind="scalar", dim=DIM_ACC, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="point",
           source="solved from the resolved baryonic source scene",
           status="derived", resolution_m=1.0 * KPC, smoothing_m=1.0 * KPC,
           uncertainty="inherits the baryon budget and the scene ensemble",
           covariance_group="baryon_budget", coarse_grain="NONLINEAR",
           causal="LOCAL_NOW", completeness="as the source scene",
           selection="as the source scene",
           depends_on=("M_enc", "r_3d"),
           exact_identities=("g_N = G M_enc / r_3d^2  (spherical case only)",),
           independently_measurable=False,
           measurability_note="constructed, not observed. In the SPHERICAL case "
                              "it is an exact function of (M_enc, r_3d), so a "
                              "law reading all three has two directions, not "
                              "three -- this is the collapse the rank test finds.",
           allowed_ops=("add_same_dim", "multiply", "divide", "power_rational",
                        "gradient", "compare_to_scale")))

    add(_q(6, name="g_total",
           definition="Magnitude of the total gravitational acceleration "
                      "predicted by the candidate law on this scene.",
           kind="scalar", dim=DIM_ACC, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="point",
           source="candidate law applied to the scene", status="derived",
           resolution_m=1.0 * KPC, smoothing_m=1.0 * KPC,
           uncertainty="law + scene", covariance_group="prediction",
           coarse_grain="NONLINEAR", causal="LOCAL_NOW", completeness="n/a",
           selection="n/a", independently_measurable=False,
           measurability_note="a prediction",
           allowed_ops=("add_same_dim", "multiply", "divide", "power_rational",
                        "gradient", "compare_to_scale")))

    add(_q(6, name="g_vec",
           definition="Gravitational acceleration vector at a point.",
           kind="vector", dim=DIM_ACC, translation="INVARIANT",
           rotation="VECTOR", boost="COVARIANT", parity="ODD",
           time_reversal="EVEN", support="point",
           source="candidate law applied to the scene", status="derived",
           resolution_m=1.0 * KPC, smoothing_m=1.0 * KPC,
           uncertainty="law + scene", covariance_group="prediction",
           coarse_grain="NONLINEAR", causal="LOCAL_NOW", completeness="n/a",
           selection="n/a", independently_measurable=False,
           measurability_note="a prediction; only its EFFECT on tracers and "
                              "photons is observable",
           allowed_ops=("add_same_dim", "multiply", "divide", "gradient",
                        "project_on_axis", "contract")))

    # THE GAUGE FAMILY.  phi_baryon cannot be built without a rule.
    _GAUGE_RULES = {
        "phi_depth_saddle":
            "zero at the nearest gravitational saddle of the baryonic "
            "potential (a physically located surface, no free constant)",
        "phi_depth_r500":
            "zero at the fixed overdensity boundary R500 of the system",
        "phi_depth_scaleradius":
            "zero at a fixed multiple (10x) of the baryonic scale radius",
        "phi_depth_volume":
            "zero at the edge of the reconstructed environmental volume "
            "(survey-footprint dependent)",
    }
    for nm, rule in _GAUGE_RULES.items():
        add(_q(6, name=nm,
               definition=f"Baryonic potential difference between the "
                          f"evaluation point and a boundary. Boundary rule: {rule}.",
               kind="scalar", dim=DIM_PHI, translation="INVARIANT",
               rotation="SCALAR", boost="INVARIANT", parity="EVEN",
               time_reversal="EVEN", support="region",
               source="solved from the resolved baryonic scene, differenced "
                      "against the named boundary",
               status="derived", resolution_m=1.0 * KPC, smoothing_m=1.0 * KPC,
               uncertainty="0.87 dex spread across the four defensible global "
                           "rules in this family (Run AH), against a 0.9 dex "
                           "gate margin -- the rule choice is NOT negligible",
               covariance_group="gauge_rule", gauge=rule,
               coarse_grain="NONLINEAR", causal="LOCAL_NOW",
               completeness="as the source scene",
               selection="the volume rule additionally inherits the survey "
                         "footprint, which is why it is the weakest of the four",
               independently_measurable=False,
               measurability_note="a constructed quantity; the point of the "
                                  "family is that a candidate whose VERDICT "
                                  "changes across the four rules is flagged",
               allowed_ops=("add_same_dim", "multiply", "divide",
                            "power_rational", "gradient", "compare_to_scale")))

    add(_q(6, name="tidal_tensor",
           definition="Hessian of the baryonic potential, d2Phi/dx_i dx_j.",
           kind="tensor2", dim=DIM_TIDAL, translation="INVARIANT",
           rotation="RANK2", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="point",
           source="second derivative of the solved baryonic potential",
           status="invariant_descriptor", resolution_m=1.0 * KPC,
           smoothing_m=1.0 * KPC,
           uncertainty="second derivatives amplify scene noise",
           covariance_group="baryon_budget", coarse_grain="NONLINEAR",
           causal="LOCAL_NOW", completeness="as the source scene",
           selection="as the source scene",
           independently_measurable=False,
           measurability_note="constructed. Its TRACE is fixed by the local "
                              "density (Poisson), so trace and density are not "
                              "two independent variables.",
           exact_identities=("trace(tidal_tensor) = 4 pi G rho_local",),
           depends_on=("rho_star",),
           allowed_ops=("add_same_dim", "multiply", "divide", "contract",
                        "trace", "eigen", "project_on_axis"),
           notes="GATE 3 hazard: sourcing this from the CATALOGUE ROW LIST "
                 "rather than from the smooth density is the named repair in "
                 "Run AB and changes the verdict"))

    add(_q(6, name="tidal_anisotropy",
           definition="Traceless part of the tidal tensor, normalised by its "
                      "own Frobenius norm; a pure shape, gauge-free.",
           kind="scalar", dim=DIMLESS, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="point",
           source="from tidal_tensor", status="invariant_descriptor",
           resolution_m=1.0 * KPC, smoothing_m=1.0 * KPC,
           uncertainty="as tidal_tensor", covariance_group="baryon_budget",
           coarse_grain="NONLINEAR", causal="LOCAL_NOW",
           completeness="as the source scene", selection="as the source scene",
           depends_on=("tidal_tensor",), independently_measurable=False,
           measurability_note="constructed",
           allowed_ops=("multiply", "divide", "power_rational",
                        "log_dimensionless", "compare_to_scale")))

    # ============================================ 7. spacetime / curvature
    #  Latent or derived.  The charter: "Invariant should not place them
    #  directly into a fitting table unless the alternate universe explains how
    #  they are generated from sources."  Hence status=latent and
    #  independently_measurable=False for both.
    add(_q(7, name="phi_lensing",
           definition="Lensing potential: the combination of metric potentials "
                      "that deflects a null geodesic, differenced against the "
                      "same boundary rule as the matter potential.",
           kind="scalar", dim=DIM_PHI, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="region",
           source="generated by a candidate universe from its sources",
           status="latent", resolution_m=1.0 * KPC, smoothing_m=1.0 * KPC,
           uncertainty="model", covariance_group="candidate_field",
           gauge="same boundary rule as the paired matter potential -- the "
                 "DIFFERENCE of the two is only meaningful if both use one rule",
           coarse_grain="NONLINEAR", causal="PAST_LIGHT_CONE",
           completeness="n/a", selection="n/a",
           independently_measurable=False,
           measurability_note="only its effect on photons is observed",
           allowed_ops=("add_same_dim", "multiply", "divide", "gradient",
                        "path_integrate")))

    add(_q(7, name="phi_slip",
           definition="Difference between the matter and lensing potentials at "
                      "the same point under one common boundary rule. Zero iff "
                      "matter and light see the same geometry.",
           kind="scalar", dim=DIM_PHI, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="region",
           source="difference of two candidate-generated potentials",
           status="latent", resolution_m=1.0 * KPC, smoothing_m=1.0 * KPC,
           uncertainty="model", covariance_group="candidate_field",
           gauge="the common rule cancels in the difference, which is what "
                 "makes this quantity gauge-safe where each term alone is not",
           coarse_grain="NONLINEAR", causal="PAST_LIGHT_CONE",
           completeness="n/a", selection="n/a",
           depends_on=("phi_lensing", "phi_depth_saddle"),
           independently_measurable=False,
           measurability_note="constructed. It is the charter's "
                              "'Do matter and light see the same geometry?' "
                              "made into a number.",
           allowed_ops=("add_same_dim", "multiply", "divide", "gradient")))

    # ============================================ 15. speculative latent state
    #  Present so a `latent_field_cell` node can legally carry an attribute.
    #  Each is admitted ONLY through a generative law: status=latent and the
    #  measurability note names what would have to generate it.
    add(_q(15, name="vacuum_order",
           definition="Scalar order parameter of a polarisable vacuum, "
                      "normalised so that 0 is the unmodified vacuum.",
           kind="scalar", dim=DIMLESS, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="region",
           source="generated by a candidate universe", status="latent",
           resolution_m=1.0 * KPC, smoothing_m=1.0 * KPC,
           uncertainty="model", covariance_group="candidate_field",
           coarse_grain="NONLINEAR", causal="LOCAL_NOW", completeness="n/a",
           selection="n/a", independently_measurable=False,
           measurability_note="NOT fittable independently at every location -- "
                              "the candidate universe must specify what "
                              "generates it and how it evolves",
           allowed_ops=("add_same_dim", "multiply", "divide", "power_rational",
                        "log_dimensionless", "gradient")))

    add(_q(15, name="vacuum_axis",
           definition="Unit vector of a latent preferred-direction field.",
           kind="vector", dim=DIMLESS, translation="INVARIANT",
           rotation="VECTOR", boost="COVARIANT", parity="ODD",
           time_reversal="EVEN", support="region",
           source="generated by a candidate universe", status="latent",
           resolution_m=1.0 * KPC, smoothing_m=1.0 * KPC,
           uncertainty="model", covariance_group="candidate_field",
           coarse_grain="NONLINEAR", causal="LOCAL_NOW", completeness="n/a",
           selection="n/a", independently_measurable=False,
           measurability_note="if it is dynamically generated by the local "
                              "source it is degenerate with source shape; only "
                              "an EXTERNALLY generated axis is identifiable "
                              "(GATE 1)",
           allowed_ops=("project_on_axis", "contract")))

    add(_q(15, name="field_memory",
           definition="Accumulated exposure of a region to a declared "
                      "environmental state, with a declared decay time.",
           kind="scalar", dim=DIM_T, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="DISSIPATIVE", support="interval",
           source="history integral generated by a candidate universe",
           status="latent", resolution_m=None, smoothing_m=None,
           uncertainty="model", covariance_group="candidate_field",
           coarse_grain="NONLINEAR", causal="PAST_LIGHT_CONE",
           completeness="n/a", selection="n/a",
           depends_on=("t_since_merger",), independently_measurable=False,
           measurability_note="a memory law must predict how the effect DECAYS; "
                              "'history matters' is not a law",
           allowed_ops=("add_same_dim", "multiply", "divide", "power_rational",
                        "time_derivative", "compare_to_scale")))

    # ============================================ 8. directional structure
    add(_q(8, name="ext_axis",
           definition="Unit vector of the externally imposed preferred axis "
                      "(cluster-centre direction, filament axis, or principal "
                      "tidal eigenvector), whichever the candidate declares.",
           kind="vector", dim=DIMLESS, translation="INVARIANT",
           rotation="VECTOR", boost="INVARIANT", parity="ODD",
           time_reversal="EVEN", support="point",
           source="from the environment reconstruction, INDEPENDENTLY of the "
                  "probe whose response is being tested",
           status="invariant_descriptor", resolution_m=100.0 * KPC,
           smoothing_m=1.0 * MPC,
           uncertainty="depends on the environment catalogue depth",
           covariance_group="environment", coarse_grain="SCALE_DEFINED",
           causal="LOCAL_NOW",
           completeness="requires a surrounding-structure catalogue, which is "
                        "the layer most often missing",
           selection="survey footprint around the line of sight",
           independently_measurable=True,
           measurability_note="THIS IS THE GATE-1 CRUX. A constant response "
                              "tensor is a coordinate stretch UNLESS its axis "
                              "is fixed by an independently measured direction "
                              "misaligned with the probe's radial direction.",
           allowed_ops=("project_on_axis", "contract", "compare_to_scale")))

    add(_q(8, name="alignment_angle",
           definition="Angle between two declared axes (e.g. a galaxy's "
                      "angular-momentum axis and the local filament axis).",
           kind="scalar", dim=DIMLESS, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="pair",
           source="from two independently measured axes",
           status="invariant_descriptor", resolution_m=None, smoothing_m=None,
           uncertainty="propagated from both axes",
           covariance_group="environment+morphology", coarse_grain="NONLINEAR",
           causal="LOCAL_NOW", completeness="where both axes exist",
           selection="joint", independently_measurable=True,
           measurability_note="yes IF both axes come from different data; an "
                              "angle between an axis and itself is not a variable",
           allowed_ops=("multiply", "divide", "power_rational",
                        "compare_to_scale"),
           notes="parity: cos(angle) is EVEN; a SIGNED angle would be ODD and "
                 "would need a handedness convention"))

    # ============================================ 9-11. environment, network, path
    add(_q(9, name="rho_env",
           definition="Baryonic mass density smoothed over a declared scale, "
                      "around the evaluation point.",
           kind="scalar", dim=DIM_RHO, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="region",
           source="environment catalogue convolved with a declared kernel",
           status="invariant_descriptor", resolution_m=100.0 * KPC,
           smoothing_m=1.0 * MPC,
           uncertainty="dominated by catalogue completeness, not by shot noise",
           covariance_group="environment", coarse_grain="SCALE_DEFINED",
           causal="LOCAL_NOW",
           completeness="the binding constraint: a magnitude-limited "
                        "environment catalogue undercounts with distance",
           selection="survey footprint and depth",
           independently_measurable=True,
           measurability_note="yes where a surrounding-structure catalogue "
                              "exists at the required radius",
           allowed_ops=("add_same_dim", "multiply", "divide", "power_rational",
                        "gradient", "smooth", "compare_to_scale")))

    add(_q(10, name="n_wells",
           definition="Number of distinct gravitational wells above a declared "
                      "mass threshold within a declared radius.",
           kind="scalar", dim=DIMLESS, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="region",
           source="counted off the member catalogue", status="derived",
           resolution_m=None, smoothing_m=1.0 * MPC,
           uncertainty="Poisson plus the deblending choice",
           covariance_group="catalogue_partition",
           coarse_grain="CATALOGUE_DEPENDENT", causal="LOCAL_NOW",
           completeness="above the catalogue threshold",
           selection="detection and deblending",
           independently_measurable=False,
           measurability_note="NOT independently measurable: its value changes "
                              "when a deblender splits one galaxy into two. A "
                              "law reading it must converge under merge/split.",
           allowed_ops=("add_same_dim", "multiply", "divide", "power_rational",
                        "graph_reduce", "compare_to_scale")))

    add(_q(10, name="graph_degree",
           definition="Weighted degree of a well in the source network, at a "
                      "declared linking length.",
           kind="scalar", dim=DIMLESS, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="graph",
           source="constructed from the member catalogue and a linking rule",
           status="invariant_descriptor", resolution_m=None,
           smoothing_m=1.0 * MPC, uncertainty="linking-length dependent",
           covariance_group="catalogue_partition",
           coarse_grain="TOPOLOGICAL", causal="LOCAL_NOW",
           completeness="as the member catalogue", selection="as above",
           depends_on=("n_wells",), independently_measurable=False,
           measurability_note="changes discretely under a merge or split, so "
                              "it must be shown convergent before use",
           allowed_ops=("multiply", "divide", "power_rational", "graph_reduce")))

    add(_q(11, name="path_density",
           definition="Baryonic mass density integrated along a source-observer "
                      "light path.",
           kind="scalar", dim=DIM_SIGMA, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="path",
           source="integral of the scene density along the ray",
           status="invariant_descriptor", resolution_m=100.0 * KPC,
           smoothing_m=100.0 * KPC,
           uncertainty="dominated by line-of-sight structure outside the "
                       "cluster, which is usually uncatalogued",
           covariance_group="los_structure", coarse_grain="INTENSIVE_LINEAR",
           causal="PAST_LIGHT_CONE",
           completeness="the foreground/background catalogue is the limiting "
                        "layer",
           selection="as the line-of-sight catalogue",
           independently_measurable=False,
           measurability_note="requires a line-of-sight structure catalogue "
                              "that mostly does not exist at the needed depth",
           allowed_ops=("add_same_dim", "multiply", "divide", "power_rational",
                        "path_integrate", "compare_to_scale")))

    add(_q(11, name="path_void_fraction",
           definition="Fraction of the path length spent below a declared "
                      "density threshold.",
           kind="scalar", dim=DIMLESS, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="path",
           source="from the path density profile and a threshold",
           status="invariant_descriptor", resolution_m=1.0 * MPC,
           smoothing_m=1.0 * MPC, uncertainty="threshold dependent",
           covariance_group="los_structure", coarse_grain="NONLINEAR",
           causal="PAST_LIGHT_CONE", completeness="as path_density",
           selection="as path_density", depends_on=("path_density",),
           independently_measurable=False,
           measurability_note="as path_density",
           allowed_ops=("multiply", "divide", "power_rational",
                        "path_integrate", "compare_to_scale")))

    # ============================================ 12. history
    add(_q(12, name="t_since_merger",
           definition="Time since the last major merger, from shock-front "
                      "geometry and gas-star offsets.",
           kind="scalar", dim=DIM_T, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="ODD", support="interval",
           source="inferred from merger diagnostics", status="derived",
           resolution_m=None, smoothing_m=None,
           uncertainty="factor-of-two at best", covariance_group="merger_state",
           coarse_grain="NONLINEAR", causal="PAST_LIGHT_CONE",
           completeness="only for visibly disturbed systems",
           selection="requires a shock or an offset to be detected",
           independently_measurable=True,
           measurability_note="crudely; the gas-star offset is a direct "
                              "observable but the conversion to a time is not",
           allowed_ops=("add_same_dim", "multiply", "divide", "power_rational",
                        "compare_to_scale")))

    # ============================================ 13. light observables
    for nm, dfn in (("e1", "First component of the measured background-source "
                           "ellipticity, in the declared sky frame."),
                    ("e2", "Second component of the measured background-source "
                           "ellipticity, in the declared sky frame.")):
        add(_q(13, name=nm, definition=dfn, kind="scalar", dim=DIMLESS,
               frame="sky_equatorial", translation="INVARIANT",
               rotation="FRAME_DEPENDENT", boost="INVARIANT",
               parity="ODD" if nm == "e2" else "EVEN", time_reversal="EVEN",
               support="point", source="MEASURED: a shape estimator on "
                                       "calibrated pixels", status="direct",
               resolution_m=None, smoothing_m=None,
               uncertainty="shape noise ~0.25-0.30 per component, plus a "
                           "multiplicative shear calibration",
               covariance_group="shear_calibration",
               coarse_grain="INTENSIVE_LINEAR", causal="PAST_LIGHT_CONE",
               completeness="source-detection limited",
               selection="size and S/N cuts; the selection is itself a "
                         "shear-dependent bias",
               independently_measurable=True,
               measurability_note="THE raw lensing observable. A convergence "
                                  "map is not: it is this, plus an inversion "
                                  "that assumes a gravity law.",
               allowed_ops=("add_same_dim", "multiply", "divide",
                            "project_on_axis")))

    add(_q(13, name="kappa",
           definition="Convergence. A DERIVED product: the shape catalogue "
                      "inverted under an assumed relation between shear and "
                      "surface density.",
           kind="scalar", dim=DIMLESS, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="region",
           source="DERIVED UNDER A THEORY from e1, e2", status="derived",
           resolution_m=None, smoothing_m=30.0 * KPC,
           uncertainty="mass-sheet degenerate", covariance_group="lens_model",
           coarse_grain="NONLINEAR", causal="PAST_LIGHT_CONE",
           completeness="as the shape catalogue", selection="as the shape catalogue",
           depends_on=("e1", "e2"), independently_measurable=False,
           derived_under_theory=True,
           measurability_note="NOT a raw observation. Charter: do not score 'a "
                              "precomputed convergence map ... as though it "
                              "were the primitive observation.'",
           allowed_ops=("add_same_dim", "multiply", "divide", "smooth"),
           notes="DM-PRESUPPOSING when the lens model assigns a halo to each "
                 "cluster galaxy by construction (e.g. the CATS Frontier "
                 "Fields maps), which makes it circular for any "
                 "does-lensing-follow-light test"))

    add(_q(13, name="image_position",
           definition="Sky position of one image in a multiply-imaged family.",
           kind="vector", dim=DIMLESS, frame="sky_equatorial",
           translation="COVARIANT", rotation="FRAME_DEPENDENT",
           boost="INVARIANT", parity="EVEN", time_reversal="EVEN",
           support="point", source="MEASURED: image astrometry",
           status="direct", resolution_m=None, smoothing_m=None,
           uncertainty="0.1-0.5 arcsec including identification ambiguity",
           covariance_group="astrometry", coarse_grain="INTENSIVE_LINEAR",
           causal="PAST_LIGHT_CONE", completeness="identified families only",
           selection="requires a spectroscopic or colour-based family "
                     "identification, which is itself model-informed",
           independently_measurable=True,
           measurability_note="the positions are raw; the FAMILY ASSIGNMENT "
                              "sometimes is not, and that distinction must be "
                              "carried",
           allowed_ops=("add_same_dim", "multiply", "divide",
                        "project_on_axis")))

    add(_q(13, name="time_delay",
           definition="Measured arrival-time difference between two images of "
                      "the same source.",
           kind="scalar", dim=DIM_T, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="ODD", support="pair",
           source="MEASURED: light-curve cross-correlation", status="direct",
           resolution_m=None, smoothing_m=None,
           uncertainty="from the light-curve sampling and microlensing",
           covariance_group="time_delay", coarse_grain="NONLINEAR",
           causal="PAST_LIGHT_CONE",
           completeness="essentially nil: cluster-scale time delays exist for "
                        "a handful of events in the whole sky",
           selection="requires a variable source behind a cluster",
           independently_measurable=True,
           measurability_note="raw and theory-free, and it is the single "
                              "strongest matter-light consistency constraint. "
                              "The problem is that almost none exist.",
           allowed_ops=("add_same_dim", "multiply", "divide", "power_rational")))

    add(_q(13, name="y_compton",
           definition="Compton-y parameter: the line-of-sight integral of the "
                      "electron pressure, from a calibrated SZ map.",
           kind="scalar", dim=DIMLESS, translation="INVARIANT",
           rotation="SCALAR", boost="INVARIANT", parity="EVEN",
           time_reversal="EVEN", support="path",
           source="MEASURED: a calibrated millimetre map", status="direct",
           resolution_m=None, smoothing_m=300.0 * KPC,
           uncertainty="beam-convolved; correlated noise across the beam",
           covariance_group="sz_map", coarse_grain="INTENSIVE_LINEAR",
           causal="PAST_LIGHT_CONE", completeness="beam and depth limited",
           selection="SZ significance",
           depends_on=("P_e",),
           exact_identities=("y = (sigma_T / m_e c^2) integral P_e dl",),
           independently_measurable=True,
           measurability_note="RAW and independent of X-ray emissivity, which "
                              "is exactly why it is worth having. An "
                              "INTEGRATED Y_500 is NOT raw: its aperture is "
                              "defined through a mass.",
           allowed_ops=("add_same_dim", "multiply", "divide", "path_integrate",
                        "smooth")))

    # ============================================ 16. constants
    for nm, dfn, dm in (("G", "Newton's gravitational constant.", DIM_G),
                        ("c_light", "Speed of light in vacuum.", DIM_C),
                        ("a0", "Candidate universal acceleration scale.",
                         DIM_ACC)):
        add(_q(16, name=nm, definition=dfn, kind="scalar", dim=dm,
               translation="INVARIANT", rotation="SCALAR", boost="INVARIANT",
               parity="EVEN", time_reversal="EVEN", support="global",
               source="laboratory" if nm != "a0" else "fitted globally, once",
               status="constant", resolution_m=None, smoothing_m=None,
               uncertainty="negligible" if nm != "a0" else "global fit",
               covariance_group="constants", coarse_grain="EXTENSIVE",
               causal="LOCAL_NOW", completeness="n/a", selection="n/a",
               independently_measurable=True,
               measurability_note="a GLOBAL constant; an object-specific value "
                                  "would violate charter criterion 3",
               allowed_ops=("multiply", "divide", "power_rational",
                            "compare_to_scale")))

    # ============================================ 17. nuisances
    _nuis = [
        ("p_member", DIMLESS, "Probability the galaxy is a cluster member.",
         "membership"),
        ("psf_fwhm", DIM_L, "Point-spread-function full width at half maximum, "
                            "as a physical length at the source.", "psf"),
        ("shear_m", DIMLESS, "Multiplicative shear calibration bias.",
         "shear_calibration"),
        ("upsilon_star", DIMLESS, "Stellar mass-to-light ratio in the declared "
                                  "band, as a GLOBAL nuisance.", "stellar_ML"),
        ("distance", DIM_L, "Angular-diameter distance to the system.",
         "cosmology"),
    ]
    for nm, dm, dfn, grp in _nuis:
        add(_q(17, name=nm, definition=dfn, kind="scalar", dim=dm,
               translation="INVARIANT", rotation="SCALAR", boost="INVARIANT",
               parity="EVEN", time_reversal="EVEN", support="point",
               source="measurement model", status="nuisance",
               resolution_m=None, smoothing_m=None,
               uncertainty=("0.06 dex, GLOBAL (0.045 dex galaxy-to-galaxy "
                            "scatter, so it is one nuisance not N)"
                            if nm == "upsilon_star" else "estimator specific"),
               covariance_group=grp, coarse_grain="NONLINEAR",
               causal="LOCAL_NOW", completeness="n/a",
               selection="n/a", independently_measurable=True,
               measurability_note="must be MARGINALISED, never promoted to a "
                                  "gravity variable without evidence",
               allowed_ops=("multiply", "divide", "power_rational")))

    return _assign_identifiability(R)


#: The five exact identities the programme has established.  Recorded here as
#: (target, inputs, relation) so `bridge.py` can take the rank of a candidate's
#: variable set before any data is opened.
EXACT_IDENTITIES = (
    ("g_N", ("M_enc", "r_3d"), "g_N = G M_enc / r_3d^2 (spherical)"),
    ("v_circ", ("g_total", "r_3d"), "v_circ^2 = g_total r_3d"),
    ("r_3d", ("r_proj", "z"), "r_3d^2 = r_proj^2 + z^2"),
    ("P_e", ("n_e", "T_x"), "P_e = n_e k_B T_x"),
    ("R500", ("M_enc",), "M(<R500) = 500 rho_crit (4/3) pi R500^3"),
)


def section_of(name: str) -> int:
    return _SECTION_OF.get(name, 0)


def coverage_by_section(reg: Registry) -> Dict[int, Dict[str, object]]:
    out: Dict[int, Dict[str, object]] = {}
    for sec, title in ONTOLOGY_SECTION.items():
        names = [q.name for q in reg.all() if section_of(q.name) == sec]
        out[sec] = {"title": title, "n": len(names), "names": names}
    return out
