"""arms.py -- the arms of the experiment, declared in one place.

An ARM is one universe with one setting of every knob, emitted on the shared
scene draw and noise realisation of a paired set:

    tag      the name results are keyed by
    uid      BF universe id (or H0_scalar_null)
    knob     deformation amplitude (None -> BF's fiducial draw)
    sys      systematics multiplier, noise  statistical-error multiplier
    halo     knob overrides for halo.NOMINAL (CDM arms only)
    cf       counterfactual edit: {'scene': {...}, 'hold_halo': bool, 'halo': {...},
             'scramble': bool}
    nuis     nuisance-amplitude multipliers on the galaxy instrument

THRESH are Run BF's threshold amplitudes (E9): the amplitude at which each
deformation's own scan first reaches the family-wise critical value against
U3, i.e. the amplitudes at which the seven families ARE one observational
class.  The class is defined at those amplitudes; the fiducials are carried as
a robustness arm because BF found them 2.9-26x above threshold.
"""
from __future__ import annotations

THRESH = {"U04_env_scalar": 0.06886, "U05_tensor_axis": 0.02003,
          "U06_wellnet": 0.00718, "U07_memory": 0.02854,
          "U08_ep_slip": 0.04666, "U09_path_redshift": 0.00814}
FID = {"U04_env_scalar": 0.60, "U05_tensor_axis": 0.50, "U06_wellnet": 0.06,
       "U07_memory": 0.20, "U08_ep_slip": 0.10, "U09_path_redshift": 0.030}
SHORT = {"U04_env_scalar": "U4", "U05_tensor_axis": "U5", "U06_wellnet": "U6",
         "U07_memory": "U7", "U08_ep_slip": "U8", "U09_path_redshift": "U9"}


def A(tag, uid, knob=None, sys=1.0, noise=1.0, halo=None, cf=None, nuis=None):
    return dict(tag=tag, uid=uid, knob=knob, sys=sys, noise=noise,
                halo=halo or {}, cf=cf or {}, nuis=nuis or {})


# the modified-gravity CLASS (seven families at threshold) plus the scalar null
CLASS_TAGS = ["U3", "H0"] + [f"{SHORT[u]}t" for u in THRESH]


def main_arms():
    arms = [A("U1", "U01_baryons_newton"),
            A("U2", "U02_cdm"),                      # the CDM prior, every nuisance drawn
            A("U3", "U03_mond_scalar"),
            A("H0", "H0_scalar_null"),
            A("U10", "U10_systematics", sys=3.0)]
    for u, a in THRESH.items():
        arms.append(A(f"{SHORT[u]}t", u, knob=a))
    for u, a in FID.items():
        arms.append(A(f"{SHORT[u]}f", u, knob=a))
    return arms


def heldout_arms():
    """On the untouched scene library: the arms the verdict needs."""
    arms = [A("U2", "U02_cdm"), A("U3", "U03_mond_scalar"), A("H0", "H0_scalar_null"),
            A("U1", "U01_baryons_newton"), A("U10", "U10_systematics", sys=3.0)]
    for u, a in THRESH.items():
        arms.append(A(f"{SHORT[u]}t", u, knob=a))
    arms += [A("U5f", "U05_tensor_axis", knob=FID["U05_tensor_axis"]),
             A("U8f", "U08_ep_slip", knob=FID["U08_ep_slip"])]
    return arms


def scan_arms():
    """CDM-only arms, one nuisance moved at a time from a NOMINAL point at which
    the drawn nuisances are frozen to their central values."""
    base = dict(f_lss=0.5, q_h_range=(0.85, 0.85), f_dd_range=(0.0, 0.0), q_max=0.05)
    arms = [A("S_nominal", "U02_cdm", halo=dict(base))]
    for f in (0.0, 0.25, 0.38, 0.75, 1.0):
        arms.append(A(f"S_flss_{f:g}", "U02_cdm", halo=dict(base, f_lss=f)))
    for s in (0.0, 0.5, 1.5, 2.0):
        arms.append(A(f"S_shmr_{s:g}", "U02_cdm", halo=dict(base, s_shmr=s)))
    for s in (0.0, 0.5, 2.0):
        arms.append(A(f"S_conc_{s:g}", "U02_cdm", halo=dict(base, s_conc=s)))
    for f in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        arms.append(A(f"S_fdd_{f:g}", "U02_cdm", halo=dict(base, f_dd_range=(f, f))))
    for q in (0.6, 0.7, 1.0):
        arms.append(A(f"S_qh_{q:g}", "U02_cdm", halo=dict(base, q_h_range=(q, q))))
    arms.append(A("S_zeroscatter", "U02_cdm",
                  halo=dict(base, s_shmr=0.0, s_conc=0.0, s_mclu=0.0, s_shape=0.0)))
    arms.append(A("S_fixhalo", "U02_cdm", halo=dict(base, fix_halo_per_object=True)))
    arms.append(A("S_sys3", "U02_cdm", halo=dict(base), sys=3.0))
    arms.append(A("S_noise0.5", "U02_cdm", halo=dict(base), noise=0.5))
    arms.append(A("S_qamp0.2", "U02_cdm", halo=dict(base, q_max=0.20)))
    # the class under the same instrument changes, for the paired comparison
    arms.append(A("S_U3_sys3", "U03_mond_scalar", sys=3.0))
    arms.append(A("S_U3_noise0.5", "U03_mond_scalar", noise=0.5))
    # nuisance-amplitude arms (C7): the instrument's own knobs moved
    for nk in ("hz", "ml", "dist", "incl", "sz_err"):
        arms.append(A(f"S_U3_nuis_{nk}", "U03_mond_scalar", nuis={nk: 2.5}))
        arms.append(A(f"S_U2_nuis_{nk}", "U02_cdm", halo=dict(base), nuis={nk: 2.5}))
    return arms


CF_BASE_HALO = dict(f_lss=0.5, q_h_range=(0.85, 0.85), f_dd_range=(0.0, 0.0), q_max=0.05)


def counterfactual_arms():
    """Paired scenes, one thing changed, everything else held.  Every edit is
    applied to U2 (halo HELD), to U3, and to the class member it targets."""
    edits = {
        "bar_mass": dict(scene=dict(dlogM=+0.10), hold_halo=True),
        "bar_size": dict(scene=dict(dlogR=+0.06), hold_halo=True),
        "bar_hz": dict(scene=dict(dlog_hz=+0.10), hold_halo=True),
        "halo_mass": dict(halo=dict(dlogM=+0.10)),
        "halo_conc": dict(halo=dict(dlogc=+0.10)),
        "halo_shape": dict(halo=dict(dell=+0.10)),
        "halo_axis": dict(halo=dict(dpa=45.0)),
        "ext_axis": dict(scene=dict(axis_rot=45.0), hold_halo=True),
        "ext_axis_follow": dict(scene=dict(axis_rot=45.0), hold_halo=False),
        "scramble": dict(scramble=True, hold_halo=True),
        "history": dict(scene=dict(tmerge_mult=3.0), hold_halo=True),
        "path": dict(scene=dict(void_flip=True), hold_halo=True),
        "bar_mass_follow": dict(scene=dict(dlogM=+0.10), hold_halo=False),
    }
    arms = [A("U2", "U02_cdm", halo=dict(CF_BASE_HALO)),
            A("U3", "U03_mond_scalar"),
            A("U5f", "U05_tensor_axis", knob=FID["U05_tensor_axis"]),
            A("U6f", "U06_wellnet", knob=FID["U06_wellnet"]),
            A("U7f", "U07_memory", knob=FID["U07_memory"]),
            A("U8f", "U08_ep_slip", knob=FID["U08_ep_slip"]),
            A("U9f", "U09_path_redshift", knob=FID["U09_path_redshift"])]
    for name, cf in edits.items():
        if "halo" in cf:
            arms.append(A(f"U2|{name}", "U02_cdm", halo=dict(CF_BASE_HALO), cf=cf))
            continue
        arms.append(A(f"U2|{name}", "U02_cdm", halo=dict(CF_BASE_HALO), cf=cf))
        arms.append(A(f"U3|{name}", "U03_mond_scalar", cf=cf))
        target = {"ext_axis": "U5f", "ext_axis_follow": "U5f", "scramble": "U6f",
                  "history": "U7f", "path": "U9f"}.get(name)
        if target:
            uid = {"U5f": "U05_tensor_axis", "U6f": "U06_wellnet", "U7f": "U07_memory",
                   "U9f": "U09_path_redshift"}[target]
            arms.append(A(f"{target}|{name}", uid, knob=FID[uid], cf=cf))
    # photon coupling changed holding matter dynamics: zeta 0 -> fiducial is U3 -> U8f,
    # already a pair of arms above (U3, U8f) on the same scene and noise.
    return arms


def scan_arms_extra():
    """The quadrupole-POWER channel does not depend on f_lss but on the halo
    ELLIPTICITY prior: scan it.  Plus the galaxy in-plane quadrupole off."""
    base = dict(f_lss=0.5, q_h_range=(0.85, 0.85), f_dd_range=(0.0, 0.0), q_max=0.05)
    arms = [A("S_eh_0", "U02_cdm", halo=dict(base, shape_corr=0.0, s_shape=0.0)),
            A("S_eh_0.36", "U02_cdm", halo=dict(base, shape_corr=0.36, s_shape=0.5)),
            A("S_eh_1.0", "U02_cdm", halo=dict(base, shape_corr=1.0, s_shape=1.5)),
            A("S_qamp0", "U02_cdm", halo=dict(base, q_max=0.0)),
            A("S_fdd_0.8", "U02_cdm", halo=dict(base, f_dd_range=(0.8, 0.8)))]
    return arms
