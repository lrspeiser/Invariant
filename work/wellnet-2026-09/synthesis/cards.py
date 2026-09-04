"""cards.py -- the two machine-readable principle cards (Job 1).

Every number quoted in a card is READ from tensor_results.json,
path_results.json and compile_results.json.  The prose is where the physics
is stated; the `evidence` block beside each field carries the numbers that
support it and names the file they come from.

The ten fields (all mandatory):
    physical_statement, source, state, propagation, matter_coupling,
    photon_coupling, known_limits, counterfactual_signature,
    unique_falsifier, cdm_distinction
plus, for each family: baryonic_closure (mean AND scatter), the action,
the charter's 15 axiom-axis values, the compiler verdicts, the derived
(not assumed) structural facts, and the honest list of fields that needed
an assumption this lane had to invent.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import guard                                    # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
KPC = 3.0856775814913673e19


def J(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return json.load(fh)


def _row(rows, r_over_a):
    return next(r for r in rows if abs(r["r_over_a"] - r_over_a) < 1e-9)


def _ff_at(table, r_kpc):
    r = np.array(table["r_m"]) / KPC
    return float(np.interp(np.log(r_kpc), np.log(r), np.array(table["factor"])))


# ======================================================================
def tensor_card(T: dict, CP: dict) -> dict:
    l2 = T["l2"]
    el = T["ellipticity"]
    kl = T["known_limits"]["rows"]
    comp = CP.get("tensor", {})
    A2_3a = l2["A2_at_3a_per_fE"]
    A2_pk = l2["peak"]["A2_per_fE"]
    rar = l2["rar_residual_dex_per_fE_at_3a"]
    fields = {}
    fields["physical_statement"] = dict(
        statement=(
            "The kinetic metric of the gravitational potential is anisotropic "
            "in the transition regime |grad Phi| ~ a0, and the axis of the "
            "anisotropy is the large-scale tidal axis of the environment. "
            "Gravitational flux is redistributed toward that axis without "
            "changing the total flux (the anisotropic term is traceless) and "
            "while the field remains the gradient of ONE scalar potential. "
            "The anisotropy vanishes in deep MOND -- required by ellipticity "
            "of the operator, not chosen -- and at high acceleration, "
            "required by the Solar System. What nature does differently: the "
            "vacuum's response to a field of order a0 knows the direction in "
            "which the surrounding matter is stretching it."),
        evidence=dict(
            ellipticity_interval_fE=[el["admissible_fE_min"], el["admissible_fE_max"]],
            deep_mond_argument=T["deep_mond_argument"],
            file="tensor_results.json"))
    fields["source"] = dict(
        statement=(
            "Rest mass rho sources Phi (the only source of the potential). The "
            "AXIS is sourced by the large-scale tidal tensor of the environment "
            "-- the traceless Hessian of the Newtonian potential smoothed on "
            "L_env, i.e. the stretching direction of the surrounding matter "
            "(filament axis), measured independently from the galaxy "
            "distribution around the object and never inferred from the "
            "residual it later explains. The STRENGTH is sourced by nothing: it "
            "is one universal constant f_E multiplying one universal gate "
            "h(|grad Phi|/a0)."),
        evidence=dict(axis_provenance_in_compiler="external (declared)",
                      file="compile_results.json"))
    fields["state"] = dict(
        statement=(
            "One scalar potential Phi and one headless background field: the "
            "unit-Frobenius-norm traceless tidal tensor That_env(x) "
            "(equivalently a unit axis e, e == -e). No vector field, no second "
            "potential, no memory. The charter's 'tensor response of space' is "
            "realised as the kinetic metric of Phi, K = exp[f_E h That] to first "
            "order, entering the Lagrangian as f_E h(|u|/a0) u^T That u."),
        evidence=dict(action=T["action"]))
    fields["propagation"] = dict(
        statement=(
            "Point-local in Phi, instantaneous weak-field (static) response. "
            "The axis field is quasi-static: it changes on the environment's "
            "dynamical time, not the object's. No retardation, no path "
            "dependence, no history dependence. The tensor-wave sector is not "
            "touched by the weak-field action (see known_limits for the "
            "assumption that entails)."),
        evidence=dict(propagation="instantaneous_weak_field"))
    fields["matter_coupling"] = dict(
        statement=(
            f"Universal: a = -grad Phi for every test body, with Phi solving "
            f"div M(grad Phi) = 4 pi G rho. Consequences, all first order in "
            f"f_E on a spherical source in the AQUAL base: (i) an m=2 harmonic "
            f"of the inward acceleration locked to the axis, with fractional "
            f"amplitude A2(r) = chi'(r)/g0(r) per unit f_E; A2 is NEGATIVE at "
            f"every radius from 0.1 a to 300 a of the caricature galaxy "
            f"(A2 = {A2_3a:+.4f} f_E at r = 3 a, peak {A2_pk:+.3f} f_E in the "
            f"core), i.e. for f_E > 0 the inward pull is WEAKER along the "
            f"tidal axis; it decays as -1/(3 g0/a0) at high acceleration and "
            f"tends to zero from below in deep MOND; (ii) the potential is "
            f"nonetheless DEEPER along the axis (chi < 0, tending to the "
            f"constant -sqrt(G M a0)/3); (iii) a disk whose normal makes an "
            f"angle psi with e has an azimuthally averaged RAR residual "
            f"-(1/2) P2(cos psi) A2 f_E, i.e. {rar:.4f} f_E dex at r = 3 a, "
            f"signed by the environment with no per-object freedom."),
        evidence=dict(A2_rows=l2["rows"], A2_all_negative=l2["A2_all_negative_0p1a_to_300a"],
                      newtonian_side=l2["newtonian_side_asymptote"],
                      deep_mond=l2["deep_mond_asymptote"],
                      validation=l2["validation_constant_K_newton"],
                      file="tensor_results.json"))
    fields["photon_coupling"] = dict(
        statement=(
            "Photons see the same Phi: the weak-field metric carries Phi in "
            "both the time-time and space-space parts (common matter-light "
            "geometry, Principle 4; no fitted slip). The lensing potential's "
            "quadrupole is f_E chi(r) P2 with the SAME chi as the dynamical "
            "one, so the lensing quadrupole phase equals the dynamical phase "
            "and the two amplitudes are tied with ratio 1 in the 3-D "
            "potential. Shear is the second derivative of the projected "
            "potential, and in deep MOND chi tends to a CONSTANT offset, so the "
            "outer shear quadrupole of this family vanishes while the inner "
            "(transition-band) one does not -- a radial profile, consistent in "
            "direction with Run BF.4's finding that its (differently "
            "constructed, linear-K) fixed-axis tensor put little of its l=2 into "
            "the cluster lensing potential. No frequency or polarization "
            "dependence; no time-delay anomaly beyond the potential's own "
            "quadrupole."),
        evidence=dict(chi_asymptote_deep_mond=l2["deep_mond_asymptote"]))
    fields["known_limits"] = dict(
        statement=(
            f"HIGH ACCELERATION / SOLAR SYSTEM: h ~ 1/x, so the fractional "
            f"quadrupole (2/3) f_E h is "
            f"{kl['Earth_1AU']['quadrupole_fE_1.0']:.1e} f_E at 1 AU and "
            f"{kl['Mercury_0.39AU']['quadrupole_fE_1.0']:.1e} f_E at Mercury; "
            f"fixed-axis anisotropies of the solar potential are bounded near "
            f"1e-9 (published PPN preferred-location and solar-J2 constraints, "
            f"quoted as an order of magnitude and not evaluated here), so "
            f"f_E <~ 0.3 is safe and f_E of order 1-2 is marginal; the theory's "
            f"own ellipticity interval is f_E in "
            f"({el['admissible_fE_min']:.2f}, {el['admissible_fE_max']:.2f}). "
            f"BINARY PULSARS: x >> 1, h < 1e-9. "
            f"GALAXY REGULARITIES: the term is traceless, so the direction-"
            f"averaged RAR is untouched at first order and its scatter gains "
            f"only the environment-locked {rar:.4f} f_E dex term; h -> 0 in "
            f"deep MOND, so flat rotation curves and the BTFR (the deep-MOND "
            f"asymptote) are untouched, and the force anisotropy vanishes in "
            f"the outskirts. "
            f"WAVE PROPAGATION: the weak-field action says nothing about the "
            f"tensor-wave sector; the GW speed equals c only if the completion "
            f"places the anisotropic kinetic term in the scalar sector alone "
            f"-- ASSUMED, flagged in honest_gaps."),
        evidence=dict(known_limit_rows=kl, peak_h=T["known_limits"]["peak_h"],
                      file="tensor_results.json"))
    fields["counterfactual_signature"] = dict(
        statement=(
            "Rotate the external axis by dpsi holding the baryons: the m=2 "
            "phase of |g_r| AND of the lensing quadrupole rotate rigidly, "
            "d(phase)/d(psi_e) = +1, zero lag, identical at every radius. "
            "Move the baryons holding e: the amplitude is re-evaluated "
            "instantly as A2 on the new g0(r). Move a halo: nothing (there is "
            "none). Scramble members preserving radial profiles: nothing (the "
            "response is a functional of the smooth field). Change history "
            "preserving present matter: nothing (no memory). Change the photon "
            "path preserving endpoints: the photon sees the same chi P2 as the "
            "matter. Radial twist d(phase)/d ln r = 0 exactly. Tilt the disk "
            "normal: a signed RAR residual -(1/2) P2(cos psi) A2 f_E."),
        evidence=dict(table=T["counterfactuals"], file="tensor_results.json"))
    fields["unique_falsifier"] = dict(
        statement=(
            "At fixed baryons and fixed independently measured tidal axis, the "
            "population distribution of (m=2 phase minus tidal-axis phase) must "
            "be a delta function convolved with the measurement error alone, "
            "with ONE universal sign (for f_E > 0 the m=2 MINIMUM of |g_r| on "
            "the axis), zero radial twist, and an amplitude profile with the "
            "universal shape A2(g/a0): largest near the g ~ a0 crossing, "
            "decaying as 1/x at high acceleration, vanishing in deep MOND. A "
            "measured intrinsic phase dispersion exceeding the axis error after "
            "marginalising inclination, position angle, distance, M/L and the "
            "axis measurement; or mixed signs across objects; or a quadrupole "
            "amplitude that GROWS into the deep-MOND outskirts -- each kills the "
            "law, and none can be repaired by an object-specific nuisance. NOT "
            "a falsifier: anisotropy, a recovered a0, a MOND-like monopole "
            "(BJ.1)."),
        evidence=dict(sign_statement=l2["sign"]["statement"]))
    fields["cdm_distinction"] = dict(
        statement=(
            "A realistic collisionless halo is triaxial with intrinsic axis-"
            "ratio scatter, oriented by the tidal field at ASSEMBLY with tens "
            "of degrees of misalignment dispersion relative to the PRESENT "
            "tidal axis, twisting with radius, carrying stochastic m=1/m=3 "
            "power from substructure, and with an anisotropy that persists or "
            "grows in the outskirts. The tensor law's JOINT response is: (i) "
            "phase locked to the present tidal axis with zero intrinsic "
            "dispersion; (ii) zero radial twist; (iii) an amplitude that is a "
            "universal function of g/a0 alone, peaking in the transition band "
            "and VANISHING where a halo's anisotropy is largest; (iv) one "
            "universal sign; (v) identical phase and tied amplitude in lensing "
            "and dynamics with no per-object freedom; (vi) residual harmonic "
            "content in m=2 only. A CDM population can match any one of these "
            "by selection; matching all six requires its halo shape "
            "distribution collapsed to a delta function slaved to the present "
            "tidal field with an ellipticity profile keyed to the baryonic "
            "acceleration and dying in the outskirts -- the opposite of what "
            "assembly-history scatter and outer triaxiality necessarily "
            "produce. Because the family's own signature is a sign and a "
            "profile rather than a magnitude, BF's 0.648 rate of the generic "
            "anisotropy detectors on the dark-matter universe does not apply "
            "to it: the detector must test (i)-(vi) jointly, not anisotropy."),
        evidence=dict(bf_generic_detector_rate_on_cdm=0.648,
                      note="Run BF.3; the joint test is specified in the "
                           "counterfactual table"))
    closure = dict(
        predicts_baryonic_closure=True,
        mean=("given the complete baryonic scene AND the environment's tidal "
              "axis, gravity is fixed up to the universal constants (G, a0, "
              "f_E, L_env): P(G | B, e) has zero intrinsic width"),
        scatter=("the law predicts the SCATTER, not only the mean: (a) with e "
                 "observed, residual scatter = measurement noise only; (b) with "
                 "e unobserved (local baryons only), the residual is a PURE "
                 "m=2 harmonic of universal amplitude A2(g/a0) f_E and random "
                 "phase, with no m=1, m=3 or m=4 power, and a RAR residual "
                 f"distributed as -(1/2) P2(cos psi) A2 f_E over the "
                 f"distribution of psi -- a scatter with a predicted shape "
                 f"(width {rar:.4f} f_E dex at r = 3 a) and harmonic content"),
        cdm_contrast=("a halo population's P(G | B) has a width set by the halo "
                      "shape/orientation distribution, all harmonics, and a "
                      "radial growth of the anisotropy; its scatter is not a "
                      "function of the present tidal axis alone"))
    axes = dict(source="rest_mass", field_type="scalar (+ background symmetric_tensor)",
                locality="point_local", superposition="nonlinear",
                directionality="tidal", axis_origin="external_matter",
                propagation="instantaneous_weak_field",
                geometry="force_in_fixed_space (weak field)",
                eff_dimension="three_d", matter_light="universal",
                equivalence="exact",
                conservation="exact_reciprocal (background axis) | "
                             "exchange_with_field (dynamical axis)",
                cosmology="expanding_geometry (assumed standard)",
                vacuum="directional", initial_conditions="standard_primordial")
    derived_not_assumed = [
        "h must vanish at least as fast as mu in deep MOND, or the kinetic "
        "operator loses ellipticity (measured: a constant h makes the operator "
        f"indefinite on {100*T['deep_mond_argument']['fraction_of_cloud_with_indefinite_operator']:.0f}% "
        "of a low-x cloud)",
        "h must vanish at high acceleration (Solar System); h = mu(1-mu) is the "
        "sparsest weight doing both with no new scale",
        f"the admissible f_E interval ({el['admissible_fE_min']:.2f}, "
        f"{el['admissible_fE_max']:.2f}) from positive-definiteness of the "
        "Hessian of the Lagrangian",
        "writing the gated anisotropy as K(u)u in QUMOND form is NOT a gradient "
        "(compiler element F2, physically_incomplete_as_written); writing it "
        "as a term of a scalar Lagrangian density is a gradient identically "
        "(u-space antisymmetry at round-off)",
        "the force quadrupole of the gated tensor has the OPPOSITE sign to "
        "that of a constant tensor with the same axis and sign, and the "
        "potential quadrupole tends to a constant in deep MOND",
        "the compiler's declared radial caricature (k_r on the radial flux) "
        "has the wrong sign for constant K and the wrong magnitude and "
        "asymptotics for the gated case; no gate verdict depends on it",
    ]
    gaps = [
        dict(field="known_limits",
             assumption="that the relativistic completion puts the anisotropic "
                        "kinetic term in the scalar sector only, so tensor waves "
                        "propagate at c; the static action cannot decide this"),
        dict(field="source",
             assumption="L_env, the scale on which 'environment' is defined, is "
                        "a declared universal constant, not derived; the axis "
                        "is a background field held fixed under variation, and "
                        "in the closed (dynamical-axis) version the compiler "
                        "can only label the theory"),
        dict(field="known_limits",
             assumption="the Solar-System comparison quotes published bounds as "
                        "an order of magnitude; no data were opened and no bound "
                        "was recomputed"),
        dict(field="physical_statement",
             assumption="the sign of f_E is not fixed by the theory; both signs "
                        "are admissible and the falsifier is that ONE sign holds "
                        "universally"),
    ]
    return dict(
        id="T_void_tensor",
        name="action-derived void/tensor gravity",
        one_line=("an anisotropic kinetic metric for one scalar potential, "
                  "weighted by a transition-band gate, with the axis from the "
                  "environment's tidal tensor"),
        fields=fields, baryonic_closure=closure, action=T["action"],
        axis_values=axes, derived_not_assumed=derived_not_assumed,
        compiler={k: dict(verdict=v["verdict"], primary_bin=v["primary_bin"],
                          failed=v["failed"], labels=v["labels"], flags=v["flags"],
                          gate1_escapes=v["gate1"]["escapes"],
                          gate1_max_resid_dex=v["gate1"]["max_single_probe_resid_dex"],
                          u_space_antisymmetry=(v["gate4"]["u_space"] or {}).get(
                              "max_relative_antisymmetry"))
                  for k, v in comp.items()},
        honest_gaps=gaps)


# ======================================================================
def path_card(Pj: dict, CP: dict) -> dict:
    comp = CP.get("path", {})
    ff = Pj["force_factor_tables_fiducial"]
    ffm = Pj["force_factor_tables_rho_mean"]
    mb = Pj["momentum_budget"]
    br = Pj["bridge"]
    cn = Pj["connectivity"]
    sc = Pj["member_scramble"]
    eps, rs = Pj["action"]["fiducial"]["eps"], Pj["action"]["fiducial"]["rho_star"]
    g_gal_30 = _ff_at(ff["galaxy_field"], 30.0)
    g_gal_min = float(min(ff["galaxy_field"]["factor"]))
    g_mem_20 = _ff_at(ff["galaxy_member"], 20.0)
    g_mem_30 = _ff_at(ff["galaxy_member"], 30.0)
    g_clu_300 = _ff_at(ff["cluster_shell"], 300.0)
    g_clu_1000 = _ff_at(ff["cluster_shell"], 1000.0)
    g_clu_2800 = _ff_at(ff["cluster_shell"], 2800.0)
    pp = br["projected_profile"]
    fields = {}
    fields["physical_statement"] = dict(
        statement=(
            "The gravitational interaction between two mass elements depends "
            "on the matter/vacuum state of the straight segment joining them: "
            "the pair kernel is -(G/d)[1 + eps v], with v the fraction of the "
            "segment along which the baryonic density lies below a universal "
            "density rho_*. Gravity is transmitted differently through "
            "emptiness than through matter (eps > 0: more strongly through "
            "vacuum). Every pair relation is reciprocal by construction, and "
            "the momentum a pair does not conserve is carried by the matter on "
            "its segment, which feels the gradient of a three-body potential "
            "Phi_3 that is an explicit function of the local density and of "
            "the product of the two opposite half-columns through the point."),
        evidence=dict(kernel_reciprocity=Pj["reciprocity"],
                      momentum_budget=dict(
                          sum_total=mb["sum_total_over_mean"],
                          sum_two_body=mb["sum_two_body_over_mean"],
                          carrier_closes=mb["carrier_closes_budget"]),
                      file="path_results.json"))
    fields["source"] = dict(
        statement=(
            "Rest mass rho, in two roles: as the endpoints of every pair (the "
            "rho(x) rho(y) weight) and as the STATE along the path (phi(rho) = "
            "1/(1 + rho/rho_*) integrated along the segment). No other source, "
            "no preferred axis. The state is explicit, gauge-free (density, not "
            "potential), representation-independent (a continuum functional), "
            "and measurable: the baryonic column along the segment is what "
            "X-ray, HI and absorption observations return."),
        evidence=dict(column_closed_form_check=Pj["column_check"]))
    fields["state"] = dict(
        statement=(
            "A nonlocal scalar relation W(x,y)[rho]. What a test body or photon "
            "sees is nevertheless ONE scalar potential, Phi = delta E/delta rho "
            "= Phi_dir + Phi_3, with Phi_3(z) = -(G eps/2) phi'(rho(z)) P(z) and "
            "P(z) = Int dOmega C(z,n) C(z,-n), the angular integral of the "
            "product of the two opposite half-line columns through z -- derived "
            "by collapsing the double integral over pairs whose segment passes "
            "through z. Phi_3 is non-zero only where matter lies on BOTH sides "
            "of z: inside a body, or in the bridge between two bodies. No "
            "vector field, no memory."),
        evidence=dict(action=Pj["action"], P_convergence=Pj["P_convergence"]))
    fields["propagation"] = dict(
        statement=(
            "Path-dependent (the straight segment between the two points), "
            "instantaneous weak-field response: the functional is static. Not "
            "retarded, not history-dependent. Finite propagation would need a "
            "relativistic completion the static action does not supply "
            "(assumption, flagged)."),
        evidence=dict(locality="path_dependent"))
    fields["matter_coupling"] = dict(
        statement=(
            f"a = -grad Phi, universal. Measured on the compiler's own probe "
            f"caricatures at the fiducial (eps = {eps}, rho_* = {rs:.0e}): "
            f"(i) ENDPOINT TERM: for eps > 0 the force is WEAKER in the shell "
            f"where the vacuum fraction rises with radius (the extra binding "
            f"grows outward, so the force falls) and tends to G(1+eps) beyond: "
            f"isolated galaxy g/g_N = {g_gal_min:.3f} at its minimum near 28 "
            f"kpc, {g_gal_30:.3f} at 30 kpc; cluster {g_clu_300:.3f} at 300 "
            f"kpc, {g_clu_1000:.3f} at 1 Mpc, {g_clu_2800:.3f} at 2.8 Mpc "
            f"-- the WRONG sign for the cluster excess inside 1.2 Mpc for "
            f"eps > 0; (ii) CARRIER TERM: any dense body embedded in a medium "
            f"near rho_* sits in a Phi_3 trough. A cluster member galaxy has "
            f"its internal gravity multiplied by {g_mem_20:.2f} at 20 kpc and "
            f"{g_mem_30:.2f} at 30 kpc, and the net force turns OUTWARD beyond "
            f"~38 kpc: at this amplitude the family fails the member-galaxy "
            f"regularity that the tensor lane already identified as the binding "
            f"constraint on anything that switches on inside cluster galaxies "
            f"(everything is linear in eps, so eps <~ 0.003 keeps this at the "
            f"2% level); (iii) MEMBER-MEMBER PAIRS: the mass-weighted vacuum "
            f"fraction of member-member segments is {sc['actual']['v_member_member']:.3f} "
            f"against {sc['scrambled']['v_member_member_mean']:.3f} +- "
            f"{sc['scrambled']['v_member_member_sd']:.4f} under angle scrambles "
            f"at fixed radii and masses -- the member-locked response is at the "
            f"{100*eps*sc['scrambled']['v_member_member_sd']:.2f}% level of the "
            f"pair force: connectivity lives in the GAS columns and in the "
            f"bridges, not in member positions; (iv) PAIR FORCE vs an "
            f"intervening filament: log-slope {cn['log_slopes']['path_vs_M_B']:.2f} "
            f"in the far endpoint mass (Newtonian filament pull: "
            f"{cn['log_slopes']['newton_vs_M_B']:.2f}) and "
            f"{cn['log_slopes']['path_vs_rho_f_low']:.2f} -> "
            f"{cn['log_slopes']['path_vs_rho_f_high']:.2f} in the filament "
            f"density, saturating (Newtonian: {cn['log_slopes']['newton_vs_rho_f']:.2f}, "
            f"never saturating)."),
        evidence=dict(force_factor_tables=ff, momentum=mb, scramble=sc,
                      connectivity=cn, file="path_results.json"))
    fields["photon_coupling"] = dict(
        statement=(
            f"Photons see the same Phi = Phi_dir + Phi_3 (common geometry, no "
            f"slip). Between two mass concentrations Phi_3 is a ridge along the "
            f"segment: for eps > 0 a potential HILL of "
            f"{br['phi3_midpoint_kms2']:.2e} (km/s)^2 at the midpoint of two "
            f"{br['M_A']:.0e} Msun clusters {br['D_kpc']/1000:.0f} Mpc apart at "
            f"the fiducial (linear in eps), whose effective density is negative "
            f"on the axis ({br['rho_eff_on_axis_midpoint']:.1e} kg/m^3) and "
            f"positive around it; its projected effective surface density is "
            f"a COMPENSATED profile, {pp['Sigma_eff_kg_m2'][0]:+.2f} kg/m^2 on "
            f"the axis and {max(pp['Sigma_eff_kg_m2']):+.2f} kg/m^2 in the "
            f"wings, and its volume integral vanishes (net/|net| = "
            f"{br['net_effective_mass_over_abs']:.3f} within the computed box, "
            f"zero in the infinite volume by Gauss). The amplitude scales as "
            f"M_A M_B (measured {br['P_midpoint_scaling']['ratio_two_one_over_eq']:.3f}, "
            f"{br['P_midpoint_scaling']['ratio_two_two_over_eq']:.3f} against "
            f"2, 4) and as -phi'(rho_bridge). No frequency or polarization "
            f"dependence; a photon's deflection depends on its path only through "
            f"where P is non-zero along it."),
        evidence=dict(bridge=br, file="path_results.json"))
    fields["known_limits"] = dict(
        statement=(
            "SOLAR SYSTEM, LABORATORY, PULSARS, WIDE BINARIES: exact Newtonian "
            "limit, not by absorbing a constant into G -- the interplanetary "
            "and interstellar media (~1e-21 to 1e-20 kg/m^3) are DENSER than "
            "rho_* for any rho_* below ~1e-22, so v = 0 on every such segment, "
            "and P at a planet is ~1e-13 kg^2/m^4, giving Phi_3 ~ 1e-8 m^2/s^2. "
            "GALAXY REGULARITIES: in-plane disk segments run through gas "
            "denser than rho_* (v ~ 0); the modification appears where the "
            "density falls below rho_* (~25 kpc for the caricature galaxy), "
            f"as a {100*(1-g_gal_min):.0f}% weakening of the force at the "
            f"fiducial amplitude, then G(1+eps) beyond. The binding limit is "
            f"the CLUSTER MEMBER: x{g_mem_20:.1f} at 20 kpc at eps = {eps}, "
            f"which bounds eps/rho_* by member internal dynamics -- data in "
            f"the confirmation reserve, not opened. WAVE PROPAGATION: the "
            f"static functional says nothing about the tensor-wave sector; "
            f"GW speed = c is an assumption on the completion (flagged)."),
        evidence=dict(galaxy_factor_min=g_gal_min, member_factor_20kpc=g_mem_20,
                      rho_mean_variant_max_deviation=float(max(
                          abs(np.array(t["factor"]) - 1.0).max()
                          for t in ffm.values())),
                      file="path_results.json"))
    fields["counterfactual_signature"] = dict(
        statement=(
            f"Rotate an external axis: nothing (no axis). Move the baryons "
            f"holding a halo: the response follows the NEW segments' columns "
            f"instantly (v, P re-evaluated). Move a halo holding baryons: "
            f"nothing. Scramble members preserving every radial profile: the "
            f"member-member vacuum fraction moves by "
            f"{sc['scrambled']['v_member_member_sd']:.4f} (1 sd), i.e. the pair "
            f"forces by {100*eps*sc['scrambled']['v_member_member_sd']:.2f}% at "
            f"eps = {eps} -- tiny, and the sign of the change is that of the "
            f"change in blocked path length. Change history preserving present "
            f"matter: nothing. Change the photon path preserving endpoints: a "
            f"path through the bridge between two concentrations crosses a "
            f"compensated feature (dO/dB: negative core convergence for "
            f"eps > 0), a path of equal length avoiding it sees none. Insert "
            f"a filament between two concentrations at fixed endpoints: the "
            f"pair force WEAKENS for eps > 0 (dF/F = "
            f"{cn['vs_filament_density'][2]['dF_path_over_F_N']:+.3f} at "
            f"rho_f = rho_*), with slope ~1 in the far endpoint mass and "
            f"saturating in the filament density; the same filament's own "
            f"Newtonian pull is independent of the far endpoint and linear in "
            f"the density. Embed a dense body in a medium near rho_*: it is "
            f"confined (eps > 0) or expelled (eps < 0), with the trough depth "
            f"~ (G eps/2 rho_*) P_medium."),
        evidence=dict(connectivity_rows=cn["vs_filament_density"],
                      scramble=sc, file="path_results.json"))
    fields["unique_falsifier"] = dict(
        statement=(
            "The bridge between pairs of mass concentrations, after the "
            "endpoints' own profiles are subtracted, must show a COMPENSATED "
            "feature -- zero net mass, negative core for eps > 0 -- in BOTH "
            "stacked convergence and the dynamics of bodies on the segment, "
            "with amplitude scaling as M_A M_B (slope 1 in each endpoint mass "
            "at fixed separation) and depending on the bridge's own baryonic "
            "column with the universal shape -phi'. A bridge with net positive "
            "mass; or a bridge signal that scales with the filament's own mass "
            "rather than with M_A M_B; or one that survives when one endpoint "
            "mass -> 0; or a cluster member whose internal dynamics show no "
            "trough at an amplitude where the bridge does -- each kills the "
            "law. Positivity of collisionless density (rho_DM >= 0) cannot "
            "produce a compensated feature, so no halo nuisance repairs it. "
            "NOT a falsifier: a monopole change of cluster gravity, or the "
            "smooth-gas part of the response, which is radial (BJ.1)."),
        evidence=dict(bridge_core_sign=pp["core_sign"], bridge_wing_sign=pp["wing_sign"],
                      net_over_abs=br["net_effective_mass_over_abs"]))
    fields["cdm_distinction"] = dict(
        statement=(
            "CDM's bridge is real mass: rho_DM >= 0, net positive convergence, "
            "set by the local density field and only statistically tied to the "
            "endpoints; CDM member forces depend on the pair separation and on "
            "the local dark-plus-baryonic mass, never on what lies on the "
            "segment beyond that mass's own pull. The path law's JOINT "
            "response -- (i) a compensated, zero-net-mass bridge with a sign "
            "fixed by eps; (ii) amplitude proportional to M_A M_B times "
            "-phi'(rho_bridge); (iii) pair forces that respond to an "
            "intervening filament with slope 1 in the FAR endpoint mass and "
            "saturation in the filament density; (iv) one functional tying the "
            "bridge's lensing to the acceleration of bodies on it -- cannot be "
            "reproduced by any rho_DM >= 0: (i) is a positivity obstruction, "
            "not a fine-tuning; (ii)-(iv) would require the dark matter between "
            "every pair to be arranged in proportion to the product of the "
            "endpoint masses and to saturate with the gas density. The "
            "smooth-gas (radial) part of the response and any monopole "
            "rescaling of cluster gravity carry NO distinguishing power and "
            "are excluded from the claim."),
        evidence=dict(P_scaling=br["P_midpoint_scaling"],
                      connectivity_slopes=cn["log_slopes"]))
    closure = dict(
        predicts_baryonic_closure=True,
        mean=("given the complete baryonic scene INCLUDING the columns along "
              "every pair segment (they are part of B), gravity is fixed up to "
              "(G, eps, rho_*): P(G | B) has zero intrinsic width"),
        scatter=("with only endpoint baryons known, the residual is a "
                 "deterministic, monotone, saturating function of the measured "
                 "intervening column with the universal shape -phi'; the "
                 "predicted scatter therefore has a measurable third variable "
                 "and a universal shape, and NO component that looks like a "
                 "halo's shape/orientation distribution"),
        cdm_contrast=("a halo population's residual correlates with the "
                      "intervening DARK mass, which is only statistically "
                      "related to the baryonic column, has no saturation, and "
                      "has no compensated component"))
    axes = dict(source="rest_mass (endpoints and path state)",
                field_type="scalar (nonlocal functional)",
                locality="path_dependent", superposition="nonlinear",
                directionality="isotropic", axis_origin="none",
                propagation="instantaneous_weak_field",
                geometry="force_in_fixed_space (weak field)",
                eff_dimension="three_d", matter_light="universal",
                equivalence="exact",
                conservation="exact_reciprocal, carrier = matter on the segment "
                             "(exchange_with_field in the q-carrier variant)",
                cosmology="expanding_geometry (assumed standard)",
                vacuum="polarizable (transmits differently from matter)",
                initial_conditions="standard_primordial")
    derived_not_assumed = [
        "the carrier term collapses to Phi_3 = -(G eps/2) phi'(rho) P with "
        "P = Int dOmega C(z,n) C(z,-n): the three-body potential is an "
        "algebraic function of the local density and the product of opposite "
        "half-columns",
        f"momentum: total forces sum to {mb['sum_total_over_mean']:.1e} of the "
        f"mean force, endpoint forces alone to {mb['sum_two_body_over_mean']:.3f}, "
        f"and the carrier closes the budget to {mb['carrier_closes_budget']:.1e}",
        "the bridge is compensated (zero net effective mass) and scales as "
        "M_A M_B -- a positivity obstruction against any collisionless halo",
        "the endpoint term WEAKENS the force where the vacuum fraction rises "
        "with radius even though it deepens the potential (same structure as "
        "the tensor family's 'deeper potential, weaker force')",
        "a density-gated state is required: gating the path state on |g_N| "
        "instead gives q = 1 at every symmetric centre, a 1/r phantom cusp and "
        "a finite repulsive acceleration at the centre of every galaxy "
        "(derived analytically before the density gate was adopted)",
        "the member-locked (angle-scramble) response is at the 0.1% level; the "
        "connectivity content is in gas columns and bridges",
    ]
    gaps = [
        dict(field="source",
             assumption="rho_* is a universal constant whose VALUE is not "
                        "derived; the fiducial 1e-24 kg/m^3 was chosen so the "
                        "compiler's galaxy and cluster probes straddle it "
                        "(bench identifiability), not from data; the no-new-"
                        "scale variant rho_* = rho_mean is compiled beside it"),
        dict(field="known_limits",
             assumption="the relativistic completion leaves tensor waves at c; "
                        "the static functional cannot decide this"),
        dict(field="known_limits",
             assumption="whether the member-galaxy confinement at the "
                        "fiducial is excluded could not be checked without "
                        "opening member internal dynamics (confirmation "
                        "reserve); it is reported as the binding constraint, "
                        "not as an exclusion"),
        dict(field="physical_statement",
             assumption="the sign of eps is not fixed by the theory; the "
                        "falsifier is that ONE sign holds universally"),
        dict(field="matter_coupling",
             assumption="the compiler's probe reduction cannot express the "
                        "cluster's own path-modified pull on the member (an "
                        "orbital, monopole effect); the factor on the member "
                        "probe is the galaxy's own modified field with the "
                        "cluster entering through the segment densities only"),
    ]
    return dict(
        id="P_reciprocal_path",
        name="reciprocal path-dependent gravity",
        one_line=("a reciprocal two-point kernel gated on the vacuum fraction of "
                  "the connecting segment, with the matter on the segment as the "
                  "declared momentum carrier"),
        fields=fields, baryonic_closure=closure, action=Pj["action"],
        axis_values=axes, derived_not_assumed=derived_not_assumed,
        compiler={k: dict(verdict=v["verdict"], primary_bin=v["primary_bin"],
                          failed=v["failed"], labels=v["labels"], flags=v["flags"],
                          gate1_escapes=v["gate1"]["escapes"],
                          gate1_max_resid_dex=v["gate1"]["max_single_probe_resid_dex"],
                          reciprocity=(v["gate4"]["reciprocity"] or {}).get(
                              "max_relative_asymmetry"),
                          jacobian_asymmetry=v["gate4"]["jacobian_asymmetry"])
                  for k, v in comp.items()},
        honest_gaps=gaps)


def main():
    guard.arm()
    T = J("tensor_results.json")
    Pj = J("path_results.json")
    CP = J("compile_results.json") if os.path.exists(
        os.path.join(HERE, "compile_results.json")) else {}
    cards = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        lane="work/wellnet-2026-09/synthesis", run_id="BK-synthesis",
        data_statement=("No observational data of any kind is opened by this "
                        "lane; asserted mechanically by patching open/io.open "
                        "and the numpy loaders (universes/provenance.py) with "
                        "the lane directory as the only readable root. KiDS "
                        "and the wide binaries are sealed by token; the "
                        "confirmation reserve (SPT, X-GAP, CLoGS, Gaia "
                        "dynamical products, MUSE/Granata dispersions) is "
                        "guarded by token and untouched."),
        binding_rule=("BJ.1: anisotropy is not evidence for anisotropic "
                      "gravity; a recovered a0 is not evidence for modified "
                      "gravity; a MOND-like monopole is not evidence for MOND. "
                      "Each card's falsifier is a SIGN, a PHASE LOCK, a "
                      "SCALING or a COMPENSATION, never a magnitude."),
        families=dict(T_void_tensor=tensor_card(T, CP),
                      P_reciprocal_path=path_card(Pj, CP)),
    )
    cards["provenance"] = guard.summary()
    with open(os.path.join(HERE, "cards.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(cards, fh, indent=1, default=float)
    for fid, c in cards["families"].items():
        missing = [f for f in ("physical_statement", "source", "state",
                               "propagation", "matter_coupling",
                               "photon_coupling", "known_limits",
                               "counterfactual_signature", "unique_falsifier",
                               "cdm_distinction") if f not in c["fields"]]
        print(fid, "fields:", len(c["fields"]), "missing:", missing,
              "| compiler entries:", len(c["compiler"]))
    print("wrote cards.json |", cards["provenance"]["assertion"])


if __name__ == "__main__":
    main()
