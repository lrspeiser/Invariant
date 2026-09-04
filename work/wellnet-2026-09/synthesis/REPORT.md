# Run BK -- the Principle Synthesis Lane: two principle cards, two actions, compiled

Generated 2026-09-04T22:41:18Z from `cards.json`, `tensor_results.json`, `path_results.json`, `compile_results.json`. Run id `BK-synthesis`, registered in `work/wellnet-2026-09/registry/registry.py` before any work.

**Data statement.** No observational data of any kind is opened by this lane; asserted mechanically by patching open/io.open and the numpy loaders (universes/provenance.py) with the lane directory as the only readable root. KiDS and the wide binaries are sealed by token; the confirmation reserve (SPT, X-GAP, CLoGS, Gaia dynamical products, MUSE/Granata dispersions) is guarded by token and untouched.

Provenance ledgers of the four lane scripts: foreign reads [0, 0, 0, 0], real-observation token matches [False, False, False, False], reserve-token matches [False, False, False, False].

## 0. Why this lane, and the rule it works under

Run BJ's verdict: the programme had built a compiler, a certificate, a ten-universe suite and an equivalence map and had NOT constructed the law that is its objective. This lane constructs. Theory construction spends no confirmation data, so the certificate gate does not block it. The binding rule: BJ.1: anisotropy is not evidence for anisotropic gravity; a recovered a0 is not evidence for modified gravity; a MOND-like monopole is not evidence for MOND. Each card's falsifier is a SIGN, a PHASE LOCK, a SCALING or a COMPENSATION, never a magnitude.

Both cards therefore stake their falsifiers on a **sign**, a **phase lock**, a **scaling** or a **compensation** -- never on the size of an anisotropy, an a0, or a monopole.

## 1. The two actions (Job 2)

### T -- action-derived void/tensor gravity

```
    L = -(1/8 pi G)[ a0^2 F(|u|^2/a0^2) + f_E h(|u|/a0) u^T That_env u ] - rho Phi,  u = grad Phi
    base:   F' = mu, mu(x) = x/(1+x) (AQUAL)
    weight: h(x) = mu(1-mu) = x/(1+x)^2
    axis:   That_env: unit-norm traceless tidal tensor of the environment (Hessian of Phi_N smoothed on L_env); principal-axis form sqrt(3/2)(e e^T - I/3) with sqrt(3/2) absorbed into f_E
    E-L:    div M(u) = 4 pi G rho, M = mu(x) u + (1/2) f_E [ h'(x)(u^T That u) uhat/a0 + 2 h(x) That u ]
    radial: g [ mu(x) + f_E lambda eta(x) ] = g_N, lambda = (e.rhat)^2 - 1/3, eta = h + x h'/2 = x(3+x)/(2(1+x)^3)
```

Universal constants: shared with the base ['G', 'a0'], new ['f_E', 'L_env'] -- **2 new**. Model class: static_scalar_potential with a background axis field (in the compiler's Gate 4 scope); the closed version with That_env promoted to a dynamical field is 'extra_propagating_field'.

Three things about it are derived rather than chosen:

* **h must vanish in deep MOND.** A weight that stays finite as x -> 0 makes the kinetic operator indefinite wherever |grad Phi| is small: measured, a constant h = 0.25 at f_E = 0.3 loses ellipticity on 59% of a cloud with x < 1, everywhere below x = 0.025. h = mu(1 - mu) is the sparsest weight that vanishes at both ends with no new scale.
* **The admissible f_E interval is (-0.95, 1.85)**, from positive-definiteness of the Hessian of the Lagrangian over a u-cloud (analytic principal-direction bounds: radial -1 < f_E < 2, transverse -2 < f_E < 6).
* **The gating must live in the Lagrangian.** The same gate written as K(u)u in QUMOND form is not a gradient (compiler element F2 below); written as a term of a scalar Lagrangian density it is one identically.

### P -- reciprocal path-dependent gravity

```
    S_P = -(1/2) Int Int rho(x) W(x,y) rho(y),  W = -(G/|x-y|)[1 + eps v(x,y)],  v = (1/|x-y|) Int_seg phi(rho) dl,  phi(rho) = 1/(1 + rho/rho_*)
    Phi = delta E/delta rho = Phi_dir + Phi_3,  Phi_3(z) = -(G eps/2) phi'(rho(z)) P(z),  P(z) = Int dOmega C(z,n) C(z,-n)
    no local PDE: Phi is the functional derivative of a nonlocal scalar functional; test bodies and photons follow -grad Phi
    carrier: matter on the connecting segment, through -grad Phi_3 (three-body term); Sum F = 0 by translation invariance
```

Universal constants: shared with the base ['G'], new ['eps', 'rho_*'] -- **2 new** (no-new-scale variant: rho_* = rho_mean). Fiducial eps = 0.3, rho_* = 1e-24 kg/m^3 (chosen so the compiler's probes straddle it; not from data). Model class: static_scalar_potential (nonlocal in rho, local in the test body): in the compiler's Gate 4 scope, as its symmetric-nonlocal control XC5 is.

The carrier term is not asserted, it is derived and then measured:

* the double integral over all pairs whose segment passes through z collapses to `Phi_3(z) = -(G eps/2) phi'(rho(z)) P(z)`, `P = Int dOmega C(z,n) C(z,-n)` -- an algebraic function of the local density and of the product of the two opposite half-columns (closed form for Plummer spheres, checked by quadrature to 2.0e-05; the angular integral converges to 2.4e-05 at 20,000 directions);
* on a 5-body configuration the total forces sum to **3.0e-09** of the mean force, the endpoint (two-body) forces alone to **0.030**, and the carrier forces to 0.030 with the opposite sign: the budget closes to 3.0e-09. The matter on the segments is the carrier, verified.
* the kernel is reciprocal to 0.0e+00 (exactly: the segment is the same set in both orders).

## 2. The principle cards (Job 1) -- ten fields each

Rendered in full in `card_tensor.md` and `card_path.md`; machine-readable in `cards.json`. The two fields the review made mandatory, in brief:

### T: action-derived void/tensor gravity

**Unique falsifier.** At fixed baryons and fixed independently measured tidal axis, the population distribution of (m=2 phase minus tidal-axis phase) must be a delta function convolved with the measurement error alone, with ONE universal sign (for f_E > 0 the m=2 MINIMUM of |g_r| on the axis), zero radial twist, and an amplitude profile with the universal shape A2(g/a0): largest near the g ~ a0 crossing, decaying as 1/x at high acceleration, vanishing in deep MOND. A measured intrinsic phase dispersion exceeding the axis error after marginalising inclination, position angle, distance, M/L and the axis measurement; or mixed signs across objects; or a quadrupole amplitude that GROWS into the deep-MOND outskirts -- each kills the law, and none can be repaired by an object-specific nuisance. NOT a falsifier: anisotropy, a recovered a0, a MOND-like monopole (BJ.1).

**CDM distinction.** A realistic collisionless halo is triaxial with intrinsic axis-ratio scatter, oriented by the tidal field at ASSEMBLY with tens of degrees of misalignment dispersion relative to the PRESENT tidal axis, twisting with radius, carrying stochastic m=1/m=3 power from substructure, and with an anisotropy that persists or grows in the outskirts. The tensor law's JOINT response is: (i) phase locked to the present tidal axis with zero intrinsic dispersion; (ii) zero radial twist; (iii) an amplitude that is a universal function of g/a0 alone, peaking in the transition band and VANISHING where a halo's anisotropy is largest; (iv) one universal sign; (v) identical phase and tied amplitude in lensing and dynamics with no per-object freedom; (vi) residual harmonic content in m=2 only. A CDM population can match any one of these by selection; matching all six requires its halo shape distribution collapsed to a delta function slaved to the present tidal field with an ellipticity profile keyed to the baryonic acceleration and dying in the outskirts -- the opposite of what assembly-history scatter and outer triaxiality necessarily produce. Because the family's own signature is a sign and a profile rather than a magnitude, BF's 0.648 rate of the generic anisotropy detectors on the dark-matter universe does not apply to it: the detector must test (i)-(vi) jointly, not anisotropy.

**Baryonic closure.** predicted: True. Mean: given the complete baryonic scene AND the environment's tidal axis, gravity is fixed up to the universal constants (G, a0, f_E, L_env): P(G | B, e) has zero intrinsic width Scatter: the law predicts the SCATTER, not only the mean: (a) with e observed, residual scatter = measurement noise only; (b) with e unobserved (local baryons only), the residual is a PURE m=2 harmonic of universal amplitude A2(g/a0) f_E and random phase, with no m=1, m=3 or m=4 power, and a RAR residual distributed as -(1/2) P2(cos psi) A2 f_E over the distribution of psi -- a scatter with a predicted shape (width 0.0127 f_E dex at r = 3 a) and harmonic content

### P: reciprocal path-dependent gravity

**Unique falsifier.** The bridge between pairs of mass concentrations, after the endpoints' own profiles are subtracted, must show a COMPENSATED feature -- zero net mass, negative core for eps > 0 -- in BOTH stacked convergence and the dynamics of bodies on the segment, with amplitude scaling as M_A M_B (slope 1 in each endpoint mass at fixed separation) and depending on the bridge's own baryonic column with the universal shape -phi'. A bridge with net positive mass; or a bridge signal that scales with the filament's own mass rather than with M_A M_B; or one that survives when one endpoint mass -> 0; or a cluster member whose internal dynamics show no trough at an amplitude where the bridge does -- each kills the law. Positivity of collisionless density (rho_DM >= 0) cannot produce a compensated feature, so no halo nuisance repairs it. NOT a falsifier: a monopole change of cluster gravity, or the smooth-gas part of the response, which is radial (BJ.1).

**CDM distinction.** CDM's bridge is real mass: rho_DM >= 0, net positive convergence, set by the local density field and only statistically tied to the endpoints; CDM member forces depend on the pair separation and on the local dark-plus-baryonic mass, never on what lies on the segment beyond that mass's own pull. The path law's JOINT response -- (i) a compensated, zero-net-mass bridge with a sign fixed by eps; (ii) amplitude proportional to M_A M_B times -phi'(rho_bridge); (iii) pair forces that respond to an intervening filament with slope 1 in the FAR endpoint mass and saturation in the filament density; (iv) one functional tying the bridge's lensing to the acceleration of bodies on it -- cannot be reproduced by any rho_DM >= 0: (i) is a positivity obstruction, not a fine-tuning; (ii)-(iv) would require the dark matter between every pair to be arranged in proportion to the product of the endpoint masses and to saturate with the gas density. The smooth-gas (radial) part of the response and any monopole rescaling of cluster gravity carry NO distinguishing power and are excluded from the claim.

**Baryonic closure.** predicted: True. Mean: given the complete baryonic scene INCLUDING the columns along every pair segment (they are part of B), gravity is fixed up to (G, eps, rho_*): P(G | B) has zero intrinsic width Scatter: with only endpoint baryons known, the residual is a deterministic, monotone, saturating function of the measured intervening column with the universal shape -phi'; the predicted scatter therefore has a measurable third variable and a universal shape, and NO component that looks like a halo's shape/orientation distribution

## 3. Compiler verdicts (Job 3)

Two basis elements were added to the compiler's grammar, `tensor_L` and `path_kernel`, each keyed on its own struct name. Additivity is asserted: 29 pre-existing candidates (known families, external-axis elements, external positive controls) re-compiled with **0 changed** verdicts, failed-gate lists, bins, defect lists or measured statistics; the external controls all agree: True. The compiler's own regression suite (`test_compiler.py`) passes 48 of 48 after the patch (`compiler_suite_run.log`).

| element | verdict | bin | failed | labels / flags | gate 1 escapes | max probe resid (dex) | u-space / Jacobian |
|---|---|---|---|---|---|---|---|
| `T_tensor_L_fE0.1` | ADMIT | `admissible` | -- | -- | b_independent_axis | 0.0064 | u 1.4e-10 / J 0.0e+00 |
| `T_tensor_L_fE0.3` | ADMIT | `admissible` | -- | -- | b_independent_axis | 0.0190 | u 1.8e-10 / J 0.0e+00 |
| `T_tensor_L_fE1.0` | ADMIT | `admissible` | -- | -- | a_spatial_variation, b_independent_axis, c_probe_disagreement | 0.0617 | u 1.8e-10 / J 0.0e+00 |
| `T_tensor_L_fE1.8` | ADMIT | `admissible` | -- | -- | a_spatial_variation, b_independent_axis, c_probe_disagreement | 0.1085 | u 2.2e-10 / J 0.0e+00 |
| `T_tensor_L_fE0.3_dynamical_axis` | OUTSIDE-CLASS | `outside_declared_model_class` | -- | action-based but OUTSIDE the scalar-potential class; momentum carrier declared but not verified | b_independent_axis | 0.0190 | -- |
| `grammar_F1_ext_axis_const` | ADMIT | `admissible` | -- | -- | a_spatial_variation, b_independent_axis, c_probe_disagreement | 0.0412 | u 0.0e+00 / J 0.0e+00 |
| `grammar_F2_ext_axis_gn_gated` | REJECT | `physically_incomplete_as_written` | gate4_scalar_potential_integrability | -- | b_independent_axis | 0.0360 | u 4.9e-02 / J 0.0e+00 |
| `grammar_F3_ext_axis_tidal_gated` | REJECT | `physically_incomplete_as_written` | gate4_scalar_potential_integrability | -- | b_independent_axis | 0.0235 |  / J 1.9e-03 |
| `P_path_kernel_fid` | ADMIT | `admissible` | -- | momentum carrier declared but not verified | a_spatial_variation, c_probe_disagreement | 0.2954 |  / J 3.1e-18 |
| `P_path_kernel_eps0.03` | ADMIT | `admissible` | -- | momentum carrier declared but not verified | a_spatial_variation, c_probe_disagreement | 0.0791 |  / J 0.0e+00 |
| `P_path_kernel_eps0.003` | REJECT | `non_identifiable_on_this_bench` | gate1_constant_K | momentum carrier declared but not verified | none | 0.0101 |  / J 0.0e+00 |
| `P_path_kernel_fid_no_carrier_declared` | ADMIT | `admissible` | -- | -- | a_spatial_variation, c_probe_disagreement | 0.2954 |  / J 3.1e-18 |
| `P_path_kernel_rho_mean` | REJECT | `non_identifiable_on_this_bench` | gate1_constant_K | momentum carrier declared but not verified | none | 0.0093 |  / J 1.4e-18 |
| `P_path_kernel_field_carrier` | OUTSIDE-CLASS | `outside_declared_model_class` | -- | action-based but OUTSIDE the scalar-potential class; momentum carrier declared but not verified | a_spatial_variation, c_probe_disagreement | 0.2954 | -- |
| `grammar_XC5_symmetric_nonlocal_action` | ADMIT | `admissible` | -- | -- | a_spatial_variation, c_probe_disagreement | 0.0518 |  / J 0.0e+00 |

Reading the table:

* **T at every f_E inside the ellipticity interval: ADMIT, `admissible`.** The u-space antisymmetry is at round-off (1e-10 against a 1e-7 floor): the flux map is a gradient by construction. Gate 1 is escaped through the independently measured axis at every amplitude; the radial residual on the bench's probes only exceeds 0.04 dex at f_E >~ 1, because the family's content is a 2-D phase, not a monopole -- which is what the compiler's radial reduction cannot see and the extraction lane must.
* **The nearest pre-existing grammar elements F2/F3 REJECT as `physically_incomplete_as_written`.** That verdict is about the QUMOND-form grammar (K(u)u with a gated f_E is not a gradient), not about the action written here; the two differ exactly by where the gate sits.
* **T with a dynamical axis field: OUTSIDE-CLASS, `outside_declared_model_class`.** A statement about the scorer (Gate 4 adjudicates the static scalar-potential class only), not about the theory; gates 1-3 still apply and pass. This is the `unsupported_by_current_scorer` situation of BA.5 and is labelled, not rejected.
* **P at eps = 0.3 and 0.03: ADMIT, `admissible`**, escaping Gate 1 by spatial variation and probe disagreement (0.30 and 0.08 dex). The kernel is reciprocal exactly and the declared Green's function is symmetric to round-off. The 'momentum carrier declared but not verified' flag is closed by the momentum budget above (the compiler cannot run that test; this lane did). The same element with no carrier declared also admits -- reciprocity by construction needs no carrier for the gate; the carrier is what makes the third law hold.
* **P at eps = 0.003 and the no-new-scale variant rho_* = rho_mean: REJECT, `non_identifiable_on_this_bench`.** The response on the bench's galaxy and cluster probes is 0.01 dex, a coordinate stretch absorbs it. This is the honest bin: the family's distinctive observable at small eps is the BRIDGE between two concentrations, and the bench has no two-body probe. A scorer statement, not a rejection of the theory.
* **P with a field carrier: OUTSIDE-CLASS.** Same scorer statement as for T.
* The compiler's admissible-branch reason string says 'the law is AQUAL/QUMOND with a redefined interpolating function' for every admitted element, including the nonlocal kernel; that wording is generic to the branch (it appears for the control XC5 too) and is not a claim about these elements.

## 4. Counterfactual signatures with signs (Job 4) -- the extraction lane's input

### T

| intervention | observable | response dO/dB | sign |
|---|---|---|---|
| rotate the external axis e by an angle dpsi, baryons fixed | phase of the m=2 harmonic of |g_r| (dynamics) and of the lensing-potential quadrupole | d(phase)/d(psi_e) = +1 exactly, zero lag, at every radius; amplitude unchanged | +1 (rigid co-rotation) |
| move the baryons holding e fixed | amplitude of the m=2 harmonic of |g_r| | A2(r) = chi'(r)/g0(r) per unit f_E re-evaluated on the NEW baryons instantly; |A2| peaks at 0.161 f_E where g0 ~ 1.18 a0, decays as -1/(3 g0/a0) at high acceleration and -> 0 in deep MOND | - along e for f_E > 0: the inward pull is WEAKER along the tidal axis (the m=2 minimum of |g_r| sits on e); the potential is nonetheless deeper along e |
| move the halo holding baryons fixed | anything | 0: there is no halo | 0 |
| scramble members preserving every radial profile | m=2 amplitude and phase | 0: the response is a functional of the smooth field and the external axis, never of the member list | 0 |
| change history preserving present matter and e | anything | 0: no memory | 0 |
| change the photon path preserving endpoints | deflection / time delay | follows the same Phi = Phi0 + f_E chi P2 the matter sees; the lensing quadrupole phase equals the dynamical one and the two amplitudes are the same chi (no slip) | fixed matter-light covariance = +1 in the potential |
| radial profile of the quadrupole at fixed e | d(phase)/d ln r | 0 exactly (e is uniform across the object); a collisionless triaxial halo twists with radius | 0 vs CDM != 0 |
| tilt the disk normal against e (angle psi) | azimuthally averaged rotation-curve residual (the RAR residual of one disk) | the in-plane average of P2(e.rhat) is -(1/2) P2(cos psi) [<(e.rhat)^2>_plane = sin^2(psi)/2], so d ln g_c = -(1/2) P2(cos psi) A2(r) f_E with A2 < 0 for f_E > 0 | + for e along the disk normal (psi = 0), - for e in the plane (psi = 90 deg), for f_E > 0: a SIGNED, environment-locked RAR residual of 0.0127 f_E dex at r = 3 a of the caricature galaxy, with no per-object freedom |

The sign is derived from the first-order l = 2 solution on a spherical source in the AQUAL base (Plummer 5e10 Msun, a = 3 kpc, AQUAL base mu = x/(1+x)), validated against the exact constant-K solution to 4.7e-04 and against both analytic asymptotes (Newtonian side A2 -> -1/(3x): ratios [0.993, 0.98, 0.95]; deep MOND chi -> -sqrt(G M a0)/3: ratio 0.984). Profile of A2 per unit f_E:

| r/a | r (kpc) | g0/a0 | h | A2 exact | A2 compiler caricature |
|---|---|---|---|---|---|
| 0.3 | 0.9 | 2.408 | 0.207 | -0.0837 | -0.1200 |
| 0.5 | 1.5 | 3.064 | 0.186 | -0.0597 | -0.0982 |
| 1 | 3.0 | 3.034 | 0.186 | -0.0458 | -0.0991 |
| 2 | 6.0 | 1.797 | 0.230 | -0.0509 | -0.1506 |
| 3 | 9.0 | 1.147 | 0.249 | -0.0583 | -0.2046 |
| 5 | 15.0 | 0.629 | 0.237 | -0.0643 | -0.2824 |
| 10 | 30.0 | 0.286 | 0.173 | -0.0588 | -0.3726 |
| 20 | 60.0 | 0.135 | 0.105 | -0.0432 | -0.4312 |
| 50 | 150.0 | 0.052 | 0.047 | -0.0234 | -0.4712 |
| 100 | 300.0 | 0.026 | 0.024 | -0.0134 | -0.4854 |

**The non-obvious result:** for f_E > 0 the inward acceleration is WEAKER along the axis e (A2 - at every radius from 0.1 a to 300 a) even though the potential is DEEPER along e (chi < 0): the potential quadrupole is a nearly constant offset that is approached from above (chi ~ -a0 r/3 on the Newtonian side, chi -> -sqrt(G M a0)/3 in deep MOND), so its radial derivative is negative. A CONSTANT anisotropy of the same axis gives the opposite force sign (chi = -GM/3r rises toward zero). This is a derived, non-obvious prediction: a transition-band-gated tensor and a constant tensor of the same axis and sign have OPPOSITE force quadrupoles. The compiler's declared radial caricature has the opposite sign to the exact solution for constant K, and the same sign but a magnitude wrong by up to 40x (and no deep-MOND decay) for the gated case; gates 1 and 4 consume only |residual| and symmetry, so no verdict depends on it, but no sign may be read off that reduction.

### P

| intervention | response dO/dB | sign (eps > 0) |
|---|---|---|
| rotate an external axis | none | 0 |
| move baryons holding a halo | v and P re-evaluated on the new segments, instantly | follows the new columns |
| move a halo holding baryons | none | 0 |
| scramble members preserving every radial profile | member-member vacuum fraction 0.333 -> 0.347 +- 0.0036; pair forces move by 0.11% | sign of the change in blocked path length; TINY |
| change history preserving present matter | none | 0 |
| change the photon path preserving endpoints | a path through the bridge crosses a compensated feature: Sigma_eff -0.49 kg/m^2 on the axis, +0.16 in the wings at the fiducial; an equal-length path avoiding it sees none | core -, wings + |
| insert a filament between two concentrations | pair force x(1 + eps dv): dF/F = -0.070 at rho_f = rho_*; log-slope 0.87 in the far endpoint mass, 0.87 -> 0.08 in rho_f (Newtonian pull of the same filament: -0.00 and 1.00) | - (weaker) |
| embed a dense body in a medium near rho_* | confined in a Phi_3 trough: cluster member x7.56 at 20 kpc at the fiducial, net force outward beyond ~38 kpc | + (confinement) |
| the M_A M_B scaling of the bridge | P at the midpoint x2.001 for 2 M_A, x4.000 for 2 M_A, 2 M_B | slope 1, 1 |

Force factors g/g_N on the compiler's three probe caricatures at the fiducial (endpoint term, carrier term, total):

| probe | r (kpc) | endpoint | carrier | total |
|---|---|---|---|---|
| galaxy_field | 10 | 0.996 | 1.006 | 1.001 |
| galaxy_field | 20 | 0.940 | 1.010 | 0.951 |
| galaxy_field | 30 | 0.912 | 0.952 | 0.864 |
| cluster_shell | 300 | 0.969 | 0.976 | 0.945 |
| cluster_shell | 700 | 0.963 | 0.957 | 0.920 |
| cluster_shell | 1400 | 1.051 | 0.914 | 0.965 |
| cluster_shell | 2800 | 1.181 | 0.990 | 1.171 |
| galaxy_member | 10 | 0.996 | 1.156 | 1.152 |
| galaxy_member | 20 | 0.964 | 7.600 | 7.565 |
| galaxy_member | 30 | 0.974 | 2.750 | 2.724 |

**The known-limits problem this exposes.** At an amplitude the bench can see, the path family multiplies the internal gravity of a cluster member by ~8 at 20 kpc and expels its stars beyond ~38 kpc, because a dense body embedded in a medium near rho_* sits in a Phi_3 trough of depth ~(G eps/2 rho_*) P_medium. This is the tensor lane's finding again from a different construction: anything that switches on with the environment switches on hardest inside cluster galaxies. Everything is linear in eps, so the amplitude at which members are safe (eps <~ 0.003) is one the bench cannot identify -- and at which the surviving distinctive signal is the compensated bridge between concentrations, at the (100 km/s)^2 level. The extraction lane needs a two-concentration scene to test this family at all.

## 5. Findings about the bench, reported rather than hidden

* The compiler's declared radial reduction for fixed-axis tensors (k_r acting on the radial flux) gives the OPPOSITE sign of angular modulation to the exact constant-K solution (1 - (2/3) f_E P2 against 1 + f_E P2/3), and for the gated element the same sign as the exact first-order solution but a magnitude wrong by up to 40x and no deep-MOND decay. Gates 1 and 4 consume only |residual| and symmetry, so no verdict depends on it; no sign may be read off that reduction, and none was.
* Gate 1 is blind to a 2-D phase by construction (a radial reduction), which is why the tensor family escapes it only through axis provenance below f_E ~ 1.
* The bench has no two-body (bridge) probe, which is why the path family at a member-safe amplitude is `non_identifiable_on_this_bench`.
* The compiler's `finite_positive_g` health check would read a genuine sign change of the net force (a nonlocal functional can do that) as 'no bounded solution'; it did not fire here because the member probe's radii stop at 30 kpc, but it is a scalar-PDE assumption that a nonlocal-functional theory does not satisfy.

## 6. Fields that could not be filled without an assumption this lane invented

* **T / known_limits** -- that the relativistic completion puts the anisotropic kinetic term in the scalar sector only, so tensor waves propagate at c; the static action cannot decide this
* **T / source** -- L_env, the scale on which 'environment' is defined, is a declared universal constant, not derived; the axis is a background field held fixed under variation, and in the closed (dynamical-axis) version the compiler can only label the theory
* **T / known_limits** -- the Solar-System comparison quotes published bounds as an order of magnitude; no data were opened and no bound was recomputed
* **T / physical_statement** -- the sign of f_E is not fixed by the theory; both signs are admissible and the falsifier is that ONE sign holds universally
* **P / source** -- rho_* is a universal constant whose VALUE is not derived; the fiducial 1e-24 kg/m^3 was chosen so the compiler's galaxy and cluster probes straddle it (bench identifiability), not from data; the no-new-scale variant rho_* = rho_mean is compiled beside it
* **P / known_limits** -- the relativistic completion leaves tensor waves at c; the static functional cannot decide this
* **P / known_limits** -- whether the member-galaxy confinement at the fiducial is excluded could not be checked without opening member internal dynamics (confirmation reserve); it is reported as the binding constraint, not as an exclusion
* **P / physical_statement** -- the sign of eps is not fixed by the theory; the falsifier is that ONE sign holds universally
* **P / matter_coupling** -- the compiler's probe reduction cannot express the cluster's own path-modified pull on the member (an orbital, monopole effect); the factor on the member probe is the galaxy's own modified field with the cluster entering through the segment densities only

Neither card's ten fields are missing; the fields above are filled with a declared assumption rather than a derivation, and they are: the tensor-wave sector for both (a static action cannot decide it), the environment scale L_env and the density scale rho_* (declared constants, not derived), the signs of f_E and eps (free; the falsifier is universality of the sign), and the order-of-magnitude Solar-System comparison for T (published bounds quoted, nothing recomputed).

## 7. What the Principle Extraction Lane should take from this

* For T: pair every universe on the same baryonic scene AND the same independently generated tidal axis; match away the monopole; test the m=2 phase lock, its radial constancy, its universal sign, its A2(g/a0) profile and its matter-light identity JOINTLY; a CDM halo drawn with assembly-history misalignment and outer triaxiality is the null. The response to 'rotate the axis' is +1 with zero lag; to 'move the halo' it is 0.
* For P: the corpus needs pairs of concentrations with a resolved bridge; the observable is the compensated Sigma_eff profile with its M_A M_B and -phi'(rho) scalings and the same feature in the dynamics of bodies on the segment; the member scramble is NOT where the signal is (0.1%); member internal dynamics bound eps/rho_*.
* Both families predict baryonic closure with a SPECIFIC scatter structure (T: pure m=2, universal amplitude, random phase when the axis is unobserved; P: a monotone saturating function of the intervening column). That structure, not the mean, is the distinguishing quantity BJ.6 asked for.

## 8. Files

`guard.py` (provenance), `tensor_family.py` -> `tensor_results.json`, `path_family.py` -> `path_results.json`, `compile_families.py` -> `compile_results.json` (+ `baseline_verdicts_prepatch.json`), `cards.py` -> `cards.json`, `render_report.py` -> this file, `card_tensor.md`, `card_path.md`; `run_all.py` runs them in order. Compiler patch: `../compiler/compiler.py` (two additive struct branches: `tensor_L`, `path_kernel`; `Candidate.force_factor`).
