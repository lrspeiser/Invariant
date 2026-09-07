# Run BL -- the Principle Extraction Lane: what separates dark matter from the modified-gravity class, located, distilled and validated across two generators

Lane `work/wellnet-2026-09/extraction/`.  Registry `BL-extraction`, VALID.  Entirely synthetic.  Provenance: foreign reads **0**; real-observation token matched **False**; the guard is installed in every worker and exercised by test T7.  Tests: **22/22** pass.

Input: Run BF's finding that the synthetic corpus separates the dark-matter universe (U2) from the seven-family modified-gravity class at z = 8.5 while the named detectors are not what carries it (BF.3, BJ.6), and Run BK's finding that the tensor/halo separation is a statement about the halo-alignment prior (BK.4).  Protocol: the fourth review's, adopted verbatim (BJ.7).

## 0  Design

* **Paired sets.** 1000 paired sets on BF's shared scene library: one scene draw (30 galaxies, 12 clusters), one set of block-seeded noise streams, one halo draw, emitted under every arm.  Test T1 checks that the seven deformations at zero amplitude reproduce U3 bit for bit and that two different universes share their sources, photo-z and shape noise.  500 sets calibrate, 500 audit; nothing is fitted and scored on the same set.
* **The class** is BF's seven families at BF's own threshold amplitudes (E9) plus the scalar-null family H0: `U3, H0, U4t, U5t, U6t, U7t, U8t, U9t`.  The fiducial amplitudes (U4f-U9f), Newton (U1) and systematics-only (U10) are reference arms.
* **The CDM prior.** Every property a halo has that the baryons do not fix is a declared knob with a nominal prior: SHMR scatter 0.16 dex, concentration scatter 0.11/0.13 dex, cluster halo mass scatter 0.05 dex, shape e_h = 0.72 ell_bar + N(0, 0.10), **halo-filament alignment f_lss ~ Beta(2, 2) per object** (BK's mixture), galaxy in-plane halo quadrupole U(0, 0.10), oblateness q_h ~ U(0.70, 1.00), dark-disc fraction f_dd ~ U(0, 0.05).  Every separation below is reported as a function of these.
* **Invariant reductions.** Rotation-, translation- and permutation-invariant by construction (test T4): per-ring harmonic decompositions of the velocity field, a PSF-forward-modelled rotation curve, vertical dispersions; per-bin m = 0, 1, 2, 4 fits of tangential AND cross ellipticity with covariance, X-ray/SZ/hydrostatic, member-dispersion and strong-lensing profiles, the member-locked well-strength correlation with its scrambled control.  Phases enter only relative to observed axes.
* **The scalar monopole is matched away.** Every gravitational quantity is a residual from the corpus's own cross-fitted universal scalar law nu-hat(g_bar), a P-spline fitted on half the galaxies and applied out of fold to the other half and to every cluster.  Test T6: an a0 shift of 0.15 dex moves the raw galaxy boost by 0.06 dex and the residuals by 0.0002 dex.
* **Two generators.** Generator 1 is BF's physics re-emitted in paired form.  Generator 2 (`forward2.py`) shares nothing: Miyamoto-Nagai discs, Einasto haloes, a quadratic SHMR, a different law family, Vikhlinin-type gas, a shell-sum projection, an Osipkov-Merritt Jeans solve, BK's lensing quadrupole and cosmology, a different instrument.

**Bugs the tests caught before any result.** T2: the first analysis projection used a stretched z grid whose first step was tens of thousands of kpc and returned Sigma too large by 6-60x with a 1/R dependence.  T5: the ring-fitted rotation curve carried a 0.1 dex beam-smearing scatter into the vertical contrast even without noise; the forward-modelled curve removed it.  T2 (generator 2): a mid-radius shell sum was 15% off the analytic NFW until the inverse-square-root singularity was integrated per shell.  T4: the scrambled network control was neither rotation- nor permutation-invariant in its first form.

## 1  Sizing first, on untouched halves

A-vs-A: one universe, random halves of its calibration sets labelled 1/0, the discriminator fitted, random halves of its audit sets scored.  24 draws: null AUC mean 0.496, sd **0.0190** (galaxies only 0.0209, clusters only 0.0233); the 95th percentile of |AUC - 0.5| under the null is 0.0304.  z = (AUC - 0.5)/sd, capped at 8.5 as in BF; a permutation p-value on the audit scores is quoted beside every z.

## 2  The discriminator: the CDM prior against the class

| class member | galaxies + clusters | galaxies only | clusters only |
|---|---|---|---|
| U3 | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 |
| H0 | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 |
| U4t | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 |
| U5t | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 |
| U6t | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 |
| U7t | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 |
| U8t | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 |
| U9t | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 |
| **pool** | **AUC 1.000, z 8.50*, p 0.0003** | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 |

Per-object discrimination (one galaxy / one cluster, U2 vs U3): galaxies **0.991**, clusters **1.000**.  At the critical value from one half of the class's audit corpora (never used in training): **power 1.000 [0.985, 1.000] on CDM at a realised class false-positive rate of 0.027 [0.021, 0.035]** (nominal 0.05); at nominal 0.01, power 1.000 [0.985, 1.000], realised 0.008 [0.005, 0.013].  The class pool holds near-duplicate corpora (the same scene and noise under deformations below detectability), so the effective number of class corpora is the number of sets, 500, and the pool's permutation p is the single-arm U3 value.

Reference arms, scored by the same discriminator (a CDM detector must not fire on the class at fiducial amplitude, on systematics, or on a Newtonian universe):

| arm | U2 vs arm | rate the arm is called CDM at the class's 0.05 critical value |
|---|---|---|
| U1 | AUC 1.000, z 8.50*, p 0.0003 | 0.224 [0.177, 0.280] |
| U10 | AUC 1.000, z 8.50*, p 0.0003 | 1.000 [0.985, 1.000] |
| U4f | AUC 1.000, z 8.50*, p 0.0003 | 0.216 [0.169, 0.271] |
| U5f | AUC 1.000, z 8.50*, p 0.0003 | 0.072 [0.046, 0.111] |
| U6f | AUC 1.000, z 8.50*, p 0.0003 | 0.116 [0.082, 0.162] |
| U7f | AUC 1.000, z 8.50*, p 0.0003 | 0.188 [0.144, 0.241] |
| U8f | AUC 1.000, z 8.50*, p 0.0003 | 0.020 [0.009, 0.046] |
| U9f | AUC 1.000, z 8.50*, p 0.0003 | 0.020 [0.009, 0.046] |
| U2 itself |  | 1.000 [0.985, 1.000] |

## 3  Which channel carries it: the ablation table

Full discriminator: AUC 1.000, z 8.50.  Leave one channel out / keep one channel in (pool AUC; z against the sized null; the per-member AUCs are in the JSON):

| channel | corpus AUC without it | z | per-object AUC without it | corpus AUC with it alone | z | per-object AUC alone |
|---|---|---|---|---|---|---|
| gal_baryons | 1.000 | 8.50 | 0.980 | 0.500 | 0.00 | 0.500 |
| gal_curve_raw | 1.000 | 8.50 | 0.987 | 1.000 | 8.50 | 0.797 |
| gal_curve_res | 1.000 | 8.50 | 0.990 | 0.994 | 8.50 | 0.744 |
| gal_harmonic_power | 1.000 | 8.50 | 0.991 | 0.953 | 8.50 | 0.588 |
| gal_m3_phase | 1.000 | 8.50 | 0.991 | 0.601 | 5.31 | 0.521 |
| gal_vertical | 1.000 | 8.50 | 0.923 | 1.000 | 8.50 | 0.938 |
| clu_baryons | 1.000 | 8.50 | 1.000 | 0.500 | 0.00 | 0.500 |
| clu_dynamics | 1.000 | 8.50 | 1.000 | 1.000 | 8.50 | 0.917 |
| clu_gas | 1.000 | 8.50 | 0.996 | 1.000 | 8.50 | 0.997 |
| clu_matter_photon | 1.000 | 8.50 | 1.000 | 0.994 | 8.50 | 0.910 |
| clu_network | 1.000 | 8.50 | 1.000 | 0.614 | 5.98 | 0.534 |
| clu_shear_mono_raw | 1.000 | 8.50 | 1.000 | 1.000 | 8.50 | 0.951 |
| clu_shear_mono_res | 1.000 | 8.50 | 1.000 | 1.000 | 8.50 | 0.915 |
| clu_shear_quad_phase | 1.000 | 8.50 | 1.000 | 1.000 | 8.50 | 0.911 |
| clu_shear_quad_power | 1.000 | 8.50 | 1.000 | 1.000 | 8.50 | 0.902 |
| clu_strong_lens | 1.000 | 8.50 | 1.000 | 0.941 | 8.50 | 0.661 |
| all quadrupole/harmonic content | 1.000 | 8.50 | 1.000 | n/a | n/a | n/a |
| shear monopole | 1.000 | 8.50 | 1.000 | n/a | n/a | n/a |
| internal galaxy dynamics | 1.000 | 8.50 | 1.000 | n/a | n/a | n/a |
| strong-lens timing | 1.000 | 8.50 | 1.000 | n/a | n/a | n/a |

The corpus AUC saturates at 30 galaxies and 12 clusters per corpus; the per-object AUC (one galaxy, one cluster, U2 vs U3) is the number that ranks the channels.

The named removals of the brief:

| ablation | corpus AUC | z | p (U3) | per-galaxy AUC | per-cluster AUC |
|---|---|---|---|---|---|
| drop:shear angular phase (all phases) | 1.000 | 8.50 | 0.0003 | 0.991 | 1.000 |
| drop:all quadrupole/harmonic content | 1.000 | 8.50 | 0.0003 | 0.991 | 1.000 |
| drop:shear monopole | 1.000 | 8.50 | 0.0003 | 0.991 | 1.000 |
| drop:member identities (network) | 1.000 | 8.50 | 0.0003 | 0.991 | 1.000 |
| drop:gas (X-ray, SZ) | 1.000 | 8.50 | 0.0003 | 0.991 | 0.996 |
| drop:internal galaxy dynamics | 1.000 | 8.50 | 0.0003 | 0.500 | 1.000 |
| drop:strong-lens timing | 1.000 | 8.50 | 0.0003 | 0.991 | 1.000 |
| drop:vertical AND scatter (curve residuals) | 1.000 | 8.50 | 0.0003 | 0.819 | 1.000 |
| randomise:baryon-field alignment | 1.000 | 8.50 | 0.0003 | 0.991 | 1.000 |
| only:non-directional | 1.000 | 8.50 | 0.0003 | 0.991 | 1.000 |
| only:directional | 1.000 | 8.50 | 0.0003 | 0.521 | 0.917 |
| matched:mean profiles removed per arm | 1.000 | 8.50 | 0.0003 | n/a | n/a |
| matched:mean profiles removed AND no vertical | 1.000 | 8.50 | 0.0003 | n/a | n/a |

Permutation importance by channel (drop in audit AUC when the channel's columns are shuffled; base AUC 1.000): `gal:gal_baryons` 0.000, `gal:gal_curve_raw` 0.000, `gal:gal_curve_res` 0.000, `gal:gal_harmonic_power` 0.000, `gal:gal_m3_phase` 0.000, `gal:gal_vertical` 0.000, `clu:clu_baryons` 0.000, `clu:clu_dynamics` 0.000.

## 4  Localisation: radius, harmonic, source class, regime

| restricted to | n features | AUC | z |
|---|---|---|---|
| galaxy inner rings (0.5-2.5 Rd) | 13 | 1.000 | 8.50 |
| galaxy outer rings (2.5-5.2 Rd) | 13 | 1.000 | 8.50 |
| cluster inner bins (0.12-0.4 R500) | 12 | 1.000 | 8.50 |
| cluster mid bins (0.4-0.9 R500) | 12 | 1.000 | 8.50 |
| cluster outer bins (0.9-2.3 R500) | 12 | 1.000 | 8.50 |
| cluster X-ray inner (<0.4 R500) | 12 | 1.000 | 8.50 |
| cluster X-ray outer (>0.4 R500) | 8 | 1.000 | 8.50 |
| m=0 only (monopole profiles, both classes) | 38 | 1.000 | 8.50 |
| m=2 only (quadrupole power + phase) | 34 | 1.000 | 8.50 |
| m=3 only (galaxy) | 12 | 0.893 | 8.50 |
| vertical only (galaxy dz, rz) | 6 | 1.000 | 8.50 |
| vertical contrast only (dz_1) | 1 | 1.000 | 8.50 |
| galaxy residual scatter only (res_sd, res_slope, res_out_in) | 3 | 0.883 | 8.50 |
| cluster profile shape only (wl_slope, t_slope, h_slope, d_slope) | 5 | 1.000 | 8.50 |
| cluster levels only (wl_lA, t_lA, h_lA, d_lA) | 5 | 1.000 | 8.50 |
| matter-photon covariance only (ep_*) | 3 | 0.994 | 8.50 |
| strong lensing only | 8 | 0.941 | 8.50 |

Per-object discrimination of the full discriminator in terciles of an observed property (AUC of object log-odds, U2 vs U3):

| property | low tercile | mid | high |
|---|---|---|---|
| galaxy outer acceleration log g_bar [SI] | 0.998 (n=9850) | 0.995 (n=9850) | 0.982 (n=9850) |
| galaxy environment log S_ext | 0.996 (n=9850) | 0.993 (n=9850) | 0.989 (n=9850) |
| galaxy stellar mass | 0.997 (n=9850) | 0.993 (n=9850) | 0.987 (n=9850) |
| galaxy inclination | 0.991 (n=9850) | 0.994 (n=9850) | 0.993 (n=9850) |
| cluster baryon ellipticity | 1.000 (n=3952) | 1.000 (n=3930) | 1.000 (n=3982) |
| cluster gas mass | 1.000 (n=3954) | 1.000 (n=3954) | 1.000 (n=3956) |
| cluster centroid shift (disturbance) | 1.000 (n=3448) | 1.000 (n=3818) | 1.000 (n=4598) |
| cluster redshift | 1.000 (n=3906) | 1.000 (n=3948) | 1.000 (n=4010) |

## 5  Causal counterfactuals: dO/dB, paired scenes, one thing changed

Per-object differences between the edited and the base emission (same scene, same noise; the halo HELD where stated), mean +- standard error across objects.  Invariants: `dz_1` the vertical/radial boost contrast at 2 R_d; `res_mean` the galaxy residual from the universal law; `res_sd` its within-galaxy scatter; `wl_lA`, `t_lA`, `d_lA` the lensing, X-ray and dynamical residual levels; `t_slope` the X-ray residual slope; `pe_tot`/`pb_tot` the quadrupole projections on the external/baryon axis; `S` the discriminator's corpus score.

| universe | edit | d dz_1 | d res_mean | d wl_lA | d t_lA | d t_slope | d pe_tot | d pb_tot | d S | d S, U2 minus U3 |
|---|---|---|---|---|---|---|---|---|---|---|
| U2 | all baryonic masses x 10^0.10, halo HELD | +0.0385 +- 0.0001 | +0.0057 +- 0.0001 | -0.002 | -0.004 | +0.008 | +0.03 | +0.03 | -26.1 +- 0.3 | -40.2 |
| U3 | all baryonic masses x 10^0.10, halo HELD | -0.0001 +- 0.0000 | -0.0002 +- 0.0001 | +0.029 | -0.003 | -0.004 | -0.00 | +0.00 | +14.1 +- 0.4 |  |
| U2 | baryonic scale lengths x 10^0.06, halo HELD | -0.0392 +- 0.0001 | -0.0052 +- 0.0001 | -0.021 | -0.043 | -0.039 | -0.00 | -0.00 | +13.9 +- 0.3 | +22.5 |
| U3 | baryonic scale lengths x 10^0.06, halo HELD | -0.0000 +- 0.0000 | +0.0003 +- 0.0001 | +0.007 | -0.025 | -0.036 | +0.00 | -0.00 | -8.6 +- 0.2 |  |
| U2 | disc scale height x 10^0.10, halo HELD | +0.0176 +- 0.0001 | +0.0000 +- 0.0000 | +0.000 | +0.000 | +0.000 | +0.00 | +0.00 | -50.3 +- 0.3 | -27.7 |
| U3 | disc scale height x 10^0.10, halo HELD | +0.0000 +- 0.0000 | +0.0000 +- 0.0000 | +0.000 | +0.000 | +0.000 | +0.00 | +0.00 | -22.6 +- 0.2 |  |
| U2 | halo mass x 10^0.10, baryons HELD | -0.0185 +- 0.0001 | -0.0024 +- 0.0001 | +0.032 | +0.015 | +0.009 | +0.75 | +0.74 | +18.5 +- 0.2 |  |
| U2 | halo concentration x 10^0.10, baryons HELD | -0.0445 +- 0.0001 | -0.0045 +- 0.0002 | -0.030 | -0.053 | -0.047 | +0.18 | +0.18 | +42.6 +- 0.4 |  |
| U2 | halo ellipticity + 0.10, baryons HELD | +0.0000 +- 0.0000 | +0.0000 +- 0.0000 | +0.027 | +0.000 | +0.000 | +1.83 | +1.77 | +0.9 +- 0.1 |  |
| U2 | halo axis rotated 45 deg, baryons HELD | +0.0006 +- 0.0000 | -0.0000 +- 0.0000 | +0.002 | +0.002 | +0.001 | -3.35 | -5.74 | -0.5 +- 0.1 |  |
| U2 | external tidal axis rotated 45 deg, local source HELD (halo held) | +0.0000 +- 0.0000 | +0.0000 +- 0.0000 | +0.000 | +0.000 | +0.000 | -5.83 | +0.00 | -0.0 +- 0.0 | -0.0 |
| U3 | external tidal axis rotated 45 deg, local source HELD (halo held) | +0.0000 +- 0.0000 | +0.0000 +- 0.0000 | +0.000 | +0.000 | +0.000 | +0.01 | +0.00 | +0.0 +- 0.0 |  |
| U5f | external tidal axis rotated 45 deg, local source HELD (halo held) | -0.0069 +- 0.0006 | -0.0043 +- 0.0007 | +0.007 | +0.022 | +0.033 | +0.01 | +0.28 | +10.0 +- 0.7 |  |
| U2 | external axis rotated 45 deg, halo REDRAWN under the f_lss prior | -0.0000 +- 0.0000 | -0.0000 +- 0.0000 | +0.001 | -0.000 | +0.000 | +2.02 | +2.42 | +0.1 +- 0.1 | +0.1 |
| U3 | external axis rotated 45 deg, halo REDRAWN under the f_lss prior | +0.0000 +- 0.0000 | +0.0000 +- 0.0000 | +0.000 | +0.000 | +0.000 | +0.01 | +0.00 | +0.0 +- 0.0 |  |
| U5f | external axis rotated 45 deg, halo REDRAWN under the f_lss prior | -0.0069 +- 0.0006 | -0.0043 +- 0.0007 | +0.007 | +0.022 | +0.033 | +0.01 | +0.28 | +10.0 +- 0.7 |  |
| U2 | member positions scrambled, every radial profile preserved | +0.0000 +- 0.0000 | +0.0000 +- 0.0000 | +0.000 | +0.000 | +0.000 | +0.00 | +0.00 | -0.1 +- 0.0 | -0.1 |
| U3 | member positions scrambled, every radial profile preserved | +0.0000 +- 0.0000 | +0.0000 +- 0.0000 | +0.000 | +0.000 | +0.000 | +0.00 | +0.00 | -0.0 +- 0.0 |  |
| U6f | member positions scrambled, every radial profile preserved | +0.0000 +- 0.0000 | +0.0000 +- 0.0000 | -0.000 | +0.000 | +0.000 | +0.00 | -0.01 | -0.0 +- 0.0 |  |
| U2 | formation history: t_merge x 3, present density field preserved | +0.0000 +- 0.0000 | +0.0000 +- 0.0000 | +0.000 | +0.000 | +0.000 | +0.00 | +0.00 | -0.4 +- 0.0 | -0.2 |
| U3 | formation history: t_merge x 3, present density field preserved | +0.0000 +- 0.0000 | +0.0000 +- 0.0000 | +0.000 | +0.000 | +0.000 | +0.00 | +0.00 | -0.2 +- 0.0 |  |
| U7f | formation history: t_merge x 3, present density field preserved | +0.0000 +- 0.0000 | +0.0005 +- 0.0001 | -0.011 | +0.000 | +0.001 | +0.00 | -0.00 | -8.2 +- 0.2 |  |
| U2 | photon path: void fraction -> 1 - void, endpoints preserved | +0.0000 +- 0.0000 | +0.0000 +- 0.0000 | +0.000 | +0.000 | +0.000 | +0.00 | +0.00 | +0.2 +- 0.0 | -0.0 |
| U3 | photon path: void fraction -> 1 - void, endpoints preserved | +0.0000 +- 0.0000 | +0.0000 +- 0.0000 | +0.000 | +0.000 | +0.000 | +0.00 | +0.00 | +0.2 +- 0.0 |  |
| U9f | photon path: void fraction -> 1 - void, endpoints preserved | +0.0000 +- 0.0000 | +0.0000 +- 0.0000 | +0.000 | +0.000 | +0.000 | +0.00 | +0.00 | +0.2 +- 0.0 |  |
| U2 | baryonic masses x 10^0.10, halo REDRAWN (follows the SHMR) | +0.0257 +- 0.0001 | -0.0091 +- 0.0002 | +0.039 | +0.026 | +0.033 | +0.78 | +0.77 | -2.6 +- 0.4 | -16.7 |
| U3 | baryonic masses x 10^0.10, halo REDRAWN (follows the SHMR) | -0.0001 +- 0.0000 | -0.0002 +- 0.0001 | +0.029 | -0.003 | -0.004 | -0.00 | +0.00 | +14.1 +- 0.4 |  |

Two things to read the table with.  The discriminator score S is CONDITIONAL on the baryonic scene (it uses the photometry as context), so it moves under a baryon edit in every universe; the CDM-specific response is the last column.  And an edit applied to every object of a corpus is partly re-absorbed by the corpus's own refitted universal law -- that is the monopole matching doing its job -- so the paired responses of the RESIDUALS to a global halo edit understate the object-to-object response, which the partial slopes below measure.

Object-level partial slopes inside the CDM prior (the invariant against the halo the object happens to have, at fixed observed baryons -- the response a real population would show):

| slope | value | se | n |
|---|---|---|---|
| gal:dz_1 vs log M200 at fixed baryons | -0.031 | 0.006 | 8848 |
| gal:dz_1 vs log c at fixed baryons | -0.410 | 0.013 | 8848 |
| gal:rz_1 vs log M200 at fixed baryons | +0.253 | 0.006 | 8848 |
| gal:rz_1 vs log c at fixed baryons | -0.029 | 0.017 | 8848 |
| gal:res_mean vs log M200 at fixed baryons | +0.295 | 0.005 | 8848 |
| gal:res_mean vs log c at fixed baryons | +0.365 | 0.013 | 8848 |
| gal:res_out_in vs log M200 at fixed baryons | +0.124 | 0.006 | 8848 |
| gal:res_out_in vs log c at fixed baryons | -0.114 | 0.014 | 8848 |
| clu:wl_lA vs log M200 at fixed baryons | +0.749 | 0.089 | 3571 |
| clu:wl_lA vs log c at fixed baryons | +0.308 | 0.048 | 3571 |
| clu:t_lA vs log M200 at fixed baryons | +0.648 | 0.024 | 3571 |
| clu:t_lA vs log c at fixed baryons | -0.012 | 0.014 | 3571 |
| clu:h_lA vs log M200 at fixed baryons | +0.579 | 0.024 | 3571 |
| clu:h_lA vs log c at fixed baryons | +0.084 | 0.014 | 3571 |
| clu:d_lA vs log M200 at fixed baryons | +0.298 | 0.014 | 3571 |
| clu:d_lA vs log c at fixed baryons | +0.069 | 0.008 | 3571 |
| clu:wl_slope vs log M200 at fixed baryons | -4.221 | 0.546 | 3571 |
| clu:wl_slope vs log c at fixed baryons | -4.039 | 0.286 | 3571 |
| clu:t_slope vs log M200 at fixed baryons | +0.146 | 0.030 | 3571 |
| clu:t_slope vs log c at fixed baryons | -0.592 | 0.012 | 3571 |

## 6  Distillation: from the discriminator to named invariants

The discriminator's separation: AUC 1.000, z 8.50.  Single corpus-level terms, ranked by audit AUC (U2 mean, class U3 mean +- corpus sd):

| term | type | AUC | z | d' | U2 | U3 | sd(U3) |
|---|---|---|---|---|---|---|---|
| `mean:dz_1` | structure | 1.000 | 8.50 | 9.4 | -0.268 | +0.005 | 0.026 |
| `mean:rz_1` | level | 1.000 | 8.50 | 8.8 | -0.336 | -0.005 | 0.029 |
| `mean:dz_0` | structure | 1.000 | 8.50 | 8.5 | -0.239 | +0.010 | 0.025 |
| `mean:rz_0` | level | 1.000 | 8.50 | 7.9 | -0.294 | -0.007 | 0.029 |
| `mean:q2_in` | structure | 1.000 | 8.50 | 5.6 | +4.910 | +0.078 | 0.216 |
| `mean:q2_grad` | structure | 1.000 | 8.50 | 5.6 | -4.529 | -0.229 | 0.242 |
| `mean:wres_2` | level | 1.000 | 8.50 | 3.7 | +5.762 | -0.044 | 0.504 |
| `mean:t_slope` | structure | 1.000 | 8.50 | 2.4 | -0.383 | -0.043 | 0.048 |
| `mean:vz_1` | level | 0.999 | 8.50 | 6.0 | +0.109 | +0.350 | 0.037 |
| `mean:hres_2` | level | 0.998 | 8.50 | 3.1 | +0.331 | +0.029 | 0.031 |
| `corpus:rar_scatter_oof` | level | 0.988 | 8.50 | 2.9 | +0.175 | +0.108 | 0.016 |
| `mean:h_slope` | structure | 0.986 | 8.50 | 2.6 | -0.264 | +0.039 | 0.085 |

`level` marks a term that carries the strength of gravity at some radius (what an extra cluster-scale component in a modified-gravity theory would remove); `structure` marks scatter, shape, geometry or the vertical contrast.

Greedy forward selection into a linear score (fitted on calibration corpora, scored on audit; ties in a saturated AUC broken by d'):

| terms | AUC | z | d' |
|---|---|---|---|
| `mean:dz_1` | 1.000 | 8.50 | 9.4 |
| `mean:dz_1`, `mean:q2_in` | 1.000 | 8.50 | 10.1 |
| `mean:dz_1`, `mean:q2_in`, `mean:dz_0` | 1.000 | 8.50 | 11.0 |

The same selection with every level term excluded (the strength-free distillation):

| terms | AUC | z | d' |
|---|---|---|---|
| `mean:dz_1` | 1.000 | 8.50 | 9.4 |
| `mean:dz_1`, `mean:q2_in` | 1.000 | 8.50 | 10.1 |
| `mean:dz_1`, `mean:q2_in`, `mean:dz_0` | 1.000 | 8.50 | 11.0 |

**Distilled invariant:** `mean:dz_1`, `mean:q2_in`, `mean:dz_0` -- AUC 1.000, z 8.50, recovering **100%** of the discriminator's AUC excess over 0.5.  Frozen, per class member: U3 1.000, H0 1.000, U4t 1.000, U5t 1.000, U6t 1.000, U7t 1.000, U8t 1.000, U9t 1.000.  On the reference arms: U5f 1.000, U8f 1.000, U10 1.000, U1 1.000.

The named physical invariants, each fitted alone and then FROZEN and transferred to the untouched scene library and to generator 2 (a frozen score can only lose separation):

| invariant | terms | AUC (G1 audit) | z | AUC untouched scenes (frozen) | AUC generator 2 vs class (frozen) | AUC generator 2 vs tensor (frozen) |
|---|---|---|---|---|---|---|
| boost_isotropy | `mean:dz_1` | 1.000 | 8.50 | 1.000 | 1.000 | 1.000 |
| closure_scatter | `sd:res_mean` | 0.976 | 8.50 | 0.962 | 0.992 | 0.989 |
| cluster_profile_freedom | `mean:t_slope`, `sd:t_lA` | 1.000 | 8.50 | 1.000 | 0.946 | 0.943 |
| cluster_level_excess | `mean:t_lA` | 0.652 | 8.00 | 0.738 | 0.024 | 0.027 |
| directional_phase | `mean:pb_tot`, `mean:pe_tot` | 0.978 | 8.50 | 0.990 | 0.844 | 0.582 |
| distilled (unrestricted) | `mean:dz_1`, `mean:q2_in`, `mean:dz_0` | 1.000 | 8.50 | 1.000 | 1.000 | 1.000 |
| distilled (strength-free) | `mean:dz_1`, `mean:q2_in`, `mean:dz_0` | 1.000 | 8.50 | 1.000 | 1.000 | 1.000 |

Values of the named terms per arm (corpus means), generator 1 and generator 2:

| arm | `mean:dz_1` | `sd:res_mean` | `corpus:rar_scatter_oof` | `mean:t_slope` | `mean:t_lA` | `mean:wl_lA` | `mean:pb_tot` | `mean:pe_tot` |
|---|---|---|---|---|---|---|---|---|
| G1 U2 | -0.268 | +0.156 | +0.175 | -0.383 | -0.093 | +0.062 | +3.747 | +3.910 |
| G1 U3 | +0.005 | +0.097 | +0.108 | -0.043 | -0.042 | -0.170 | -0.002 | -0.003 |
| G1 H0 | +0.006 | +0.102 | +0.116 | +0.151 | +0.465 | -0.678 | -0.003 | -0.008 |
| G1 U5f | +0.011 | +0.106 | +0.123 | -0.053 | -0.043 | -0.223 | -0.314 | +0.720 |
| G1 U8f | +0.005 | +0.097 | +0.108 | -0.043 | -0.042 | -0.090 | -0.002 | -0.003 |
| G1 U10 | +0.067 | +0.341 | +0.383 | -0.286 | -0.487 | -0.703 | +6.876 | -2.912 |
| G1 U1 | +0.009 | +0.130 | +0.154 | -0.177 | -0.155 | -0.626 | -0.002 | -0.003 |
| G2 C | -0.263 | +0.094 | +0.108 | -0.156 | -0.094 | +0.006 | -0.024 | -0.002 |
| G2 D | -0.469 | +0.175 | +0.195 | -0.282 | +0.149 | +0.319 | +0.586 | +0.621 |
| G2 T | -0.257 | +0.098 | +0.114 | -0.155 | -0.091 | +0.008 | -0.010 | +0.903 |
| G2 N | -0.255 | +0.126 | +0.152 | -0.204 | -0.101 | -0.376 | -0.024 | -0.002 |

## 7  Stage 4 certificates

3 issued, 33 refused, seven checks each, typed identifiers.  C2 uses a MEASURED control (the largest effect of the same statistic on anything that is not a halo: the class at fiducial amplitude, systematics-only, Newton, and the instrument-nuisance arms); C3 is the realised false-positive rate on the untouched audit half; C6 is the fraction of the generator-1 effect recovered on generator 2; C7 is the response pattern across the 20-statistic set.

| candidate | effect (U2 - U3) | class corpus sd | verdict | C2 control/target | C4 sigma at predicted | C6 G2 recovery | C7 worst |r| |
|---|---|---|---|---|---|---|---|
| `CAND.CDM.BOOST_ISOTROPY.AT_PRIOR_FDD_0-0.05` | -0.274 | 0.026 | **ISSUED** | 0.22 | 10.5 | 0.75 | 0.53 |
| `CAND.CDM.BOOST_ISOTROPY.AT_FDD_0.1` | -0.215 | 0.026 | **ISSUED** | 0.28 | 8.3 | 0.96 | 0.53 |
| `CAND.CDM.BOOST_ISOTROPY.AT_FDD_0.3` | -0.086 | 0.026 | **ISSUED** | 0.71 | 3.3 | 2.39 | 0.53 |
| `CAND.CDM.BOOST_ISOTROPY.AT_FDD_0.5` | +0.011 | 0.026 | refused: C2, C4, C6 | 5.53 | 0.4 | -18.61 | 0.53 |
| `CAND.CDM.BOOST_ISOTROPY.AT_FDD_1.0` | +0.186 | 0.026 | refused: C6 | 0.33 | 7.1 | -1.11 | 0.53 |
| `CAND.CDM.CLOSURE_SCATTER.AT_PRIOR_SHMR_0.16` | +0.060 | 0.018 | refused: C2, C4 | 4.09 | 0.5 | 1.34 | 0.53 |
| `CAND.CDM.CLOSURE_SCATTER.AT_SHMR_0.08` | +0.056 | 0.018 | refused: C2, C4 | 4.33 | 0.3 | 1.42 | 0.53 |
| `CAND.CDM.CLOSURE_SCATTER.AT_SHMR_0` | +0.055 | 0.018 | refused: C2, C4 | 4.43 | 0.0 | 1.46 | 0.53 |
| `CAND.CDM.CLOSURE_SCATTER.AT_SHMR_0.32` | +0.074 | 0.018 | refused: C2, C4 | 3.30 | 1.0 | 1.08 | 0.53 |
| `CAND.CDM.CLOSURE_SCATTER.AT_QAMP_0` | +0.061 | 0.018 | refused: C1, C2, C4 | 4.03 | 0.0 | 1.33 | 0.53 |
| `CAND.CDM.CLOSURE_SCATTER.AT_QAMP_0.05` | +0.060 | 0.018 | refused: C1, C2, C4 | 4.06 | 0.0 | 1.33 | 0.53 |
| `CAND.CDM.CLOSURE_SCATTER.AT_QAMP_0.2` | +0.060 | 0.018 | refused: C1, C2, C4 | 4.06 | 0.0 | 1.33 | 0.53 |
| `CAND.CDM.CLUSTER_PROFILE_SLOPE.AT_PRIOR` | -0.340 | 0.192 | refused: C4, C6 | 0.71 | 1.8 | 0.37 | 0.53 |
| `CAND.CDM.CLUSTER_PROFILE_SLOPE.AT_CONC_SCATTER_0` | -0.345 | 0.192 | refused: C4, C6 | 0.70 | 1.8 | 0.36 | 0.53 |
| `CAND.CDM.CLUSTER_PROFILE_SLOPE.AT_FLSS_0` | -0.339 | 0.192 | refused: C4, C6 | 0.72 | 1.8 | 0.37 | 0.53 |
| `CAND.CDM.CLUSTER_PROFILE_SLOPE.AT_FLSS_1` | -0.339 | 0.192 | refused: C4, C6 | 0.72 | 1.8 | 0.37 | 0.53 |
| `CAND.CDM.CLUSTER_LEVEL_SCATTER.AT_PRIOR` | +0.004 | 0.018 | refused: C2, C4 | 16.82 | 0.2 | 5.49 | 0.53 |
| `CAND.CDM.CLUSTER_LEVEL_SCATTER.AT_CONC_SCATTER_0` | +0.002 | 0.018 | refused: C2, C4 | 28.98 | 0.1 | 9.46 | 0.53 |
| `CAND.CDM.CLUSTER_LEVEL_SCATTER.AT_FLSS_0` | +0.002 | 0.018 | refused: C2, C4 | 27.75 | 0.1 | 9.06 | 0.53 |
| `CAND.CDM.CLUSTER_LEVEL_SCATTER.AT_FLSS_1` | +0.002 | 0.018 | refused: C2, C4 | 27.84 | 0.1 | 9.08 | 0.53 |
| `CAND.CDM.CLUSTER_LEVEL_EXCESS.AT_PRIOR` | -0.051 | 0.493 | refused: C2, C4, C6 | 8.67 | 0.1 | -4.74 | 0.53 |
| `CAND.CDM.CLUSTER_LEVEL_EXCESS.AT_CONC_SCATTER_0` | -0.046 | 0.493 | refused: C2, C4, C6 | 9.74 | 0.1 | -5.33 | 0.53 |
| `CAND.CDM.CLUSTER_LEVEL_EXCESS.AT_FLSS_0` | -0.050 | 0.493 | refused: C2, C4, C6 | 8.83 | 0.1 | -4.83 | 0.53 |
| `CAND.CDM.CLUSTER_LEVEL_EXCESS.AT_FLSS_1` | -0.048 | 0.493 | refused: C2, C4, C6 | 9.24 | 0.1 | -5.05 | 0.53 |
| `CAND.CDM.LENSING_LEVEL_EXCESS.AT_PRIOR` | +0.232 | 0.422 | refused: C2, C4 | 2.30 | 0.6 | 1.35 | 0.53 |
| `CAND.CDM.LENSING_LEVEL_EXCESS.AT_CONC_SCATTER_0` | +0.251 | 0.422 | refused: C2, C4 | 2.12 | 0.6 | 1.25 | 0.53 |
| `CAND.CDM.LENSING_LEVEL_EXCESS.AT_FLSS_0` | +0.242 | 0.422 | refused: C2, C4 | 2.20 | 0.6 | 1.29 | 0.53 |
| `CAND.CDM.LENSING_LEVEL_EXCESS.AT_FLSS_1` | +0.245 | 0.422 | refused: C2, C4 | 2.18 | 0.6 | 1.28 | 0.53 |
| `CAND.CDM.BARYON_AXIS_QUAD.AT_PRIOR` | +3.749 | 0.522 | refused: C2, C3, C6 | 1.83 | 7.2 | 0.16 | 0.53 |
| `CAND.CDM.BARYON_AXIS_QUAD.AT_CONC_SCATTER_0` | +4.860 | 0.522 | refused: C2, C3, C6 | 1.42 | 9.3 | 0.13 | 0.53 |
| `CAND.CDM.BARYON_AXIS_QUAD.AT_FLSS_0` | +10.474 | 0.522 | refused: C3, C6 | 0.66 | 20.1 | 0.06 | 0.53 |
| `CAND.CDM.BARYON_AXIS_QUAD.AT_FLSS_1` | -4.278 | 0.522 | refused: C2, C6 | 1.61 | 8.2 | -0.14 | 0.53 |
| `CAND.CDM.EXTERNAL_AXIS_QUAD.AT_PRIOR` | +3.912 | 0.535 | refused: C6 | 0.74 | 7.3 | 0.16 | 0.53 |
| `CAND.CDM.EXTERNAL_AXIS_QUAD.AT_CONC_SCATTER_0` | +4.247 | 0.535 | refused: C6 | 0.69 | 7.9 | 0.15 | 0.53 |
| `CAND.CDM.EXTERNAL_AXIS_QUAD.AT_FLSS_0` | -4.304 | 0.535 | refused: C6 | 0.68 | 8.0 | -0.14 | 0.53 |
| `CAND.CDM.EXTERNAL_AXIS_QUAD.AT_FLSS_1` | +10.132 | 0.535 | refused: C6 | 0.29 | 18.9 | 0.06 | 0.53 |

* `CAND.CDM.BOOST_ISOTROPY.AT_FDD_0.5`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 5.528x the effect of a collisionless halo (0.0612 against 0.0111); C4_powered: the theory predicts 0.0104251; through a pipeline of responsiveness 0.999 that is 0.40 sigma; C6_out_of_grammar: recovers -1861% of an out-of-grammar injection
* `CAND.CDM.BOOST_ISOTROPY.AT_FDD_1.0`: C6_out_of_grammar: recovers -111% of an out-of-grammar injection
* `CAND.CDM.CLOSURE_SCATTER.AT_PRIOR_SHMR_0.16`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 4.085x the effect of halo scatter (0.2445 against 0.0599); C4_powered: the theory predicts 0.16; through a pipeline of responsiveness 0.060 that is 0.52 sigma
* `CAND.CDM.CLOSURE_SCATTER.AT_SHMR_0.08`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 4.332x the effect of halo scatter (0.2445 against 0.0565); C4_powered: the theory predicts 0.08; through a pipeline of responsiveness 0.060 that is 0.26 sigma
* `CAND.CDM.CLOSURE_SCATTER.AT_SHMR_0`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 4.431x the effect of halo scatter (0.2445 against 0.0552); C4_powered: the theory predicts 0; through a pipeline of responsiveness 0.060 that is 0.00 sigma
* `CAND.CDM.CLOSURE_SCATTER.AT_SHMR_0.32`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 3.299x the effect of halo scatter (0.2445 against 0.0741); C4_powered: the theory predicts 0.32; through a pipeline of responsiveness 0.060 that is 1.04 sigma
* `CAND.CDM.CLOSURE_SCATTER.AT_QAMP_0`: C1_responsive: statistic moves 4.262e-04 over the effect range; C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 4.033x the effect of the halo quadrupole (0.2445 against 0.0606); C4_powered: the theory predicts 1e-09; through a pipeline of responsiveness 0.001 that is 0.00 sigma
* `CAND.CDM.CLOSURE_SCATTER.AT_QAMP_0.05`: C1_responsive: statistic moves 4.262e-04 over the effect range; C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 4.062x the effect of the halo quadrupole (0.2445 against 0.0602); C4_powered: the theory predicts 0.05; through a pipeline of responsiveness 0.001 that is 0.00 sigma
* `CAND.CDM.CLOSURE_SCATTER.AT_QAMP_0.2`: C1_responsive: statistic moves 4.262e-04 over the effect range; C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 4.057x the effect of the halo quadrupole (0.2445 against 0.0603); C4_powered: the theory predicts 0.2; through a pipeline of responsiveness 0.001 that is 0.01 sigma
* `CAND.CDM.CLUSTER_PROFILE_SLOPE.AT_PRIOR`: C4_powered: the theory predicts 0.340205; through a pipeline of responsiveness 1.000 that is 1.77 sigma; C6_out_of_grammar: recovers 37% of an out-of-grammar injection
* `CAND.CDM.CLUSTER_PROFILE_SLOPE.AT_CONC_SCATTER_0`: C4_powered: the theory predicts 0.344786; through a pipeline of responsiveness 1.000 that is 1.80 sigma; C6_out_of_grammar: recovers 36% of an out-of-grammar injection
* `CAND.CDM.CLUSTER_PROFILE_SLOPE.AT_FLSS_0`: C4_powered: the theory predicts 0.33914; through a pipeline of responsiveness 1.000 that is 1.77 sigma; C6_out_of_grammar: recovers 37% of an out-of-grammar injection
* `CAND.CDM.CLUSTER_PROFILE_SLOPE.AT_FLSS_1`: C4_powered: the theory predicts 0.338936; through a pipeline of responsiveness 1.000 that is 1.77 sigma; C6_out_of_grammar: recovers 37% of an out-of-grammar injection
* `CAND.CDM.CLUSTER_LEVEL_SCATTER.AT_PRIOR`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 16.823x the effect of a collisionless halo (0.0597 against 0.0035); C4_powered: the theory predicts 0.00354843; through a pipeline of responsiveness 1.000 that is 0.20 sigma
* `CAND.CDM.CLUSTER_LEVEL_SCATTER.AT_CONC_SCATTER_0`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 28.982x the effect of a collisionless halo (0.0597 against 0.0021); C4_powered: the theory predicts 0.00205978; through a pipeline of responsiveness 1.000 that is 0.12 sigma
* `CAND.CDM.CLUSTER_LEVEL_SCATTER.AT_FLSS_0`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 27.752x the effect of a collisionless halo (0.0597 against 0.0022); C4_powered: the theory predicts 0.00215103; through a pipeline of responsiveness 1.000 that is 0.12 sigma
* `CAND.CDM.CLUSTER_LEVEL_SCATTER.AT_FLSS_1`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 27.835x the effect of a collisionless halo (0.0597 against 0.0021); C4_powered: the theory predicts 0.00214461; through a pipeline of responsiveness 1.000 that is 0.12 sigma
* `CAND.CDM.CLUSTER_LEVEL_EXCESS.AT_PRIOR`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 8.672x the effect of a collisionless halo (0.4443 against 0.0512); C4_powered: the theory predicts 0.0512333; through a pipeline of responsiveness 1.000 that is 0.10 sigma; C6_out_of_grammar: recovers -474% of an out-of-grammar injection
* `CAND.CDM.CLUSTER_LEVEL_EXCESS.AT_CONC_SCATTER_0`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 9.743x the effect of a collisionless halo (0.4443 against 0.0456); C4_powered: the theory predicts 0.0456023; through a pipeline of responsiveness 1.000 that is 0.09 sigma; C6_out_of_grammar: recovers -533% of an out-of-grammar injection
* `CAND.CDM.CLUSTER_LEVEL_EXCESS.AT_FLSS_0`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 8.831x the effect of a collisionless halo (0.4443 against 0.0503); C4_powered: the theory predicts 0.0503088; through a pipeline of responsiveness 1.000 that is 0.10 sigma; C6_out_of_grammar: recovers -483% of an out-of-grammar injection
* `CAND.CDM.CLUSTER_LEVEL_EXCESS.AT_FLSS_1`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 9.238x the effect of a collisionless halo (0.4443 against 0.0481); C4_powered: the theory predicts 0.0480943; through a pipeline of responsiveness 1.000 that is 0.10 sigma; C6_out_of_grammar: recovers -505% of an out-of-grammar injection
* `CAND.CDM.LENSING_LEVEL_EXCESS.AT_PRIOR`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 2.296x the effect of a collisionless halo (0.5329 against 0.2321); C4_powered: the theory predicts 0.232114; through a pipeline of responsiveness 1.000 that is 0.55 sigma
* `CAND.CDM.LENSING_LEVEL_EXCESS.AT_CONC_SCATTER_0`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 2.124x the effect of a collisionless halo (0.5329 against 0.2509); C4_powered: the theory predicts 0.250851; through a pipeline of responsiveness 1.000 that is 0.59 sigma
* `CAND.CDM.LENSING_LEVEL_EXCESS.AT_FLSS_0`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 2.199x the effect of a collisionless halo (0.5329 against 0.2423); C4_powered: the theory predicts 0.242296; through a pipeline of responsiveness 1.000 that is 0.57 sigma
* `CAND.CDM.LENSING_LEVEL_EXCESS.AT_FLSS_1`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 2.176x the effect of a collisionless halo (0.5329 against 0.2449); C4_powered: the theory predicts 0.244943; through a pipeline of responsiveness 1.000 that is 0.58 sigma
* `CAND.CDM.BARYON_AXIS_QUAD.AT_PRIOR`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 1.835x the effect of a collisionless halo (6.8773 against 3.7488); C3_exchangeable: untouched audit half: realised 0.068 (upper) / 0.041 (lower) at 0.05, 0.023 at 0.01; C6_out_of_grammar: recovers 16% of an out-of-grammar injection
* `CAND.CDM.BARYON_AXIS_QUAD.AT_CONC_SCATTER_0`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 1.415x the effect of a collisionless halo (6.8773 against 4.8597); C3_exchangeable: untouched audit half: realised 0.068 (upper) / 0.041 (lower) at 0.05, 0.023 at 0.01; C6_out_of_grammar: recovers 13% of an out-of-grammar injection
* `CAND.CDM.BARYON_AXIS_QUAD.AT_FLSS_0`: C3_exchangeable: untouched audit half: realised 0.068 (upper) / 0.041 (lower) at 0.05, 0.023 at 0.01; C6_out_of_grammar: recovers 6% of an out-of-grammar injection
* `CAND.CDM.BARYON_AXIS_QUAD.AT_FLSS_1`: C2_not_a_restatement: the largest non-halo effect (systematics-only universe (U10)) reproduces 1.608x the effect of a collisionless halo (6.8773 against 4.2776); C6_out_of_grammar: recovers -14% of an out-of-grammar injection
* `CAND.CDM.EXTERNAL_AXIS_QUAD.AT_PRIOR`: C6_out_of_grammar: recovers 16% of an out-of-grammar injection
* `CAND.CDM.EXTERNAL_AXIS_QUAD.AT_CONC_SCATTER_0`: C6_out_of_grammar: recovers 15% of an out-of-grammar injection
* `CAND.CDM.EXTERNAL_AXIS_QUAD.AT_FLSS_0`: C6_out_of_grammar: recovers -14% of an out-of-grammar injection
* `CAND.CDM.EXTERNAL_AXIS_QUAD.AT_FLSS_1`: C6_out_of_grammar: recovers 6% of an out-of-grammar injection

## 8  The separation as a function of f_lss, and of every other halo nuisance

The discriminator trained on the CDM PRIOR (f_lss ~ Beta(2, 2)), read on CDM with one nuisance fixed, against the class's audit corpora.  `directional` uses only the quadrupole phases and powers; `non-directional` drops every quadrupole/harmonic and network feature; `vertical` and `curve residuals` are the single-channel galaxy discriminators (the latter carries residual SHAPE as well as scatter).

| CDM arm | full | directional (power + phase) | phase only | non-directional | vertical only | curve residuals only | phase only vs TENSOR (U5f) | non-directional vs TENSOR (U5f) |
|---|---|---|---|---|---|---|---|---|
| f_lss = 0 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) | 1.000 | 1.000 |
| f_lss = 0.25 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) | 1.000 | 1.000 |
| f_lss = 0.38 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) | 1.000 | 1.000 |
| f_lss = 0.5 (nominal) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) | 1.000 | 1.000 |
| f_lss = 0.75 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) | 1.000 | 1.000 |
| f_lss = 1 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) | 1.000 | 1.000 |

The last two columns restate BK's question: the CDM arm against the tensor universe at fiducial amplitude.  A phase discriminator must lose that separation as the halo's alignment moves onto the external axis; a non-directional one must not.

Every other scan arm (S_U3_* are the CLASS under the same instrument change and must sit at chance):

| arm | full | directional only | non-directional only | vertical only | curve residuals only |
|---|---|---|---|---|---|
| S_shmr_0 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.996 (z 8.5) |
| S_shmr_0.5 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.996 (z 8.5) |
| S_shmr_1.5 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) |
| S_shmr_2 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.998 (z 8.5) |
| S_conc_0 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.989 (z 8.5) |
| S_conc_0.5 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.992 (z 8.5) |
| S_conc_2 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) |
| S_fdd_0.05 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) |
| S_fdd_0.1 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) |
| S_fdd_0.2 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) |
| S_fdd_0.3 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) |
| S_fdd_0.5 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.982 (z 8.5) | 0.997 (z 8.5) |
| S_fdd_0.7 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.749 (z 8.5) | 0.997 (z 8.5) |
| S_fdd_1 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.201 (z 8.5) | 0.997 (z 8.5) |
| S_qh_0.6 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) |
| S_qh_0.7 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) |
| S_qh_1 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) |
| S_zeroscatter | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.986 (z 8.5) |
| S_fixhalo | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) |
| S_sys3 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) |
| S_noise0.5 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.992 (z 8.5) |
| S_qamp0.2 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) |
| S_U3_sys3 | 1.000 (z 8.5) | 0.557 (z 3.0) | 1.000 (z 8.5) | 0.998 (z 8.5) | 1.000 (z 8.5) |
| S_U3_noise0.5 | 0.272 (z 8.5) | 0.591 (z 4.8) | 0.258 (z 8.5) | 0.434 (z 3.2) | 0.322 (z 8.5) |
| S_U3_nuis_hz | 0.708 (z 8.5) | 0.417 (z 4.3) | 0.708 (z 8.5) | 0.623 (z 5.9) | 0.525 (z 1.2) |
| S_U2_nuis_hz | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) |
| S_U3_nuis_ml | 0.912 (z 8.5) | 0.417 (z 4.3) | 0.916 (z 8.5) | 0.864 (z 8.5) | 0.993 (z 8.5) |
| S_U2_nuis_ml | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) |
| S_U3_nuis_dist | 0.868 (z 8.5) | 0.417 (z 4.3) | 0.872 (z 8.5) | 0.842 (z 8.5) | 0.941 (z 8.5) |
| S_U2_nuis_dist | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.999 (z 8.5) |
| S_U3_nuis_incl | 0.822 (z 8.5) | 0.543 (z 2.2) | 0.821 (z 8.5) | 0.846 (z 8.5) | 0.847 (z 8.5) |
| S_U2_nuis_incl | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.999 (z 8.5) |
| S_U3_nuis_sz_err | 0.889 (z 8.5) | 0.417 (z 4.3) | 0.885 (z 8.5) | 0.864 (z 8.5) | 0.525 (z 1.2) |
| S_U2_nuis_sz_err | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) |
| S_eh_0 | 1.000 (z 8.5) | 0.500 (z 0.0) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) |
| S_eh_0.36 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) |
| S_eh_1.0 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) |
| S_qamp0 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.997 (z 8.5) |
| S_fdd_0.8 | 1.000 (z 8.5) | 1.000 (z 8.5) | 1.000 (z 8.5) | 0.528 (z 1.3) | 0.997 (z 8.5) |

Generator 2, its own discriminator (D vs C), the same scans:

| arm | full | directional only | non-directional only |
|---|---|---|---|
| D_nominal | 1.000 | 0.936 | 1.000 |
| D_flss_0 | 1.000 | 0.909 | 1.000 |
| D_flss_0.25 | 1.000 | 0.920 | 1.000 |
| D_flss_0.5 | 1.000 | 0.936 | 1.000 |
| D_flss_0.75 | 1.000 | 0.928 | 1.000 |
| D_flss_1 | 1.000 | 0.918 | 1.000 |
| D_fdd_0.1 | 1.000 | 0.936 | 1.000 |
| D_fdd_0.3 | 1.000 | 0.936 | 1.000 |
| D_fdd_0.5 | 1.000 | 0.936 | 1.000 |
| D_fdd_1 | 1.000 | 0.936 | 1.000 |
| D_zeroscatter | 1.000 | 0.886 | 1.000 |
| D_shmr2 | 1.000 | 0.936 | 1.000 |

## 9  Sample size: where the answer changes with N

| galaxies per corpus | clusters per corpus | U2 vs U3 |
|---|---|---|
| 1 | 0 | AUC 0.992, z 8.50*, p 0.0003 |
| 3 | 0 | AUC 1.000, z 8.50*, p 0.0003 |
| 10 | 0 | AUC 1.000, z 8.50*, p 0.0003 |
| 30 | 0 | AUC 1.000, z 8.50*, p 0.0003 |
| 0 | 1 | AUC 1.000, z 8.50*, p 0.0003 |
| 0 | 3 | AUC 1.000, z 8.50*, p 0.0003 |
| 0 | 12 | AUC 1.000, z 8.50*, p 0.0003 |
| 3 | 1 | AUC 1.000, z 8.50*, p 0.0003 |
| 10 | 3 | AUC 1.000, z 8.50*, p 0.0003 |
| 30 | 12 | AUC 1.000, z 8.50*, p 0.0003 |

## 10  Transfer: untouched scenes and the independent generator

**Untouched scene library** (a library BF never drew).  The generator-1 discriminator transferred unchanged: U2 vs class pool AUC 1.000, z 8.50*, p 0.0003; refitted on the held-out library (null sd 0.0507): AUC 1.000, z 8.50*, p 0.0003; vertical channel alone, refitted: AUC 1.000, z 8.50*, p 0.0003.

| class member | transferred | refitted |
|---|---|---|
| U3 | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 |
| H0 | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 |
| U4t | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 |
| U5t | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 |
| U6t | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 |
| U7t | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 |
| U8t | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 |
| U9t | AUC 1.000, z 8.50*, p 0.0003 | AUC 1.000, z 8.50*, p 0.0003 |

**Generator 2.**  Transferred unchanged (a fingerprint test -- absolute zero points differ between the generators' disc conventions, see section 12): D vs C AUC 1.000, z 8.50*, p 0.0003, D vs tensor AUC 1.000, z 8.50*, p 0.0003, D vs Newton AUC 1.000, z 8.50*, p 0.0003.  Transferred after calibrating every feature's zero point on the two generators' Newtonian arms (label-free with respect to CDM vs class; possible in simulation only): D vs C AUC 1.000, z 8.50*, p 0.0003, D vs tensor AUC 1.000, z 8.50*, p 0.0003.  Refitted on generator 2 (null sd 0.0236): D vs C **AUC 1.000, z 8.50*, p 0.0003**, D vs tensor AUC 1.000, z 8.50*, p 0.0003, D vs Newton AUC 1.000, z 8.50*, p 0.0003.

| generator-2 ablation | AUC | z |
|---|---|---|
| only:gal_vertical | 1.000 | 8.50 |
| only:gal_curve_res | 1.000 | 8.50 |
| drop:gal_vertical | 1.000 | 8.50 |
| drop:all quadrupole/harmonic content | 1.000 | 8.50 |
| only:directional | 0.928 | 8.50 |
| only:clu_gas | 1.000 | 8.50 |
| only:clu_shear_mono_res | 1.000 | 8.50 |
| only:clu_strong_lens | 1.000 | 8.50 |
| only:clu_dynamics | 1.000 | 8.50 |

## 11  Baryonic closure: does P(G | B) differ, and by how much

**Stochastic closure.**  The same library object emitted across sets: the sd of its residual across sets (noise only for the class; noise plus the halo draw for CDM), pooled over objects.  `fixhalo` keeps ONE halo per object across sets, so `halo-isolated` = sqrt(nominal^2 - fixhalo^2) is the halo's own contribution and `excess over class` = sqrt(nominal^2 - U3^2).

| residual | U3 | H0 | U2 prior | CDM nominal | CDM halo held | CDM zero scatter | halo-isolated sd | excess over class |
|---|---|---|---|---|---|---|---|---|
| `gal:res_mean` | 0.098 | 0.104 | 0.140 | 0.141 | 0.114 | 0.110 | **0.083** | 0.101 |
| `gal:res_out_in` | 0.119 | 0.131 | 0.147 | 0.149 | 0.146 | 0.136 | **0.030** | 0.090 |
| `gal:res_slope` | 0.215 | 0.235 | 0.263 | 0.264 | 0.259 | 0.242 | **0.053** | 0.154 |
| `gal:rz_1` | 0.162 | 0.166 | 0.177 | 0.179 | 0.175 | 0.172 | **0.036** | 0.076 |
| `gal:dz_1` | 0.139 | 0.140 | 0.150 | 0.151 | 0.141 | 0.140 | **0.054** | 0.059 |
| `gal:res_6` | 0.106 | 0.114 | 0.157 | 0.159 | 0.133 | 0.128 | **0.087** | 0.119 |
| `clu:wl_lA` | 0.469 | 1.149 | 0.374 | 0.336 | 0.333 | 0.325 | **0.047** | 0.000 |
| `clu:t_lA` | 0.062 | 1.306 | 0.110 | 0.106 | 0.104 | 0.093 | **0.019** | 0.085 |
| `clu:h_lA` | 0.081 | 0.558 | 0.111 | 0.107 | 0.100 | 0.095 | **0.040** | 0.071 |
| `clu:d_lA` | 0.045 | 0.295 | 0.061 | 0.061 | 0.057 | 0.055 | **0.021** | 0.041 |
| `clu:wl_slope` | 1.976 | 5322.457 | 2.291 | 2.264 | 2.109 | 1.994 | **0.823** | 1.105 |
| `clu:t_slope` | 0.056 | 0.501 | 0.114 | 0.112 | 0.091 | 0.078 | **0.066** | 0.098 |
| `clu:sl_res` | 0.000 | 1.091 | 0.016 | 0.016 | 0.010 | 0.009 | **0.012** | 0.016 |

The galaxy residual scatter as a function of the SHMR scatter prior (dex of halo mass at fixed stellar mass):

| arm | out-of-fold RAR scatter | within-galaxy residual sd | between-object sd of res_mean | within-object sd of res_mean |
|---|---|---|---|---|
| S_shmr_0 | 0.171 | 0.058 | 0.079 | 0.135 |
| S_shmr_0.5 | 0.172 | 0.058 | 0.079 | 0.137 |
| S_nominal | 0.176 | 0.059 | 0.080 | 0.141 |
| S_shmr_1.5 | 0.183 | 0.060 | 0.080 | 0.148 |
| S_shmr_2 | 0.192 | 0.061 | 0.081 | 0.157 |
| U3 | 0.108 | 0.047 | 0.013 | 0.098 |

**Covariance of the increments.**  Per-object increment of each residual, arm minus U3 on the same scene and noise; slopes between channels (a halo moves lensing, X-ray and dynamics together; a photon-coupling change does not):

| arm - U3 | slope d_lA on wl_lA | slope t_lA on wl_lA | slope dz_1 on res_mean | mean d wl_lA | mean d d_lA | mean d t_lA | mean d dz_1 |
|---|---|---|---|---|---|---|---|
| U2-U3 | +0.07 +- 0.00 | +0.13 +- 0.00 | -0.34 +- 0.00 | +0.232 | +0.033 | -0.051 | -0.274 |
| U8f-U3 | +0.00 +- 0.00 | +0.00 +- 0.00 | n/a +- n/a | +0.081 | +0.000 | +0.000 | +0.000 |
| U5f-U3 | +0.02 +- 0.00 | +0.05 +- 0.00 | -0.56 +- 0.00 | -0.053 | +0.001 | -0.000 | +0.006 |
| U6f-U3 | +0.02 +- 0.00 | +0.03 +- 0.00 | -0.13 +- 0.00 | +0.125 | +0.013 | +0.021 | -0.000 |
| U7f-U3 | +0.02 +- 0.00 | +0.04 +- 0.00 | -0.00 +- 0.00 | +0.035 | +0.006 | +0.011 | -0.000 |
| U4f-U3 | +0.00 +- 0.00 | +0.00 +- 0.00 | -0.00 +- 0.00 | +0.068 | +0.021 | +0.028 | +0.000 |

**Structural closure.**  Corpus means with every halo scatter set to zero (G a deterministic function of B):

| arm | `dz_1` | `rz_1` | `res_mean` | `res_out_in` | `wl_lA` | `t_lA` | `t_slope` | `d_lA` | `sl_res` | `rar_scatter_oof` |
|---|---|---|---|---|---|---|---|---|---|---|
| U3 | +0.005 | -0.005 | -0.010 | -0.000 | -0.170 | -0.042 | -0.043 | -0.049 | +0.000 | +0.108 |
| U2 | -0.268 | -0.336 | -0.061 | +0.010 | +0.062 | -0.093 | -0.383 | -0.016 | +0.006 | +0.175 |
| S_nominal | -0.294 | -0.364 | -0.064 | +0.011 | +0.073 | -0.092 | -0.382 | -0.016 | +0.006 | +0.176 |
| S_zeroscatter | -0.292 | -0.350 | -0.052 | +0.012 | +0.083 | -0.084 | -0.390 | -0.011 | +0.003 | +0.152 |
| S_fixhalo | -0.289 | -0.357 | -0.061 | +0.014 | +0.095 | -0.071 | -0.382 | -0.006 | +0.004 | +0.171 |

The discriminator on zero-scatter CDM: full 1.000 (z 8.5), non-directional 1.000, vertical only 1.000, scatter only 0.986.

## 12  What this establishes, as bounds

**Bound on distinguishability.**  Under the corpus, amplitudes, geometries and nuisance priors stated in section 0, the calibrated discriminator separates the dark-matter universe from the modified-gravity class at AUC 1.000 (z 8.50*) on generator 1, 1.000 on untouched scenes, and 1.000 on the independent generator; the vertical channel alone reaches 1.000 and survives every alignment prior (f_lss from 0 to 1: 1.00, 1.00, 1.00, 1.00, 1.00, 1.00).  This is a statement about these source distributions, noise levels, nuisance models and channels, not a theorem about every future observation.

**Where the vertical answer changes.**  Vertical-only discrimination against the dark-disc fraction: f_dd 0: 1.00, f_dd 0.05: 1.00, f_dd 0.1: 1.00, f_dd 0.2: 1.00, f_dd 0.3: 1.00, f_dd 0.5: 0.98, f_dd 0.7: 0.75, f_dd 1: 0.20.  A razor-thin dark component carrying the corresponding fraction of the halo's enclosed-mass profile is the one collisionless degree of freedom that mimics an isotropic boost; the report quotes the crossing as the amplitude at which the vertical statistic stops separating.

**Zero point.**  The vertical contrast's absolute value depends on the disc's vertical-structure convention: generator 1 (razor-thin, sigma_z^2 = g_z h_z) puts the class at +0.01 dex and CDM at -0.27; generator 2 (Miyamoto-Nagai discs, k_z in [0.8, 1.25]) puts the class at -0.26 and CDM at -0.47.  The CDM-minus-class CONTRAST (-0.27 vs -0.21 dex) transfers; the zero point does not, and on real data it is the systematic the test lives or dies by.

**Responsiveness.**  d(dz_1 estimate)/d(true contrast) = 0.999 +- 0.000 across the dark-disc scan; d(closure scatter)/d(SHMR scatter, dex) = 0.060 +- 0.010.
