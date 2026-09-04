# Run BI -- the void path-length map cross-correlated with Planck

Generated 2026-09-04T21:34:34Z from `voidcmb_results.json`, `certificate_voidcmb.json`, `isw_separation.json`, `systematics.json`. Run id `BI-voidcmb`, registered in `work/wellnet-2026-09/registry/registry.py`.

Run AK partitioned the path-redshift hypothesis and killed one half: an energy-drain or tired-light mechanism carrying the redshift predicts `b = 0` in `dt_obs = dt_em (1+z)^b`, which is 90 sigma from the DES supernova measurement. The GEOMETRIC half predicts `b = 1` identically, so that test has zero power against it. AK derived, but never ran, the one test that reaches it: an achromatic path redshift gives `dT/T = -c2 dI_q` on the CMB. This run turns that derivation into a measurement.

## The result

    c2/c1 = -0.0266% +- 0.0206%   (-1.28 sigma null-calibrated, p = 0.206)
    |c2/c1| < 0.0678% at 95%
    responsiveness d(estimate)/d(injected) = 0.9814

Against Run AK's derived CMB gate of 0.28-0.44% (and 0.35% recomputed on this map's own sd(dI_q)), the 95% limit is **4.1 to 6.5 times tighter**. At AK's own predicted amplitude this pipeline would have seen 13.6 sigma (at 0.28%) or 21.3 sigma (at 0.44%). It saw 1.28 sigma. The 3-sigma floor is 0.0631%, against the 3.9% (statistical) / 5.9% (with systematics) floor Run AK quoted for the supernova-based fit -- a factor of 62 to 94.

| injected c2/c1 | recovered | in-grammar slope |
|---|---|---|
| +0.10% | +0.0739% | 1.0000 |
| +0.20% | +0.1739% | 1.0000 |
| +0.40% | +0.3739% | 1.0000 |
| -0.40% | -0.4261% | 1.0000 |

**The headline model was declared before any value existed.** It is the ISW-marginalised fit, which is the conservative of the three; the physically normalised ISW treatment is tighter and is reported below.

## The finding that is not the measurement

The OLS covariance says sigma = 0.0044%. The rotation null says 0.0206%. The ratio is **4.72**, so the analytic error bar would have announced this null as a **6.0 sigma detection**.

Run AK's lane found the same thing (6.1 sigma analytic against 1.8 simulated) on a different dataset with a different estimator. This run reproduces it on a third. The gap is not a subtlety: the CMB is a correlated field, OLS assumes independent pixels, and the number of independent modes under a 5,810 deg^2 footprint is of order the number of degree-scale patches, not the number of pixels.

The clearest demonstration is the resolution arm: nside 128 uses 26,944 pixels against nside 64's 6,753, a factor of four. Its analytic error falls to 0.0027% while its NULL width is 0.0198% against 0.0206% -- unchanged. Four times the pixels carry no extra information, and only the null knows it.

## Job 0 -- the Stage 4 certificate, issued before any value was read

`certificate_voidcmb.json`, generated 2026-09-04T21:30:18Z, `opened_true_footprint_temperature = False`. A blind guard refused any pixel set overlapping the true footprint by more than 5%; it was consulted 7,085 times during certification. Every temperature the certificate read came from a rotated placement, so the test could be sized against the real sky -- foregrounds, noise and mask included -- without the measurement being visible.

| check | verdict | number that decides it |
|---|---|---|
| C1 responsive | PASS | d(estimate)/d(injected) = 1.0000 in grammar, x 0.9814 for pixelisation |
| C2 not a restatement | PASS | a 100 uK monopole moves c2/c1 by +1.2e-18, a 0.1% gain error by +3.5e-07, against 4.0e-03 for the predicted signal |
| C3 exchangeable | PASS | realised FPR **0.050** (leave-one-out) and **0.052** (independent Gaussian skies) against nominal 0.05; sd(sim)/sd(null) = 0.98 |
| C4 powered at the PREDICTED amplitude | PASS | 13.6 sigma at 0.28%, 21.3 sigma at 0.44% -- AK's own bound, not a convenient amplitude |
| C5 support | PASS | reads chi in [0, 332.4] Mpc/h and z <= 0.1125, exactly the catalogue's declared limits; 97.5% of the eroded footprint survives the Planck mask |
| C6 out-of-grammar | PASS | recovery I_R 0.31, I_sat 0.67, N_v 0.64, I_phi_in 0.00, I_q_far 0.68 |
| C7 nuisance-distinct | PASS | closest nuisance `ISW_phi_in` at |r| = 0.763 |

**CERTIFICATE ISSUED.** The null was built from 2,000 admissible rotations and reflections; the two halves give sd 0.0207% (proper) and 0.0204% (reflected).

C3 is the check that matters, because two earlier runs in this programme found permutations running at FPR 0.53-0.70 and 0.855-0.970 against a nominal 0.05. Here the size was measured two independent ways -- leave-one-out over the rotation bank, and Gaussian CMB skies drawn from the published Planck TT spectrum on the true footprint geometry -- and both land on nominal. The two null widths agree to 2%.

C7's worst nuisance is the ISW template itself at |r| = 0.763. Every other systematic is far away: ISW_phi_k3 +0.413, ame -0.151, dipole_x -0.144, edge_deg +0.131, sync -0.106, dust -0.099.

## Job 1 -- the path-length map

HEALPix nside 64, NESTED, GALACTIC -- the native Planck grid, so the CMB is never resampled. 39,735 holes forming 1,163 voids from the SDSS DR7/NSA VAST VoidFinder catalogue (Planck2018 comoving), exact ray-sphere intersection with interval-union algebra, integrated over chi in [0, 332.4] Mpc/h.

    footprint          6,923 pixels, 5810 deg^2 after 5 deg erosion
    I_q                mean 203.3, sd 34.1 Mpc/h, range 69.1-295.7
    void path fraction 0.612 of the ray
    median void Reff   15.5 Mpc/h

AK's warning that the watershed finders TILE the volume rather than select voids -- `corr(dI_q, mean LOS density) = +0.319` for REVOLVER and -0.190 against the true underdensity path integral -- is the reason VoidFinder is the only arm here. A VoidFinder hole is a sphere certified empty, so its union path length is an emptiness measure by construction, not by inference. The watershed arms are not used, and the caveat is therefore carried rather than inherited.

## Job 2 -- Planck

9 products from the IRSA/IPAC mirror (the PLA AIO endpoint is a recorded trap: 503 for a whole session while its landing page returns 200). Three detectors were required of every one, none sufficient alone -- transport (bytes == Content-Length), structure (NSIDE/ORDERING/COORDSYS as assumed), identity (a header provenance string naming the expected product and release). All 9 passed.

| product | MB | validated |
|---|---|---|
| `COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits` | 402.7 | yes |
| `COM_CMB_IQU-smica_1024_R2.02_full.fits` | 176.2 | yes |
| `COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits` | 201.3 | yes |
| `COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt` | 0.2 | yes |
| `COM_CompMap_dust-commander_0256_R2.00.fits` | 28.3 | yes |
| `COM_CompMap_Synchrotron-commander_0256_R2.00.fits` | 9.5 | yes |
| `COM_CompMap_freefree-commander_0256_R2.00.fits` | 18.9 | yes |
| `COM_CompMap_AME-commander_0256_R2.00.fits` | 28.4 | yes |
| `COM_CompMap_CO-commander_0256_R2.00.fits` | 28.3 | yes |

## Job 3 -- the measurement, and every declared variant

| arm | c2/c1 | null sd | null-calibrated | analytic sd | what the analytic error would have said |
|---|---|---|---|---|---|
| M1_no_isw | -0.0036% | 0.0139% | -0.28 sigma | 0.0029% | -1.3 sigma |
| M2_isw_marginalised | -0.0261% | 0.0206% | -1.28 sigma | 0.0044% | -6.0 sigma |
| M3_hardened | -0.0227% | 0.0194% | -1.18 sigma | 0.0045% | -5.1 sigma |
| S_near | -0.0246% | 0.0288% | -0.89 sigma | 0.0050% | -4.9 sigma |
| S_far | -0.0088% | 0.0183% | -0.52 sigma | 0.0048% | -1.8 sigma |
| S_nonedge | -0.0222% | 0.0176% | -1.31 sigma | 0.0027% | -8.1 sigma |
| S_isw_k3 | -0.0012% | 0.0135% | -0.13 sigma | 0.0032% | -0.4 sigma |
| map_smica_PR2_nside1024 | -0.0272% | 0.0200% | -1.36 sigma | 0.0043% | -6.3 sigma |
| erode_2deg | -0.0269% | 0.0177% | -1.48 sigma | 0.0040% | -6.6 sigma |
| erode_8deg | -0.0269% | 0.0193% | -1.37 sigma | 0.0047% | -5.8 sigma |
| nside_128 | -0.0254% | 0.0198% | -1.25 sigma | 0.0027% | -9.5 sigma |

Nothing moves. Two component-separation pipelines from two Planck releases, three footprint erosions, two map resolutions, a tomographic split of the path integral, and a cut to voids that never touch the survey boundary all agree inside a fraction of the null width.

## The ISW separation

The integrated Sachs-Wolfe effect is a real physical contaminant with the same sign structure -- voids are cold spots in both. It is separated by the two templates weighting the void radius function differently: a top-hat void of radius R contributes a chord ~2R to I_q but ~R^3 to the potential integral.

    A  free-amplitude marginalisation (headline)  c2/c1 = -0.0261% +- 0.0205%  (-1.30 sigma)
    B  LCDM template, amplitude FIXED by theory   c2/c1 = -0.0044% +- 0.0138%  (-0.29 sigma)
    C  no ISW term at all                         c2/c1 = -0.0036% +- 0.0138%  (-0.29 sigma)

**The decisive number is not any of those three.** The LCDM ISW, normalised by theory rather than fitted (Omega_m = 0.315, |delta| = 0.7, f = 0.530), would bias an ISW-free fit by +0.0008%, which is **0.057 sigma**. The ISW is not a limiting systematic at this sensitivity; it is 17 times below the noise.

The free-amplitude fit returns an ISW coefficient 28x the LCDM value (9.4 uK rms against 0.33 uK predicted), but that coefficient is only 1.40 sigma in its OWN rotation null. It is absorbing large-scale CMB variance, not measuring the ISW, and it costs 1.49x in error for doing so. Reporting it as the headline is the conservative choice and was declared as such before any value existed.

## Systematic splits

| split | n | c2/c1 | null-calibrated |
|---|---|---|---|
| b_gt_30 | 6,534 | -0.0268% | -1.35 sigma |
| b_gt_45 | 5,334 | -0.0277% | -1.36 sigma |
| b_below_median | 3,439 | -0.0376% | -1.33 sigma |
| b_above_median | 3,314 | -0.0169% | -0.76 sigma |
| l_below_median | 3,379 | -0.0180% | -0.99 sigma |
| l_above_median | 3,374 | -0.0010% | +0.03 sigma |
| interior_edge_gt_15deg | 3,788 | -0.0288% | -1.33 sigma |
| near_edge_le_15deg | 2,965 | -0.0013% | -0.07 sigma |
| dust_quiet_half | 3,377 | -0.0353% | -1.91 sigma |
| dust_loud_half | 3,376 | -0.0077% | -0.33 sigma |
| fully_unmasked_only | 6,210 | -0.0254% | -1.32 sigma |

Every split is re-nulled inside its own geometry. The largest excursion is the dust-quiet half at -1.91 sigma, with the dust-loud half at -0.33 -- the wrong way round for a dust systematic, and unremarkable among eleven splits.

The decile relation is linear: the slope through the ten dI_q deciles is +0.0247 uK per Mpc/h with a quadratic curvature of -5.61e-04.

## What this settles, and what it does not

**Settles.** The geometric half of the path-redshift class -- the half supernova time dilation cannot touch, because it predicts b = 1 identically -- is now measured, not bounded by an anisotropy budget. |c2|/c1 < 0.0678% at 95% under the conservative headline model and 0.0307% under the physically normalised ISW treatment. Run AK's gate of 0.28-0.44% is excluded: the pipeline had 14-21 sigma of power at exactly that amplitude and found nothing.

**Does not settle.**

1. A mechanism whose coefficient is REDSHIFT-DEPENDENT such that it vanishes below z = 0.11 and revives above it. The map reaches z = 0.1125 and no further; that is stated as support, not assumed away.
2. A mechanism keyed to a different environmental functional. The out-of-grammar recoveries measure exactly how much of that this test reaches: I_R 0.31, I_sat 0.67, N_v 0.64, I_q_far 0.68. A law expressible as a smooth monotone functional of the void path length is covered at 0.31-0.68 of full sensitivity; one orthogonal to it is not covered at all.
3. The tidal coefficients c3 and c6. AK.5 showed they are separable only on watershed geometry, which the footprint analysis restricts to n = 46. Nothing here changes that.
4. Anything in the gravity lanes. This branch is logically independent; no data, fit, calibration or model-selection step is shared with them.

## Can the data reach the amplitude, and what would go deeper

Yes, comfortably. AK's gate sits at 14-21 sigma of this pipeline's null width, and the 3-sigma floor is 0.0631%. The question the charter would ask next is what it would take to go further, and the answer is: not more Planck.

The limit is cosmic-variance-limited by the CMB's own anisotropy projected onto this specific template over 5810 deg^2. The nside-128 arm proves it directly -- four times the pixels, the same null width to 4%. Instrumental noise, component separation and the ISW are all far below the floor (the LCDM ISW at 0.057 sigma). What buys sensitivity is sky area and path length in the VOID map: the error scales as 1/(sd(dI_q) sqrt(area)). DESIVAST VoidFinder over the DESI BGS footprint with sd(dI_q) = 35.1 Mpc/h against this map's 34.1 would gain of order the square root of the area ratio, i.e. a factor near two -- not an order of magnitude. **This observable is within roughly a factor of two of its ultimate reach with existing data.**

**Sealed and reserved, untouched.** KiDS and the wide binaries are sealed. SPT, X-GAP, CLoGS, Gaia dynamical products and MUSE/Granata dispersions are the confirmation reserve. This run read Planck and the SDSS void catalogue only, both explicitly unreserved.

**Admissibility grade (BE.6).** The Planck temperature map is T1 (a calibrated detector observable after component separation, not a fit under any gravity law). The VoidFinder catalogue is T1-T2: hole positions and radii follow from galaxy positions and a fixed distance-redshift relation, with no dynamical mass modelling anywhere. The LCDM ISW template is T3 -- model-derived -- which is why the headline marginalises its amplitude rather than trusting it, and why its theory-normalised value is reported separately as a sizing argument rather than as a subtraction.
