# Temperature-clamp audit of `invariant_bench._cluster_profile`

Run 2026-09-04T12:03:09, numpy 2.2.6, Python 3.13.5. Sealed probes never loaded: kids, widebin.

`_cluster_profile` interpolated the coarse X-COP temperature profile onto the fine density grid with a bare `np.interp` -- no `left`, no `right` -- so `kT` was **clamped** to the endpoint value outside the measured range. The hydrostatic acceleration carries `(dln n_e/dln r + dln kT/dln r)`, so beyond the last measured temperature bin `dln kT/dln r` was forced to **exactly zero** and the temperature-gradient term was silently deleted -- in the outskirts, where this programme reads its radial trend, and where the true temperature is falling.

Bench sha256 before `00cfbf287b4ad9798ab77ac03b93f5d2aec37dd299c9c80ba3df8cfbdaf25d44`  
Bench sha256 after  `bed02ff466da83e64b1fac1e4f93f525832e2017bbc05983a80efcc2ebc4c77a`

## Headline

* **93 of 588 X-COP points (15.82%) rest on clamped temperature**, all of them the outermost points of their cluster (0 inner extrapolations, so the clamped set is exactly the outer set).
* The measured temperature grid ends at a median **r/R500 = 0.915** (range 0.789 to 1.097); the relation is quoted out to **r/R500 = 1.519**.
* **3 of 9 recorded verdicts change** when the clamped points are dropped. The three that move are the X-COP temperature correlation (`rho_T`), its permutation p, and the pressure amplitude `kappa`.
* Against a synthetic truth with a genuinely falling temperature, the clamp biases the within-cluster radial slope by **-0.0765 dex/dex**, i.e. 15.9% of the observed slope, and it is 91.7% of the pipeline's entire flat-truth bias.
* Inside Run AT's **own** noiseless forward null, reproduced bit-identically, the clamp is **64.2% of the 29%** (S3 bias -0.1359 -> -0.0487 with the clamp removed and nothing else changed).
* The trend itself **survives**: the within-cluster radial slope moves from -0.4817 to -0.4499, 6.6%.

## Job 1 -- consumer inventory

Derived by walking the repo with `ast`, not asserted (`inventory.py`). **0** direct callers outside the bench, **15** importers of `Bench` of which **15** consume X-COP, and **9** files that re-implement the identical unguarded interpolation. This lane's own files are excluded.

### Direct callers of `_cluster_profile`

* none outside the bench. `_xcop` is the only in-repo caller, which is why patching `_cluster_profile` reaches every lane at once.

### Lanes that import `Bench`, and therefore run `_xcop` -> `_cluster_profile`

**All fifteen consume X-COP.** The first pass of this inventory said three of them did not; that was wrong -- `p03`, `p04` and `p05` never write `d["xcop"]`, they call `b.confound(...)`, `b.score(...)` or iterate `b.d`, every one of which pulls X-COP in.

| lane | consumes X-COP via |
| --- | --- |
| `work/gravity-cluster-audit-2026-09/c70_endogeneity.py` | `d["xcop"]`, `b.d[` |
| `work/gravity-cluster-audit-2026-09/c71_amplitude.py` | `d["xcop"]`, `b.d[` |
| `work/gravity-cluster-audit-2026-09/mirror/mirror_run.py` | `b.d[` |
| `work/gravity-cluster-audit-2026-09/power/power_analysis.py` | `d["xcop"]`, `b.d[` |
| `work/gravity-wells-2026-09/m10_xcop_temperature.py` | `d["xcop"]`, `b.d[` |
| `work/gravity-wells-2026-09/m11_verify_identity.py` | `d["xcop"]`, `b.d[` |
| `work/gravity-wells-2026-09/m12_redo_correct.py` | `d["xcop"]`, `b.d[` |
| `work/gravity-wells-2026-09/p01_rigorous.py` | `d["xcop"]`, `b.d[` |
| `work/gravity-wells-2026-09/p03_audit_confound.py` | `.d.items()` |
| `work/gravity-wells-2026-09/p04_reaudit.py` | `.d.items()` |
| `work/gravity-wells-2026-09/p05_floor.py` | `b.d[` |
| `work/gravity-wells-2026-09/v01_extract.py` | `d["xcop"]`, `b.d[` |
| `work/gravity-wells-2026-09/v03_lens_and_substructure.py` | `d["xcop"]`, `b.d[` |
| `work/gravity-wells-2026-09/v04_lumpiness.py` | `d["xcop"]`, `b.d[` |
| `work/gravity-wells-2026-09/v05_realizations.py` | `d["xcop"]`, `b.d[` |

### Independent re-implementations of the same clamp

These do not call `_cluster_profile`, but they interpolate the X-COP temperature table with their own bare `np.interp`. Every `np.interp` call in the repo was parsed; **not one** passes `left=` or `right=` on a temperature, so the patch to the bench does not fix any of them.

| file | unguarded temperature interps | direction |
| --- | --- | --- |
| `work/gravity-cluster-audit-2026-09/c70_endogeneity.py` | lines [128, 205] | **same as the bench: coarse T on a finer grid** |
| `work/gravity-cluster-audit-2026-09/c71_amplitude.py` | lines [66, 77] | **same as the bench: coarse T on a finer grid** |
| `work/gravity-cluster-audit-2026-09/power/power_analysis.py` | lines [143, 144, 176, 192, 196, 199, 313] | **same as the bench: coarse T on a finer grid** |
| `work/gravity-wells-2026-09/m10_xcop_temperature.py` | lines [126] | **same as the bench: coarse T on a finer grid** |
| `work/gravity-wells-2026-09/m12_redo_correct.py` | lines [77] | **same as the bench: coarse T on a finer grid** |
| `work/gravity-wells-2026-09/p01_rigorous.py` | lines [112, 113] | **same as the bench: coarse T on a finer grid** |
| `work/wellnet-2026-09/r500-audit/ingest.py` | lines [193] | **same as the bench: coarse T on a finer grid** |
| `work/wellnet-2026-09/r500-audit/nullsim.py` | lines [177, 204, 224, 261] | **same as the bench: coarse T on a finer grid** |
| `work/wellnet-2026-09/r500-audit/run_job2d.py` | lines [64] | **same as the bench: coarse T on a finer grid** |
| `src/sigma_theory_compiler/gravity_shared_target_blind_ben_development_executor_v4.py` | lines [1841, 1849] | onto the T grid -- same mechanism, different exposure, **not measured here** |
| `tests/test_gravity_shared_target_blind_ben_development_executor_v4.py` | lines [428, 429, 465, 468] | onto the T grid -- same mechanism, different exposure, **not measured here** |
| `work/gravity-ben-development-executor-v4-superseded-audit1/gravity_shared_target_blind_ben_development_executor_v4.py` | lines [1016, 1021] | onto the T grid -- same mechanism, different exposure, **not measured here** |
| `work/gravity-ben-development-executor-v4-superseded-audit1/test_gravity_shared_target_blind_ben_development_executor_v4.py` | lines [171, 173] | onto the T grid -- same mechanism, different exposure, **not measured here** |
| `work/gravity-ben-development-executor-v4-superseded-audit2/gravity_shared_target_blind_ben_development_executor_v4.py` | lines [1416, 1427] | onto the T grid -- same mechanism, different exposure, **not measured here** |
| `work/gravity-ben-development-executor-v4-superseded-audit2/test_gravity_shared_target_blind_ben_development_executor_v4.py` | lines [263, 265, 301, 305] | onto the T grid -- same mechanism, different exposure, **not measured here** |
| `work/gravity-ben-development-executor-v4-superseded-audit3/gravity_shared_target_blind_ben_development_executor_v4.py` | lines [1674, 1682] | onto the T grid -- same mechanism, different exposure, **not measured here** |
| `work/gravity-ben-development-executor-v4-superseded-audit3/test_gravity_shared_target_blind_ben_development_executor_v4.py` | lines [300, 301, 337, 340] | onto the T grid -- same mechanism, different exposure, **not measured here** |

### Recorded JSON downstream

| file | producer | present |
| --- | --- | --- |
| `work/gravity-wells-2026-09/paper_results.json` | `p01_rigorous.py` | yes |
| `work/gravity-cluster-audit-2026-09/paper_results.json` | `p01_rigorous.py (copy)` | yes |
| `work/gravity-wells-2026-09/xcop_identity.json` | `m11_verify_identity.py` | yes |
| `work/gravity-cluster-audit-2026-09/c70_endogeneity.json` | `c70_endogeneity.py` | yes |
| `work/gravity-cluster-audit-2026-09/c71.json` | `c71_amplitude.py` | yes |
| `work/gravity-cluster-audit-2026-09/power/power_results.json` | `power_analysis.py` | yes |
| `work/wellnet-2026-09/r500-audit/results.json` | `r500-audit (Run AT)` | yes |
| `work/wellnet-2026-09/r500-audit/job1_results.json` | `r500-audit (Run AT)` | yes |
| `work/wellnet-2026-09/r500-audit/job2_results.json` | `r500-audit (Run AT)` | yes |
| `work/wellnet-2026-09/r500-audit/job2d_results.json` | `r500-audit (Run AT)` | yes |
| `work/wellnet-2026-09/r500-audit/job3_results.json` | `r500-audit (Run AT)` | yes |
| `work/wellnet-2026-09/r500-audit/identity_results.json` | `r500-audit (Run AT)` | yes |

## Exposure, per cluster

| cluster | n | clamped | % | stencil | last measured T (r/R500) | outermost point (r/R500) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1644 | 56 | 20 | 35.7% | 21 | 0.848 | 1.519 |
| A1795 | 38 | 5 | 13.2% | 6 | 0.940 | 1.352 |
| A2029 | 38 | 5 | 13.2% | 6 | 0.928 | 1.163 |
| A2142 | 46 | 3 | 6.5% | 4 | 0.958 | 1.147 |
| A2255 | 49 | 8 | 16.3% | 9 | 0.960 | 1.369 |
| A2319 | 54 | 2 | 3.7% | 3 | 1.097 | 1.199 |
| A3158 | 53 | 8 | 15.1% | 9 | 0.929 | 1.395 |
| A3266 | 63 | 9 | 14.3% | 10 | 0.790 | 1.090 |
| A644 | 47 | 4 | 8.5% | 5 | 0.902 | 1.325 |
| A85 | 46 | 8 | 17.4% | 9 | 0.789 | 1.285 |
| RXC1825 | 49 | 15 | 30.6% | 16 | 0.850 | 1.483 |
| ZW1215 | 49 | 6 | 12.2% | 7 | 0.868 | 1.204 |
| **total** | **588** | **93** | **15.82%** | | median **0.915** | max **1.519** |

`stencil` counts points whose `np.gradient` three-point stencil touches a clamped point, so their `dlnT/dlnr` is contaminated even though they are themselves inside the measured range.

## Job 2 -- the fix

`_cluster_profile(d, temp_extrapolation=...)` with four modes:

| mode | behaviour |
| --- | --- |
| `clamp` | **default**; `np.interp`'s own clamping, bit-identical to every recorded number |
| `drop` | keep only points inside the measured temperature range |
| `loglinear` | continue the measured outer log-log temperature slope |
| `forbid` | raise `TemperatureExtrapolationError` |

The return value is a `ClusterProfile`, a 4-tuple subclass, so every existing `r, gb, go, R500 = self._cluster_profile(d)` keeps working unchanged. It carries `.extrapolated`, `.stencil`, `.frac_extrapolated`, `.r_tmin`, `.r_tmax`, `.kT`, `.mode`. `Bench(temp_extrapolation=...)` threads the choice through, `Bench.extrapolation_report` holds one structured dict per cluster, and `Bench._announce_extrapolation` emits a `TemperatureExtrapolationWarning` carrying the extrapolated fraction on every load that touches a clamped point.

**Default behaviour is bit-identical.** 12 of 12 recorded quantities reproduce under `mode='clamp'`:

| recorded quantity | recorded | reproduced | \|diff\| |
| --- | ---: | ---: | ---: |
| per-cluster excess, worst of 12 | 0 | 0 | 0.00e+00 |
| rho_T  Spearman(kT, excess) | 0.615385 | 0.615385 | 0.00e+00 |
| p_T  permutation | 0.036975 | 0.037445 | 4.70e-04 |
| kappa best fit | 156315 | 156315 | 0.00e+00 |
| c70 d ln excess / d ln kT | 0.516532 | 0.516532 | 0.00e+00 |
| c70 slope error | 0.205051 | 0.205051 | 0.00e+00 |
| c70 within-cluster rho median | 0.642557 | 0.642557 | 0.00e+00 |
| c70 normalisation ratio median | 1.03206 | 1.03206 | 9.64e-10 |
| c71 within-cluster scatter | 0.168939 | 0.168939 | 6.00e-12 |
| c71 median-ratio scatter | 0.0682513 | 0.0682513 | 1.12e-11 |
| power A (= c70 within rho median) | 0.642557 | 0.642557 | 1.11e-16 |
| power B (= c71 within scatter) | 0.168939 | 0.168939 | 6.00e-12 |

### Regression test

`test_tempclamp.py`, 93 tests. `prove_test_fails_prepatch.py` loads the untouched pre-patch source and shows **7 of 7** of the suite's demands fail against it:

| demand | fails pre-patch | detail |
| --- | --- | --- |
| forbid raises TemperatureExtrapolationError | **yes** | TypeError instead: Bench._cluster_profile() got an unexpected keyword argument 'temp_extrapolation' |
| profile carries an .extrapolated mask | **yes** | type=tuple, len=4, attrs=ABSENT -- the output cannot tell you |
| a warning carries the extrapolated fraction | **yes** | 2 warnings emitted, 0 of them about extrapolation (['ResourceWarning', 'ResourceWarning']) |
| module exports TemperatureExtrapolationError | **yes** | absent |
| module exports TemperatureExtrapolationWarning | **yes** | absent |
| module exports ClusterProfile | **yes** | absent |
| module exports TEMP_MODES | **yes** | absent |

The load-bearing one is `test_forbid_raises_on_xcop`: pre-patch there was no way to say *do not extrapolate*, so the call raised `TypeError` instead of the specific error and the clamped points went through unannounced. Its counterweight is `test_default_is_bit_identical`, which inlines the pre-patch body verbatim and requires `np.array_equal` on `r`, `gb` and `go` for all twelve clusters.

### Bugs this lane's own tests found

1. **My first `deep`-mask test was wrong.** It let the innermost extrapolated point (a one-sided `np.gradient` stencil) into a set asserted to be exactly flat. Also, a clamped stencil returns ~5.7e-14, not literal zero: `np.gradient`'s unequal-spacing coefficients sum to zero only to rounding. Fixed to exclude endpoints and use a 1e-11 tolerance -- four orders below the smallest measured slope.
2. **My first synthetic truth produced negative temperatures.** Anchoring the hydrostatic pressure at the middle bin let `P(r_out)` go negative for five of twelve clusters; `log(kT)` became `nan` and the reconstruction then *silently dropped* those points -- reproducing, inside the audit, the exact class of silent mask the audit exists to catch. It also flipped the sign of the reported bias. Fixed by anchoring at the outer boundary, where positivity is automatic, plus an assertion.
3. **`mode='drop'` went silent.** The first version of the patch counted only extrapolated points still present, so `drop` reported zero extrapolation and said nothing about the 93 points it had just deleted. Now covered by `test_drop_mode_still_reports_what_it_removed`.
4. **`Bench._load` swallowed the `forbid` error.** Its bare `except Exception` turned a deliberate hard stop into a silently missing probe. `_load` now re-raises `TemperatureExtrapolationError` and records every other swallowed loader failure on `self.load_errors`. Covered by `test_load_does_not_swallow_the_forbid_error`.
5. **My pre-patch proof script counted warnings instead of reading them.** astropy emits two FITS warnings, so `len(w) == 0` reported a pass for a module that says nothing about extrapolation at all. Now matched on content.

## Job 3 -- impact audit

| quantity | recorded (clamp) | drop | loglinear | drop - clamp | change |
| --- | ---: | ---: | ---: | ---: | ---: |
| n points | 588 | 495 | 588 | -93 | -15.8% |
| within-cluster radial slope | -0.4817 | -0.4499 | -0.4590 | 0.0317 | -6.6% |
| its standard error | 0.0227 | 0.0287 | 0.0217 | 0.0060 | +26.5% |
| Spearman(r/R500, residual) | -0.7871 | -0.7310 | -0.7862 | 0.0560 | -7.1% |
| rho_T = Spearman(kT, excess) | 0.6154 | 0.3287 | 0.6503 | -0.2867 | -46.6% |
| p_T (permutation) | 0.0374 | 0.2970 | 0.0260 | 0.2595 | +693.0% |
| c70 dln(excess)/dln(kT) | 0.5165 | 0.2923 | 0.5491 | -0.2243 | -43.4% |
| c70 within-cluster rho median | 0.6426 | 0.5533 | 0.6477 | -0.0892 | -13.9% |
| c70 normalisation ratio median | 1.0321 | 1.1229 | 1.0321 | 0.0909 | +8.8% |
| c71 within-cluster scatter | 0.1689 | 0.1327 | 0.1599 | -0.0362 | -21.4% |
| c71 median-ratio scatter | 0.0683 | 0.0504 | 0.0692 | -0.0178 | -26.1% |
| kappa | 156315 | 189234 | 155597 | 32920 | +21.1% |
| chi2 of the kappa fit | 41.49 | 29.82 | 42.94 | -11.68 | -28.1% |

The bench's own headline score moves too:

| mode | n X-COP | median \|err\| dex, X-COP |
| --- | ---: | ---: |
| `clamp` | 588 | 0.3894 |
| `drop` | 495 | 0.4219 |
| `loglinear` | 588 | 0.3859 |

### Per-cluster median excess

| cluster | clamp | drop | loglinear | drop - clamp |
| --- | ---: | ---: | ---: | ---: |
| A1644 | 1.6334 | 2.2979 | 1.5619 | +40.7% |
| A1795 | 2.5706 | 2.6592 | 2.5706 | +3.4% |
| A2029 | 2.6493 | 2.8684 | 2.6493 | +8.3% |
| A2142 | 2.5438 | 2.5680 | 2.5438 | +1.0% |
| A2255 | 1.5954 | 1.9282 | 1.5954 | +20.9% |
| A2319 | 2.5947 | 2.6385 | 2.5947 | +1.7% |
| A3158 | 2.1229 | 2.3316 | 2.1229 | +9.8% |
| A3266 | 2.0462 | 2.5547 | 2.0462 | +24.8% |
| A644 | 3.5225 | 3.6065 | 3.5225 | +2.4% |
| A85 | 2.4793 | 2.7305 | 2.4793 | +10.1% |
| RXC1825 | 2.2753 | 2.7148 | 2.2753 | +19.3% |
| ZW1215 | 2.5272 | 2.6355 | 2.5272 | +4.3% |

### Does the verdict change, not just the digit?

| quantity | clamp | drop | recorded verdict | verdict after dropping | changes |
| --- | ---: | ---: | --- | --- | --- |
| rho_T = Spearman(kT, excess), n=12 | 0.6154 | 0.3287 | rho = +0.615, p = 0.037 -> H2 confirmed at alpha 0.05 | rho = +0.329, p = 0.297 -> NOT significant | **YES** |
| p_T (permutation) | 0.03744 | 0.297 | p < 0.05 | p > 0.05 | **YES** |
| c70 d ln(excess)/d ln(kT) | 0.5165 | 0.2923 | +0.517 +- 0.205: 2.4s from algebraic(+1), 0.6s from pressure(+0.392) | +0.292 +- 0.205: 3.4s from algebraic, 0.5s from pressure | no |
| c70 within-cluster rho median (= power A) | 0.6426 | 0.5533 | +0.643, null fraction 0.3565 -> NOT significant | +0.553, null fraction 0.5610 -> NOT significant | no |
| c71 within-cluster scatter (= power B) | 0.1689 | 0.1327 | 0.1689, p = 0.708 -> true pairing NOT tighter | 0.1327, p = 0.734 -> true pairing NOT tighter | no |
| c71 median-ratio scatter (= power C) | 0.06825 | 0.05044 | 0.0683, p = 0.064 -> borderline, called NOT tighter | 0.0504, p = 0.072 | no |
| kappa (pressure amplitude) | 1.563e+05 | 1.892e+05 | 1.563e5, 68% [1.479e+05, 1.648e+05] | 1.892e+05, 68% [1.807e+05, 1.982e+05] | **YES** |
| within-cluster radial slope | -0.4817 | -0.4499 | -0.480: a cluster-only excess organised by radius | -0.4499: same statement, 6.6% smaller | no |
| Spearman(r/R500, RAR residual) | -0.7871 | -0.731 | -0.788 | -0.7310 | no |

* **rho_T = Spearman(kT, excess), n=12** -- the single largest verdict change in the audit
* **p_T (permutation)** -- test is correctly sized (FPR 0.040 vs nominal 0.05)
* **c70 d ln(excess)/d ln(kT)** -- the number moves 43% but the conclusion -- not algebraic, consistent with pressure -- strengthens rather than flips
* **c70 within-cluster rho median (= power A)** -- was never significant; still is not
* **c71 median-ratio scatter (= power C)** -- drop does not flip it, but LOGLINEAR does: p = 0.0487 < 0.05, so this verdict is decided by the extrapolation policy and by nothing else
* **kappa (pressure amplitude)** -- the drop-mode value lies OUTSIDE the recorded 68% interval, so the recorded interval understates the systematic
* **within-cluster radial slope** -- survives; the trend is not manufactured by the clamp
* **Spearman(r/R500, RAR residual)** -- correlations saturate; quote the slope instead (Run AT.6)

### Placebo, and what it cannot do

Dropping the clamped points moves the within-cluster slope by +0.0317. Dropping the **same number of points per cluster at random positions** moves it by -0.0030 +- 0.0115 (95% [-0.0254, +0.0183]), so the real change sits at p = 0.0050 against that null: it is not the loss of 93 points, it is *which* points.

But the second placebo is degenerate, and this matters: the clamped set IS the outermost set (True). So the drop-versus-clamp difference **cannot on its own** attribute the change to clamping rather than to radial range. That attribution requires a synthetic truth, which is Job 4.

### Calibration of the test that produced p_T

Run AT found an obvious permutation test running at FPR 0.53-0.70 against a nominal 0.05, so this one was checked before being read. Against a null with excess independent of kT at the same n = 12 and the same marginal spread, the realised FPR is **0.040 +- 0.010** over 400 draws against a nominal 0.05: correctly sized. The p-values above are therefore usable at face value -- but at n = 12 a single cluster's median excess moving by 4% takes rho_T from 0.6154 to 0.6503, which is the underpowering the register already records, now demonstrated.

## The honest reach of the X-COP relation

| r/R500 | clusters with a MEASURED temperature there |
| ---: | ---: |
| 0.70 | 12 of 12 |
| 0.80 | 10 of 12 |
| 0.90 | 7 of 12 |
| 1.00 | 1 of 12 |
| 1.10 | 0 of 12 |
| 1.52 | 0 of 12 |
| 1.90 | 0 of 12 |
| 2.50 | 0 of 12 |

* All twelve clusters have a measured temperature only out to **r/R500 = 0.78**; at least half out to **0.92**.
* **R200 = 1.52 R500: 0 of 12** clusters have a measured temperature there.
* **The a0 crossing at r/R500 = 1.9-2.5: 0 of 12.** The outermost X-COP data point of any kind sits at r/R500 = 1.519, so the crossing is beyond every X-COP point, not merely beyond every measured temperature.
* `g_bar = a0` is directly bracketed in 0 of 12 clusters within the bench's own 120-1650 kpc radial cut.

### What can honestly be said about "factor 1.4 at R200"

Fitting each cluster's excess against r/R500 using **only points with a measured temperature**, the within-cluster slope is -0.3755 +- 0.1535 (spread across clusters) and the excess at the last measured temperature radius is **1.493**. Continuing that fit to R200 -- 0.220 dex beyond the data -- gives

> excess(R200) = **1.22**, 16-84 range **[0.93, 1.54]**, with 3 of 12 clusters extrapolating below 1.0.

So X-COP does not contradict "factor 1.4 at R200", but it does not support it either: its own extrapolation centres nearer 1.2 with a 1-sigma interval that reaches below 1.0, i.e. *no excess at all*. The recorded 1.4 came from lensing; **X-COP cannot corroborate it, because X-COP measures no temperature at R200 in any of the twelve clusters.** The honest statement is that the X-COP relation is measured to **r/R500 = 0.78 in all twelve clusters and to 0.92 in half of them**, and everything beyond r/R500 = 0.91 is extrapolation.

### What can honestly be said about "crossing at r/R500 = 1.9-2.5"

Continuing the same measured-temperature fits until the excess reaches 1.0 gives r/R500 = **2.49**, 16-84 range **[1.36, 6.89]** -- 0.44 dex beyond the last measured temperature. The central value happens to land inside the recorded 1.9-2.5, but that interval spans a factor 1.32 in radius while the honest 1-sigma range spans a factor 5.1 -- **3.8 times wider than what was recorded**, and 12 of 12 clusters have no measured temperature anywhere near it. Quote the range or do not quote the number.

## Job 4 -- is clamping the right default?

A synthetic truth is built on each cluster's **real** n_e: the target excess is exactly `C0 * (r/r_mid)^s_true`, and the temperature is solved from hydrostatic equilibrium, anchored at the outer boundary by continuing that cluster's own measured outer log-slope. The synthetic temperature falls outward in **12 of 12** clusters, which is the condition the job asked for. No noise at all.

| cluster | measured dlnT/dlnr | synthetic dlnT/dlnr |
| --- | ---: | ---: |
| A1644 | -0.432 | -0.949 |
| RXC1825 | -0.233 | -0.743 |
| A3158 | -0.391 | -0.627 |
| A1795 | -0.922 | -1.016 |
| A2255 | -0.446 | -0.725 |
| A85 | -0.341 | -0.694 |
| A644 | -1.310 | -1.042 |
| ZW1215 | -0.417 | -0.363 |
| A2319 | -0.041 | -0.226 |
| A2029 | -0.765 | -1.064 |
| A2142 | -0.736 | -0.387 |
| A3266 | -0.854 | -0.849 |

Two statistics are reported because they disagree about how the clamp matters: `within` is the fixed-effects within-cluster slope this programme actually reads, `pooled_S3` is Run AT's statistic (pooled slope beyond 0.25 R500, no per-cluster level).

### `within`

| policy | bias at a FLAT truth | response d(measured)/d(true) | observed | bias as % of observed | de-biased truth |
| --- | ---: | ---: | ---: | ---: | ---: |
| `clamp` | -0.0765 | 0.8489 | -0.4817 | 15.9% | -0.4755 |
| `drop` | -0.0339 | 0.9289 | -0.4499 | 7.5% | -0.4484 |
| `loglinear` | -0.0613 | 0.8498 | -0.4590 | 13.4% | -0.4669 |
| `full_coverage` | -0.0064 | 0.9780 | -0.4817 | 1.3% | -0.4855 |
| `perfect` | -0.0042 | 0.9930 | -0.4817 | 0.9% | -0.4809 |

Flat-truth bias -0.0765 decomposes as `np.gradient` discretisation -0.0042 (5.4%), coarse temperature grid at **full** radial coverage -0.0022 (2.9%), and **the clamp -0.0702 (91.7%)**.

### `pooled_S3`

| policy | bias at a FLAT truth | response d(measured)/d(true) | observed | bias as % of observed | de-biased truth |
| --- | ---: | ---: | ---: | ---: | ---: |
| `clamp` | -0.0795 | 0.7324 | -0.4796 | 16.6% | -0.5395 |
| `drop` | +0.0401 | 0.9947 | -0.4054 | -9.9% | -0.4438 |
| `loglinear` | -0.0536 | 0.7307 | -0.4401 | 12.2% | -0.5235 |
| `full_coverage` | +0.0107 | 0.8911 | -0.4796 | -2.2% | -0.5491 |
| `perfect` | -0.0054 | 0.9091 | -0.4796 | 1.1% | -0.5217 |

Flat-truth bias -0.0795 decomposes as `np.gradient` discretisation -0.0054 (6.8%), coarse temperature grid at **full** radial coverage +0.0161 (-20.3%), and **the clamp -0.0902 (113.5%)**.

`full_coverage` uses a coarse grid with the same number of bins and the same log spacing but covering the whole radial range, so it isolates coarsening from extrapolation; `perfect` uses the true fine-grid temperature and isolates `np.gradient` alone.

### Which policy is least biased

On the statistic this programme reads, ordered by |bias| at a flat truth: **`drop`** -0.0339, **`loglinear`** -0.0613, **`clamp`** -0.0765.

`drop` is the least biased of the three (-0.0339, 7.5% of the observed slope) and has much the best response (0.929 against 0.849 for the clamp), i.e. it both mis-states the zero point least and tracks a real signal most faithfully. `loglinear` sits between them and buys back only 20% of the clamp's bias, because continuing a slope fitted on three noisy outer bins is itself an assumption. **Clamping is the worst of the three and should not be the scientific default** -- it stays the code default only so that no recorded number moves without someone choosing it.

### How much of Run AT's 29% is this bug

Run AT's noiseless forward null was reproduced **bit-identically** (S1 -0.2067, S3 -0.1359; recorded -0.2067 and -0.1359). One switch was then flipped inside Run AT's own machinery: the simulated cluster is observed on a temperature grid extended outward to the last density bin (37 bins added across the sample), so nothing is ever extrapolated. The truth, the boundary pressure, the R500 inference and the statistics are the same code.

| | S1 (correlation) | S3 (slope) |
| --- | ---: | ---: |
| observed on real data | -0.7884 | -0.4803 |
| total pipeline bias | -0.2067 | -0.1359 |
| of which **the clamp** | -0.1522 | -0.0873 |
| of which everything else | -0.0546 | -0.0487 |
| bias as % of observed | 26.2% | 28.3% |
| **clamp share of that bias** | **73.6%** | **64.2%** |
| clamp as % of observed | 19.3% | 18.2% |

So of the 29% that Run AT attributed to the pipeline, **64% is this bug** and 36% is everything else -- chiefly the R500 inference step, whose recovery ratio improves from 0.9186 to 0.9887 once the clamp is removed. **The clamp was also biasing the inferred R500 low by 8%**, which is a second, unrecorded consequence of the same defect.

## What this audit did NOT establish

* **Whether the residual trend is physics or hydrostatic bias.** Removing the clamp reduces the pipeline's flat-truth bias but does not separate modified gravity from an outward-rising non-thermal pressure. X-ray data alone still cannot.
* **Whether `loglinear` is right.** It continues a slope fitted on three outer bins with no uncertainty propagated into the downstream statistics. It is offered as a sensitivity, not a recommendation.
* **CLASH.** Its binned table does not pass through `_cluster_profile` and was not touched here; Run AT already flagged it as the open half of that audit.
* **The re-implementations.** 9 files carry their own bare `np.interp` on the same table, in the same direction: `c70_endogeneity.py`, `c71_amplitude.py`, `power_analysis.py`, `m10_xcop_temperature.py`, `m12_redo_correct.py`, `p01_rigorous.py`, `ingest.py`, `nullsim.py`, `run_job2d.py`. Their exposure is quantified above through the recomputed statistics, but the fix was applied only to the shared bench; patching each lane individually is a separate job. `r500-audit/nullsim.py` is a deliberate reproduction of the bug for Run AT's forward null and should keep it.
* **Provenance.** `repro/inputs_c70.json` and `repro/inputs_c71.json` pin a 20,139-byte `invariant_bench.py` at sha256 `fe817b22...` that lived in a scratchpad and no longer exists; the surviving repo copy was 19,707 bytes at `00cfbf28...` before this patch. Those receipts already pinned a file that cannot be checked, and this patch changes the repo file's hash to `bed02ff4...`. They need a legitimate reseal, not a hash edit.
* **A standing seal hazard, pre-existing.** `Bench.__init__` calls `_widebin()`, which returns hard-coded El-Badry boosts from the source itself, so any bare `Bench()` loads a sealed probe. Nothing in this lane constructs a Bench without stubbing `_kids` and `_widebin` first.

---

Rendered programmatically from `results.json` by `report.py`. Audit runtime 25.6 s.
