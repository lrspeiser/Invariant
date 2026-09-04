# env-data lane -- report

Lane: `work/wellnet-2026-09/env-data/`
Programme brief: `work/wellnet-2026-09/BRIEF.md`

**Sealed holdouts: KiDS and wide binaries were never loaded, listed, queried or
referenced by this lane.** Nothing here touches them.

Two tests were commissioned, both aimed at adding a measurement direction that
the SPARC rotation-curve bench cannot supply. Test 1 asks whether the internal
gravitational relation of a galaxy depends on the depth of the external
potential it sits in. Test 2 asks for systems in which two nearly perpendicular
directions of the field are measured in the *same* baryonic system.

---

## Headline

**Test 1 was built and is real. In MaNGA alone its strictest form is
underpowered; adding SAMI fixes that.** 10,071 unique MaNGA DR17 galaxies were
cross-matched against the Tempel+2014 and Tempel+2017 SDSS friends-of-friends
group catalogues and the MCXC X-ray cluster meta-catalogue, and matched
field/cluster pairs were built at seven declared tiers. The strictest MaNGA tier
yields only **23 pairs**; the largest defensible one yields **281**, and
the X-ray-confirmed tier -- the one whose environment rests on a direct
observable -- yields **218**. Resolved kinematics (DAP `MAPS`) were downloaded
for all 645 galaxies appearing in any tier, 4.99 GB.

**The decisive addition is SAMI DR3, which deliberately observed eight rich
clusters where MaNGA observed none on purpose.** The identical build on SAMI
yields **108 morphologically clean pairs** (5.7x MaNGA's primary tier) at a
higher median host dispersion (690 versus 605 km/s) and deeper inside the
potential (0.46 R200 versus 0.77 R_vir), plus **364** disk-bearing pairs.
With resolved rotation curves the clean SAMI tier reaches a 3-sigma sensitivity
of 0.022 dex, about 5% in velocity -- inside the range where a plausible
external-field effect at |g_ext| ~ 0.2 a_0 would live. The two surveys are kept
in separate files and must not be pooled without a cross-calibration.

**The gas-matched tier yields zero pairs, and that is a physical result, not a
catalogue gap.** Neutral hydrogen is stripped out of cluster galaxies: among
MaNGA galaxies covered by HI-MaNGA in hosts with sigma_v >= 400 km/s, 17 of 494
are HI detections (14 of 156 among late types), against 572 of 1603 in the field
arm. The gas fraction cannot be matched across the two arms. The environment
whose effect is under test has destroyed one of the control variables.

**Two of the five matching variables are the same variable.** `log Sigma_b` and
`log g_bar(2.2 R_d)` correlate at **r = 0.9955** in the MaNGA build and
at exactly 1 in the SAMI build. That is arithmetic, not coincidence: for a
razor-thin exponential disk evaluated at a fixed multiple of R_d, the Freeman
formula makes g_bar equal to Sigma_b times a pure constant, and MaNGA departs
from unity only through its bulge term. `f_gas` correlates with `log M_star` at
r = -0.778. The effective number of independent matching directions is
about **three**, not five, and the result must be stated that way.

**Two silent bugs were found and fixed, both by validating against something
whose answer was known in advance.** (i) The X-ray confirmation flag matched the
Tempel *group centre* to an MCXC cluster within 10 arcmin, and Coma's group
centre sits 12.4 arcmin from its X-ray peak -- so the flag was failing on
precisely the richest systems. Re-flagging on the *galaxy* raised the X-ray
sample from 222 to 978 galaxies across 92 clusters. (ii) The 103 galaxies
observed by both MaNGA and SAMI disagreed in stellar mass by -0.308 dex, a
factor of two. The cause is that NSA quantities are computed with **h = 1**
while this lane works at H0 = 70; the required rescaling is +0.3098 dex. After
correcting, the offset is **+0.002 dex**. Before the fix every baryonic quantity
here -- M_b, Sigma_b, g_bar, f_gas -- was 0.31 dex low, which for a programme
built on the radial acceleration relation is not a rounding error. Neither bug
raised an error, a warning, or an implausible number.

**A separate result the task note asked for: a genuinely resolved sigma_LOS(R)
profile now exists for 240 near-face-on MaNGA disks**, 4-9 radial points
each, 1671 radial points in total, median formal error 1.02 km/s.
DiskMass VI/VII gives an exponential *fit* for 30 galaxies with an *inferred*
scale height; this gives measured radial profiles for 240. Its limits are
stated below and they are real.

---

## Test 1 -- galaxies in clusters as controlled experiments

### What was acquired

| product | rows | source |
|---|---|---|
| MaNGA DRPall v3_1_1 | 11,273 | `data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/` |
| MaNGA DAPall 3.1.0, HYB10-MILESHC-MASTARSSP | 10,782 | `.../spectro/analysis/v3_1_1/3.1.0/` |
| MaNGA PyMorph DR17 photometry (g, r, i) | 10,293 per band | `.../photo/pymorph/1.1.1/` |
| MaNGA deep-learning morphology DR17 | 10,293 | `.../morphology/deep_learning/1.1.1/` |
| MaNGA visual morphology 2.0.1 | 10,126 | `.../morphology/manga_visual_morpho/2.0.1/` |
| HI-MaNGA v2_0_1 | 6,632 rows / 6,442 galaxies | `.../manga/HI/v2_0_1/` |
| GEMA 2.0.2 environment VAC | 9,670 / 10,086 / 3,287 | `.../manga/gema/2.0.2/` |
| Tempel+2014 SDSS DR10 galaxies | 588,193 | VizieR `J/A+A/566/A1/galaxies` |
| Tempel+2014 groups | 82,458 | VizieR `J/A+A/566/A1/groups` |
| Tempel+2017 SDSS DR12 galaxies | 584,449 | VizieR `J/A+A/602/A100/table1` |
| Tempel+2017 groups | 88,662 | VizieR `J/A+A/602/A100/table2` |
| MCXC X-ray clusters | 1,743 | VizieR `J/A+A/534/A109/mcxc` |
| DAP `MAPS` cubes | 902 files, 5.30 GB | per galaxy, DR17 |

Every row count was asserted in code against the number stated by the source
paper or data model; a mismatch aborts the ingest. Every file has a sibling
`.manifest.json` with source URL, retrieval timestamp, SHA-256, byte size, row
and column counts, column names with units, and the exact query issued.

### The joined sample

`clean/manga_env_master.csv` -- 10,071 rows x 186 columns, one row per unique
MaNGA galaxy after quality cuts (`srvymode == 'MaNGA dither'`, `0.001 < z < 0.2`,
DRP3QUAL CRITICAL bit clear) and deduplication of repeat observations by
`mangaid`. Of these: 7,814 cross-match to Tempel+2014 within 3 arcsec and
|dz| < 0.002; 5,331 sit in a group and 2,483 are friends-of-friends singletons;
587 sit in hosts with sigma_v >= 500 km/s; 222 sit in hosts matching an MCXC
X-ray cluster within 10 arcmin and dz < 0.01.

### Host potential depth: what is an observation and what is not

**Usable as observations:** the host member rms velocity `sigma_v`
(Tempel+2014, from member redshifts alone); the projected clustercentric radius
(sky geometry times angular-diameter distance); the host X-ray luminosity `L500`
(MCXC). The potential-depth proxy is `sigma_v^2` and the external-field proxy is
`sigma_v^2 / R_proj`. Both are built from member kinematics and geometry only --
no mass model, no NFW profile, no dark matter.

**Ranking only, dark-matter dependent:** `MNFW` (Tempel+2014, assumed NFW),
`M200` (Tempel+2017, assumed NFW), `M500` and `R500` (MCXC, from an L-M scaling
relation calibrated on hydrostatic masses). Every one carries the literal suffix
`_rank_only` in its column name throughout this lane so it cannot be picked up
as an observable by accident.

The one soft spot is `R_vir`, Tempel's projected harmonic mean radius. It is
computed from member positions so it is geometric, but calling it a *virial*
radius imports an equilibrium assumption. It is used only to define the "inside
the cluster" gate, never as a measurement.

### Matched pairs

Full detail, including the achieved tolerance on every matching variable in
every tier, is in `MATCHED_PAIRS.md`.


| tier | morphology | environment | pairs |
|---|---|---|---|
| `B1_primary` | deep-learning late type | host sigma_v>=400 km/s, Ngal>=10, R_proj/R_vir<=1.0 | **23** |
| `A1_gas_matched` | deep-learning late type | host sigma_v>=400 km/s, Ngal>=10, R_proj/R_vir<=1.0 | **0** |
| `B2_disk_strict` | disk-bearing (S0 or later) | host sigma_v>=400 km/s, Ngal>=10, R_proj/R_vir<=1.0 | **121** |
| `B3_late_wide` | deep-learning late type | host sigma_v>=300 km/s, Ngal>=5, R_proj/R_vir<=1.5 | **80** |
| `B4_disk_wide` | disk-bearing (S0 or later) | host sigma_v>=300 km/s, Ngal>=5, R_proj/R_vir<=1.5 | **281** |
| `C1_xray_late` | deep-learning late type | galaxy lies within 2 Mpc projected of an MCXC X-ray peak at |dz|<0.01 (L500 is a direct observable), and R_proj/R_vir<=1.5 of its Tempel host | **61** |
| `C2_xray_disk` | disk-bearing (S0 or later) | galaxy lies within 2 Mpc projected of an MCXC X-ray peak at |dz|<0.01 (L500 is a direct observable), and R_proj/R_vir<=1.5 of its Tempel host | **218** |

Declared tolerances: 0.10 dex on log M_star, 0.10 dex on log R_d, 0.15 dex on
log Sigma_b, 0.10 dex on log g_bar(2.2 R_d), 0.10 absolute on f_gas, plus
nuisance controls of 10 deg on inclination, 0.15 on B/T and 0.010 on redshift.
Every tolerance is met by construction (a hard box); the achieved rms sits well
inside it. Primary tier:


| matching variable | declared tol | max abs(delta) | rms delta |
|---|---|---|---|
| `logMstar_nsa` | 0.1 | 0.0992 | 0.0451 |
| `logRd` | 0.1 | 0.0943 | 0.0403 |
| `logSigma_b` | 0.15 | 0.0984 | 0.0647 |
| `log_gbar_2p2Rd` | 0.1 | 0.0989 | 0.0639 |
| `incl_deg` | 10 | 9.3346 | 5.5907 |
| `pym_r_BT_SE` | 0.15 | 0.1424 | 0.0888 |
| `z` | 0.01 | 0.0088 | 0.0041 |

`B1_primary` is the primary sample. The other tiers were declared after seeing
that the primary cluster arm held only 48 galaxies -- a sample-size observation,
not a residual observation, so blindness is intact -- and they are reported
separately and must not be merged.

### Environmental contrast actually achieved

For the cluster members of the primary tier: host sigma_v spans 407 to
840 km/s, projected radius 0.23 to 0.97 R_vir, and the
external-field proxy `|g_ext|/a_0 = (sigma_v^2/R_proj)/a_0` spans **0.08
to 0.77**, median 0.17. Across the largest tier it spans
0.028 to 2.16, a factor of 77. This is a real external
field: at |g_ext| ~ 0.2 a_0 an algebraic-MOND external-field effect is predicted
to suppress the internal boost at the several-per-cent level, so the sample sits
in the regime where the effect exists rather than one where it is negligible by
construction.

The angle between the disk normal and the direction to the host centre spans
about 12 to 90 degrees with median near 60 degrees in every tier, so an angular
test is possible in principle. But **this angle is not the 3-D angle.** The
line-of-sight offset between a galaxy and its host centre is not measurable
(redshift differences are dominated by peculiar velocity) and the near side of a
disk is not determined by the photometry, so what is reported is
`psi = arccos( sin(i) |sin(PA_host - PA_disk)| )`, computed assuming the offset
lies in the plane of the sky and folded to [0,90] deg. It is a projected bound,
not the angle itself. Any alignment test must propagate that.

### Is the sample large enough? -- the honest answer

Scatter was measured **on the field arm alone**, so the cluster-versus-field
contrast this sample exists to measure has not been looked at anywhere in this
lane.

The stellar-mass Tully-Fisher residual scatter in the field arm, using the DAP
summary velocity half-range as a rotation proxy, is **sigma = 0.141 dex**
(n = 1081). That proxy is crude; a proper tilted-ring fit to the `MAPS`
cubes should reach the literature sTFR scatter of about 0.055 dex. Both
cases are quoted.


| tier | N | 3-sigma detectable, DAP proxy | 3-sigma detectable, resolved RC |
|---|---|---|---|
| `B1_primary` | 23 | 0.125 dex | 0.049 dex |
| `B2_disk_strict` | 121 | 0.054 dex | 0.021 dex |
| `B3_late_wide` | 80 | 0.067 dex | 0.026 dex |
| `B4_disk_wide` | 281 | 0.036 dex | 0.014 dex |
| `C1_xray_late` | 61 | 0.077 dex | 0.030 dex |
| `C2_xray_disk` | 218 | 0.041 dex | 0.016 dex |

Pairs needed for a 3-sigma detection of a given mean offset:

| effect size | percent in V | with the DAP proxy | with resolved rotation curves |
|---|---|---|---|
| 0.02 dex | 4.7% | 896 | 137 |
| 0.03 dex | 7.2% | 399 | 61 |
| 0.05 dex | 12.2% | 144 | 22 |
| 0.10 dex | 25.9% | 36 | 6 |

**Verdict.** The primary 23-pair sample can only detect a mean shift of
about 0.049 dex (12% in velocity) at 3 sigma even with perfect
rotation curves. That is larger than what a plausible potential-depth or
external-field theory would produce at |g_ext| ~ 0.2 a_0, so **the primary
sample is underpowered for its intended purpose.** The 281-pair
`B4_disk_wide` tier reaches 0.014 dex (3.3% in velocity) with
resolved rotation curves, which *is* in the interesting range -- but it buys
that power by admitting S0s and by widening the environment gate to
sigma_v >= 300 km/s, which lowers the median external field from 0.24 to
0.12 a_0. There is a genuine trade between statistical power and
environmental contrast, and MaNGA alone does not resolve it.

The best-founded tier is neither of those. `C2_xray_disk` has **218 pairs**
whose cluster members are confirmed to sit inside an X-ray emitting
intracluster medium -- environment evidence that rests on a direct observable
(L500) rather than on friends-of-friends bookkeeping -- at a median external
field of 0.13 a_0, and reaches **0.016 dex (3.7% in
velocity)** at 3 sigma with resolved rotation curves. Its late-type-only
counterpart `C1_xray_late` has 61 pairs.

The correct conclusion: **MaNGA can bound a potential-depth effect at the
few-per-cent level in velocity, but only in the larger tiers, and only after
rotation curves are actually fitted from the MAPS cubes.** Reaching the
same power in the high-contrast, morphologically clean tier needs a survey that
deliberately targeted clusters -- which is why SAMI was added.

### Caveats a downstream analysis must carry

1. **M_b is stellar-only for the cluster arm.** With HI undetected in 97% of
   cluster galaxies, `M_b` there is `M_star` plus nothing. HI upper limits are
   carried per galaxy (`cl_logMHI_limit`) so the omitted gas can be bounded, but
   it is not measured. Molecular gas is not available at all.
2. **R_d is a half-light radius divided by 1.678.** PyMorph's `A_HL_SE_DISK` is
   explicitly documented as the disk half-light semi-major axis, *not* the scale
   length. The conversion is exact only for a pure exponential; `N_SE_DISK` is
   carried so departures can be checked.
3. **Inclination comes from an axis ratio with an assumed intrinsic thickness**
   q0 = 0.20. For S0s in the disk-bearing tiers this is a worse assumption than
   for late-type disks.
4. **The environment axes are not independent.** `R_proj` appears in both
   `gext_proxy` and `R_over_Rvir`; redshift enters the distance scaling of both
   `R_d` and `R_proj`, so a distance error moves them together.
5. **Tempel group membership is projection-contaminated.** Friends-of-friends in
   redshift space assigns interlopers. Tempel+2017 is carried as an independent
   membership determination for exactly that reason; 7,814 and 7,841 galaxies
   match the two catalogues respectively and the disagreements are visible in
   the master table.

### A bug found by validating against known clusters, and fixed

The X-ray confirmation flag was first built by matching the **Tempel group
centre** to an MCXC cluster within 10 arcmin. Validating the pipeline against
clusters whose velocity dispersion is known independently
(`code/validate_env_against_known_clusters.py`) showed Coma recovered correctly
on kinematics -- 306 MaNGA galaxies, host sigma_v 840 km/s against a literature
value near 1000, richness 680 -- but carrying **no X-ray flag at all**. The
cause: Coma's Tempel luminosity-weighted centre sits **12.4 arcmin** from the
MCXC X-ray peak, just outside the 10 arcmin window. The flag was therefore
failing on exactly the richest and most valuable systems, silently.

The fix was to flag on the **galaxy** rather than the group centre: a galaxy is
X-ray confirmed if it lies within 2 Mpc projected of an MCXC peak at
|dz| < 0.01, which is the physically meaningful statement "this galaxy sits in
an X-ray emitting intracluster medium". That raises the flagged sample from 222
galaxies to **978 across 92 distinct clusters**, and the X-ray-confirmed pair
tiers from 21 and 64 pairs to **61 and 218**. Both flags are retained in
the master table (`t14_mcxc_*` for the group-centre version, `xray_*` for the
galaxy-centred one) so the difference is auditable.

This is worth recording as a methodology point: the bug produced no error, no
warning and no implausible number. It was only visible because the pipeline was
checked against a system whose answer was known in advance.

### A second bug, found the same way: the NSA h = 1 convention

103 galaxies were observed by **both** MaNGA and SAMI. Comparing them
(`code/crosscal_manga_sami.py`) showed the redshifts agreeing to a median of
0.0000 -- so the cross-match is right -- while the stellar masses disagreed by
**-0.308 dex**, a factor of two. MaNGA was low.

The cause is a units convention. The SDSS data model states that
`NSA_ELPETRO_MASS` and `NSA_ELPETRO_ABSMAG` are computed with
(Om=0.3, OL=0.7, **h=1**): masses in h^-2 Msun, absolute magnitudes on the h=1
distance scale. This lane works at H0 = 70. The required rescaling is
`-2 log10(0.7) = +0.3098` dex in mass and `5 log10(0.7) = -0.7745` mag, and the
measured offset was -0.308 dex. The agreement to 0.002 dex identifies the cause
beyond reasonable doubt.

After applying the correction the MaNGA-minus-SAMI stellar-mass offset is
**+0.002 dex**. Every baryonic quantity in this lane -- M_b, Sigma_b,
g_bar(2.2 R_d), f_gas -- was 0.31 dex low before the fix.

What this did and did not affect:

- **Matched-pair differences in log M_star: unaffected.** A constant offset
  applied to both arms cancels exactly in the difference, so the matching
  tolerances were never violated.
- **f_gas: genuinely wrong before the fix**, because M_HI carries no such
  convention. The gas fraction was over-stated by roughly a factor of two.
- **Sigma_b and g_bar: shifted, and not by a constant**, because M_b = M_star +
  M_gas mixes an h-scaled term with an unscaled one. Pair counts moved slightly
  as a result (the primary tier from 19 to 23).
- **Any absolute comparison to a_0: was wrong by 0.31 dex.** For a programme
  whose central object is the radial acceleration relation, that is not a
  rounding error.

The residual R_d offset between the two surveys is **+0.179 dex** and is *not* a
bug: PyMorph's `A_HL_SE_DISK` is the half-light radius of the **disk component**
of a bulge+disk decomposition, while SAMI's `ReMGE` is a **total-light**
effective radius. They are different quantities and should differ. Anyone
pooling the two tables must correct for it; the measured value is in
`clean/manga_sami_crosscal_summary.json`.

### End-to-end validation of the baryonic bookkeeping

A units bug does not raise an exception; it produces plausible numbers. After
the h correction, `code/sanity_physical.py` checks every derived baryonic
quantity against the range known a priori for disk galaxies, on the 3609
late-type disks with 25 < i < 75 deg:

| quantity | p10 | p50 | p90 |
|---|---|---|---|
| log10 M_star [Msun] | 9.36 | 10.19 | 10.97 |
| R_d [kpc] | 1.34 | 2.80 | 5.66 |
| Sigma_b [Msun/pc^2] | 115 | 399 | 1189 |
| f_gas | 0.19 | 0.44 | 0.73 |
| g_bar(2.2 R_d) / a_0 | 0.15 | 0.51 | 1.52 |
| V_bar(2.2 R_d) [km/s], baryons only | 54 | 112 | 189 |
| V_obs [km/s], DAP proxy | 81 | 157 | 276 |

All inside range. The sharper check is the last two rows together: the median
observed-to-baryonic velocity ratio is **1.40**, and the radial acceleration
relation predicts **1.40** at the sample's median g_bar of 0.51 a_0. The lane's
independently computed baryonic accelerations put the galaxies exactly where the
RAR says they should sit.

That is a consistency check on the bookkeeping, not evidence for or against any
gravity law -- the RAR is the programme's incumbent and reproducing it is the
minimum bar, not a result. Its value here is diagnostic: before the h fix,
g_bar/a_0 sat at 0.25 while V_obs was unchanged, so the same comparison would
have demanded a boost of 1.55 against an observed 1.40. The disagreement would
have been visible had it been looked for, which is the argument for computing
it routinely.

### A source that did not contain what the brief assumed

The **GEMA 2.0.2** environment VAC was acquired on the expectation that it would
supply host halo identification, clustercentric radius, host velocity dispersion
and potential depth. **It does not.** What it actually contains is 15 binary
tables of *local* environment statistics: tidal strength `Q` at 1 Mpc and 5 Mpc
apertures and several magnitude limits, nearest-neighbour distances, a group
tidal strength `Q_group` with `GroupSize`, an overdensity to the 5th nearest
neighbour (only 3,287 galaxies), and a large-scale-structure table with tidal
tensor eigenvalues `t1,t2,t3` and major/minor axis directions. There is no
clustercentric radius, no host velocity dispersion and no host identifier.
Worse, the VOTable in its primary HDU carries **no `DESCRIPTION` and no `unit`
attribute for any field in the tables this lane needs**, so the physical meaning
and units of `Q_nn`, `Q_group`, `mh`, `den1-3` and `t1-3` cannot be recovered
from the file alone. GEMA is therefore carried as auxiliary columns and is *not*
used for the environment definition; Tempel+2014 supplies that. The LSS
tidal-tensor columns are, however, exactly the tidal-tensor information Test 1
asks about, and are retained for that reason with the units caveat attached.

---

## Failure modes from the brief -- explicitly checked

Machine-readable verdicts in `clean/checks.json`.


| failure mode | verdict | what was checked |
|---|---|---|
| Shared-denominator artefacts | **PASS** | No galaxy-internal measurement enters any environment variable, and no environment variable enters any internal variable. Overlap of the two column sets: []. Environment provenance: {'t14_grp_sigma_v': 'member redshifts of the host group only', 't14_Rproj_kpc': 'sky separation to the group centre times the angular-diameter distance of the galaxy', 'R_over_Rvir_t14': "the above divided by the group's projected harmonic radius", 'gext_proxy_ms2': 'sigma_v^2 / R_proj', 't14_mcxc_L500_1e44': 'ROSAT/other X-ray luminosity of the host', 't14_Ngal': 'FoF richness of the host'}. NOTE for the analysis lane: R_proj appears in BOTH gext_proxy and R_over_Rvir, so those two are NOT independent environment axes (Pearson r = -0.180 on the cluster arm); and the redshift z enters both the distance scaling of R_d and of R_proj, so a distance error moves both together. |
| Monotone-invariant statistics | **PASS** | median \|g_ext\|/a_0 versus the sigma_v threshold: sigma_v>=300: N=591, g_ext/a0=0.210; sigma_v>=350: N=434, g_ext/a0=0.238; sigma_v>=400: N=357, g_ext/a0=0.255; sigma_v>=500: N=281, g_ext/a0=0.272; sigma_v>=600: N=244, g_ext/a0=0.283; sigma_v>=700: N=172, g_ext/a0=0.337; sigma_v>=800: N=162, g_ext/a0=0.333. Spread over the tested range = 0.127 in units of a_0 (61% of the smallest value), so d(statistic)/d(threshold) != 0 and the environment ranking is not degenerate. |
| Environment contrast dynamic range | **PASS** | In the largest tier the cluster-side \|g_ext\|/a_0 spans 0.028 to 2.161 (factor 77.2), 10th-90th percentile 0.051 to 0.442. An external-field effect of any monotone form must vary across this range; if a headline statistic does not, that is a bug, not a null. |
| Refitting on the held-out set / blind protection | **PASS** | Matching space = ['f_gas_or_nan', 'incl_deg', 'logMstar_nsa', 'logRd', 'logSigma_b', 'log_gbar_2p2Rd', 'pym_r_BT_SE', 'z']. Kinematic columns in it: []. The field/cluster split uses only sigma_v, richness and R_proj/R_vir. The only kinematic quantity read anywhere in the pair builder is the field-arm Tully-Fisher scatter used for the power calculation, and the cluster arm is excluded from that fit, so the contrast under test has not been looked at. |
| Silent extraction failures | **PASS** | Every ingest asserts its row count against the number stated by the source (mismatches: none). The VizieR reader additionally asserts that the response carries #Table and #Column lines and that the header names match the #Column declarations one for one, which is what catches VizieR's HTTP-200-generic-page failure. MAPS download: 902 of 882 files retrieved, 0 failures. |
| Dark matter used as an observation | **PASS** | Dark-matter-dependent columns, all suffixed `_rank_only`: ['t14_grp_MNFW_rank_only', 't17_grp_M200_rank_only', 't14_mcxc_M500_rank_only', 't14_mcxc_R500_rank_only', 't17_mcxc_M500_rank_only', 't17_mcxc_R500_rank_only', 'xray_M500_rank_only', 'xray_R500_rank_only', 'xray_R_over_R500_rank_only']. None of them enters any matching variable, the quality gate, the field/cluster split, or any derived baryonic quantity. The environment ranking uses sigma_v (member kinematics) and L500 (X-ray), both observables. |
| Test bugs that look like solver bugs | **N/A** | No PDE solver is exercised in this lane; this failure mode belongs to the gravitylab solver lanes. |
| Non-monotonic M(r) in deprojection | **N/A** | No lensing deprojection is performed in this lane. The only lensing-adjacent quantity touched is MCXC L500, an X-ray luminosity. |

---

## Test 1, second survey -- SAMI DR3

MaNGA was not a cluster survey.  SAMI deliberately observed eight rich clusters,
so the same build on SAMI DR3 produces a far larger cluster arm at the same
morphological purity.  SAMI DR3 was pulled anonymously from Data Central's IVOA
TAP service; all 14 DR3 catalogues plus the Owers+2017 cluster table were
acquired, row-count-asserted, and merged into a 3068 x 81 master inventory.


| tier | morphology | pairs | cluster arm | field arm | 3-sigma detectable, resolved RC |
|---|---|---|---|---|---|
| `S1_latetype` | spirals (morph_type >= 2.0), 20 < i < 80 | **108** | 160 | 1197 | 0.0225 dex |
| `S2_diskbearing` | S0 and later (morph_type >= 1.0), 20 < i < 80 | **364** | 449 | 1552 | 0.0122 dex |

**SAMI's clean tier is 5.7x larger than MaNGA's and sits deeper in the
potential.**  `S1_latetype` has 108 pairs against MaNGA's primary 23,
with host sigma_v from 492 to 1002 km/s (median 690,
versus MaNGA's 605 median) and a median projected radius of 0.46 R200
(MaNGA's primary tier sits at 0.77 R_vir).  Median external field
0.17 a_0.  Hosts: APMCC 0917, Abell 119, Abell 168, Abell 2399, Abell 3880, Abell 4038, Abell 85, EDCC 442.

Sensitivity with resolved rotation curves: **0.0225 dex (5.3% in
velocity)** for the 108-pair clean tier and **0.0122 dex (2.9%)**
for the 364-pair disk-bearing tier.  Both are inside the range where a
plausible external-field effect at |g_ext| ~ 0.2 a_0 would live.  This is the
single biggest improvement in the lane: **the strict-morphology arm of Test 1
is underpowered in MaNGA and adequately powered in SAMI.**

Three caveats, all real:

1. **The two surveys must not be pooled.**  Stellar masses, effective radii and
   morphologies come from different pipelines.  A zero-point offset between them
   would appear as a field/cluster signal if the surveys contributed unequally
   to the two arms.  They are kept in separate files for that reason.
2. **SAMI's cluster arm has no Sersic index**, because the cluster structural
   fits (Owers et al. 2019) publish no per-galaxy table.  The SAMI match
   therefore uses MGE photometry -- which IS homogeneous across both arms -- and
   does **not** control bulge fraction, unlike the MaNGA build which matches B/T
   to 0.15.  That is a genuine weakening.
3. **No gas fraction at all.**  SAMI has no HI counterpart in this lane, so
   f_gas is neither matched nor bounded.  The MaNGA build at least carries HI
   upper limits.

**A shared-denominator hazard in the SAMI environment columns**, inherited from
the source and flagged during acquisition: `R_on_rtwo = R_proj / R200` and
`R200 = 0.17 sigma_200 / H(z)`, so `R/R200` and `v_pec/sigma_200` both carry
sigma_200 in the denominator.  Putting sigma_200 or M200 on the other axis
reproduces exactly the structure that retracted rho_p = -0.304.  Two sigma-free
columns are therefore carried and are what the external-field proxy is built
from: `R_proj_Mpc_from_cat` and `v_pec_kms`.

### The collinearity is exact, not merely tight

In the MaNGA build `log Sigma_b` and `log g_bar(2.2 R_d)` correlate at r = 0.996.
In the SAMI build they correlate at **exactly 1** -- the achieved max and rms
tolerances are identical to four decimal places.  The reason is arithmetic
rather than empirical: for a razor-thin exponential disk evaluated at a fixed
multiple of R_d, the Freeman formula gives
`g_bar(2.2 R_d) = 4 pi G Sigma_0 R_d y^2 [I0 K0 - I1 K1] / R` with y = 1.1
fixed, and `Sigma_b = Sigma_0`, so g_bar is Sigma_b times a pure constant.  The
0.996 in MaNGA is a departure from unity caused only by the bulge term.

So the brief's fourth and fifth matching variables are **the same measurement**
whenever the baryons are modelled as a single exponential disk.  Stating five
independently satisfied tolerances would misrepresent how completely the
internal structure has been controlled.


---

## The resolved vertical dispersion profile

`clean/manga_faceon_sigma_profiles.csv` -- 1671 radial points across
**240 near-face-on MaNGA disks**, 4-9 points per galaxy.

Selection, structural only: inclination < 30 deg from the PyMorph r-band axis
ratio with q0 = 0.20; deep-learning late type; DAPQUAL clean; PyMorph
`FLAG_FIT != 3`; `STELLAR_SIGMA_1RE > 50` km/s; median r-band S/N > 5.

Method: one entry per independent Voronoi bin, deduplicated on the
stellar-continuum `BINID` channel so radial bins are not inflated by spaxel
repetition; astrophysical dispersion
`sqrt(STELLAR_SIGMA^2 - STELLAR_SIGMACORR^2)` with errors propagated from the
DAP inverse variance; `STELLAR_SIGMA_MASK != 0` dropped; radii from the DAP
elliptical polar radius map.

Results: sigma_LOS spans 34.8 to 237.5 km/s, median 67.4, with
median formal error **1.02 km/s**. The profiles decline outward in
81% of galaxies with median d log sigma / d log R = -0.20
(16-84% range -0.39 to 0.02) -- the expected behaviour for a disk, and
a sanity check that the extraction is not returning noise.

**What this is and is not.**

- It **is** a radially resolved, per-galaxy line-of-sight dispersion profile for
  240 disks, against DiskMass VI/VII's exponential *fit* (a central value
  plus a scale length in arcsec) for 30 galaxies. On radial resolution it is a
  clear upgrade.
- It is **not** sigma_z. At inclination i the in-plane components leak in at
  order sin^2 i, which is under 0.25 here but not zero. Converting sigma_LOS to
  sigma_z requires an assumed sigma_R/sigma_z, which is a model. The measured
  quantity is reported; the conversion is left to whoever wants to state that
  assumption explicitly.
- It does **not** supply a scale height and does **not** supply Sigma_dyn. The
  DiskMass complaint that h_z is inferred from h_R is not fixed here; it is
  sidestepped, because no scale height is claimed at all.
- **Resolution honesty.** MaNGA's stellar instrumental resolution corresponds to
  sigma_inst of roughly 70 km/s, and the DAP is increasingly systematics-limited
  below about 50 km/s. `STELLAR_SIGMACORR` is only the template-versus-data
  resolution difference (median 24 km/s here), *not* the full
  instrumental sigma, so a ratio test against it is weak and must not be read as
  "resolved". The honest absolute figures: **92% of radial points exceed
  50 km/s and 46% exceed 70 km/s; 171 of 240 galaxies have every
  radial point above 50 km/s and 45 have every point above 70 km/s.** For
  the coldest disks -- exactly the ones where the vertical force is most
  interesting -- MaNGA is at or below its limit, which is precisely why DiskMass
  used SparsePak/PPak instead. The 45-galaxy fully-above-70 km/s subset is
  the one to trust without argument.

---

## Test 2 -- systems that measure two gravitational directions at once

Full system-by-system inventory: `TWO_DIRECTION_INVENTORY.md`.

| category | systems with BOTH directions measured | numbers tabulated? |
|---|---|---|
| (a) polar rings | 9 with rotation measured independently in both planes; 40 confirmed PRGs | **no** |
| (b) warped H I disks | 15 with i(R), PA(R) and V(R) in one table, ~9 credibly warped, +1 | yes |
| (c) stellar streams | 60 Milky Way streams with a measured 3-D track, 30 with 6-D | yes |
| (d) satellite systems | 101 SAGA hosts with orientation, 378 satellites | yes, but no in-plane curve |
| (e) two-component galaxies | 105 near-orthogonal + 38 counter-rotating measured here; 447 + 261 from SAMI | yes |
| vertical dispersion | 240 MaNGA face-on disks measured here; 2 external galaxies + the Milky Way | yes |

**The sharpest finding is a negative on the configuration the task named as most
powerful.** No polar-ring galaxy reachable from arXiv, VizieR, CDS or NED has a
numerically tabulated rotation curve in **both** planes. Exactly two have one
plane tabulated (NGC 4650A's host disk, 23 points; NGC 2685's warped H I disk,
21 rings). Nine systems do have rotation measured independently in both planes
-- NGC 4650A, NGC 4262, SPRC-7, SPRC-260, NGC 4632, NGC 6156, A0136-0801,
UGC 7576, UGC 9796 -- and every one of them exists only as a figure. The physics
is published; the numbers are not. Recovering them means digitising figures,
re-reducing archival cubes, or asking the authors, and none of that is a
data-acquisition task.

**The category the task listed last turned out to be the largest by two orders
of magnitude.** Integral-field spectroscopy measures two planes of the same
galaxy by construction, because the stellar and ionised-gas velocity fields are
independent tracers of the same potential. Measuring the gas-versus-stellar
kinematic misalignment directly from the MAPS cubes this lane already holds
(`clean/manga_gas_star_misalignment.csv`) gives 105 near-orthogonal and 38
counter-rotating systems out of 891 with both components rotating -- with the
resolved velocity fields in both planes already on disk, not merely cited. SAMI's
published kinematic position angles give 447 and 261 more. That is a screen, not
a measurement of record: a linear-gradient estimator cannot see a decoupled core
or a warp, so anything used as a detection needs a proper kinematic-PA fit first.

**Two structural obstructions worth recording, because they are not survey
defects and no amount of further acquisition removes them:**

1. **No external galaxy anywhere has a *measured* scale height alongside a
   resolved sigma_z(R).** Measuring h_z requires an edge-on view; measuring
   sigma_z requires a face-on one. Every face-on sigma_z in the literature --
   DiskMass and Aniyan alike -- pairs with an h_z inferred from an h_R/h_z
   relation. The correlated-by-construction problem the programme recorded for
   DiskMass is therefore a geometric obstruction, not a DiskMass defect, and any
   Sigma_dyn built from sigma_z^2/(2 pi G h_z) inherits it.
2. **SAGA publishes no in-plane rotation curve for its hosts**, so category (d)
   delivers excellent angular sampling of the out-of-plane field around 101
   hosts with no in-plane field to compare it against. The Milky Way and M31
   remain the only satellite systems where both directions are genuinely
   available, at a few tens of objects each.

**The largest untapped dataset found in the whole lane** is H I layer thickness
in edge-on galaxies: gas dispersion plus a *measured* thickness gives the
vertical force directly, and O'Brien et al. 2010 and Peters et al. 2017 jointly
fit rotation curve, surface density, thickness and dispersion for 8 galaxies
(the two series overlap, so the union is 8, not 14). All of it is published as
figures. The sources are on disk.

---

## Files

```
env-data/
  MATCHED_PAIRS.md                     field/cluster pairs, tolerances achieved
  TWO_DIRECTION_INVENTORY.md           Test 2 inventory, sections (a)-(e)
  REPORT.md                            this file
  clean/                               every file has a sibling .manifest.json
    manga_env_master.csv               10,071 x 196, the joined MaNGA sample
    matched_pairs.csv                  784 rows across 7 declared tiers
    matched_pairs_summary.json         per-tier achieved tolerances and power
    sami_matched_pairs.csv             472 rows across 2 tiers, SAMI DR3
    sami_matched_pairs_summary.json
    manga_sami_crosscal.csv            103 galaxies observed by both surveys
    manga_sami_crosscal_summary.json   the measured inter-survey offsets
    manga_faceon_sigma_profiles.csv    1,671 radial points, 240 face-on disks
    faceon_sample.csv                  the 240 selected disks
    manga_gas_star_misalignment.csv    gas-vs-stellar kinematic PA, 900 cubes
    checks.json                        failure-mode verdicts
  raw/
    manga/        DRPall, DAPall, PyMorph, morphology, HI-MaNGA, GEMA
    manga/maps/   902 DAP MAPS cubes, 5.30 GB
    groups/       Tempel+2014, Tempel+2017, MCXC
    sami/         SAMI DR3, 14 catalogues + Owers+2017, 3068 x 81 inventory
    polar-rings/       Test 2a  -> POLAR_RINGS.md
    warps-vertical/    Test 2b and the sigma_z literature -> WARPS_AND_VERTICAL.md
    streams-satellites/ Test 2c, 2d, 2e
  code/
    vizier_tsv.py              VizieR TSV reader with the HTTP-200 trap assertion
    build_manga_env.py         ingest, cross-match, derived quantities
    build_matched_pairs.py     tiered matching, blind-protected
    build_sami_pairs.py        the same build on SAMI DR3
    crosscal_manga_sami.py     inter-survey offsets on the 103 shared galaxies
    extract_sigma_profiles.py  resolved dispersion profiles from the MAPS cubes
    extract_aniyan2018_ngc628.py  rotated-PDF table recovery, with validation
    measure_gas_star_misalignment.py  Test 2(e) screen
    fetch_maps.py              MAPS downloader
    write_manifests.py         manifest generation
    checks.py                  failure-mode checks
    verify_lane.py             re-hash every manifest against its file
    sanity_physical.py         end-to-end physical sanity check
    validate_env_against_known_clusters.py   Coma / A2199 / Hercules check
    diagnose_mcxc_match.py     why the first X-ray flag failed
    check_manga_sami_overlap.py
    write_matched_pairs_md.py, write_report_md.py, write_two_direction_md.py
```

Every downloaded file carries a `<name>.manifest.json` with source URL,
retrieval timestamp (UTC, ISO-8601), SHA-256, byte size, row count, column names
with units, and the exact query issued. `code/verify_lane.py` re-hashes all of
them: **330 manifests, 330 targets verified, zero mismatches.**

