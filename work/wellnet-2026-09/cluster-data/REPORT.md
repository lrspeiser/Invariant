# REPORT — cluster data acquisition for a raw-observable gravity test

Lane: `work/wellnet-2026-09/cluster-data/`
Compiled 2026-09-04 (UTC).
Companion documents: `INVENTORY.md` (the cluster × product matrix and every gap
reason), `inventory.json`, `validation_report.json`, `contamination_register.json`,
`cone_search_audit.json`, `los_depth_argument.json`,
`weaklensing/WEAK_LENSING_AVAILABILITY_AUDIT.json`, `gas/gas_profile_qa.json`.

---

## 1. The headline, stated first because it changes the experiment design

**A public raw weak-lensing shear catalogue exists for exactly one of the seven
target clusters: Abell 370.** For the other six there is no per-source shear
catalogue and — this is the part that was not anticipated — **no public shear
PROFILE table either.** The binned tangential shear profiles for A2744
(Medezinski 2016), AS1063 (Gruen 2013) and the CLASH clusters (Umetsu 2014,
2016) exist only as figures. What those papers tabulate is NFW masses and
concentrations, which presuppose a dark-matter halo profile and are therefore
inadmissible here.

The brief anticipated this possibility and asked that it be reported prominently
rather than papered over with substituted profiles. Nothing was substituted.

There is a second, sharper finding stacked on top of it:

> **The one cluster with raw weak lensing is one of the two clusters without
> resolved member-galaxy structural parameters.**

Abell 370 has 18,556 shear measurements reaching 6.2 Mpc, and its 870
red-sequence members carry only SExtractor axis ratio and position angle — no
effective radius, no Sérsic index. Meanwhile A2744, MACS J0416, MACS J1149 and
AS1063 have full seven-band Sérsic fits for their members and no weak lensing at
all. **No cluster in the target list has both.** Any experiment that needs a
fully resolved lumpy baryonic model *and* an outer-radius lensing constraint has
to bridge that gap explicitly rather than assume it away.

---

## 2. What was acquired

302 files, 154 with individual manifests, 148 raw upstream envelopes checksummed
in `validation_report.json`. `scripts/validate_manifests.py` re-verifies every
SHA-256, byte count and row count and currently reports **0 problems**.

| Product | Clusters with it | Notes |
|---|---|---|
| 1. Member galaxies | 7 of 7 | Full Sérsic (R_e, n, q, θ) in 7 HST bands for 4 (A2744, MACS J0416, MACS J1149, AS1063) |
| 2. BCG | 7 of 7, but only A2029 as a light *profile* | No published light profile for any HFF BCG |
| 3. ICL | 6 of 7 as *fractions*; **0 of 7** as a profile | A2029 has no ICL measurement at all |
| 4. Gas | 6 of 7 with resolved n_e(r) + T(r); 6 of 7 with SZ | MACS J0416 has SZ but no X-ray profile |
| 5. Weak lensing | **1 of 7** | Abell 370 only |
| 6. Strong lensing | 6 of 7 | A2029 is not a strong lens; MACS J1149 has time delays |
| 7. Member velocities | 7 of 7, radial coverage very uneven | A2029 gives 1215 members to 8.75 Mpc; MACS J1149 gives 151 to 0.65 Mpc |

Four clusters — **Abell 2744, MACS J1149, Abell S1063 and Abell 370** — carry
six of the seven products at acquired-or-partial status with no outright gap
except weak lensing (and A370 has that too). That meets the brief's "at least
four clusters" bar.

### Products worth singling out

**SN Refsdal time delays** (`stronglensing/MACSJ1149_SNRefsdal_time_delays_Kelly2023.tsv`).
Four image pairs × five independent measurement methods × {delay, magnification
ratio}. Adopted SX−S1 = 376.02 d, 16th–84th percentiles 370.50–381.65 — about
1.4% precision on a pure light-curve measurement with no mass model in it. This
is the single most constraining raw observable in the whole set, and MACS J1149
is the only target cluster that has one.

**Measured internal stellar velocity dispersions for 213 member galaxies**
(Granata et al. 2026, in `members/*_velocity_dispersions.csv`; 76 A2744, 52
MACS J0416, 51 MACS J1149, 34 AS1063). pPXF fits to VLT/MUSE spectra, joinable
by ID to the same paper's Sérsic structural fits. This lets the member-galaxy
mass components be constrained by *measured* kinematics plus *measured* light
profiles instead of an assumed sigma–luminosity scaling — which is exactly what a
lumpy solver needs and is unusual to have.

**Bolocam SZ maps for all six HFF clusters** (`gas/bolocam_sz/`, IRSA DOI
10.26131/IRSA562). Not in the original brief but directly on-target: the SZ
decrement is proportional to the line-of-sight integral of electron pressure, an
independent thermal observable that needs neither a deprojection nor a
hydrostatic assumption. All six verified: map centres land 0.00 arcmin from the
archive positions and every map shows a clear decrement (−522 to −1255 µK). This
is what keeps MACS J0416 in the sample at all.

**A370 in ACCEPT.** ACCEPT zero-pads Abell numbers below 1000, so Abell 370 is
keyed `ABELL_0370` and a search for `ABELL_370` returns nothing. The first pass
concluded the cluster was absent and fell back on a 4-shell deprojection; the
real ACCEPT entry has 32 bins to 806 kpc and is now in
`gas/accept_ABELL_0370.tsv`.

---

## 3. What does not exist

These are **confirmed absences established by exhausting named leads with
identifiers echoed back**, not untried gaps. Each is a result in its own right
because each changes which clusters can carry which channel of the test.

1. **Raw weak-lensing shear for six of seven clusters.** Documented
   source-by-source in `weaklensing/WEAK_LENSING_AVAILABILITY_AUDIT.json`:
   BUFFALO HLSP directory listings (re-verified live), CLASH HLSP, Umetsu 2014
   and 2016 table-by-table from the arXiv sources, six per-cluster WL papers,
   HFF lens-model deliverables, the pyRRG GitHub tree, and Zenodo. The
   six-cluster BUFFALO release *is announced* — "Non-spherical BUFFALOs"
   (arXiv:2602.06904) states the catalogues "will be made available upon
   acceptance" at the BUFFALO HLSP — but as of 2026-09-04 they are not there.
   **This gap has a known expiry date and is worth re-checking.**
2. **A shear profile table for any cluster other than A370.** Figures only.
3. **Any tabulated ICL surface-brightness profile, for any of the seven.**
   Verified directly against the Montes & Trujillo 2018 arXiv source: its three
   tables are cluster properties/SB limits, ICL fractions, and age/metallicity
   radial profiles. The stellar-mass-density profiles are figures. DeMaio 2018 is
   the same. What exists is ICL *fractions* under five definitions, plus a fitted
   log-linear slope beyond 50 kpc reported graphically.
4. **A BCG light profile for any HFF cluster.** Only total fluxes and aperture
   luminosities. Worse, the HFF-DeepSpace catalogue construction models and
   *subtracts* the bCG+ICL before measuring other sources, so even its bCG
   photometry is not a clean total.
5. **Any ICL measurement for Abell 2029.** arXiv full-text search for "Abell
   2029" + "intracluster light" returns zero. Kluge 2020 classified it
   single-Sérsic so the double-Sérsic ICL columns are blank; Donzelli 2011
   independently fitted no exponential envelope; Kluge 2021 reports ICL fractions
   only as 170-cluster averages.
6. **A radial n_e(r) or T(r) for MACS J0416.** Five leads exhausted
   (Andrade-Santos 2017 and 2021: zero hits across 4 tables / 538 rows;
   Donahue 2014 CLASH-X: in the sample but 3 bins to 1.39 arcmin with the
   profiles as figures only; CLASH HLSP `data/`: optical only; Mantz WtG: not in
   VizieR; CHEX-MATE: no per-cluster profile catalogue). Also absent from all 243
   ACCEPT keys.
7. **Arc orientations / position angles, for any cluster.** The only PA-shaped
   column in any strong-lensing file is `theta` in the CATS Lenstool inputs,
   which those files' own headers call a placeholder — verified constant per file.
8. **Strong lensing for Abell 2029.** It is a relaxed low-z cool-core cluster,
   not a lens.

One apparent absence turned out to be an ACCESS-ROUTE problem rather than a real
one, and it is worth separating from the list above. The wide-field spectroscopic
catalogues for Abell S1063 and MACS J0717 are **not** missing; they are simply not
reachable by the CDS route:

- **Abell S1063**: the CLASH-VLT public release (Mercurio et al. 2021, A&A 656,
  A147) is distributed from the project's own site, not CDS. VizieR
  `J/A+A/656/A147` does not exist and silently serves the Cooper+2013 fallback;
  cdsarc returns 404. Acquired: 3850 redshifts, replacing the 290-row NED cone.
- **MACS J0717**: VizieR `J/ApJS/211/21` (Ebeling, Ma & Barrett 2014) *is* real and
  *is* a spectroscopic catalogue. Acquired: 1266 rows for the J0717.5+3745 field,
  asserted against the CDS ReadMe entry `table4.dat 70 1266`, replacing the NED
  cone with a single homogeneous Keck/DEIMOS + LRIS + GMOS survey. The `+` in the
  `MACS=` filter must be percent-encoded as `%2B` or the filter matches nothing.
  Incidentally that parent table also holds 65 redshifts in the MACS J0416 field.

Three access routes are simply not scriptable and produced false negatives
elsewhere in this programme: `cdsarc.cds.unistra.fr/ftp/...` **data** files sit
behind an Anubis bot-check returning a 4.4 KB HTML challenge instead of data (the
ReadMe works with a browser user-agent; use the unprotected VizieR `asu-*`
services for data); IOPscience 302-redirects to `validate.perfdrive.com`, so
machine-readable tables cannot be pulled there; and `www.aanda.org` returns HTTP
403 to all automated clients, which this lane hit directly.

---

## 3b. Velocity coverage is the second uneven axis

All seven clusters have spectroscopic redshifts, but the downstream use is a
projected dispersion **field** sigma(R), and radial coverage varies by more than
an order of magnitude. N counts galaxies within rest-frame |dv| < 3000 km/s of
the cluster redshift; no dispersion was computed anywhere in this lane.

| Cluster | Best catalogue | N members | Max member R | Resolved sigma(R)? |
|---|---|---|---|---|
| Abell 2029 | Sohn 2019b (MMT/Hectospec, no colour cut) | 1215 | 8.75 Mpc | yes, comfortably |
| Abell S1063 | CLASH-VLT / Mercurio 2021 | 1192 | full VIMOS field | yes |
| MACS J0416 | Caminha 2017 (CLASH-VLT VIMOS) | 982 | 5.49 Mpc | yes |
| MACS J0717 | Ebeling 2014 (Keck/DEIMOS + LRIS + GMOS) | 559 | wide field | yes |
| Abell 2744 | Owers 2011 (AAOmega + literature) | 418 | 4.19 Mpc | yes |
| Abell 370 | PilotWINGS Lagattuta 2022 | 382 | 0.90 Mpc | core only |
| MACS J1149 | Schuldt 2024 | 151 | 0.65 Mpc | **marginal, core only** |

**Membership was re-derived, not inherited.** Neither the AS1063 nor the
MACS J0717 catalogue ships a membership column, so both were cut at rest-frame
|dv| < 3000 km/s about the cluster redshift (z = 0.3480 and 0.5458). That yields
1192 members for AS1063 against the 1234 that Mercurio et al. publish from a
peak-plus-gap selection, and 559 for MACS J0717 against the 537 that Limousin
et al. and Jauzac et al. publish from the same data. Both agree at the 3-4% level
with a genuinely different procedure, which is the expected size of disagreement
and not a signal worth chasing. **The counts above are a sanity check on the
ingest, not a reproduction of the published member lists** — the downstream test
should re-derive membership deliberately.

Three caveats that change how these should be weighted:

- **The MACS J0717 GLASS grism catalogue is unusable for kinematics.** Its
  redshift errors are sigma_z ~ 0.003-0.01 against a cluster velocity signal of
  ~0.005 in z: the error is comparable to the quantity being measured. Retained
  for membership only. The Ebeling catalogue replaces it.
- **CLASH-VLT applied a colour preselection** (R <~ 24 with colour cuts), so
  radial completeness for MACS J0416 and AS1063 is not uniform even though N and
  reach are good.
- MUSE catalogues are deep but confined to footprints under about 0.5 Mpc and
  give no radial leverage on their own.

**MACS J1149 is now the only cluster whose velocity field cannot support a
resolved sigma(R).** That is a change from the earlier reading, in which AS1063
and MACS J0717 also looked thin; both were artefacts of the CDS access route
rather than real data gaps.

---

## 4. The binding constraint

**Product 5, raw weak lensing, is the binding constraint on the downstream
test.** The argument is about radial reach, and it is quantitative.

Measured coverage for Abell 370 (`weaklensing/a370_coverage_diagnostics.json`,
FlatLambdaCDM H0=70 Om=0.3 used only for unit conversion):

| Probe | Radial range (proper kpc) | Median |
|---|---|---|
| Strong-lensing images | 12 – 332 | 193 |
| Cluster members | 21 – 2857 | 706 |
| Weak-lensing sources (HST+Subaru) | 282 – 6208 | 3209 |

For the five HFF clusters without weak lensing, the outermost probe that
constrains a *field* is the X-ray gas, which stops at 0.71–1.07 R500 (roughly
800–1330 kpc). Strong lensing stops near 300 kpc. So without weak lensing the
test is confined to r <~ 1.3 Mpc — and the programme's own prior result is that
what survives elimination is a **cluster-only excess organised by r/R500**,
which is to say the interesting regime is precisely the one that weak lensing
alone reaches. One cluster is not a sample there.

The Bolocam SZ maps partially mitigate this: they constrain integrated electron
pressure to roughly 3.5 R500. But at a 58 arcsec beam on 30×30-pixel maps they
give an integrated quantity, not a resolved field, and they probe the gas rather
than the potential.

### Why the ICL gap is *not* the binding constraint

It is the obvious rival candidate — no tabulated mu_ICL(r) exists anywhere — so it
is worth ruling out numerically. Within R500 the gas mass fraction is about 0.10
(eRASS1 gives fgas500 = 0.097 for MACS J0416) and the stellar mass fraction is
about 0.015, so stars are about 13% of the baryons. ICL is 5–25% of the stellar
light (Montes & Trujillo 2018 range across the six HFF clusters), hence **about
1–4% of the baryonic mass.** Getting its radial distribution wrong is a
percent-level error on rho_b.

That said, the ICL gap binds hard on one specific question: any candidate law
whose coupling is to the *stellar* or *luminous* component rather than to total
baryonic mass will be limited by it, because the ICL is the most spatially
extended stellar component and its profile is exactly what is unmeasured. If the
downstream test includes a mass-follows-light family, ICL becomes the binding
constraint for that family and must be sampled as a nuisance function.

### Consequences for experiment design

- **Outer-radius lensing test** → Abell 370 only, and with degraded member
  resolution (no R_e, no n).
- **Fully resolved lumpy baryonic model + strong lensing + a time delay** →
  MACS J1149 is the strongest single target, but it has no weak lensing.
- **Longest gas lever arm** → Abell 2029 (X-COP, 0.002–1.56 R500), but it has no
  lensing of any kind and no ICL.
- No single cluster carries everything. A design that requires all seven products
  on one object has no valid target in this sample.

---

## 5. Products contaminated by a dark-matter or mass-follows-light assumption

`contamination_register.json` flags 63 files programmatically. The ones that
matter, and the rule for each:

| Product | Contamination | Rule |
|---|---|---|
| ACCEPT `Mgrav`, `Merr` | Newtonian hydrostatic equilibrium | Never an observation. `nelec` and `Tx` in the same file are fine. |
| X-COP `A2029_hydro_mass.fits` | Newtonian HSE | Never an observation. Flagged `derived_assumes_newtonian_hse: true`. |
| MCXC M500 / R500 | L–M scaling relation calibrated on hydrostatic masses | Never an observation. MCXC has no temperature column at all. |
| NFW masses and concentrations (Umetsu 2014/2016, Gruen 2013, Medezinski 2016, Herbonnet) | NFW halo profile assumed | Presupposes dark matter. Not acquired as data. |
| Bolocam `gnfw_fit_map.fits`, `gnfw_fit_mcmc.fits` | Generalised-NFW parametric fit | Model. The unfiltered and filtered SZ images in the same tarball are observations. |
| Montes & Trujillo `fICL_R_lt_R500_pct` | Aperture defined by R500, which needs a mass model | Use the surface-brightness-cut columns instead; they are purely photometric. |
| DeMaio 2018 sample M500 / r500 | X-ray hydrostatic (Vikhlinin 2009 kT–M) | Context only. |
| Kluge 2021 `Radg`, `logMg` | Dynamical virial-type estimates | Context only. |
| HFF-DeepSpace `.fout` stellar masses | SED fitting: BC03, Chabrier IMF, delayed-exponential SFH, Calzetti dust | Model-derived but **not** DM-dependent. Usable as an input with a stated IMF systematic. |
| BUFFALO `galcat-*.dat` `z` column | Hard-coded 0.0 LENSTOOL placeholder | Not a measured redshift. Trap. |
| CATS lens-model `arcs.txt` redshifts | Undifferentiated adopted values mixing spectroscopic, photometric and model-optimised with no provenance flag | Use the published per-paper image catalogues instead; those carry real flags. |
| Zitrin lens-model arcs | Carries a real `Z_SPEC` flag plus `F?` = redshift free in the model | Usable if the flag is respected. |
| AS1063 CATS file | Model-optimised redshifts wrapped in parentheses, e.g. `(2.16)` | 22 rows; a naive parser drops them silently. |
| Simard 2011 `Rhlr`, `Rchl_r` | Angular size measured, but kpc conversion uses redshift plus assumed cosmology | Semi-derived. Keep the angular quantity. |
| ACCEPT `Tx` | **Projected** spectroscopic temperature interpolated onto the density bins, not a deprojected 3D temperature | Geometry assumption. X-COP and Umetsu 2022 give genuinely deprojected T. |
| All deprojected n_e(r) | Spherical symmetry assumed | Explicitly wrong for A2744 (major merger) and MACS J0416 (bimodal pre-merger). |

**No mass map, convergence map, magnification map or deflection map was
downloaded at all**, so no `debug_only` labelling was needed. They were located —
BUFFALO Niemiec a/b1/b2/b3/c1/c2/d1/d2/e; HFF CATS, Sharon, GLAFIC, Zitrin,
Keeton, Williams, Diego, Bradac, Merten; CLASH Merten and Zitrin — and left
alone.

---

## 6. Coordinates: which of x_a, y_a, z_a are measured

**x_a and y_a are measured.** RA and Dec are image centroids on astrometrically
calibrated frames. Two independent confirmations from this run: Granata's LaTeX
appendix versus its VizieR table agree to 0.0000 arcsec; SDSS imaging versus
MMT/Hectospec targeting agree to a median 0.026 arcsec. At 0.10 arcsec the
transverse position error is 0.15–0.64 kpc across the seven clusters — three to
four orders of magnitude below the scales any model here resolves.

**z_a, the physical line-of-sight depth, is NOT measured and cannot be recovered
from these data.** The only line-of-sight observable is a single scalar
redshift, in which Hubble flow and peculiar velocity are exactly degenerate:
cz_obs = H(z)·d + v_pec, one equation, two unknowns, and nothing in any acquired
catalogue breaks it. Quantified in `los_depth_argument.json`:

| Cluster | z | 1 Mpc of depth gives | as a fraction of sigma_v | spurious depth if sigma_v read as Hubble flow, over 2R200 |
|---|---|---|---|---|
| Abell 2744 | 0.308 | 82.0 km/s | 0.055 | 4.6 |
| MACS J0416 | 0.396 | 86.2 km/s | 0.086 | 3.2 |
| MACS J0717 | 0.546 | 94.1 km/s | 0.057 | 4.2 |
| MACS J1149 | 0.542 | 93.9 km/s | 0.051 | 4.7 |
| Abell S1063 | 0.348 | 83.8 km/s | 0.061 | 3.9 |
| Abell 370 | 0.375 | 85.2 km/s | 0.073 | 3.4 |
| Abell 2029 | 0.077 | 72.6 km/s | 0.063 | 4.0 |

A full megaparsec of real depth moves the observed velocity by 5–9% of the
dispersion. Reading the dispersion as Hubble flow implies a depth spread 3.2–4.7
times the cluster's own diameter. It is worse than noise: Finger-of-God
distortion makes inferred depth *anti*-correlate with true 3D radius, with a
coefficient set by the unknown orbital anisotropy.

**Only the projected separation R_perp is observable. The downstream code must
sample z_a and marginalise over it.** Photometric redshifts (sigma_z about
0.03–0.05(1+z)) are hundreds of Mpc worse and carry no depth information
whatsoever. The source redshift of a strong-lensing image *is* a genuine third
measured quantity, but it fixes the source's cosmological distance, not the depth
of any cluster member.

---

## 7. Failure modes checked — as the brief requires, explicitly

**Shared-denominator artefacts — checked, and several live cases found in the
acquired data.** No correlation was computed in this lane, but the acquired
tables contain quantities built from common measured inputs whose errors are
therefore correlated, and these are exactly the trap that retracted rho_p = −0.304:
- ACCEPT `Kitpl`, `Pitpl` and `Mgrav` are all constructed from the *same* `nelec`
  and `Tx`. Entropy versus density, or pressure versus density, has a non-zero
  naive null.
- eRASS1 `fgas500` = Mgas500/M500 and `R500` is derived from M500, so fgas versus
  M500 or versus R500 shares a factor.
- ICL fractions have the total cluster light in the denominator and part of it in
  the numerator; f_ICL errors are anti-correlated with total-light errors.
- Within any single-Sérsic fit, R_e and mu_e are strongly anti-correlated and n
  covaries with both (Granata, Simard, Kluge, Donzelli all inherit this).
Any downstream correlation using these must simulate the null with the actual
error covariance or use errors-in-variables.

**Monotone-invariant statistics — checked, not applicable.** No rank statistic
and no headline statistic S(theta) was computed in this lane. The only numbers
produced are descriptive coverage percentiles and count-based QA.

**Refitting on the held-out set — checked, not applicable.** No fitting of any
kind was performed. **KiDS and wide binaries were not loaded, opened or
inspected.** A Zenodo keyword search surfaced a KiDS weak-lensing catalogue (DOI
10.5281/zenodo.16366035); it was deliberately not downloaded, and is recorded in
the weak-lensing audit as untouched. It covers none of these clusters anyway.

**Silent extraction failures — checked; the traps fired repeatedly and were
caught.**
- *VizieR HTTP-200 generic response*: detected via the explicit
  `#INFO Error=Table or Catalog not found` string, plus a check that the echoed
  table identifier matches what was requested. Nine of fourteen probed BCG/ICL
  catalogues and three strong-lensing catalogues were NOT FOUND. A subtler
  variant also appeared: `J/A+A/587/A80`, `J/A+A/588/A99` and `J/A+A/645/A140`
  returned HTTP 200 carrying an *unrelated* catalogue (`I/16`) plus
  `CatalogsExamined=10213`. A **third** variant appeared in the velocity lane:
  for a nonexistent `-source=`, VizieR returned HTTP 200 echoing
  `#Name: J/MNRAS/430/1125` (Cooper et al. 2013, an RMS near-infrared YSO
  survey) -- a completely unrelated real catalogue served silently in place of
  the request, and URL-encoding the `+` does not help. **The only detector that
  works across all three variants is to check that the response echoes back the
  exact identifier requested.** Twelve identifiers were rejected this way in the
  velocity lane and three more in the strong-lensing lane, five of them supplied
  by the task brief itself. Nothing was substituted in any case.
- *A merged-table trap, new to this programme*: VizieR fuses Braglia et al.
  2009's two separate published tables -- A2744 (395 rows) and A2537 (809 rows) --
  into a single 1204-row table distinguished only by an `A` column. Ingested
  unfiltered it injects galaxies roughly 530 Mpc away into the A2744 velocity
  field. The delivered file is filtered to `A=2744`, exactly 395 rows.
- *A whitespace-parsing trap that silently loses a row*: the CLASH-VLT AS1063
  catalogue has exactly one line with a SPACE INSIDE ITS OBJECT ID
  (`CLASHVLTJ2249 9.98-442802.3`, which should read `CLASHVLTJ224959.98-442802.3`),
  so that row splits into 8 fields while the other 3849 split into 7, shifting
  every column. The failure is silent and quantifiable: the quality-flag counts
  reported to this lane by a whitespace parse summed to 3849, one short of the
  file's 3850 data lines. Repairing the identifier and asserting the field count
  per row recovers it -- flag 3 goes from 3004 to 3005 and the total reconciles.
- *Row counts asserted against the authoritative source, not the paper text*:
  every VizieR table's row count was checked against the CDS ReadMe `Records`
  column (ReadMes preserved in `velocities/raw/`). All matched exactly.
- *LaTeX table split across two `table*` environments*: fired twice for real.
  DeMaio 2018's colour profiles span two environments and parsing only the first
  drops 7 of 23 systems including target MACS J1149 — and the last two blocks of
  file 1 are commented out and duplicated in file 2, so naive parsing
  double-counts. Limousin 2016 splits its image list 48 + 117; parsing only the
  first would have discarded 71% of the catalogue. Both asserted against the
  paper's stated totals (23 systems; "61 systems, comprised of 165 individual
  images").
- *A third, novel variant*: Granata's velocity dispersions live in four
  `longtable`s and a bare `%` comment line was glued onto the first data row of
  three of them, eating one row each. First parse gave 210, correct answer 213.
  Caught only by asserting against the abstract.
- *Cone-search nulls*: a reported failure mode in which VizieR `-c`/`-c.rs`
  returns zero rows with no error (it invalidated a claim elsewhere in this
  programme — MACS J0416 was said to be absent from PSZ2 but is present as PSZ2
  G221.06-44.05). Audited across this lane in `cone_search_audit.json`: **no null
  here rests on a cone search.** Every negative came from a directory listing, a
  `-source=` probe with an explicit error string, arXiv source-tarball table
  enumeration, or a GitHub tree listing. The lane's single cone-derived product
  is a *positive* result and was additionally checked for silent radial
  truncation — complete to 110.0 of 110 arcmin requested, outermost equal-area
  annulus at 1.14 times the median density.
- *Identifier traps*: ACCEPT zero-pads Abell numbers below 1000 (`ABELL_0370`,
  and 37 such keys); ACCEPT's `ABELL_1063S` is Abell S1063; the MAST BUFFALO
  landing page carries a **broken link** pointing at an
  `abells1063/.../niemiec-lensing-dr1/..._abell370_..._readme.txt` path that 404s
  and must not be mistaken for an AS1063 lensing release; the brief's
  `archive.stsci.edu/prepds/hff-deepspace/` URL is dead, the live path is
  `/hlsps/hff-deepspace/`; the arXiv id 1710.07300 suggested for Montes &
  Trujillo 2018 resolves to a machine-learning paper, the correct id is
  1710.03240.
- *Foreground interloper*: the brightest bCG-flagged source within 60 arcsec of
  the MACS J0416 centre is at z_spec = 0.1137 — a foreground galaxy, not the
  z = 0.396 BCG. A spectroscopic membership cut was added.
- *Row and column counts were asserted after every ingest.*
  `scripts/validate_manifests.py` re-verifies all of them and reports 0 problems.

**Test bugs that look like solver bugs — the acquisition analogue was found.**
`gas/gas_profile_qa.json`: **ACCEPT stores its radial bins in DESCENDING radius
order in every file.** Code that assumes ascending order integrates the profile
backwards, silently, and would produce a "solver" error that is really an ingest
error.

**Non-monotonic profiles — found and quantified.** n_e(r) is not monotonic in any
ACCEPT profile. Abell 2744 has 27 upward steps in 58 bin-to-bin differences: the
spherical deprojection of that major merger is close to noise-dominated and a
single spherical n_e(r) is a poor description of it. A2029 has 16/72 (cool-core
sloshing spiral), MACS J1149 9/40, A370 7/31; AS1063 and MACS J0717 are cleanest
at 5 each.

---

## 8. What I could not establish

- Whether the announced six-cluster BUFFALO weak-lensing release will appear, or
  when. It is stated as "upon acceptance" of arXiv:2602.06904. **This is the
  single highest-value thing to re-check before the downstream test is designed.**
- Whether a usable A2029 shear catalogue could be produced from the public
  LoVoCCS DECam exposures on the NOIRLab Astro Data Archive. The imaging is
  public and the source density is stated (7–12 per square arcmin, effective
  source z about 0.51), but reducing it is an image-processing project, not
  acquisition.
- Whether the STScI-authenticated `frontier/internal/` area holds anything
  relevant. It redirects to an SSO login and was not pursued.
- Whether ICL surface-brightness profiles exist as unpublished author data. Only
  their published form was checked, and in published form they are figures.
- Anything about the *quality* of these data as constraints. This lane acquired
  and characterised; it fitted nothing and scored nothing.

Two process items worth recording rather than hiding. A background VizieR fetch
in the strong-lensing lane **exited non-zero**, crashing after writing all 22
catalogue files but before emitting its summary, with an empty log that swallowed
the traceback; that lane's first report described the run as clean, which was
wrong. The products were reconstructed by parsing the downloaded files directly,
and an independent re-fetch has since confirmed that all nine VizieR catalogues
the products depend on return **byte-identical data blocks** -- so no truncation
occurred, and that now rests on direct comparison rather than inference.
Separately, two mid-build bugs were caught and fixed: an early builder wrote
strong-lensing products before applying redshift propagation, so its manifests
briefly described counts the TSVs did not contain; and cz-sourced velocity tables
(Owers, Sohn 2017, Boschin) had their redshifts converted while their errors were
left in km/s. Both chains were rebuilt and re-verified.

---

## 9. Recommended sample for the downstream test

Given the above, the defensible design is a **tiered** one rather than a uniform
seven-cluster run:

- **Tier 1, resolved baryonic model plus inner-region lensing plus kinematics** —
  A2744, MACS J0416, AS1063, MACS J1149. All four have seven-band Sérsic member
  fits, measured *internal* member velocity dispersions from MUSE, strong-lensing
  image lists with large spectroscopic fractions, gas (SZ for MACS J0416, X-ray
  profiles for the others) and ICL fractions. **They are not equal on the
  cluster-kinematics axis**: A2744, MACS J0416 and AS1063 all support a resolved
  sigma(R) with 400-1200 members, whereas MACS J1149 has only 151 members inside
  0.65 Mpc. MACS J1149 earns its place regardless, because it is the only cluster
  in the entire sample carrying measured time delays.
- **Tier 2, the only outer-radius lensing target** — Abell 370, accepting
  degraded member structural resolution (q and theta but no R_e or n) as a stated
  systematic.
- **Tier 3, gas lever arm only** — Abell 2029, out to 1.56 R500 with X-COP, and
  with a 1054-member caustic-flagged velocity field. No lensing, no ICL.
- **MACS J0717** has good cluster kinematics after all -- 559 members from the
  Ebeling 2014 Keck survey, not the 17 usable redshifts its GLASS grism catalogue
  offers -- but it has **no member Sérsic fits at all**, and it is a quadruple
  merger. Include it where resolved member structure is not required.

The honest summary is that the seven products are available on four-to-six
clusters each, but their *intersection* is thin, and weak lensing is what makes
it thin.
