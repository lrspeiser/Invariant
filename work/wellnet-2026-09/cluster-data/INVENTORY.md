# INVENTORY — cluster × product matrix

Lane: `work/wellnet-2026-09/cluster-data/`
Compiled 2026-09-04 (UTC). Machine-readable companions: `inventory.json`,
`validation_report.json`, `contamination_register.json`, `cone_search_audit.json`,
`los_depth_argument.json`, `gas/gas_profile_qa.json`,
`weaklensing/WEAK_LENSING_AVAILABILITY_AUDIT.json`,
`weaklensing/a370_coverage_diagnostics.json`,
`velocities/_PRODUCT7_INDEX.json`, `stronglensing/NOT_FOUND_and_negative_results.json`.
All acquisition and audit code is in `scripts/`; `scripts/agent_scratch/` holds
working scripts kept for reproducibility.

Legend
- **A** — acquired: the product exists, is public, and is on disk with a manifest.
- **P** — partial: something is on disk but it is materially weaker than the brief asked for. The shortfall is named.
- **U** — unavailable: no such public product exists. The reason is given and **nothing was substituted**.

| Cluster | P1 members | P2 BCG | P3 ICL | P4 gas | P5 weak lensing | P6 strong lensing | P7 velocities |
|---|---|---|---|---|---|---|---|
| Abell 2744            | **A** | P | **A** | **A** | **U** | **A** | **A** |
| MACS J0416.1-2403     | **A** | P | **A** | P | **U** | **A** | **A** |
| MACS J0717.5+3745     | P | P | **A** | **A** | **U** | **A** | **A** |
| MACS J1149.5+2223     | **A** | P | **A** | **A** | **U** | **A** | P |
| Abell S1063 (RXC J2248.7-4431) | **A** | P | **A** | **A** | **U** | **A** | **A** |
| Abell 370             | P | P | **A** | **A** | **A** | **A** | **A** |
| Abell 2029            | P | **A** | **U** | **A** | **U** | **U** | **A** |

Totals: 25 acquired, 16 partial, 8 unavailable, of 49 cells.

The eight unavailable cells are: weak lensing for six clusters, ICL for
Abell 2029, and strong lensing for Abell 2029.

---

## P1 — Member galaxy catalogues

Every cluster has RA, Dec, redshift, membership and stellar mass. The axis on
which clusters differ is **resolved structural parameters** (R_e, Sérsic n,
axis ratio q, position angle θ), which is what a spatially resolved baryonic
model actually needs.

| Cluster | Status | Structural parameters | Key files |
|---|---|---|---|
| Abell 2744 | A | Full Sérsic in 7 HST bands, 225 members | `members/A2744_Granata2026_AA709_A254_members_structural.raw.tsv` |
| MACS J0416 | A | Full Sérsic, 224 members | `members/MACS0416_Granata2026_...structural.raw.tsv` |
| MACS J1149 | A | Full Sérsic, 279 members | `members/MACS1149_Granata2026_...structural.raw.tsv` |
| Abell S1063 | A | Full Sérsic, 222 members | `members/AS1063_Granata2026_...structural.raw.tsv` |
| Abell 370 | **P** | q and θ only, from SExtractor moments. **No R_e, no n.** | `weaklensing/buffalo_a370/hlsp_buffalo_hst_multi_abell370_f814w_v1.0_galcat-redseq.cat` (870 members) |
| MACS J0717 | **P** | q and θ only, from SExtractor moments. **No R_e, no n.** | `members/HFFDS_macs0717clu_v3.9.*`, `members/MACS0717_Molino2017_...tsv` |
| Abell 2029 | **P** | Sérsic n, q, θ, R_e for **388 of 1054 members (36.8%)** | `members/A2029_members_Sohn2019_x_Simard2011_structural.csv` |

**Gap reasons.**
- *MACS J0717 and Abell 370*: Granata et al. 2026 (VizieR `J/A+A/709/A254`) is the
  only published Sérsic catalogue of HFF cluster members and it contains exactly
  four tables — `a2744`, `as1063`, `m0416`, `m1149`. MACS J0717 is deliberately
  excluded by those authors (it is a quadruple merger with no single anchoring
  BCG). No substitute catalogue exists for either cluster; VizieR description
  searches, positional-metadata searches and arXiv searches all came back empty.
- *Abell 2029*: Simard et al. 2011 (`J/ApJS/196/11`) fits only the SDSS DR7
  **spectroscopic** sample, which is complete to r = 17.77. Sohn et al. 2019
  reaches r = 21.3 with MMT/Hectospec. So the 63% unmatched are real members
  that Simard never fitted — a selection effect, not a cross-match failure
  (matched pairs agree to a median 0.026 arcsec). All 1054 members are retained
  with blank structural columns rather than being silently dropped.
- The A2029 fits are also **ground-based SDSS at ~1.4 arcsec seeing** and are not
  on the same measurement footing as the HST fits for the HFF clusters. Do not
  pool them.

**Bonus product not asked for but worth flagging.** Granata et al. 2026 also
publish **measured internal stellar velocity dispersions** for 213 member
galaxies (76 A2744, 52 MACS J0416, 51 MACS J1149, 34 AS1063) from pPXF fits to
VLT/MUSE spectra, joinable by ID to the structural tables. That lets the
member-galaxy mass components be constrained by measured kinematics plus
measured light profiles rather than by an assumed σ–luminosity scaling — which
is precisely what the lumpy-solver work needs.

### x_a, y_a, z_a — the brief's explicit question

**x_a and y_a are MEASURED.** RA and Dec are image centroids on astrometrically
calibrated frames. Two independent confirmations from this run: Granata's LaTeX
appendix versus its VizieR table agree to 0.0000 arcsec, and SDSS imaging versus
Hectospec targeting agree to a median 0.026 arcsec. At 0.10 arcsec the implied
transverse position error is 0.15–0.64 kpc depending on cluster redshift.

**z_a, the physical line-of-sight depth, is NOT measured and cannot be recovered
from any of these data.** The only line-of-sight observable is a single scalar
redshift, in which Hubble flow and peculiar velocity are exactly degenerate:
`cz_obs = H(z)·d + v_pec`, one equation, two unknowns. Quantified in
`los_depth_argument.json`: 1 Mpc of real depth produces only 5.1–8.6% of the
measured velocity dispersion, and reading the dispersion as Hubble flow implies
a spurious depth 3.2–4.7× the cluster's own diameter. It is worse than noise —
the Finger-of-God distortion makes inferred depth *anti*-correlate with true 3D
radius, with a coefficient set by the unknown orbital anisotropy. **The
downstream code must sample z_a and marginalise over it.** Photometric redshifts
are hundreds of Mpc worse and carry no depth information at all.

---

## P2 — BCG, separately from the other members

| Cluster | Status | What exists |
|---|---|---|
| Abell 2029 | **A** | Three independent products for BCG = IC 1101. Kluge et al. 2020 single-Sérsic fit (n = 5.55 ± 0.26, r_e = 261 arcsec ≈ 387 kpc, μ_e = 26.08, M_tot = −25.85, g′, reaching μ = 30 g′ mag/arcsec²); Donzelli et al. 2011 (n = 5.78, r_e = 439 kpc, plus ellipticity and PA inner and outer); Lauer et al. 2014 three-point curve of growth at 14.3/28.6/57.2 kpc plus σ* = 386 km/s. |
| All six HFF | **P** | Total fluxes only, in up to 17 bands (`bcg/shipley2018_hff_bCG_photometry.tsv`, 391 rows). DeMaio et al. 2018 adds aperture luminosity and stellar mass inside r < 10/50/100 kpc for AS1063, MACS J0416 and MACS J1149. **No light profile for any HFF BCG.** |

**Gap reason.** No published Sérsic or tabulated μ(r) fit exists for any HFF BCG.
Worse, the HFF-DeepSpace catalogue construction **models and subtracts the
bCG+ICL** before measuring the other sources, so its bCG photometry is not a
clean total either. DeMaio's decomposition treats BCG+ICL as one continuous
system and does not separate them at all.

---

## P3 — Intracluster light

| Cluster | Status | What exists |
|---|---|---|
| All six HFF | **A** (fractions) | Two independent, methodologically different measurements per cluster. Montes & Trujillo 2018 gives f_ICL under five definitions — two surface-brightness cuts (μ_V > 26; 26 < μ_V < 27) and three radius-based (50 kpc < R < R_limit; R < R_limit; R < R500). Jiménez-Teja et al. 2018 (A2744, MACS J0416, MACS J0717, MACS J1149) and de Oliveira et al. 2022 (A370, AS1063) give f_ICL from CICLE wavelet decomposition — so all six have a homogeneous CICLE set as well. |
| Abell 2029 | **U** | **No published ICL measurement exists.** |

**Critical caveats.**
- **No tabulated μ_ICL(r) exists for any of the seven clusters.** Montes &
  Trujillo 2018 and DeMaio et al. 2018 both publish surface-brightness and
  stellar-mass-density profiles as **figures only**; the paper's three tables are
  cluster properties/SB limits, the ICL fractions, and age/metallicity radial
  profiles. Verified directly against the arXiv source. The only ICL profile
  information in parametric form anywhere is the Kluge 2020 and Donzelli 2011
  single-Sérsic fits for A2029 — and those do not separate ICL from BCG.
- The five Montes & Trujillo definitions use **two incompatible methods** in one
  table and must not be mixed.
- The `fICL_R_lt_R500_pct` column requires R500 and is therefore **not purely
  photometric** — it inherits a mass model. See the contamination register.

**A2029 gap reason, stated plainly.** An arXiv full-text search for "Abell 2029"
+ "intracluster light" returns zero results. Kluge et al. 2020 measured the
A2029 BCG+ICL profile down to 30 g′ mag/arcsec² but classified the system as
**single-Sérsic**, so the double-Sérsic columns (n2, re2, SBe2, f2) are blank —
there is no BCG/ICL split. Donzelli et al. 2011 independently fitted **no**
exponential envelope for A2029. Kluge et al. 2021 publishes ICL fractions only
as 170-cluster sample averages, never per cluster. Uson et al. 1991 measured
diffuse light to ~607 kpc but as a single de Vaucouleurs halo. Nothing was
substituted.

---

## P4 — Gas

Two independent classes of observable were acquired: X-ray (deprojected electron
density and temperature) and SZ (line-of-sight electron pressure).

| Cluster | X-ray n_e(r) + T(r) | Radial range | SZ | Status |
|---|---|---|---|---|
| Abell 2029 | ACCEPT 73 bins **and** X-COP (XMM) | 2.3–2212 kpc = 0.002–1.56 R500 | not in BOXSZ (z < 0.15) | **A** |
| Abell 2744 | ACCEPT 59 bins | 0–1326 kpc = 0–1.07 R500 | Bolocam | **A** |
| Abell S1063 | ACCEPT 41 bins | 0–1012 kpc = 0–0.71 R500 | Bolocam | **A** |
| MACS J0717 | ACCEPT 39 bins | 0–1191 kpc = 0–0.88 R500 | Bolocam | **A** |
| MACS J1149 | ACCEPT 41 bins | 0–1301 kpc = 0–1.07 R500 | Bolocam | **A** |
| Abell 370 | ACCEPT 32 bins; plus Umetsu 2022 4-shell deprojection with a *deprojected* T and a core-excised global kT | 0–806 kpc | Bolocam | **A** |
| MACS J0416 | **none** — global kT = 10.06 keV only, plus an eRASS1 global entry and Bonamigo 2017's dPIE fit to the Chandra surface brightness | — | Bolocam | **P** |

**MACS J0416 — a confirmed absence, not an untried gap.** No radial n_e(r) or
T(r) exists for this cluster in the published literature. Five named leads were
exhausted, with VizieR identifiers echoed back in each case:

1. Andrade-Santos 2017 (`J/ApJ/843/76`) and 2021 (`J/ApJ/914/58`) — both found in
   VizieR, **zero** hits for MACS J0416 across 4 tables and 538 rows. These are
   Planck-ESZ / X-ray-selected samples it is not a member of.
2. Donahue et al. 2014 CLASH-X (arXiv:1405.7876) — MACS J0416 **is** in the
   sample, Chandra only (every XMM column is `\nodata`), but with just 3 radial
   bins to R_max = 1.39 arcmin, and the published tables carry only fit
   statistics and hydrostatic masses. The JACO n_e(r) and T(r) exist **as figures
   only**. Donahue et al. 2016 is not in VizieR.
3. CLASH HLSP on MAST — `macs0416/data/` contains only `hst/` and `subaru/`.
   Optical throughout; no X-ray product.
4. Mantz WtG (`J/MNRAS/463/3582`) — not in VizieR (explicit error).
5. CHEX-MATE — no per-cluster profile catalogue in VizieR, so membership is moot.

It is also genuinely absent from ACCEPT: verified against all 243 ACCEPT cluster
keys, nearest neighbour is the unrelated MACS_J0417.5-1154. Ogrean et al. 2015's
β-model and broken-power-law fits are narrow *sectors* chosen for shock analysis
and their parameters appear only in figures. The cluster is a bimodal pre-merger,
so a single spherical profile would be a poor description even if one existed.

**One genuine addition.** MACS J0416 **is** in eRASS1 (Bulbul et al. 2024, A&A
685, A106) as `1eRASS J041610.3-240351` at 0.66 arcmin offset, identity confirmed
by the catalogue's own MatchName cross-ID to ACT-CL J0416.1-2404
(`gas/macsj0416_erass1_global.tsv`). L500T = 3.61e44 erg/s, Mgas500 = 8.1e13
M_sun, R500 = 1242 kpc, fgas500 = 0.097. **Global only, no profile**, and its
temperature is unusable — the KT column is empty with only a lower bound
b_KT = 2.38 keV from a 143 s exposure yielding 144 counts inside R500. Ogrean's
Chandra kT = 10.06 keV remains far superior. M500, Mgas500 and R500 are
scaling-relation products; eRASS1 calibrates on weak lensing rather than
hydrostatic masses, so the HSE flag was deliberately **not** set on that file.

The Bolocam SZ map partly compensates for the missing profile: it constrains the
integrated electron pressure over roughly 0.1–3.5 R500. That makes MACS J0416 the
one cluster whose gas channel rests on SZ rather than on a resolved X-ray
deprojection — which changes what it can contribute to the resolved test, and is
why its P4 cell is **P** rather than **A**.

**Two acquisition QA findings the downstream code must handle**
(`gas/gas_profile_qa.json`):
1. **ACCEPT stores its radial bins in DESCENDING radius order in every file.**
   Code assuming ascending order integrates the profile backwards, silently.
2. **n_e(r) is not monotonic.** Abell 2744 has 27 upward steps in 58 bin-to-bin
   differences — the spherical deprojection of that major merger is close to
   noise-dominated and a single spherical n_e(r) is a poor description of it.
   A2029 has 16/72 (cool-core sloshing spiral), MACS J1149 9/40, A370 7/31;
   AS1063 and MACS J0717 are cleanest at 5 each.

**Identifier trap, found and fixed.** ACCEPT names Abell 370 `ABELL_0370` with a
**zero-padded** Abell number. A search for `ABELL_370` returns nothing, and the
first pass therefore concluded the cluster was absent from ACCEPT and fell back
on a 4-shell deprojection. The 32-bin ACCEPT profile is now in
`gas/accept_ABELL_0370.tsv`. Same class of trap as ACCEPT's `ABELL_1063S`
standing for Abell S1063.

---

## P5 — Weak lensing (raw shear or ellipticity catalogue)

| Cluster | Status | What exists |
|---|---|---|
| Abell 370 | **A** | Per-source shear catalogues from the BUFFALO lensing DR1. HST-only: 3557 sources with RA, Dec, e1, e2, var_e1, var_e2, a, b, θ, three-band magnitudes and shape S/N. HST+Subaru combined: 18556 sources spanning 0.43° × 0.51°, of which 877 carry a source redshift. |
| Every other target | **U** | No public per-source shear catalogue. **No public shear PROFILE table either.** |

This is the headline gap and it is documented source-by-source in
`weaklensing/WEAK_LENSING_AVAILABILITY_AUDIT.json`, which records every archive,
VizieR identifier, arXiv source tarball, GitHub repository and Zenodo query
checked, with the verbatim outcome. Summary of why each avenue failed:

- **BUFFALO HLSP**: the `niemiec-lensing-dr1` directory exists for `abell370`
  only. The other five cluster directories contain the Pagul photometric
  catalogue and nothing else. Re-verified live.
- **The six-cluster release is announced but not yet public.** "Non-spherical
  BUFFALOs" (arXiv:2602.06904, Feb 2026) analyses all six HFF clusters with
  pyRRG on ACS/F814W — 2917 to 3708 background galaxies each — and states the
  catalogues "will be made available upon acceptance of the publication at
  https://archive.stsci.edu/hlsp/buffalo". As of 2026-09-04 they are not there.
- **CLASH**: the HLSP ships Suprime-Cam *photometry* and photo-z only. Its own
  readme says which image "should be used for shape measurements" — the shapes
  were never released. The RXJ2248 Subaru directory is **empty**.
- **Umetsu et al. 2014 and 2016**: checked table by table in the arXiv sources.
  Table 3 gives background-sample summary statistics, Table 5 gives only the
  *signal-to-noise* of the g_+ profile, and the rest are NFW masses and
  concentrations. The binned shear profile is never tabulated. Neither paper is
  in VizieR.
- **Per-cluster WL papers** (Medezinski 2016 for A2744, Gruen 2013 for AS1063,
  Jauzac 2018 for MACS J0717, Finner 2017 for MACS J1149, Abriola 2024 and Kim
  2026 for A2744, Harvey 2024's JWST analysis): none releases a catalogue. Shear
  profiles appear as figures. Harvey 2024's data availability statement points at
  a *code* repository whose 1645-path tree contains only TinyTim PSF libraries.
- **Abell 2029**: LoVoCCS states "the LoVoCCS DECam exposures are publicly
  available on the NOIRLab Astro Data Archive" — the *exposures*, not the shear
  catalogue. The LoVoCCS GitHub organisation has seven repositories covering
  X-ray, BCG and SZ work and no lensing catalogue. A2029 shear is recoverable in
  principle by re-reducing public DECam imaging; that is an image-processing
  project, not an acquisition task.
- **HFF lens-model deliverables**: only CATS and Zitrin ship input catalogues at
  all, and those are multiple-image lists, not shear. `frontier/internal/`
  redirects to an STScI SSO login and was not pursued.

**Deliberately not touched.** A Zenodo keyword search surfaced a KiDS weak
lensing catalogue (DOI 10.5281/zenodo.16366035). KiDS is a permanent sealed
holdout under the programme brief; it was not downloaded, opened or inspected,
and it covers none of these clusters anyway.

**Deliberately not downloaded.** Convergence, shear, deflection, magnification
and mass maps exist in bulk for every HFF cluster (BUFFALO Niemiec a/b1/b2/b3/
c1/c2/d1/d2/e; HFF CATS, Sharon, GLAFIC, Zitrin, Keeton, Williams, Diego,
Bradač, Merten; CLASH Merten and Zitrin). None was taken, because a mass map is
not lensing data.

---

## P6 — Strong lensing

| Cluster | Systems | Images | Spectroscopic z | Status |
|---|---|---|---|---|
| Abell 2744 | 59 / 38 | 185 / 149 | 92 / 121 | **A** — Mahler 2018; Bergamini 2023 GLASS-JWST |
| MACS J0416 | 58 / 37 | 237 / 114 | **237 (all)** / 114 | **A** — Bergamini 2023; Caminha 2017 |
| MACS J0717 | 61 | 165 | 27 | **A** — Limousin 2016 |
| MACS J1149 | 34 | 97 | 34 | **A** — Treu 2016, **plus SN Refsdal** |
| Abell S1063 | 16 / 42 | 48 / 121 | 30 / 31 | **A** — Caminha 2016; Chan 2020 |
| Abell 370 | 45 / 59 | 138 / 176 | 123 / — | **A** — Lagattuta 2019; BUFFALO DR1 sl-final |
| Abell 2029 | — | — | — | **U** |

Plus `HFF6_multiple_images_Richard2014.tsv`, a uniform six-cluster baseline
(256 images, 97 spectroscopic) in a single schema.

**Time delays — the single most constraining observable in the set.** SN Refsdal
in MACS J1149: `MACSJ1149_SNRefsdal_time_delays_Kelly2023.tsv` (4 image pairs ×
5 independent measurement methods × {delay, magnification ratio}), adopted
SX−S1 = **376.02 d** with 16th–84th percentiles 370.50–381.65, i.e. ~1.4%
precision; plus Rodney 2016 for S2/S3/S4 relative to S1. **MACS J1149 is the only
target cluster with measured time delays.**

**Gaps.**
- **Arc orientations / position angles do not exist for any cluster.** The only
  PA-shaped column in any of these files is `theta` in the CATS Lenstool inputs,
  which those files' own headers call a placeholder — verified constant per file.
- *Abell 2029*: confirmed not a strong lens. It is a low-redshift (z = 0.0773)
  relaxed cool-core cluster; VizieR holds only redshift and member catalogues for
  it, and arXiv returns zero hits for A2029 combined with strong lensing,
  multiple images or giant arc.

---

## P7 — Member velocities (spectroscopic redshift catalogues)

All seven clusters have redshifts, but **radial coverage is very uneven, and
radial coverage is what a projected dispersion field σ(R) actually needs.** The
table below reports, per cluster, the best available file by N of galaxies within
±3000 km/s of the cluster and the maximum projected radius those reach. Full
per-file detail is in `velocities/_PRODUCT7_INDEX.json` and `_summary.json`.

| Cluster | Best catalogue | N members (\|dv\| < 3000 km/s) | Max member R | σ(R) field supportable? |
|---|---|---|---|---|
| Abell 2029 | Sohn 2019b (MMT/Hectospec, no colour cut) | 1215 | 8.75 Mpc | **Yes, comfortably** |
| Abell S1063 | CLASH-VLT / Mercurio 2021 (3850 z) | 1192 | full VIMOS field | **Yes** |
| MACS J0416 | Caminha 2017 / Balestra 2016 (CLASH-VLT) | 982 / 903 | 5.49 Mpc | **Yes** |
| MACS J0717 | Ebeling 2014 (Keck/DEIMOS+LRIS, Gemini/GMOS) | 559 | wide field | **Yes** |
| Abell 2744 | Owers 2011 (AAOmega + literature) | 418 | 4.19 Mpc | **Yes** |
| Abell 370 | PilotWINGS Lagattuta 2022 | 382 | 0.90 Mpc | Inner region only |
| MACS J1149 | Schuldt 2024 | 151 | 0.65 Mpc | **Marginal — the one weak cell** |

**Two catalogues recovered from outside CDS.** The AS1063 and MACS J0717 cells
initially looked thin (95 and 17 usable members, from a NED cone and a grism
survey). Both were access-route artefacts, not data gaps:
- **Abell S1063** — the CLASH-VLT public release *is* the Mercurio et al. 2021
  catalogue, distributed from the project's own site. VizieR `J/A+A/656/A147`
  does not exist and silently serves the Cooper+2013 fallback; cdsarc returns
  404. Acquired: 3850 redshifts, `velocities/AS1063_CLASHVLT_Mercurio2021_zcat.tsv`.
  (The task brief attributed this reference to MACS J0416; it is AS1063's.)
- **MACS J0717** — VizieR `J/ApJS/211/21` (Ebeling, Ma & Barrett 2014) *is* real
  and *is* spectroscopic. Acquired: 1266 rows for the J0717.5+3745 field,
  asserted against the CDS ReadMe entry `table4.dat 70 1266`. The `+` in the
  `MACS=` filter must be percent-encoded as `%2B`. That parent table also holds
  65 redshifts in the MACS J0416 field.

**Membership was re-derived, not inherited.** Neither new catalogue ships a
membership column. Both were cut at rest-frame |dv| < 3000 km/s about the cluster
redshift, giving 1192 members for AS1063 against the published 1234 (peak+gap
selection) and 559 for MACS J0717 against the published 537 — both agreeing at
the 3–4% level with a genuinely different procedure. These are a sanity check on
the ingest, **not** a reproduction of the published member lists.

**Remaining caveats.**
- MUSE catalogues (Richard 2021, Grillo 2016) are deep but confined to footprints
  under ~0.5 Mpc; they give no radial leverage on their own.
- The MACS J0717 **GLASS grism** catalogue is unusable for kinematics: σ_z ≈
  0.003–0.01 against a cluster velocity signal of ~0.005 in z. Membership only.
- CLASH-VLT applied a colour preselection (R ≲ 24), so radial completeness for
  MACS J0416 and AS1063 is not uniform even though N and reach are good.

**A third VizieR trap variant, caught here.** For a nonexistent `-source=`,
VizieR does not always emit `#INFO Error=Table or Catalog not found`. In this
lane it sometimes returned HTTP 200 with a *completely unrelated catalogue*,
echoing `#Name: J/MNRAS/430/1125` (Cooper et al. 2013, an RMS near-infrared YSO
survey), and URL-encoding the `+` does not help. The only reliable detector is to
check that the response echoes the **exact identifier requested**. Twelve
identifiers were probed and rejected this way; raw evidence for each is preserved
in `velocities/raw/probe_*.tsv`.

**A whitespace-parsing trap that silently loses a row.** The CLASH-VLT AS1063
catalogue has exactly one line with a space inside its object ID
(`CLASHVLTJ2249 9.98-442802.3`, which should read `CLASHVLTJ224959.98-442802.3`),
so that row splits into 8 whitespace fields while the other 3849 split into 7,
shifting every column. The failure is silent and measurable: a whitespace parse
returns quality-flag counts summing to 3849 against the file's 3850 data lines.
Rejoining the identifier and asserting the field count per row recovers it — flag
3 goes 3004 → 3005 and the total reconciles to 3850.

---

## Cross-cutting notes

**Provenance.** Every data file carries a sibling `.manifest.json` with source
URL, exact query, UTC retrieval timestamp, SHA-256, byte size, row and column
counts, column names with units, extraction method and a note. Raw upstream
responses are preserved unmodified in `raw/` subdirectories and are checksummed
in `validation_report.json` even where they have no manifest of their own.
`scripts/validate_manifests.py` re-verifies every checksum, byte size and row
count; the current run reports **0 problems**.

**Cone-search audit.** A reported failure mode — VizieR `-c` / `-c.rs` cone
searches returning zero rows with no error, which invalidated one null elsewhere
in this programme (MACS J0416 was wrongly reported absent from PSZ2; it is
present as PSZ2 G221.06-44.05 at 1.08 arcmin) — was audited across this lane.
Result in `cone_search_audit.json`: **no null in this lane rests on a cone
search.** Every negative result came from a directory listing, a `-source=` probe
returning an explicit `#INFO Error=Table or Catalog not found`, arXiv
source-tarball table enumeration, or a GitHub tree listing. The lane contains
exactly one cone-derived product — the Simard 2011 fits around A2029 — and it is
a positive result of 2853 rows, checked here for silent radial truncation and
found complete: rows reach exactly 110.0 of the 110 arcmin requested and the
outermost equal-area annulus carries 1.14× the median annular surface density.

**Not observations.** `contamination_register.json` lists 63 files carrying a
dependence on an NFW halo profile, Newtonian hydrostatic equilibrium, a scaling
relation, a mass-model-defined aperture, SED-fitted stellar mass, or a lens-model
output. Those may be used as inputs or context; none may be the quantity a
candidate gravity law is scored against.
