# SAMI Galaxy Survey DR3 — acquisition inventory

Lane: `work/wellnet-2026-09/env-data/raw/sami/`
Acquired 2026-09-04 (UTC). All row counts below are asserted in code, not eyeballed.

---

## 1. Access: what was and was not needed

**No account was created, no credentials were entered, no terms or consent dialogue
was accepted.** Everything here came off anonymous public endpoints.

| Route | Status |
|---|---|
| Data Central **IVOA TAP** `https://datacentral.org.au/vo/tap/sync` | **Fully anonymous.** Served all 14 SAMI DR3 catalogue tables via `SELECT *` ADQL. Cross-schema joins to `gama_dr4` work too. |
| Data Central **REST API** `https://datacentral.org.au/api/schema/` | **403 "Authentication credentials were not provided."** Not needed — TAP covers the same catalogues. |
| Data Central `/api/services/query/`, `/api/services/download/` | Reachable anonymously (POST services). Not used; cube-level downloads were out of scope. |
| **VizieR** | **Does not hold SAMI DR3, Owers 2017, Bryant 2015 or Owers 2019.** See §5. |
| **arXiv** e-prints | Used for the two cluster-property tables. |

There is **no access blocker** for anything this lane needs.

---

## 2. What was downloaded

### 2.1 SAMI DR3 catalogues — Data Central TAP, all 14 tables

Raw VOTable bytes (`.vot`) kept unmodified; cleaned `.tsv` beside each; both carry a
manifest. Row counts were checked against an independent `COUNT(*)` query *and*
against Table 5 of Croom et al. 2021.

| Table | Rows got | Rows per Croom+2021 Table 5 | Cols | Contents |
|---|---:|---:|---:|---|
| `InputCatGAMADR3` | **5536** | 5536 ✓ | 21 | Field/group input catalogue: z, r-band mags, **Re, ellip, PA, mu(Re), (g−i), log M\*** |
| `InputCatClustersDR3` | **1433** | 1433 ✓ | 20 | Cluster input catalogue: same photometry **plus R/R200, v_pec/sigma_200, member flag** |
| `InputCatFiller` | **2980** | 2980 ✓ | 5 | Filler targets (minimal columns) |
| `CubeObs` | **3712** | 3712 ✓ | 28 | One row per cube; `ISBEST`, `CATSOURCE`, 20 `WARN*` quality flags |
| `samiDR3Stelkin` | **3426** | 3426 ✓ | 44 | **sigma in Sersic-Re / MGE-Re / 3 kpc / 1.4–4 arcsec, lambda_R(Re), V/sigma(Re), stellar kinematic PA, kinemetry k5/k1** |
| `samiDR3gaskinPA` | **3426** | 3426 ✓ | 6 | Gas kinematic PA + error |
| `EmissionLine1compDR3` | **3425** | 3245 ✗ | 271 | Aperture emission-line fluxes, **gas v and sigma**, SFR, in 7 apertures |
| `IndexAperturesDR3` | **3375** | 3375 ✓ | 290 | Lick indices |
| `SSPAperturesDR3` | **3375** | 3375 ✓ | 45 | SSP age, [Z/H], [alpha/Fe] |
| `MGEPhotomUnregDR3` | **3150** | 3150 ✓ | 10 | **MGE Re (circularised), total mag, PA, ellipticity at Re and light-weighted** |
| `VisualMorphologyDR3` | **3068** | 3068 ✓ | 2 | Visual morphological type |
| `DensityCatDR3` | **6969** | 6969 ✓ | 11 | 5th-nearest-neighbour surface density |
| `FstarCatGAMA` | **2578** | 2578 ✓ | 5 | Calibration stars |
| `FstarCatClusters` | **183** | 183 ✓ | 5 | Calibration stars |

**Two discrepancies against the published documentation, reported rather than papered over:**

1. `EmissionLine1compDR3` has **3425** rows in the live archive, not the 3245 printed
   in Croom+2021 Table 5. 3425/3426 matches the two kinematics tables, so the paper
   number looks like a 3425 → 3245 digit transposition. The archive is self-consistent;
   the paper is not.
2. `EmissionLineRecomcompDR3` (recommended-component emission-line fits, 3245 rows) is
   listed in Croom+2021 Table 5 but **is not present in the `sami_dr3` TAP schema**.
   `EmissionLine1compDR3` is the only emission-line table Data Central exposes. If the
   lane needs multi-component gas fits, that table is a genuine gap.

### 2.2 Cluster properties — Owers et al. 2017 Table 1

`owers2017_table1_clusters.tsv` — 8 rows, 18 columns, transcribed verbatim from the
arXiv LaTeX source `1703.00997` (`clusters.tex`, the single `table*` labelled
`clus_table`). The tarball is kept unmodified as `owers2017_arxiv_1703.00997.tar.gz`.

**The eight clusters are confirmed from the source** (paper abstract and Table 1), and
match the list in the brief exactly: APMCC 0917, Abell 168, Abell 4038, EDCC 442,
Abell 3880, Abell 2399, Abell 119, Abell 85.

### 2.3 Cluster targets — Croom et al. 2021 Table 3

`croom2021_table3_cluster_targets.tsv` — 8 rows, from arXiv `2101.12224`. Adds the
per-cluster observed/good/all target counts and the **second (northern) pointing centre
used for Abell 168** (18.739974, +0.430807), which appears nowhere in the catalogues.
Used as an independent cross-check: it reproduces the Owers+2017 RA, Dec, z_clus,
sigma_200, R_200 and virial M_200 exactly for all eight clusters, and its target totals
sum to 1433 — the `InputCatClustersDR3` row count.

### 2.4 Sersic index supplement (GAMA arm only)

`gama_dr4_SersicCatSDSSv09_SAMIsubset.tsv` — 5536 rows, r-band GALFIT single-Sersic
parameters (**`GALINDEX_r` = Sersic n**, Re, ellipticity, PA, mu_0, mu_e, R90, chi2 and
errors) for every SAMI GAMA-region target. Kelvin et al. 2012, served as
`gama_dr4.SersicCatSDSSv09` on the same anonymous TAP endpoint. SAMI's GAMA-region CATID
*is* the GAMA CATAID: all 5536 join. See §6 for why this covers only one arm.

### 2.5 Derived join products

| File | Rows | What it is |
|---|---:|---|
| `sami_cluster_env_hosts.tsv` | 1433 | Every cluster-region target with its **host cluster named**, plus observable projected radius in Mpc and observable v_pec in km/s |
| `sami_dr3_master_galaxy_inventory.tsv` | 3068 | One row per SAMI galaxy with a cube: arm, kinematics, structure, morphology, environment, availability flags |
| `sami_inventory_counts.json` | — | The counts in §4, machine-readable |
| `vizier_negative_probe.json` | 14 | The VizieR negative result, recorded with its guard |

---

## 3. Cluster membership vs field/group — the split

`CubeObs.CATSOURCE` labels the input catalogue each cube came from. Restricting to
`ISBEST = 1` and `WARNSTAR = 0` gives **3068 unique galaxies**, exactly the DR3 headline
sample:

| Arm | CATSOURCE | N | Environment available |
|---|---|---:|---|
| **Cluster** | 2 | **896** | R/R200, v_pec/sigma_200, membership flag, named host, host sigma_200 |
| **GAMA field/group** | 1 | **2100** | 5th-nearest-neighbour surface density only (2081 of them) |
| Filler | 3 | 72 | none |

`InputCatClustersDR3` carries **no host-cluster column**. Host was assigned here from the
CATID prefix (the first four digits encode the observing field) and **validated** by
recomputing R/R200 from scratch: angular separation × angular diameter distance at
z_clus (FlatLambdaCDM H0 = 70, Om = 0.3, the cosmology stated in Owers+2017 §1) divided
by R200. Result: median fractional difference **+0.20 %**, p68 **0.20 %**, max **0.32 %**
over the 1385 galaxies with R/R200 > 0.05; **1433/1433 pass**.

Per-galaxy nearest-centre assignment does *not* work: APMCC 0917 and Abell 4038 sit
1.9 deg apart and shared 2dF fields (Owers+2017 Table 2), so 10 Abell 4038 targets are
closer to the APMCC 0917 centre than to their own. The prefix map is one-to-one:

| Prefix | Cluster | N targets | N with cubes | N members | N members inside R200 |
|---|---|---:|---:|---:|---:|
| 9091 | APMCC 0917 | 47 | 33 | 33 | 28 |
| 9016 | Abell 168 | 173 | 96 | 95 | 81 |
| 9044 | EDCC 442 | 97 | 47 | 47 | 44 |
| 9403 | Abell 4038 | 198 | 119 | 100 | 86 |
| 9388 | Abell 3880 | 116 | 93 | 89 | 60 |
| 9239 | Abell 2399 | 143 | 131 | 118 | 99 |
| 9011 | Abell 119 | 417 | 202 | 196 | 196 |
| 9008 | Abell 85 | 242 | 175 | 171 | 164 |
| | **Total** | **1433** | **896** | **849** | **758** |

Cluster-arm coverage: R/R200 from 0 to 1.96, projected radius 0 to 3.26 Mpc,
|v_pec| up to 3182 km/s, z from 0.024 to 0.066.

---

## 4. What internal kinematics each galaxy has

All counts are over the 3068 galaxies with a best cube.

| Product | Total | Cluster | GAMA | Filler |
|---|---:|---:|---:|---:|
| 2-moment stellar kinematic maps **and** emission-line maps produced | **3068** | 896 | 2100 | 72 |
| …with clean flags (`WARNSKER`, `WARNMULT`, `WARNZ`, `WARNWCS` all 0) | 2865 | 826 | 1985 | 54 |
| Aperture sigma within **Sersic Re** | 2251 | **745** | 1505 | 1 |
| Aperture sigma within **MGE Re** | 2377 | 794 | 1563 | 20 |
| **lambda_R(Re)** spin proxy (and V/sigma(Re)) | 1930 | **712** | 1216 | 2 |
| Stellar kinematic PA | 2815 | 844 | 1914 | 57 |
| Gas kinematic PA | 3067 | 896 | 2099 | 72 |
| Ionised-gas sigma and v within Re | 2898 | **851** | 2045 | 2 |

Every DR3 cube has resolved 2-moment stellar kinematic maps and emission-line maps
(`WARNSK2M` and `WARNEMFT` are 0 for all 3068). What varies is whether the aperture
summary quantities could be measured. Cluster-arm sigma(Re) spans 38.5–345.4 km/s.

**Structural photometry:**

| Product | Total | Cluster | GAMA | Filler |
|---|---:|---:|---:|---:|
| Sersic Re + ellipticity + PA + log M\* | 2964 | **885** | 2079 | 0 |
| MGE Re + ellipticity (homogeneous, both arms) | 3016 | **895** | 2100 | 21 |
| Visual morphology | 3068 | 896 | 2100 | 72 |
| log stellar mass | 2996 | 896 | 2100 | 0 |
| **Sersic index n** | 2100 | **0** | 2100 | 0 |

**Matched-sample yields** (kinematics = sigma(Re); structure = Sersic Re/ellip/M\* *and*
MGE; plus visual morphology):

| Cut | Cluster | GAMA (comparison pool) |
|---|---:|---:|
| Full measurement set | 744 | 1505 |
| …late-type (TYPE ≥ 2) | 191 | **878** |
| …late-type, confirmed member, inside R200 | **152** | — |
| …same, with clean kinematic flags | 150 | — |
| …late-type, confirmed member, any radius (≤ 2 R200) | **178** | — |
| …TYPE ≥ 2.5 (early/late and late spirals), member, inside R200 | 102 | — |

Without the measurement requirements there are **237** late-type cluster-arm galaxies
with cubes, **217** of them confirmed members, **178** inside R200.

**Against the MaNGA cluster arm (~48 quality-passing late-type disks inside R_vir of
hosts with sigma_v ≥ 400 km/s): SAMI contributes 152 on the strictest equivalent cut,
178 if the radius limit is relaxed to the survey edge — a 3–4x increase.** All eight
SAMI hosts clear the sigma_v ≥ 400 km/s bar by construction (492–1002 km/s), so that
cut costs nothing.

Stellar-mass ranges (log10 M\*/Msun, 5th–95th percentile): cluster 9.60–11.05
(median 10.28), GAMA 8.34–11.07 (median 10.04). The cluster arm has a harder faint-end
cut — matching against the field must be done inside the overlap.

---

## 5. VizieR: negative result, recorded

VizieR returns HTTP 200 with a generic page for a nonexistent `-source=`, so a hit was
only counted when the body contained `#Table` **and** `#Column` **and** echoed the
identifier back. Probed: `J/MNRAS/505/991`, `/990`, `/992`, `/1`; `J/MNRAS/468/1824`,
`/1823`, `/1825`; `J/MNRAS/447/2857`, `/2856`, `/2858`; `J/ApJ/873/52`, `/51`, `/53`.
**All thirteen returned "Table or Catalog not found".** Metadata searches on "SAMI
Galaxy Survey", "SAMI", "Owers" and "Croom SAMI cluster" return only
`J/MNRAS/446/1567` — the 2015 SAMI **Early Data Release** (107 galaxies), superseded by
DR3 and containing no cluster galaxies. Recorded in `vizier_negative_probe.json`.

Data Central TAP and the arXiv sources are the only routes.

---

## 6. Corrected premises — read these before using the data

**(a) There is no Sersic index for the cluster arm, and there never was one published.**
The brief asked for Sersic parameters. SAMI DR3 releases Re, ellipticity and PA that
*are* Sersic-derived, but no table in DR3 carries n.

- *GAMA arm:* n is recoverable from `gama_dr4.SersicCatSDSSv09` (§2.4). Done: 2100/2100.
- *Cluster arm:* Croom+2021 states the cluster Re/ellip/PA come from the Sersic fits of
  **Owers et al. 2019** (arXiv 1901.08185). That paper describes PROFIT fits to r-band
  SDSS DR9 / VST-ATLAS imaging but **publishes no per-galaxy structural table** — its
  three deluxetables are spectral-classification summaries — and the index is not
  propagated into DR3. It is not in VizieR either. **n is unavailable for all 896
  cluster galaxies.**

  Do **not** match cluster against field on Sersic n; the covariate exists for one arm
  only, and a one-sided covariate is an environment proxy in disguise. The homogeneous
  basis that does cover both arms is **`MGEPhotomUnregDR3`** — 895/896 cluster and
  2100/2100 GAMA — giving circularised Re, total magnitude, PA, ellipticity at Re and
  light-weighted ellipticity from the same MGE pipeline on both. Croom+2021 quote an
  rms of 0.067 dex between MGE and Sersic half-light radii, i.e. ~0.047 dex measurement
  uncertainty on the MGE Re.

  Note also that Owers et al. **2017** contains no Sersic fitting at all — its
  ellipticities are SExtractor shape parameters. The Sersic provenance is 2019, not 2017.

**(b) Owers+2017 is internally inconsistent on member counts.** Summing Table 1 gives
1941 members inside R200 and 2901 inside 2 R200; the abstract and §5.1 state 1935 and
2899. The LaTeX source carries commented-out superseded rows for Abell 168 (192/276 live
vs 195/279 commented) and Abell 119 (372/578 vs 370/576), so the table was revised
without the text being re-summed. Neither variant reproduces 1935/2899. The
transcription reproduces the **table**, which is the per-cluster breakdown actually
needed. Deltas +6 and +2 — nowhere near a truncated-table failure, and recorded in the
manifest.

**(c) Croom+2021 Table 3 counts 888 observed cluster targets; `CubeObs` has 896 unique
cluster-region galaxies with a best cube.** An 8-galaxy surplus of cluster-region
objects with cubes that are not counted as observed primary or secondary targets.

---

## 7. Host-environment provenance — observable vs model-derived

This is the point the brief flagged. Per Owers+2017 §5.1 and §6, and Croom+2021 §8.2:

| Quantity | Derivation | Verdict |
|---|---|---|
| `z_clus` | Biweight location (Beers+1990) of member redshifts within 2 R200 | **OBSERVABLE** |
| `sigma_200` | Biweight **scale** estimator on member line-of-sight velocities inside R200 | **OBSERVABLE** (up to the R200 aperture) |
| `e_sigma_200`, `N_mem` | Uncertainty and the member count it rests on (86–590 per cluster; 1941 total) | **OBSERVABLE** |
| Projected separation | Angular separation × angular diameter distance at z_clus | **OBSERVABLE** |
| `v_pec` | c[(1+z_pec)²−1]/[(1+z_pec)²+1] from member and cluster redshifts | **OBSERVABLE** |
| **`R_200`** | **`R_200 = 0.17 sigma_200 / H(z)` Mpc, iterated. Carlberg+1997 *singular isothermal sphere*.** | **MODEL-DERIVED** — assumes an isothermal, virialised, dark-matter-dominated halo |
| **`M_200` (virial)** | Girardi+1998 corrected virial mass, surface-pressure term C ≈ 0.19 M_vir. Assumes **virial equilibrium and spherical symmetry**. | **MODEL-DERIVED — RANK ONLY** |
| **`M_200` (caustic)** | Diaferio 1999 escape-velocity/caustic estimator with **F_beta = 0.7** calibrated on simulations (Serra+2011). Assumes spherical symmetry; the caustic placement itself minimises (⟨v_esc(R200)²⟩ − 4 sigma_200²)², i.e. an isothermal-sphere assumption. | **MODEL-DERIVED — RANK ONLY** |
| `R/R200`, `v_pec/sigma_200` | Observable numerators divided by model-derived / observable denominators | **MIXED** — see below |

Both mass estimators are total dynamical masses, so both presuppose dark matter under
the brief's constraint 2. **Neither may enter a fit as an observation.** They may rank
the eight environments — and the ranking is unambiguous, since virial and caustic masses
order the clusters identically and monotonically with sigma_200.

Owers+2017 also warns that A2399 and A85 show substructure (h3 ~ 0.1) and their masses
may be overestimated by up to ~50 %, and that combining cluster masses with GAMA group
masses needs a ×1.25 rescaling on top of a cosmology correction. Both are additional
reasons to treat them as ordinal only.

### Shared-denominator hazard — this lane has one

`R_on_rtwo = R_proj / R200` and `R200 ∝ sigma_200`, so **`R_on_rtwo` carries sigma_200 in
its denominator**. So does `V_on_sigma = v_pec / sigma_200`. Any statistic that puts
sigma_200 (or M_200, which scales as sigma³) on one axis and R/R200 or v/sigma on the
other **shares sigma_200 across both axes** — precisely the structure that produced the
retracted rho_p = −0.304.

Two sigma-free columns are therefore provided in `sami_cluster_env_hosts.tsv` and the
master table:

- **`R_proj_Mpc_from_cat` = `R_on_rtwo` × R200** — the projected physical clustercentric
  radius in Mpc, with the model normalisation undone. Verified against an independent
  recomputation from RA/Dec (`R_proj_Mpc_direct`) to 0.2 %.
- **`v_pec_kms` = `V_on_sigma` × sigma_200** — the line-of-sight peculiar velocity in km/s.

**Use these, not R/R200 and v/sigma, whenever sigma_200 or a cluster mass is on the other
axis.** With only eight hosts the null must be simulated with the actual error
covariance regardless.

---

## 8. Failure modes explicitly checked

- **Silent extraction failures** — every ingest asserts a row count against an
  independent source (TAP `COUNT(*)`; Croom+2021 Table 5; the paper's stated eight
  clusters; the 1433 target total). Every `table*` environment in both LaTeX sources was
  enumerated before parsing, and the count of environments carrying the wanted label was
  asserted to be exactly 1 — this is what catches a table split across two environments.
  Every parsed row was asserted to have the expected cell count. Commented-out superseded
  LaTeX rows were skipped and are documented.
- **VizieR HTTP-200-on-missing** — guarded, and the negative recorded (§5).
- **Shared-denominator artefacts** — found one, documented, and sigma-free replacement
  columns provided (§7).
- **Dark-matter-dependent quantities as observations** — R200 and both M200 estimators
  identified and tagged RANK-ONLY in every manifest that touches them (§7).
- **Corrected premise over substituted proxy** — the missing cluster-arm Sersic index is
  reported as a gap with the reason, not filled with a different survey's n (§6a).
- **Monotone-invariant statistics, refitting on held-out data, solver/test bugs** — not
  applicable to an acquisition lane; no fitting or model selection was performed here.
- **KiDS and wide binaries** — not touched; nothing in this lane goes near them.

---

## 9. Files

All paths relative to
`C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data\raw\sami\`.
38 data files, 38 manifests, ~112 MB. Every data file has a sibling
`<name>.manifest.json` (audited in code).

**Raw downloads (unmodified):**
`sami_dr3_<TABLE>.vot` × 14 · `gama_dr4_SersicCatSDSSv09_SAMIsubset.vot` ·
`owers2017_arxiv_1703.00997.tar.gz` · `croom2021_dr3_arxiv_2101.12224.tar.gz`

**Cleaned:**
`sami_dr3_<TABLE>.tsv` × 14 · `gama_dr4_SersicCatSDSSv09_SAMIsubset.tsv` ·
`owers2017_table1_clusters.tsv` · `croom2021_table3_cluster_targets.tsv`

**Derived:**
`sami_cluster_env_hosts.tsv` · `sami_dr3_master_galaxy_inventory.tsv` ·
`sami_inventory_counts.json` · `vizier_negative_probe.json`

**Code (all re-runnable, all assertions live):**
`_fetch_dc_tap.py` · `_fetch_gama_sersic.py` · `_transcribe_owers2017.py` ·
`_transcribe_croom2021.py` · `_probe_vizier.py` · `_build_cluster_env.py` ·
`_build_master.py` · `_summary_stats.py` · `_fill_sidecar_manifests.py` ·
`_verify.py`

`python _verify.py` re-hashes every described file and re-counts its rows.
Last run: **38/38 manifests match on sha256, byte size and row count.**
