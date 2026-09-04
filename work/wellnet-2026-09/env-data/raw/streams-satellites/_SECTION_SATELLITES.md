# Section: Satellites as out-of-plane tracers of the host gravitational field

JOB 2 of the three-job acquisition. Lane directory:
`C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data\raw\streams-satellites\`

Every file below has a sibling `<name>.manifest.json` carrying source URL, UTC
retrieval timestamp, SHA-256, byte size, row count, column names with units, the
exact query issued, and a `measurement_or_model` label. 46 data files, 46
manifests, 167.8 MB.

**Sealed holdouts:** KiDS and wide binaries were not loaded, queried or looked at.

---

## 0. The one thing that decides whether this lane is usable

The scientific goal needs, for a single baryonic system, BOTH

* (a) the host disk orientation — inclination (or axis ratio `b/a`) AND position
  angle, because the disk *normal* is what defines "out of plane"; and
* (b) each satellite's projected position and line-of-sight velocity.

(b) is easy and every catalogue below has it. **(a) is the bottleneck, and it is
the thing the satellite surveys themselves mostly do not publish.** Findings:

| Survey | host orientation in the survey's own host table? |
|---|---|
| **SAGA DR3** (Mao+2024) | **YES** — `ba` (axis ratio) + `PA` + `Sersic`, from DESI Legacy Imaging, for all 101 hosts |
| SAGA DR2 (Mao+2021) | **NO** — host table has only position, v, distance, K magnitude, coverage |
| **ELVES** (Carlsten+2022) | **NO** — host table has distance, v_rec, M_Ks, M_V, B−V, log M*, r_cover, and nothing else |
| Carlsten+2021 ELVES host table (VizieR `J/ApJ/922/267/table1`) | **NO** — 24 hosts, no shape columns |
| McConnachie 2012 / LVDB | host = MW or M31; orientation is external literature knowledge, not a table column |

Because ELVES ships no orientation at all, orientation was acquired separately
from two independent sources and joined (§6). After that join, **125 of 132 host
entries have both an axis ratio and a position angle: 101 of 101 SAGA hosts, and
24 of 31 ELVES hosts.**

The 7 ELVES entries without a usable PA:

* **MW** — the Milky Way, not an external galaxy; orientation is definitional.
* **NGC628, NGC3344, NGC5236, NGC5457** — HyperLEDA returns `--` for PA and all
  four have `b/a > 0.93`. These are genuinely near face-on, where the major-axis
  PA is *physically* ill-defined. Not a data defect: for these hosts the disk
  normal points nearly along the line of sight, so a satellite's angle from the
  normal is set by projected radius alone. They must not be silently treated as
  PA = 0.
* **NGC3379** — `b/a = 0.91`, an elliptical; PA ill-defined for the same reason,
  and it has no disk to be in or out of.
* **NGC4258** — the one real gap. `b/a = 0.389`, `i = 71°`, so the PA is
  perfectly well defined physically, but HyperLEDA simply carries no PA
  measurement for it. Recoverable from another source if this host matters.

`--` in HyperLEDA means "no PA measurement", not "face-on"; the two cases are
distinguished above by inspecting `b/a`. In `sat_host_orientation_crossref.tsv`
`--` is normalised to empty so that `has_orientation` is honest.

For the 101 SAGA hosts the survey's own DESI `PA` is populated for **all 101**
(0 blank, 0 `--`). The HyperLEDA cross-check PA is blank for 9 of them, so those
9 have orientation from one source rather than two.

---

## 1. SAGA DR3 — the primary dataset for this goal

**Mao et al. 2024, ApJ 976, 117.** Official release at `sagasurvey.org/data/`,
AAS machine-readable-table format. **Not in VizieR** — `J/ApJ/976/117` returns an
empty response; the VizieR route reaches only SAGA DR2 (`J/ApJ/907/85`).

| File | Rows | Content |
|---|---|---|
| `sat_saga_dr3_tableC1_hosts.mrt` / `.tsv` | **101** | host catalogue, 31 columns |
| `sat_saga_dr3_tableC3_satellites.mrt` / `.tsv` | **378** | confirmed satellites, 44 columns |
| `sat_saga_dr3_tableC4_candidates.mrt` / `.tsv` | **156** | candidates without reliable redshift, 16 columns |
| `sat_saga_dr3_tableC2_bkgz.mrt` | **75704** | background galaxy redshifts, 17 columns |

Row counts 101 / 378 / 75704 match the release page's stated
"101 MW-mass hosts", "378 identified satellites across 101 systems",
"75,704 background galaxies" **exactly**. `.tsv` files are faithful byte-range
conversions of the fixed-width MRT; the MRT originals are kept unmodified.

**Join integrity (asserted):** all 378 satellites carry a `HOSTID` present in the
host table — **0 orphans**. 378/378 have `DVhost`; 378/378 have `Rhost`. The
satellites are distributed over **97** of the 101 hosts (min 1, median 3, max 13
per host).

### What is measured, and in which direction
* **Out of plane:** `RAdeg`, `DEdeg`, `Rhost` (projected separation, kpc) and
  `DVhost` (line-of-sight velocity offset from the host, km/s) per satellite.
  Combined with host `ba` + `PA` this gives the angle between the satellite's
  projected direction and the host's projected major axis, and hence its position
  relative to the disk plane. **No proper motions** — these hosts are at 25–40 Mpc,
  so the only kinematic component is line-of-sight.
* **In plane:** **NOT PRESENT.** No rotation curve, no HI line width, no
  `W50` for SAGA hosts. The host table gives `log(MHI)` (an HI *mass*) and
  `KsMag`/`rmag`/`gr`/`sb` photometry — enough for a baryonic mass, **not** enough
  for an in-plane rotation velocity. SAGA hosts are all outside the D < 11 Mpc
  Local Volume catalogues, so `W50` could not be recovered from UNGC either
  (0 of 101 matched). **This is the main gap in the SAGA leg.**

### MEASUREMENT vs MODEL — SAGA DR3 host table
* **MEASUREMENT:** `RAdeg`, `DEdeg`, `HRV`, `Dist`, `DistMod`, `KsMag`, `rmag`,
  `gr`, `sb`, `ba`, `PA`, `Sersic`, `log(MHI)`, `nz-saga`, `nz-total`, `nsat-*`,
  `sep-MW`, `sep-massive`.
* **MODEL — presupposes dark matter:** `log(Mhalo)`, a halo mass taken from the
  Lim et al. (2017) group catalogue. **Must never be used as an observation.**
* **Derived with stellar-population assumptions (no dark matter):** `log(M*)`,
  `log(sfr)`.

### MEASUREMENT vs MODEL — SAGA DR3 satellite table
* **MEASUREMENT:** `RAdeg`, `DEdeg`, `Rhost`, `rmag`, `gr`, `sb`, `ba`, `PA`,
  `Sersic`, `z`, `DVhost`, all emission-line fluxes and equivalent widths,
  `NUVmag`, `log(MHI)`.
* **Derived with stellar-population assumptions:** `log(M*)`, `log(sfr)*`,
  `quenched`.
* **No dark-matter-dependent column exists in this table.**

SAGA DR2 was also taken twice for cross-checking — official CSV
(`sat_saga_dr2_hosts_official.csv`, **36**; `sat_saga_dr2_sats_official.csv`,
**127**) and VizieR (`sat_saga_dr2_table2.tsv`, **36**; `sat_saga_dr2_table3.tsv`,
**127**). Counts agree across both routes.

---

## 2. Milky Way satellites

| File | Rows | Source | Notes |
|---|---|---|---|
| `sat_mcconnachie2012_catalog.tsv` | **102** | VizieR `J/AJ/144/4/catalog` | the 2012 published table, 94 columns |
| `sat_mcconnachie2012_refs.tsv` | **319** | VizieR `J/AJ/144/4/refs` | |
| `sat_mcconnachie_updated_Jan2021.fits` | **144** | NRC/CADC PANDAS vault | **the maintained version**, 49 columns, adds `pmra`/`pmdec` |
| `sat_mcconnachie_updated_References.dat` | 689 lines | NRC/CADC | |
| `sat_mcconnachie_updated_Comments.dat` | 36 lines | NRC/CADC | |
| `sat_mcconnachie_updated_Update.log` | 85 lines | NRC/CADC | last update January 2021 |
| `sat_lvdb_v1.1.1_comb_all.csv` / `.ecsv` | **1727** | LVDB v1.1.1 GitHub release | 122 columns |
| `sat_lvdb_v1.1.1_pm_overview.csv` | **790** | LVDB v1.1.1 | per-system systemic PM **with per-entry literature provenance** |
| `sat_pace2022_table1_dsph_properties.tsv` | **54** | arXiv:2205.05699 `table_overview.tex` | |
| `sat_pace2022_table2_systemic_pm.tsv` | **52** | arXiv:2205.05699 `table_results.tex` | |
| `sat_pace2022_table3_orbits_MODEL.tsv` | **46** | arXiv:2205.05699 `table_orbit.tex` | **MODEL** |
| `sat_battaglia2022_pmem.tsv` | **645720** | VizieR `J/A+A/657/A54/pmem` | **member-star probabilities only** |

### Row-count reconciliations (all three checked against the papers, none is an extraction failure)
* **Pace+2022 Table 1 has 54 rows, not the 52 in the abstract.** The abstract's 52
  is the *proper-motion* sample (Table 2). Table 1 additionally lists **Cet III**
  and **Vir I**, which have no PM measurement. Verified by set difference
  Table1 − Table2 = {Cet III, Vir I}.
* **Pace+2022 Table 3 has 46 rows, not 52.** Six dwarfs with no systemic
  line-of-sight velocity cannot be orbit-integrated and are absent: **Boo IV,
  Cen I, Cet II, Hor II, Pic I, Pic II**. Note Table 3 abbreviates Pisces II as
  `Pis II` where Tables 1–2 use `Psc II` — same object; do not treat as a 53rd.
* Each `.tex` was checked for the split-table failure mode: **exactly one
  `\startdata` per file**, so no silently truncated table.

### The Battaglia+2022 premise is wrong — stated plainly
The brief expected a consolidated systemic proper-motion catalogue at
`J/A+A/657/A54`. **It is not that.** The only VizieR table in that catalogue is
`pmem`, which is **645,720 rows of individual Gaia EDR3 stars with a membership
probability** (`Galaxy`, `GaiaEDR3`, `RA_ICRS`, `DE_ICRS`, `Pmemb`). It contains
**no systemic proper motion, no velocity, no distance**. The Battaglia+2022
*systemic* PMs were obtained instead through the LVDB `pm_overview` table, which
carries 69 entries attributed to `Battaglia2022A&A...657A..54B` with the bibcode
recorded per row.

### MEASUREMENT vs MODEL
* **MEASUREMENT:** positions, `distance_modulus`, `distance`, `apparent_magnitude_v`,
  `M_V`, `rhalf`, `position_angle`, `ellipticity`, `vlos_systemic`, `vlos_sigma`,
  `pmra`, `pmdec`, metallicities, `flux_HI`. Also all of Pace+2022 Tables 1 and 2.
* **MODEL — presupposes dark matter:** `sat_pace2022_table3_orbits_MODEL.tsv` in
  its entirety. `r_peri`, `r_apo`, `ecc`, `f_peri` and their `nL` variants are
  produced by integrating each dwarf's measured 6-D phase-space vector **in an
  assumed Milky Way (+LMC) dark-matter halo potential**. The filename carries the
  label. Retained for cross-checking only.
* **Derived, equilibrium-dependent (no NFW halo, but not an observation either):**
  LVDB `mass_dynamical_wolf` — the Wolf et al. (2010) estimator, which assumes
  dynamical equilibrium and spherical symmetry. McConnachie's `Mdyn` likewise.
  Treat as derived, never as data.

### Direction sampled
The MW is the one host where the field is sampled in **three dimensions**:
projected position + line-of-sight velocity + **proper motion** (2 tangential
components). This is the only system in this lane with full 6-D phase space per
tracer. The MW's own disk orientation is definitional (Galactic coordinates), and
`GLON`/`GLAT` are already columns in the McConnachie catalogue, so the angle out
of the Galactic plane is directly available as `GLAT`.

**In-plane data for the MW exists and is excellent** (the Galactic rotation curve),
but is not part of this acquisition.

---

## 3. M31 satellites (PAndAS)

| File | Rows | Source | Content |
|---|---|---|---|
| `sat_martin2016_dsph.tsv` | **23** | VizieR `J/ApJ/833/167/dsph` | structural parameters of 23 M31 dwarfs |
| `sat_martin2016_table3.tsv` | **11500** | VizieR `J/ApJ/833/167/table3` | MCMC chains for those 23 dwarfs (500 samples each) |
| `sat_collins2013_dsph.tsv` | **18** | VizieR `J/ApJ/768/172/dsph` | structural + kinematic properties of 18 M31 dSphs |
| `sat_collins2013_table2.tsv` | **22** | VizieR `J/ApJ/768/172/table2` | spectroscopic observing log |
| `sat_collins2013_table3.tsv` | **295** | VizieR `J/ApJ/768/172/table3` | 295 individual probable member stars with `HRV` |
| LVDB `table == dwarf_m31` | **43** | LVDB v1.1.1 | 37 with `vlos_systemic`, 34 with `vlos_sigma`, only 8 with a PM |

**Caution — the VizieR table names are not `table1`.** `J/ApJ/833/167/table1` and
`J/ApJ/768/172/table1` **do not exist**; VizieR returns HTTP 200 with
`#INFO Error=Table 'table1' does not exist in catalog`. The real names are
`dsph` in both cases. This is the documented VizieR trap in a second form —
`assert_vizier_tsv` caught it.

Martin+2016 has 23 rows matching "the 23 dwarf spheroidal galaxies" in the
catalogue's own table-3 description; 11500 = 23 × 500 chain samples, consistent.

### Direction sampled
Projected position + line-of-sight velocity only (8 of 43 have a PM). M31 is at
~780 kpc, so proper motions are marginal.

### Host disk orientation — available and excellent
M31's disk orientation is among the best-determined of any galaxy. Independently
recovered here from two catalogues rather than asserted:
* UNGC (Karachentsev+2013): `b/a = 0.33`, **inclination i = 78°**
* HyperLEDA (VizieR VII/237): `b/a = 0.355`, **PA = 35.0°**, PGC 2557

Both sit on the standard literature values (i ≈ 77°, PA ≈ 38°), which is a useful
end-to-end validation of the orientation pipeline in §6. Both rows are in
`sat_host_orientation_crossref.tsv`.

**In-plane data for M31 is available:** UNGC gives `W50 = 510 km/s`,
`vAmp = 249 km/s` for M31. So **M31 pairs an out-of-plane tracer population with
an in-plane rotation measure for the same system.**

### MEASUREMENT vs MODEL
* **MEASUREMENT:** all of Martin+2016 (`RAJ2000`, `DEJ2000`, `ell`, `PA`, `rh`,
  `rhpc`, `Vmag`, `VMag`, `logL`, `mu0`); Collins+2013 `RAJ2000`, `DEJ2000`,
  `VMag`, `rh`, `Dist`, `RV`, `sigV`, `[Fe/H]`, and the 295 stellar `HRV` values.
* **Derived, equilibrium-dependent:** Collins+2013 `Mh` (mass within the
  half-light radius) and `[M/L]`. These use a dynamical mass estimator assuming
  equilibrium. **Not an observation.** No NFW fit is involved, but they must not
  enter as data.

---

## 4. ELVES

| File | Rows | Source | Content |
|---|---|---|---|
| `sat_elves_table1_hosts.tsv` | **31** | arXiv:2203.00014 `host_table.tex` | host list — **no orientation columns** |
| `sat_elves_overview_table.tsv` | **31** | arXiv:2203.00014 `overview_table.tex` | satellite counts per host |
| `sat_elves_table9.tsv` | **444** | VizieR `J/ApJ/933/47/table9` | satellite photometry, incl. `Rproj` (kpc) |
| `sat_elves_table6.tsv` | **251** | VizieR `J/ApJ/933/47/table6` | confirmed satellite distances, incl. `vrec` |
| `sat_elves_table7.tsv` | **196** | VizieR `J/ApJ/933/47/table7` | rejected background contaminants |
| `sat_elves_table8.tsv` | **106** | VizieR `J/ApJ/933/47/table8` | unconfirmed candidates |
| `sat_elves_table10.tsv` | **271** | VizieR `J/ApJ/933/47/table10` | GALEX photometry |
| `sat_elves2_table4.tsv` | **177** | VizieR `J/ApJ/927/44/table4` | ELVES II early-type satellites + GC counts |
| `sat_carlsten2021_table1.tsv` | **24** | VizieR `J/ApJ/922/267/table1` | earlier 24-host ELVES host table |
| `sat_carlsten2021_table4.tsv` | **128** | VizieR `J/ApJ/922/267/table4` | LV field dwarf photometry |
| `sat_carlsten2021_table5.tsv` | **223** | VizieR `J/ApJ/922/267/table5` | main LV satellite dwarf photometry |

**Row-count cross-check:** the paper states 338 confirmed satellites and 106
candidates. `table9` has **444 = 338 + 106** exactly. `table6` (251) is smaller
because it lists only satellites with a *new ELVES distance measurement*, not all
confirmed ones — do not mistake 251 for the confirmed-satellite count.

**Host count reconciliation:** the host table has **31** rows. The paper says
"31 such hosts" from the luminosity + distance cut and "30 surveyed hosts". Row 31
is the Milky Way, listed as a reference with `dist = 0`; `NGC3621` has
`r_cover = 0`, i.e. it is in the parent sample but unsurveyed. 31 rows is the
published content.

### Direction sampled
Projected position (`RAJ2000`, `DEJ2000`, `Rproj` in kpc) plus a recession
velocity `vrec` where a redshift exists (`table6`, `table7`). Note that many ELVES
satellites are confirmed by surface-brightness fluctuation or TRGB distance rather
than by redshift, so **the velocity coverage is far from complete** — unlike SAGA,
where all 378 have `DVhost`.

### In-plane data — the strongest pairing in this lane
Every ELVES host is a bright galaxy at D < 12 Mpc, i.e. exactly the population
with resolved HI and long-slit/IFU rotation curves. From UNGC alone, **27 of 31
ELVES hosts have an HI line width `W50` and rotation amplitude `vAmp`** — an
in-plane rotation measure for the same system whose satellites are catalogued.
The four without are NGC1023, NGC1808, NGC3379 (an elliptical) and NGC4565.

### MEASUREMENT vs MODEL
All ELVES tables acquired here are **MEASUREMENT**: sky positions, projected
separations, apparent and absolute magnitudes, colours, surface brightnesses,
Sérsic/effective radii, ellipticities, recession velocities, TRGB and SBF
distances, GC counts. `logMs` (stellar mass) is derived with a colour–M/L
relation — stellar-population assumption, no dark matter. **No ELVES table
acquired contains a halo mass, an NFW quantity or a dynamical mass.**

---

## 5. Cross-cutting catalogue: Local Volume Database

`sat_lvdb_v1.1.1_comb_all.csv` (**1727** rows × 122 columns) is the modern
maintained successor to McConnachie 2012 (Pace 2024, arXiv:2411.07424; OJAp 8,
142). Subsets via the `table` column: `dwarf_mw` **68**, `dwarf_m31` **43**,
`dwarf_local_field` **64**, `dwarf_local_field_distant` **736**, plus star-cluster
and candidate tables. `dwarf_mw`: 62/68 have a proper motion, 56/68 a systemic
`vlos`, 45/68 a `vlos_sigma`.

`sat_lvdb_v1.1.1_pm_overview.csv` (**790** rows, **307** distinct systems) is the
single most useful PM product here because every row names its source bibcode:
Vasiliev2021 162, Battaglia2022 69, McConnachie2020 116 (two papers), **Pace2022
52**, Li2021 46, **Fritz2018 39**, Libralato2026 28, Simon2018 17, and ~20 more.
It therefore covers all three PM catalogues the brief asked for, with provenance,
in one file.

`comb_all.ecsv` is kept alongside the CSV because its YAML header carries the
per-column units that the CSV does not.

---

## 6. Host disk orientation — separately acquired, then joined

Because the ELVES host table publishes no shape information at all, orientation
was acquired from two independent catalogues:

| File | Rows | Source | Gives |
|---|---|---|---|
| `sat_hyperleda_saga_hosts_PA.tsv` | **101** | VizieR `VII/237/pgc`, queried by the PGC ids carried natively in the SAGA host table | `logD25`, `logR25`, `PA` |
| `sat_hyperleda_elves_hosts_PA.tsv` | **30** | VizieR `VII/237/pgc`, 1.5′ cone at Sesame-resolved coordinates, nearest match | `logD25`, `logR25`, `PA` |
| `sat_ungc_karachentsev2013_catalog.tsv` | **869** | VizieR `J/AJ/145/101/catalog` | `b/a`, **inclination `i`**, **`W50`**, **`vAmp`**, `Bmag`, `Kmag` |

The HyperLEDA query returned **101 rows for 101 requested PGCs** — a complete
match. All **30** external ELVES hosts were found in HyperLEDA within the 1.5′
cone; only `MW` is unresolvable, correctly. Of those 30, **24 carry a numeric PA**
and 6 return `--` (see §0).

`sat_host_orientation_crossref.tsv` (**132** rows = 101 SAGA + 31 ELVES) is a
**DERIVED JOIN**, not an upstream product, and is labelled as such in its
manifest. The only arithmetic performed is `b/a = 10^(−logR25)`, HyperLEDA's own
definition. Columns: survey, host name/id, PGC, satellite count, survey `ba`/`PA`/
Sérsic, HyperLEDA `logR25`/`b/a`/`PA`, UNGC `b/a`/`i`/`W50`/`vAmp`/`Kmag`/`Bmag`,
distance, and two boolean summary flags.

**Independent cross-check passes.** For SAGA hosts the DESI Legacy Imaging `ba`/`PA`
and the HyperLEDA `b/a`/`PA` are two independent measurements of the same isophotal
shape and they agree closely (e.g. ESO079-003: 0.196 / 131.6° vs 0.178 / 129°;
ESO288-025: 0.165 / 53.2° vs 0.117 / 54°). Disagreements should be treated as the
systematic floor on the disk-normal direction, not as noise to be averaged away.

Inclination `i` in UNGC is a **geometric inversion of the observed axis ratio**
assuming an oblate disk of finite intrinsic thickness. It is a measured shape, not
a dark-matter-dependent quantity — but it is a *derived* geometric quantity and
should be recomputed from `b/a` if a different thickness convention is wanted.

---

## 7. Which systems pair an out-of-plane tracer with in-plane data

| System | Out-of-plane tracer | Host orientation | In-plane rotation | In-plane photometry |
|---|---|---|---|---|
| **Milky Way** | 68 dwarfs, **full 6-D** (pos + v_los + PM) | definitional (`GLAT`) | not in this lane (exists) | yes |
| **M31** | 43 dwarfs (37 v_los, 8 PM); Martin+2016 23; Collins+2013 18 | **i = 78° (UNGC), PA = 35° (LEDA)** | **W50 = 510, vAmp = 249 km/s** | yes |
| **ELVES hosts, 24 of 31** | 444 satellites, `Rproj`, partial v_rec | **b/a + PA (LEDA), i (UNGC)** | 27 of 31 have **W50 + vAmp (UNGC)** | M_Ks, M_V, B−V, log M* |
| ELVES: NGC628, NGC3344, NGC5236, NGC5457, NGC3379 | same | b/a + i only — near face-on, PA ill-defined | yes except NGC3379 | yes |
| ELVES: NGC4258 | same | b/a + i, **PA missing from HyperLEDA** | yes | yes |
| ELVES: NGC1023, NGC1808, NGC4565 | same | **b/a + PA** | **NO** (absent from UNGC) | yes |
| **101 SAGA hosts** | **378 satellites, all with Rhost + DVhost** | **ba + PA + Sérsic (DESI), cross-checked vs LEDA** | **NO — none** | KsMag, rmag, g−r, sb, log MHI |

The honest summary: **SAGA gives by far the largest and cleanest out-of-plane
sample with host orientation (101 hosts, 378 satellites, zero join orphans), but
no in-plane rotation velocity for any host. ELVES and M31 give the in-plane leg
but with sparser and less uniform satellite kinematics.** Closing the SAGA in-plane
gap would need a separate acquisition — DESI/SDSS spectroscopy or an HI survey
(ALFALFA covers part of the SAGA footprint) for the 101 hosts. That was not part
of this job.

---

## 8. What failed, with the corrected premise

1. **SAGA DR3 is not in VizieR.** `J/ApJ/976/117` returns a zero-length response.
   Corrected by going to the official `sagasurvey.org/data/` release. VizieR
   reaches only DR2.
2. **Pace, Erkal & Li 2022 has no VizieR catalogue.** `J/ApJ/940/136` returns
   `{"error":"ReadMe is not found"}`; a VizieR full-text search on bibcode
   `2022ApJ...940..136P` returns no catalogue id. Corrected via the arXiv source
   tarball (arXiv:2205.05699) plus LVDB `pm_overview`.
3. **Fritz et al. 2018 has no VizieR catalogue.** `J/A+A/619/A103` likewise
   returns `ReadMe is not found`, and a bibcode search on
   `2018A&A...619A.103F` yields nothing. **Its systemic PMs were obtained only
   indirectly**, as 39 provenance-tagged rows in LVDB `pm_overview`. The original
   tables were not retrieved.
4. **IOPscience is bot-blocked.** The Pace+2022 machine-readable tables at
   `iopscience.iop.org/.../apjac997bt{1,2}_ascii.txt` return HTTP 200 with
   **0 bytes**. Corrected via arXiv source.
5. **Battaglia+2022 does not contain what the brief assumed** — see §2.
6. **VizieR table names are not the `.dat` filenames.** The catalogue JSON
   metadata lists `table1.dat`, but the queryable table is `catalog` (McConnachie)
   or `dsph` (Martin, Collins). Five initial fetches returned HTTP 200 carrying
   `#INFO Error=Table 'table1' does not exist`. Caught by `assert_vizier_tsv`;
   resolved by enumerating real names via `Vizier.get_catalogs`.
7. **`NearbyGalaxies.dat` no longer exists** at the CADC vault (HTTP 404). The
   maintained catalogue is now distributed as
   `NearbyGalaxies_Jan2021_PUBLIC.fits` (**144** rows, 49 columns). Last updated
   **January 2021** — it is *not* being refreshed at the promised six-month
   cadence, and the LVDB is the live successor.
8. **The LVDB latest release (v1.1.1) ships fewer per-subset files** than v1.0.6:
   only `comb_all` (csv/fits/ecsv), `pm_overview`, `j_factor`. The `dwarf_mw.csv` /
   `dwarf_m31.csv` splits are not in v1.1.1. Not a problem — `comb_all` carries a
   `table` column that reproduces them exactly. Versions were not mixed.
9. **3 of 31 ELVES hosts are absent from UNGC** (NGC1023, NGC1808, NGC4565):
   their UNGC distances exceed the catalogue's 11 Mpc limit. They have HyperLEDA
   orientation but no UNGC inclination or `W50`.
10. **6 of 30 external ELVES hosts have no HyperLEDA position angle.** Five are
    near face-on (`b/a > 0.9`) where PA is physically ill-defined; **NGC4258 is a
    genuine gap** — `b/a = 0.389`, `i = 71°`, so the PA is well defined but simply
    absent from HyperLEDA. Not substituted with a proxy.
11. **UNGC names Messier objects as `MESSIER031`/`MESSIER081`/`MESSIER101`** and
    zero-pads NGC numbers to four digits (`NGC0253`). A naive name join silently
    loses 6 of 31 ELVES hosts. Fixed with an explicit alias + zero-padding
    normaliser; both the naive and corrected match counts were checked.

## 9. Failure modes from the standing brief — explicitly checked

* **Silent extraction failures:** every VizieR fetch ran through
  `assert_vizier_tsv` with the catalogue id echoed back; five bad table names were
  caught this way rather than being written as empty files. Every LaTeX table was
  checked for the split-`table*` mode (`\startdata` count == 1 in all five files).
  Every row count was compared against the paper's stated sample size, and the
  three that disagreed (Pace T1 54≠52, Pace T3 46≠52, ELVES 31≠30) were each
  traced to a real property of the publication and documented above rather than
  patched.
* **Shared-denominator artefacts:** flagged for downstream use. `Rhost` (projected
  separation) is computed from the host distance, and any host-distance error
  propagates into both `Rhost` and any physical satellite size derived from the
  same distance. Do not correlate a `Rhost`-normalised quantity against another
  distance-scaled quantity without simulating the null with the actual error
  covariance. Likewise `b/a` and `i` are **not independent** — `i` is a
  deterministic function of `b/a`; never use both as separate predictors.
* **Monotone-invariant statistics, refitting on held-out data, solver/test bugs,
  non-monotonic M(r):** not applicable to an acquisition job; no fitting,
  statistics or deprojection was performed here.
* **Sealed holdouts:** KiDS and wide binaries were never loaded, queried or
  inspected.

## 10. Code written (reproduces every file above)

`sat_fetch_vizier.py`, `sat_fetch_web.py`, `sat_extract_latex.py`,
`sat_fetch_host_orientation.py`, `sat_fetch_elves_leda.py`, `sat_build_clean.py`.
No downloaded code was executed; the arXiv tarballs were opened read-only to
extract `.tex` table bodies.
