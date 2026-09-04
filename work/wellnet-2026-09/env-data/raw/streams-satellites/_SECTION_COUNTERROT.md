# Counter-rotators, kinematically decoupled cores and POLAR gas discs

Lane: `env-data/raw/streams-satellites/`, file prefix `crot_`.
Acquired 2026-09-03/04 UTC. Every file listed here has a sibling
`<name>.manifest.json` carrying source URL, UTC ISO-8601 retrieval timestamp,
SHA-256, byte size, row count, column names with units, the exact query issued,
and a `measurement_or_model` label. 50 data files, 50 manifests, 0 unlabelled.

## Why this lane exists, and the one number that matters

A rotation curve measures the gravitational field in **one** plane. What this
programme needs is a *second, independent direction* of the field in the *same*
baryonic system. Ranked by how much new directional information each
configuration actually buys:

| Configuration | Angle between the two components | Directional information |
|---|---|---|
| **Polar gas disc / inner polar structure** | **~90°** | **Two orthogonal planes — the prize** |
| Kinematic misalignment, intermediate | 30–150° | Partial second direction |
| Counter-rotating disc / 2σ / CRC / KDC | ~180° | **Same plane, opposite sense — NO new direction** |

This distinction is load-bearing and is applied throughout below. A
counter-rotating disc is *not* a substitute for a polar one: anti-parallel
angular momentum probes the identical plane. Several of the catalogues the brief
named are ~180° catalogues, and they are labelled as such rather than being
counted as polar.

---

## HEADLINE: systems with BOTH components kinematically measured

| Catalogue | Survey | N with both components measured | Misalignment angle measured? |
|---|---|---|---|
| Ristea+ 2024 `J/MNRAS/527/7438` | MaNGA | **1899** (stellar **and** gas V at 1 R_e) | No — amplitudes only |
| Ristea+ 2022 (arXiv:2210.01147) | SAMI DR3 | **1445** | Yes, but aggregate counts only |
| Raimundo+ 2023 `J/other/NatAs/7.463` | SAMI DR3 | **1310** | **Yes — per galaxy, with 3σ errors** |
| Bryant+ 2019 (arXiv:1811.09298) | SAMI | **622** (486 GAMA + 136 cluster) | Yes, but aggregate counts only |
| Barrera-Ballesteros+ 2014 (arXiv:1405.5222) | CALIFA | **80** | **Yes — per galaxy** |
| Long-slit series (Corsini/Pizzella/Vega Beltrán/Sarzi) | — | **49** | Resolved curves for both components |
| Bevacqua+ 2022 `J/MNRAS/511/139` | MaNGA | **42** of 64 CRD candidates carry ΔPA | Yes — per galaxy |
| Moiseev 2012 (arXiv:1204.4437) | literature | **39** (host PA₀ + polar PA₁); **28** with Δi | **Yes — DEPROJECTED Δi** |
| ATLAS3D II (arXiv:1102.3801) | ATLAS3D | **30** (two *stellar* components) | ~180°, not an angle catalogue |
| Beom+ 2022 (arXiv:2206.00682) | MaNGA edge-on | **10** | ~180° counter-rotation |
| Combes+ 2013 `J/A+A/554/A11` | SPRC | **9** kinematically confirmed polar rings | Confirmation flag only |

## HEADLINE: the ~90° POLAR subsets, with exact counts

| Source | Polar criterion | **N polar** | Angle type |
|---|---|---|---|
| **Raimundo+ 2023** | \|ΔPA−90\| ≤ 30° (60–120°) | **47** | projected, gas vs stars |
| | \|ΔPA−90\| ≤ 20° (70–110°) | **31** | |
| | \|ΔPA−90\| ≤ 10° (80–100°) | **19** | |
| | as above **and** 3σ error ≤ 30° | **45** | |
| **Moiseev 2012 inner polar structures** | \|Δi−90\| ≤ 20° | **22** | **deprojected, plane vs plane** |
| | \|Δi−90\| ≤ 15° | **20** | |
| | \|Δi−90\| ≤ 10° | **18** | |
| **Combes+ 2013** | kinematically confirmed polar ring | **9** | ring vs host |
| García-Lorenzo+ 2015 | \|PA_gas,kin − PA_phot\| ≥ 75° | **7** (16 at ≥60°) | gas vs **photometry** |
| Bevacqua+ 2022 | \|ΔPA−90\| ≤ 30° | **4** (3 at ≤20°) | projected, gas vs stars |
| Bryant+ 2019 | 40–140° band (published binning) | **29** (24 GAMA + 5 cluster) | aggregate only |
| ATLAS3D II | Ψ ≥ 75° | **4** | **stellar kin vs photometric — see caveat** |
| Barrera-Ballesteros+ 2014 | any | **0** | non-interacting control, all aligned |

**The two catalogues that actually deliver polar systems with both components
measured are Raimundo+ 2023 (47 projected) and Moiseev 2012 (22 deprojected).**
Moiseev is the smaller but *stronger* set: Δi is a genuine angle between two
planes, not an on-sky projection.

---

## 1. ATLAS3D kinematic classification

### 1a. ATLAS3D II — Krajnović+ 2011, MNRAS 414, 2923 — **NOT IN VizieR**

`crot_atlas3d_II_krajnovic2011_kinclass.tsv` — **260 rows** (asserted == 260, the
ATLAS3D early-type sample; 260 unique names).

`Vizier.find_catalogs` over the full VizieR description index returns ATLAS3D
**I, III, IV, VII, XXIII, XXIX, XXX, XXXI** but **no II**. Also absent from
`J/A+A/635/A129` (Krajnović+ 2020). Obtained instead from the arXiv LaTeX source
(arXiv:1102.3801, member `krajnovic_A3D_kinmis.tex`), which holds the table as a
single `deluxetable` — verified: 1 deluxetable, 1 `\startdata` block, 0 `table*`.

Columns: `Name, PAphot, e_PAphot, eps, e_eps, PAkin, e_PAkin, Psi, k5k1, e_k5k1,
k1max, Morph, Dust, KinStruct, Group`.

**Subsample counts (exact):**

| Subsample | N | Kinematic group |
|---|---|---|
| **2σ (counter-rotating disc)** | **11** | group d = 11 ✓ |
| KDC | 11 | group c = 19 ✓ |
| CRC (counter-rotating core) | 8 | |
| KDC + CRC | **19** | |
| **Combined 2σ+KDC+CRC subset file** | **30** | |

`KinStruct` breakdown: RR/NF 171, RR/2m 36, NRR/NF 12, NRR/KDC 11, NRR/2s 7,
NRR/CRC 7, NRR/LV 7, RR/2s 4, RR/KT 2, RR/CRC 1, U 2.
Group breakdown: e 209, c 19, b 12, d 11, a 7, f 2.

Subset file: `crot_atlas3d_II_2sigma_KDC_CRC_subset.tsv` (30 rows).

> **CAVEAT THAT CHANGES HOW THIS TABLE MAY BE USED.** The Ψ column is the angle
> between the **stellar kinematic axis** and the **photometric axis of the same
> stellar body**. It is **NOT** a gas-versus-stars misalignment. Ψ ≥ 75° holds
> for only 4 of 260 and Ψ ≥ 85° for **none** (median Ψ = 2.7°, max 81.4°).
> Reading ATLAS3D II's Ψ as a polar-gas indicator would be a category error.
>
> Likewise the 2σ, KDC and CRC systems are **~180° anti-parallel**, not polar:
> two stellar components sharing one plane. They give a second *sense*, not a
> second *direction*.

**MEASUREMENT.** PAphot/eps photometric; PAkin from the SAURON mean **stellar**
velocity map by kinemetry; Ψ arithmetic from two measurements; k5/k1 and k1max
are kinemetric harmonic amplitudes of the observed field; KinStruct/Group are
classifications *of* measurements. No DM halo, no Jeans/JAM, no halo-dependent
M/L in any column.

### 1b. Supporting ATLAS3D tables (VizieR)

| File | Catalogue | Rows | Content | Label |
|---|---|---|---|---|
| `crot_atlas3d_I_cappellari2011.raw.tsv` | `J/MNRAS/413/813` | **871** | Parent sample: positions, V_hel, distances, M_K, T-type | MEASUREMENT |
| `crot_atlas3d_III_emsellem2011.raw.tsv` | `J/MNRAS/414/888` | **260** | λ_Re, V/σ, ε, slow/fast rotator | MEASUREMENT |
| `crot_atlas3d_XXIII_krajnovic2013.raw.tsv` | `J/MNRAS/433/2812` | **122** | Photometric (MGE) decomposition | MEASUREMENT (light-profile fit) |

**Host photometry is available** for the whole ATLAS3D sample via I + XXIII.

**ATLAS3D project page: FAILED.** `https://www-astro.physics.ox.ac.uk/atlas3d/`
returns an 867-byte HTML **frameset** and nothing else; requesting the frame
targets it names (`menu.html`, `main.html`) returns the byte-identical frameset
(MD5 `2c4456014e018c964ac0f371240ba07a` for all three). No data page, table or
download link is reachable. Recorded, not worked around.

---

## 2. MaNGA counter-rotating galaxies

### 2a. Bevacqua+ 2022, MNRAS 511, 139 — `J/MNRAS/511/139` ✓

`crot_manga_crd_bevacqua2022.raw.tsv`. Two tables:
- `table1`: **64** counter-rotating-disc (CRD) candidates ✓ matches the paper
- `tableg18`: **4460** parent MaNGA galaxies with λ_Re, ellipticity, fast/slow

Per-galaxy `MaNGAId` present throughout, so every object is directly
look-up-able in the MaNGA DAP. `DPA` = measured stellar-vs-gas kinematic PA
difference, present for **42** of the 64.

ΔPA distribution (15° bins): 0–15: 23, 15–30: 9, 30–45: 4, 45–60: 0, 60–75: 0,
75–90: 1, 90–105: 1, 105–120: 1, 120–135: 1, 135–150: 1, 150–165: 1.
**Polar \|ΔPA−90\| ≤ 30°: 4** (≤20°: 3). Counter-rotating ΔPA ≥ 150°: 1.

Note `Pop`: gas co-rotates with the *young* disc in 16, the *old* disc in 2.
Most of the 64 sit at ΔPA ≈ 0 because the counter-rotation is between two
**stellar** discs found by population decomposition, not gas versus stars.

**MEASUREMENT** for kinematics; `logM*` is a stellar-population **model**
quantity. No DM halo, no JAM.

### 2b. Gasymov+ 2025 — `J/ApJS/281/19` ✓ (bonus, not in the brief)

`crot_manga_counterrot_gasymov2025.raw.tsv`. Tables: **65** / **55** / **468**.
Stellar counter-rotation classification from MaNGA velocity maps.
`CRConfig` (table2): inner 35, outer 23, unclear 5, undef 2.
**No misalignment-angle column.** MEASUREMENT (classification of measured maps).

### 2c. Ristea+ 2024 — `J/MNRAS/527/7438` ✓ — the biggest two-component set

`crot_manga_kincat_ristea2024.raw.tsv`, **4215 rows**. Carries **both** stellar
and ionised-gas rotation velocities at 1, 1.3 and 2 R_e, inclination-corrected,
with errors, plus V/σ for each component.

| Radius | Stellar V | Gas V | **BOTH** |
|---|---|---|---|
| 1 R_e | 2683 | 3430 | **1899** |
| 1.3 R_e | 2211 | 3047 | **1458** |
| 2 R_e | 530 | 1019 | **235** |

**No misalignment-angle column** — it gives the two rotation *amplitudes*, not
the angle between the two spin axes. To use it for this programme it must be
joined to an angle catalogue on `MaNGAId`/`Plateifu`.
**MEASUREMENT**; `logMstar`, `logSFR` are stellar-population model quantities.

### 2d. Beom+ 2022 — arXiv:2206.00682 ✓

`crot_beom2022_manga_edgeon_counterrot_gas.tsv`, **10 rows** (asserted == 10).
MaNGA **edge-on** galaxies with a counter-rotating **gaseous** disc, each with
MaNGA-ID, plate-IFU, SDSS ID, RA/Dec, z, distance, log M*.
Geometry ~180°, **not** polar — but because the hosts are edge-on the projection
factor is near unity, so both rotation curves are measured with minimal
inclination correction. Table pulled from `Beom_2022_arXiv.tex`.

### 2e. Xu+ 2022 — arXiv:2202.04937 — **aggregate counts only**

`crot_xu2022_manga_misalign_counts.tsv`, 5 rows. Aligned **6502**; misaligned
**456** (SF 72, GV 142, QS 242); total **6958** → 6.6% misaligned.
**No per-galaxy list exists in the source.** The table is `\input{}` from
`sections/2-data.tex`; parsing only `paper.tex` finds zero tables.
This same 456-galaxy misaligned sample underlies **Zhou+ 2022** (MNRAS 515,
5081), which likewise publishes no per-galaxy list.

### 2f. MaNGA papers that yielded NOTHING — stated plainly

| Paper | Result |
|---|---|
| **Jin+ 2016, MNRAS 463, 913** (arXiv:1611.00528) | **NO DATA TABLE AT ALL.** Zero `tabular`/`deluxetable`/`\input{}` table environments in the source; no VizieR catalogue; no data-availability statement. The **66** kinematically misaligned MaNGA galaxies are **not recoverable** as IDs. |
| **Bao+ 2022** (arXiv:2202.03848) | Zero table environments. Interpretation paper on formation scenarios. |
| **Zinchenko+ 2023, A&A 674, L7** (arXiv:2305.13387) | Zero table environments. **CORRECTION: this is Zinchenko, not Bao** — I initially mislabelled it. The file is still named `crot_bao2023_*` (kept so the SHA-256 stays valid); its manifest carries the correction and `filename_is_misleading: true`. |
| **Zhou+ 2022, MNRAS 515, 5081** | No per-galaxy list; shares Xu+ 2022's 456-galaxy sample. |

---

## 3. CALIFA

### 3a. Barrera-Ballesteros+ 2014, A&A 568, A70 — arXiv:1405.5222 ✓ — **a clean negative**

`crot_bb2014_califa_gas_vs_stellar_PAkin.tsv`, **80 rows** (asserted == 80).
Per-galaxy **stellar** and **ionised-gas** kinematic PAs (approaching and
receding sides fitted separately, with errors), joined on CALIFA id, plus the
derived gas-vs-stars ΔPA.

> **Measured ΔPA(gas−stars) spans only 0.3°–23.1°, median 4.7°. All 80 are
> ALIGNED (≤30°). ZERO polar. ZERO counter-rotating.**

This is the paper's own result, not an extraction defect: the sample is
**non-interacting by selection** and the paper is titled *"Kinematic **alignment**
of non-interacting CALIFA galaxies"*. **Use it as the aligned control
population.** Anyone coming here for CALIFA counter-rotators will find none.

*Extraction guard:* both tables are `\input{}` from separate members
(`tabletex_Skin_prop.txt`, `tabletex_Gkin_prop.txt`); the main
`noInter_final.tex` has **zero** tabular environments. Worse, **each of those
two files itself contains TWO `tabular` environments** — my first parse took the
first `\midrule…\bottomrule` block only and silently returned **51 of 80** rows.
The row-count assertion caught it. This is the brief's "59 of 100" trap, live.

### 3b. Other CALIFA tables

| File | Catalogue | Rows | Content | Label |
|---|---|---|---|---|
| `crot_califa_gaskin_garcialorenzo2015.raw.tsv` | `J/A+A/573/A59` | **177** | Ionised-gas velocity fields: V_sys, gas kinematic PAs (PA_rec, PA_app, PA_min), kinematic centre offsets | MEASUREMENT |
| `crot_califa_starkin_falconbarroso2017.raw.tsv` | `J/A+A/597/A48` | **300** | Stellar kinematics: PA, ε, R_eff, λ_Re | MEASUREMENT |
| `crot_califa_angmom_falconbarroso2019.raw.tsv` | `J/A+A/632/A59` | **291** | Stellar angular momentum λ_Re | MEASUREMENT |
| `crot_s0_morphokin_mendezabreu2018.raw.tsv` | `J/MNRAS/474/1307` | 102×3 | S0 bulge/disc/bar photometric decomposition | MEASUREMENT (parametric light fit) |
| `crot_califa_kinclass_kalinova2017.raw.tsv` | `J/MNRAS/469/2539` | 238 + 50 + 238 | Circular-velocity-curve classes | ⚠ **MODEL** |

**García-Lorenzo+ 2015 misalignment:** 165 galaxies have both a morphological PA
and a gas kinematic PA. \|PA_gas,kin − PA_phot\| folded to 0–90°: 0–15: 99,
15–30: 25, 30–45: 17, 45–60: 8, 60–75: 9, 75–90: 7. **≥60°: 16; ≥75°: 7.**
Weaker than a gas-vs-*stellar-kinematics* misalignment, because PA1 is
photometric, not the stellar kinematic axis.

> ⚠ **Kalinova+ 2017 is MODEL-CONTAMINATED.** `(M/L)dyn` is the *"median of the
> posterior distributions of the dynamical mass-to-light ratios, **from JAM-MCMC
> model**"*, and `betaz` likewise. The circular-velocity curves and CVC classes
> are JAM products. `PA, eps, Incl, Re, Rmax, Vsys` are measurements. Do **not**
> treat the V_circ curves or (M/L)dyn as observations.

---

## 4. Gas–stellar misalignment catalogues with a MEASURED angle

### 4a. Raimundo+ 2023, Nat. Astron. 7, 463 — `J/other/NatAs/7.463` ✓ **BEST IN LANE**

`crot_kinangles_raimundo2023.raw.tsv`, **1310 rows** — SAMI DR3.
Every row carries `PAs` (stellar kinematic axis), `PAg` (gas kinematic axis),
`DPA = |PAs − PAg|`, **and 3σ uncertainties on all three**.
**All 1310 have both components kinematically measured.**

ΔPA distribution (15° bins): 0–15: **961**, 15–30: 179, 30–45: 36, 45–60: 19,
60–75: 15, 75–90: 14, 90–105: 13, 105–120: 5, 120–135: 7, 135–150: 10,
150–165: 11, 165–180: **36**.

| Class | Criterion | N |
|---|---|---|
| Aligned | ΔPA ≤ 30° | 1140 |
| **POLAR** | 60–120° | **47** |
| **POLAR** | 70–110° | **31** |
| **POLAR** | 80–100° | **19** |
| POLAR + 3σ err ≤ 30° | 60–120° | **45** |
| Counter-rotating | ΔPA ≥ 150° | 51 |

Polar subset written to `crot_raimundo2023_POLAR_subset.tsv` (**47 rows**,
sorted by \|ΔPA−90\|). **MEASUREMENT**; `logMstar` is a stellar-population model
quantity; no DM halo, no JAM.
*Caveat:* ΔPA is a **projected** on-sky angle, not a deprojected 3-D angle
between angular-momentum vectors.

### 4b. Bryant+ 2019, MNRAS 483, 458 — arXiv:1811.09298 — **aggregate only**

`crot_bryant2019_sami_misalignment_stats.tsv` (7 rows) and
`crot_bryant2019_sami_misalignment_by_morphology.tsv` (28 rows).

**No per-galaxy misalignment catalogue exists** — the arXiv tarball holds
exactly two tables, both statistical summaries, and there is no VizieR
catalogue. Individual IDs and their PA offsets are **not recoverable**.

| PA offset | GAMA (All) | Field | Groups | Clusters |
|---|---|---|---|---|
| 0–180° (denominator) | **486** | 192 | 294 | **136** |
| > 30° | 55 (0.11±0.01) | 14 | 41 | 15 (0.11±0.03) |
| > 40° | 42 | 11 | 31 | 9 |
| 30–150° | 38 | 12 | 26 | 11 |
| **40–140° (polar band)** | **24** | 8 | 16 | **5** |
| > 140° | 18 | 3 | 15 | 4 |
| > 150° (counter-rot) | 17 | 2 | 15 | 4 |

**622 galaxies have both stellar and gas PAs fitted**; the 40–140° band —
the closest published proxy for polar — holds **29** (24 + 5).

*Extraction guard:* both tables are `\input{}` from separate archive members
(`MisalignmentSummaryTable`, `MislignmentStatsByMophology`); parsing only the
main `.tex` returns zero rows.

### 4c. Ristea+ 2022, MNRAS 517, 2677 — arXiv:2210.01147 — aggregate only

`crot_ristea2022_sami_misalignment_fractions.tsv`, 5 rows. SAMI DR3 parent
sample = **1445** galaxies with both components measured:
all misaligned (30–180°) **169**; "unstable" 30–150° **95**;
counter-rotating 150–180° **53**; aligned <30° **1276**.
The 95 in the 30–150° band **contain** the ~90° polar systems but the paper does
not separate them. No per-galaxy list.

### 4d. Jin+ 2016 — see §2f. **Nothing obtainable.**

---

## 5. Polar-ring galaxies

### 5a. Moiseev 2012, "Inner Polar Rings and Disks" — arXiv:1204.4437 ✓ **STRONGEST GEOMETRY**

`crot_moiseev2012_inner_polar_structures.tsv`, **47 rows** (asserted == 47, the
paper's own stated count).

Columns include host disc `PA0`/`i0`, inner polar structure `PA1`/`i1`, and
**`Delta_i` — the angle between the two planes, DEPROJECTED where both
inclinations are known.** This is a genuine plane-vs-plane angle, strictly
stronger than the projected ΔPA of the IFU surveys.

- **39** rows have both host PA₀ and polar-structure PA₁.
- **28** rows have a Δi value (the paper states 27 — see reconciliation below).
- Sorted Δi: 26, 50, 54, 55, 62.5, 65, 73, 74, 76, 77.5, 80, 80, 80, 84, 86, 86,
  87, 87, 87, 88, 88, 89, 89, 90, 90, 90, 90, 90.

| Polar criterion | N |
|---|---|
| \|Δi−90\| ≤ 20° | **22** |
| \|Δi−90\| ≤ 15° | **20** |
| \|Δi−90\| ≤ 10° | **18** |
| Δi ≥ 80° | **18** |

Tracer of the inner polar structure: Hα/ionised gas 41, stars 10, CO 5, HI 2 →
**45 gas-traced, 10 star-traced** (some galaxies have both).

*Count reconciliation:* I recover **28** non-empty Δi cells against the paper's
**27**. The extra one is **NGC 4233**, whose Δi cell in the source literally
reads `--    80` — simultaneously a dash and the number 80. Excluding it gives
exactly 27. **Treat NGC 4233's Δi as unreliable**; subtract 1 from the ≥80° and
≤10° counts to exclude it.

*Extraction guard:* Table 1 is **split across two `tabular` environments** (the
second captioned `(continue)` after `\setcounter{table}{0}`). Block 1 alone
gives **32 of 47**. Both parsed; total asserted. A second bug: TeX superscript
stripping left a bare `^` in values like `90^*`, silently dropping every starred
(projected-estimate) value and yielding 19 instead of 28 — caught by comparing
against the paper's stated 27.

### 5b. Moiseev+ 2011 SPRC — `J/MNRAS/418/244` ✓

`crot_sprc_moiseev2011.raw.tsv`, **275 rows** ✓ matches the paper.

| Type | N |
|---|---|
| B = best candidate | **70** |
| G = good candidate | 115 |
| R = related object | 53 |
| P = possible face-on ring | 37 |

248 have a heliocentric cz.

> **The SPRC contains NO kinematics — neither of the ring nor of the host.**
> `Type` is a **morphological** classification from SDSS imaging. On its own this
> catalogue does **not** establish that any ring is kinematically decoupled. It
> is a target list, not a measurement of two components.

### 5c. Combes+ 2013 CO — `J/A+A/554/A11` ✓ — the kinematic confirmations

`crot_prg_co_combes2013.raw.tsv`: **21** SPRC galaxies observed in CO(1–0)/(2–1)
(+ 10 spectra rows). The `n_SPRC = 'C'` flag marks objects **kinematically
confirmed** as polar rings: **9 galaxies** — SPRC **7, 10, 14, 33, 39, 60, 67,
69, 260**. MEASUREMENT.

### 5d. HI surveys of polar-ring galaxies ✓

| File | Catalogue | Rows | Content |
|---|---|---|---|
| `crot_prg_hi_huchtmeier1997.raw.tsv` | `J/A+A/319/401` | 44 + 38 | HI velocities, widths, M_HI, M_HI/L_B |
| `crot_prg_hi_vandriel2002.raw.tsv` | `J/A+A/386/140` | 33 | V_opt, V_HI, W50, W20, HI flux, M_HI |

van Driel+ 2002: 17 have V_opt, 18 have V_HI, 18 have W50, **10 have both V_opt
and V_HI**. Types: P 22, G 9, K 2.
Huchtmeier 1997 table1: 32 have optical HRV, 36 have V_HI, 36 have dv20.

> **W50/W20 are GLOBAL single-dish HI linewidths.** They are an integrated
> kinematic *amplitude*, **not** a resolved ring rotation curve, and they do
> **not** separate ring gas from host gas. These catalogues do not by themselves
> give "ring AND host kinematics".

### 5e. Whitmore+ 1990 PRC — **not obtained directly; 88 IDs recovered indirectly**

The original Polar Ring Catalogue (Whitmore+ 1990, AJ 100, 1489; 157 objects,
of which only 6 "class A" were kinematically confirmed at publication) is **not
in VizieR** and predates arXiv, so no e-print exists. **Not acquired.**

However **88 distinct PRC identifiers are recoverable** from the `PRC` columns of
the two HI catalogues above — by class: **C 47, D 18, B 17, A 6**. All **6
class-A (kinematically confirmed) objects are present**. This is a partial
recovery of the PRC's identifier space, not the catalogue itself.

### 5f. Also acquired, not yet transcribed

`crot_moiseev2014_prg_kinematics.eprint.tar.gz` (arXiv:1410.3607, "Structure and
kinematics of the polar ring galaxies: new observations and estimation of the
dark halo shape") and `crot_moiseev2011_sprc_paper.eprint.tar.gz`
(arXiv:1107.1966, the SPRC paper, 6 table environments). Raw sources with
manifests. **Note for the 2014 paper: its dark-halo-shape estimates are MODEL
output and must not be used as observations; the ring/host velocity fields
themselves are measurements.**

---

## 6. Long-slit gas + stellar kinematics — 49 galaxies, both components resolved

Small but exceptionally clean: these give **resolved rotation curves for BOTH
the ionised gas and the stars along the same slit**, which is exactly the
two-component measurement this programme needs, decades before IFUs.

| File | Catalogue | Gas gals | Stellar gals | **BOTH** | Data points (gas + stars) |
|---|---|---|---|---|---|
| `crot_cp_corsini1999.raw.tsv` | `J/A+A/342/671` | 6 | 6 | **6** | 304 + 292 |
| `crot_cp_vegabeltran2001.raw.tsv` | `J/A+A/374/394` | 15 | 17 | **15** | 666 + 453 |
| `crot_cp_corsini2003.raw.tsv` | `J/A+A/408/873` | 9 | 9 | **9** | 892 + 316 |
| `crot_cp_pizzella2004.raw.tsv` | `J/A+A/424/447` | 17 | 17 | **17** | 2339 + 366 |
| `crot_cp_corsini2002.raw.tsv` | `J/A+A/382/488` | NGC 2855 | | **1** | 126 + 32 |
| `crot_cp_sarzi2000.raw.tsv` | `J/A+A/360/439` | NGC 4672 | | **1** | 59 gas + 35 stars |
| | | | | **49 total** | |

Sarzi+ 2000 covers **NGC 4672** along **both major and minor axes** (maj 72,
min 22 points) — NGC 4672 also appears in Moiseev 2012 with **Δi = 88–90°**, so
this is a genuinely polar system with resolved long-slit kinematics on two axes.
All MEASUREMENT; no DM model anywhere in this series.

---

## 7. Model-contaminated or failed — do not treat as observations

| Item | Status |
|---|---|
| **MaNGA DynPop VII, Zhu+ 2025** `J/ApJS/280/55` | **ACQUISITION FAILED.** Four VizieR query forms (`-out.all&-out.max=unlimited`; `&-out.max=999999`; `-source=…/table1`; `-out.max=unlimited&-out=**`) all returned HTTP 200 with the full metadata/column header block and **zero data rows**. No substitute used. Low cost: its headline `Vc(Re)/Vcmax` circular-velocity curves are **JAM model** output — the table carries an explicit `Qual = JAM model quality (−1 to 3)` column — so the hard rule forbids treating them as observations anyway. Header-only response kept as the record. |
| **Kalinova+ 2017** `J/MNRAS/469/2539` | ⚠ **MODEL.** `(M/L)dyn` and `betaz` from JAM-MCMC; circular-velocity curves and CVC classes are model products. PA, eps, Incl, Re, Rmax, Vsys are measurements. |
| **Zhong+ 2026** `J/A+A/707/A137` (193 ATLAS3D + 933 MaNGA) | ⚠ **MIXED.** `LR(Re)`, `Ell(Re)` are MEASUREMENTS. `LRintr(Re)`, `Inc`, `kapparot`, `fspheroid`, `fhalo` are orbit-superposition (Schwarzschild) **model** outputs. Note `fhalo` here is the **stellar** halo mass fraction, *not* a dark-matter fraction — but it is still model output, and the orbit machinery embeds a mass model, so it is DM-assumption contaminated at one remove. |
| **ATLAS3D project page** | **FAILED** — frameset only, byte-identical for every path. |
| **Jin+ 2016 / Bao+ 2022 / Zinchenko+ 2023** | **No data tables exist** in the sources. |
| `logM*`, `logMstar`, `logSFR` everywhere | Stellar-population **model** quantities, not dynamical masses. Labelled in every manifest. |

---

## 8. Failure-mode checks the brief demands

- **VizieR HTTP-200-HTML for a nonexistent `-source=`** — `assert_vizier_tsv()`
  run on **all 24** VizieR fetches with `expect_catalog`; every catalogue id
  echoed back and verified. 1 of 24 failed (`J/ApJS/280/55`) and is recorded as
  a failure, **not silently substituted**. 23 succeeded.
- **LaTeX table split across environments** — hit **three times**, caught three
  times by row-count assertions: Moiseev 2012 (32 of 47 from block 1 alone);
  Barrera-Ballesteros 2014 (**51 of 80** from the first `\midrule` block);
  Bryant 2019 and Xu 2022 (tables `\input{}` from separate archive members —
  zero rows from a main-file-only parse). ATLAS3D II verified clean
  (1 deluxetable, 1 `\startdata`, 0 `table*`) before extraction.
- **Row counts asserted and cross-checked against stated sample sizes** — 10/10
  PASS: ATLAS3D 260/260, 871/871, 2σ 11/11, group c 19/19; Bevacqua 64/64;
  SPRC 275/275; Bryant 622/622; Ristea 1445/1445; Raimundo 1310/1310;
  Moiseev 47/47; BB2014 80/80; Beom 10/10.
- **Multi-table VizieR responses** — `asu-tsv` concatenates every table of a
  catalogue into one payload, so a naive line count is inflated by the
  interleaved headers of later tables. Per-table counts parsed and stored in
  each manifest's `tables_detail`.
- **Shared-denominator artefacts** — not applicable here (no correlation or
  ratio statistic computed; this is acquisition only). **Flag for the analysis
  stage:** ΔPA is built *from* PA_stellar and PA_gas, so correlating ΔPA against
  anything that itself depends on those angles (inclination-corrected V_rot,
  λ_Re) shares an input — simulate the null with the real error covariance.
- **Monotone-invariant statistics / refitting on held-out sets** — not
  applicable; no rank statistic and no fitting in this lane.
- **KiDS and wide binaries** — **neither touched**. Nothing here loads, reads or
  references KiDS or any wide-binary catalogue.
- **No downloaded code executed.** Tarballs were read only; extraction rejects
  absolute paths and `..` traversal.

## 9. Honest assessment of what this lane can and cannot support

**Can support.** A polar-configuration test with **47 projected-polar SAMI
galaxies** (Raimundo+ 2023, with 3σ errors and full velocity fields retrievable
from SAMI DR3) plus **22 deprojected-polar systems** (Moiseev 2012, Δi within
20° of 90°, tracer identified per object). Both components are kinematically
measured in every one. Add **9 kinematically confirmed polar rings** (Combes+
2013) and **49 long-slit galaxies** with resolved two-component rotation curves.

**Cannot support, and should not be claimed.**
1. **No catalogue in this lane provides a deprojected polar rotation curve
   ready to use.** Raimundo gives an *angle*, not the two rotation curves;
   Ristea 2024 gives two *amplitudes*, not the angle. Getting both for the same
   galaxy requires joining catalogues on MaNGA-ID/SAMI CATID and pulling the
   velocity fields from the DAP — that work is **not** done here.
2. **The polar samples are small.** 47 (projected) and 22 (deprojected), with
   only 18–19 within 10° of 90°. Any test built on these is power-limited from
   the outset; state the power before looking at residuals.
3. **Projected vs deprojected is not a detail.** A projected ΔPA of 90° does not
   imply a 90° angle between angular-momentum vectors. Only the Moiseev Δi
   column is a true plane-vs-plane angle, and 1 of its 28 values (NGC 4233) is
   unreliable.
4. **The largest requested MaNGA misalignment samples are unobtainable as
   per-galaxy lists** — Jin+ 2016 (66), Xu+ 2022 / Zhou+ 2022 (456), Bryant+ 2019
   (622), Ristea+ 2022 (1445) all publish counts only.
5. **Host baryonic photometry** is available for ATLAS3D (I + XXIII), CALIFA S0s
   (Méndez-Abreu+ 2018) and the Pizzella/Vega Beltrán galaxies, but **not**
   uniformly for the Raimundo polar subset — that must be sourced separately
   before any baryonic-field calculation.
