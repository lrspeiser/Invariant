# Section 1 — Stellar streams as out-of-plane tracers

## 1.0 What decides whether this lane is usable

A rotation curve measures the gravitational field almost entirely **in** the
disc plane. A stellar stream at high Galactic latitude traces the field far
**off** that plane. For the Milky Way, and only for the Milky Way, we hold both
in the same system at once, so this section is built around one question per
track: *does it carry a genuine measured 3-D position and velocity, or only a
sky projection and some filler?*

Two geometric quantities are computed here from the published tracks by pure
coordinate transform — **no potential, no mass model and no halo enters any
column**:

* `orbit_inc_to_disc_deg` = `90 − |pole_b|`, the inclination of the stream's
  orbital plane to the Galactic disc. **0° = the orbit lies in the disc plane
  (no new direction); 90° = a polar orbit, the maximum possible out-of-plane
  leverage.**
* `absz_max_kpc` = the greatest height above the Galactic plane reached along
  the track (astropy default Galactocentric frame, R₀ = 8.122 kpc,
  z_sun = 20.8 pc).

## 1.1 galstreams v1.2.1 — the compiled library (Mateu 2023)

**There is no Zenodo DOI.** The Zenodo API returns `"total": 0` for
`q=galstreams`. The canonical distribution is GitHub + PyPI plus the journal
DOI `10.1093/mnras/stad321`. Acquired from the release tarball
`codeload.github.com/cmateu/galstreams/tar.gz/refs/tags/v1.2.1`
(35,394,796 bytes, tag published 2026-06-03). Only ECSV **data** files were
read; no package code was imported or executed.

* `galstreams_track_summary.tsv` — **217 rows**, one per track, 47 columns.
* `galstreams_paper_supertable.tsv` — **126 rows**, the published table.

### The flag legend, verbatim from `galstreams/core.py`

```
# bit 0: 0 = great circle by construction
# bit 1: 0 = no distance track available (only mean or central value reported)
# bit 2: 0 = no proper motion data available (only mean or central value reported)
# bit 3: 0 = no radial velocity data available (only mean or central value reported)
```

and from the paper: *the first character is 0 if the stream is assumed to be a
great circle and 1 if not; the second, third and fourth characters indicate
whether distance, proper motions and radial velocity tracks are available (1) or
not (0)*, with `2` marking "available but with a caveat".

### Provenance strength of the library itself

The paper states galstreams compiles **direct observables**: *predicted* proper
motion and radial velocity tracks reported in some papers are **excluded**, and
values are given as **observed heliocentric** quantities without solar-reflex
correction. **No Galactic potential is presupposed anywhere in the library.**
Caveat: many tracks were digitised from published figures with WebPlotDigitizer,
so they carry digitisation error, not just measurement error.

### MEASUREMENT vs MODEL census (217 tracks)

| Quantity | Status | N |
|---|---|---:|
| Celestial track | `EMPIRICAL_TRACK` — **MEASUREMENT** | 192 |
| Celestial track | `GREAT_CIRCLE_ASSUMED` — **MODEL of the track shape** | 25 |
| Distance | `MEASURED_TRACK` — **MEASUREMENT** | 69 |
| Distance | `PLACEHOLDER_1KPC` — **NOT DATA** | 68 |
| Distance | `GEOMETRIC_INTERPOLATION` — **MODEL** | 6 |
| Distance | `SINGLE_MEAN_VALUE` / `ABSENT` | 2 / 72 |
| Proper motion | `MEASURED_TRACK` — **MEASUREMENT** | 165 |
| Proper motion | `SINGLE_MEAN_VALUE` / `ABSENT` | 6 / 46 |
| Radial velocity | `MEASURED_TRACK` — **MEASUREMENT** | 98 |
| Radial velocity | `UNPHYSICAL` — **NOT DATA** | 15 |
| Radial velocity | `ABSENT` | 104 |

`GREAT_CIRCLE_ASSUMED` is a **geometric** model of the track shape — a great
circle interpolated between measured end points or about a measured pole. It is
*not* a dark-matter model; no potential is involved. It is still not a measured
track and must not be treated as one.

**Usable subsets:**
* `usable_3d` = empirical sky track **and** measured distance track → **69 tracks / 60 distinct streams**
* `usable_6d` = that, plus measured proper-motion **and** radial-velocity tracks → **33 tracks / 30 distinct streams**

### Upstream defects found in galstreams v1.2.1 — 102 of 217 tracks affected

These are **silent**: the library's own flags advertise data that is not there.
Anyone reading `InfoFlags` at face value ingests placeholders as measurements.

1. **68 `ibata2024` tracks assert a distance track while the distance column is
   identically 1.000 kpc.** (85 ibata2024 tracks carry the placeholder in total;
   68 of them additionally have the flag set.) GD-1 is among them — a stream at
   ~8–10 kpc recorded at 1 kpc. The values carry float round-trip noise
   (`0.9999999999999946`), so an exact `== 1.0` test **silently finds nothing**;
   a tolerance is required. Their `z`/`R_gc` columns are left blank here.
2. **15 tracks advertise `InfoFlags=1111` (full 6-D) but carry unphysical radial
   velocities**, against a Galactic escape speed of ~550 km/s:
   * `track.st.Hydrus.ibata2024` → **9,561,412 km/s (32× the speed of light)**
   * `track.st.NGC1261b.ibata2024` → **−32,929,072 km/s**
   * `track.st.Gaia-2.ibata2021` → −129,958 to +87,217 km/s
   The percentile pattern (sane median, divergent tails) indicates spline
   blow-up at the track ends.
3. **16 tracks have the Vrad flag clear yet a populated, non-constant Vrad
   column** — filler that must not be read as data.
4. **`track.st.Pal5.pricewhelan2019` has Vrad identically 999.0**, a null
   sentinel.
5. **6 tracks have the PM flag set but a constant PM column.**
6. **3 summary files have no track file at all** in the tarball
   (`Jhelum-broad/-narrow/-spur`, viswanathan2023): 220 summaries, 217 tracks.

**Consequence, and the rule applied here:** `InfoFlags` must not be trusted
alone, and the data must not be trusted alone either. The classification rule
used is **the flag governs; the data may only downgrade it, never promote it**.
Letting the data promote a flag-0 column is what wrongly admitted the corrupt
`ibata2024` velocities in an earlier pass. Every disagreement is recorded
per-row in the `data_defects` column.

### Out-of-plane leverage of the usable set

| Orbit inclination to the disc | `usable_3d` | `usable_6d` |
|---|---:|---:|
| 0–30° (nearly in-plane) | 3 | 2 |
| 30–60° | 29 | 14 |
| 60–80° | 21 | 9 |
| **80–90° (nearly polar)** | **16** | **8** |

`|z|max` across `usable_3d`: min 1.06, median 7.94, max **84.30 kpc**.
47 tracks exceed 5 kpc, **26 exceed 10 kpc**, 11 exceed 20 kpc, 2 exceed 40 kpc.

`R_gc,max`: median 13.33 kpc, max 97.54 kpc. **14 tracks extend beyond 25 kpc**,
i.e. past the outer edge of the best in-plane Milky Way rotation curve.

### The best individual out-of-plane tracks

| Track | inc to disc | \|z\|max kpc | R_gc range kpc | pts |
|---|---:|---:|---|---:|
| `Orphan-Chenab.koposov2019` | 56.4° | **84.30** | 14.7–97.5 | 23000 |
| `Orphan-Chenab.koposov2023` | 56.4° | 37.21 | 15.0–58.6 | 18113 |
| `AAU-AliqaUma.li2021` | 87.0° | 27.69 | 28.1–33.3 | 950 |
| `AAU-ATLAS.li2021` | 87.0° | 24.45 | 20.7–27.9 | 2350 |
| `Orphan-Chenab.ibata2021` | 53.4° | 23.01 | 14.1–36.2 | 8660 |
| `LMS-1.yuan2020` | 77.6° | 17.42 | 12.9–21.2 | 17261 |
| `Pal5.ibata2021` | 47.7° | 16.07 | 13.5–17.7 | 2174 |
| `M3-Svol.yang2023` | 88.4° | 13.53 | 8.2–16.0 | 4290 |
| `GD-1.ibata2021` | 60.1° | 9.46 | 14.0–15.5 | 10209 |

`Cetus-Palca.yuan2021` reaches |z| = 38.25 kpc at inclination 81.9° but is
`usable_3d`, not `usable_6d` (no measured Vrad track).

## 1.2 Individual streams — original published tables

Acquired from VizieR with hard validation. **VizieR returns HTTP 200 and a
`#INFO Error=Table or Catalog not found: <id>` line for a nonexistent
`-source=`, and that error line echoes the requested id back**, so echo-checking
alone is insufficient; the shared validator was hardened to fail on any
`Error=` line first.

| File | Catalogue | Rows | Content | Label |
|---|---|---:|---|---|
| `stream_koposov2019_orphan_rrl.vizier.tsv` | `J/MNRAS/485/4726` | **109** | Orphan RR Lyrae: positions, Gaia DR2 PM, PL-relation distances | MEASUREMENT |
| `stream_ibata2021_streamfinder_members.vizier.tsv` | `J/ApJ/914/123` | **5960** | Per-star Gaia EDR3 astrometry + HRV + stream label 1–32 | MEASUREMENT (except `dSF`) |
| `stream_vasiliev2021_sagittarius.vizier.tsv` | `J/MNRAS/501/2279` | **55192** | Sagittarius: astrometry, distance, vLOS, [Fe/H], Λ/β | MEASUREMENT |
| `stream_antoja2020_sgr_pm_map.vizier.tsv` | `J/A+A/635/L3` | **294344** | All-sky Sgr proper-motion map (no distance, no RV) | MEASUREMENT |
| `stream_ishigaki2016_pal5_focas_members.vizier.tsv` | `J/ApJ/823/157/table4` | **19** | Pal 5 FOCAS members, Vlos | MEASUREMENT |
| `stream_ishigaki2016_pal5_deimos_stars.vizier.tsv` | `J/ApJ/823/157/table5` | **130** | Pal 5 DEIMOS stars, HRV | MEASUREMENT |

**Do not query the parent `J/ApJ/823/157`.** It returns table4 and table5
*concatenated* with different column sets, which cannot be described by one
column list; the parent query also yields a misleading row count. Each table was
fetched separately (19 + 130 = 149).

### Catalogues that do NOT exist in VizieR — corrected premises

Probed and confirmed absent by the `#INFO Error=` line:
`J/ApJ/863/L20` (Price-Whelan & Bonaca 2018 GD-1), `J/ApJ/892/L37`
(Bonaca+2020), `J/ApJ/891/161` (Ibata+2020), `J/ApJ/819/1` returns only
**photometry** (1,002,771 rows of CFHT/MegaCam and KPNO g,r magnitudes — no
kinematics, not acquired), `J/ApJ/889/70` and `J/MNRAS/516/731` (S5 survey),
`J/MNRAS/520/5225` (the galstreams paper itself).

### Ibata+2021 STREAMFINDER — two things the brief did not anticipate

1. **`dSF` is model-dependent.** The CDS ReadMe defines it as *"Distance to the
   star estimated by STREAMFINDER"*. STREAMFINDER finds streams by searching for
   stars consistent with a common **orbit in an assumed Galactic potential**, so
   `dSF` — and the membership assignment itself — is **MODEL**. The Gaia
   astrometry, photometry and the compiled HRVs are independent MEASUREMENTS.
2. **The stream label 1–32 has no published name mapping.** The paper states the
   label only colours the streams in a figure; the ReadMe says only *"[1/32] A
   unique stream identification label"*. A probable mapping was recovered by
   nearest-track assignment against the galstreams `ibata2021` tracks and written
   to `stream_ibata2021_label_to_name_DERIVED.tsv` (**32 rows**), explicitly
   labelled **DERIVED — not a measurement**. It is validated by two counts quoted
   in the paper's own prose reproducing exactly: **276 stars** for the NGC 6397
   stream (derived label 26) and **388 stars** for NGC 3201 (derived label 9).
   29 of 32 labels match within 1° median separation; labels 16 (Fimbulthul,
   separation ratio only 1.2), 28 and 11 are uncertain.

**Velocity yield:** of 5960 member stars, `r_HRV = 0` ("no measurement") for
5275, leaving **685 with a velocity**; the per-reference counts in the ReadMe
sum to 685 and were verified against the download exactly. Two of those 685 are
themselves unphysical (−3837.77 and +16856.83 km/s), leaving **683 credible**.
`HRV` uses **1000 km/s as a null sentinel** — VizieR blanked it correctly here,
but a raw-file reader would not.

## 1.3 Orphan–Chenab: the tables VizieR does not hold

VizieR's Koposov+2019 record contains **only** the RR Lyrae subset. The on-sky
track and — critically — the **radial-velocity track** exist only in the paper.
Transcribed verbatim from `arXiv:1812.08172` `main.tex`; all 8 table
environments were enumerated and each target table confirmed to live in a
**single** environment, so the split-table failure mode does not apply. Row
counts reproduce the source exactly.

| File | Rows | Label |
|---|---:|---|
| `stream_koposov2019_orphan_track_gaia_rgb.tsv` | 8 | **MEASUREMENT** — φ2 centroid + width vs φ1, Gaia RGB counts |
| `stream_koposov2019_orphan_track_decals.tsv` | 10 | **MEASUREMENT** — same from DECaLS matched filter |
| `stream_koposov2019_orphan_velocity_track.tsv` | 8 | **MEASUREMENT** — SDSS radial-velocity track (3rd column is the measurement *uncertainty*, not a dispersion) |
| `stream_koposov2019_orphan_selection_spline_track.tsv` | 12 | **SELECTION FUNCTION — not a measurement** |
| `stream_koposov2019_orphan_selection_spline_distance.tsv` | 12 | **SELECTION FUNCTION — not a measurement** |
| `stream_koposov2019_orphan_selection_spline_pm.tsv` | 10 | **SELECTION FUNCTION — not a measurement** |

The three spline tables define the authors' candidate-selection window. They
look like tracks and would be easy to mistake for one; they are not.

## 1.4 The in-plane counterpart — the Milky Way rotation curve

`stream_eilers2019_MW_rotation_curve.tsv` — **38 rows**, R = 5.27 → 24.82 kpc,
columns `R_kpc`, `vc_kms`, `e_vc_minus_kms`, `e_vc_plus_kms`.

**Neither Eilers+2019 nor Mróz+2019 has a CDS/VizieR catalogue** — both ReadMe
URLs return HTTP 404. Transcribed from `arXiv:1810.09466`, which contains
exactly one table environment. The radial span was asserted against the paper's
stated 5–25 kpc coverage (the paper gives no bin count in prose, so a row-count
assertion was not possible).

**Label: MEASUREMENT of the in-plane circular velocity, with a stated caveat.**
It is derived from Gaia DR2 + APOGEE red giants by Jeans modelling of an
axisymmetric disc. It assumes axisymmetry and equilibrium **of the disc
tracers** but assumes **no dark-matter halo and no parametric mass model**. It
is an inferred kinematic quantity, not a raw observable.

## 1.5 The pairing — what this section actually delivers

**The Milky Way is the pairing.** It is the one system in this whole lane where
an out-of-plane tracer and an in-plane rotation curve are both measured, both
public, and both free of any assumed halo:

* **In-plane:** 38 points of v_c(R) over 5.27–24.82 kpc.
* **Out-of-plane:** 69 stream tracks with genuine 3-D geometry, of which 33 are
  full 6-D; 16 of the 69 are within 10° of polar; 26 reach |z| ≥ 10 kpc; and
  **14 extend beyond 25 kpc, past the outer edge of the rotation curve.**

The two probe the same baryonic system in different directions, which is exactly
the second independent direction the programme lacked.

Streams around **external** galaxies — where the host's own rotation curve would
provide the in-plane leg — are covered in §1.6 below.
