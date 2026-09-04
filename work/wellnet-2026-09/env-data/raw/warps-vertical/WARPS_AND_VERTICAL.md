# Warped H I disks and resolved vertical-field measurements

Lane: `work/wellnet-2026-09/env-data/raw/warps-vertical/`
Acquired 2026-09-03/04 UTC. Every downloaded file has a sibling `<name>.manifest.json`
with source URL, UTC retrieval timestamp, SHA-256, byte size, row count, column names
with units, and the exact query issued. Raw upstream responses are kept unmodified;
cleaned TSVs sit beside them.

**Purpose.** Rotation curves constrain the in-plane radial field only. This lane looked
for systems where a *second, perpendicular* direction of the gravitational field is
measured in the same galaxy: (JOB 1) warp geometry `i(R)`, `PA(R)` alongside `V_rot(R)`,
and (JOB 2) a radially resolved vertical velocity dispersion `sigma_z(R)` or a vertical
force `K_z`.

---

## Headline findings, stated plainly

1. **The published-table premise is largely wrong.** Tilted-ring `i(R)` and `PA(R)`
   solutions are computed by almost every modern H I paper and published almost
   universally as **per-galaxy figures**, not tables. An exhaustive ADQL query over
   VizieR's full `TAP_SCHEMA` (64,325 tables) for any table carrying a rotation-velocity
   column *and* an inclination column *and* a radius column returned **exactly one**
   relevant catalogue: `J/A+A/370/765/table4` (Verheijen & Sancisi 2001). See
   "Query that establishes the negative" below.
2. **15 galaxies** have warp geometry and rotation tabulated together in machine-readable
   form, all from that one catalogue, and all 15 also have B/R/I/K' photometry and H I
   fluxes in sibling tables, so `g_bar` is computable for every one of them.
3. **A genuinely resolved `sigma_z(R)` profile exists**, published as a table, for
   **NGC 6946** (Aniyan et al. 2021, 5 radii) and **NGC 628** (Aniyan et al. 2018,
   5 radii; PDF-only, not transcribed here), and for the **Milky Way** (Gaia DR3
   Drimmel et al. 2023: 68 radial bins of `sigma_vZ`, plus Bovy & Rix 2013: 43 points of
   `Sigma_1.1(R)` and `K_Z,1.1(R)`). This is a *narrow* but real yes, not a proxy.
4. **The DiskMass limitation the programme already knew about is confirmed and is
   universal to that survey.** No DMS paper (I–XI) publishes a per-radius stellar
   dispersion table; DMS VI Table 6 is explicitly captioned "Exponential fits to the
   sigma_LOS and sigma_z radial profiles". The resolved profiles are Appendix-atlas
   figures only.
5. **Garcia-Ruiz, Sancisi & Kuijken 2002 does not contain what the brief assumed.** It
   publishes one global `PA` and one warp *angle* per side per galaxy — not a tilted-ring
   `i(R)`/`PA(R)` table. Details in §1.3.

---

# Section 1 — Warped disks

Column key: **i(R)** = inclination tabulated as a function of radius; **PA(R)** = position
angle as a function of radius; **V(R)** = rotation curve tabulated; **Phot** = baryonic
photometry / H I surface density available to compute `g_bar`.
`T` = tabulated (machine-readable), `F` = published only as a figure, `G` = single global
value, `—` = absent.

## 1.1 The one source with everything tabulated — Verheijen & Sancisi 2001

**Verheijen & Sancisi 2001, A&A 370, 765** — WSRT H I synthesis of the Ursa Major cluster.
VizieR `J/A+A/370/765`.

| item | status |
|---|---|
| i(R) | **T** — `Incl` column, per ring |
| PA(R) | **T** — `PA` column, per ring |
| V(R) | **T** — `Vrot`, plus separate approaching/receding curves with max/min bounds |
| Phot | **T** — table2 gives B, R, I, K' apparent and absolute magnitudes, extinctions, D25; table5 gives W20, W50, integrated H I flux, R_HI |

Files:
- `vizier_verheijen2001_uma_tiltedring_rotcurves.tsv` — **437 rows × 13 cols**, 41 galaxies.
  Columns: `Sample, Name, Rad (arcsec), VrotApp, VrotAppM, VrotAppm, VrotRec, VrotRecM,
  VrotRecm, Vrot (km/s), Incl (deg), PA (deg)`.
- `vizier_verheijen2001_uma_photometry.tsv` — 52 rows × 18 cols.
- `vizier_verheijen2001_uma_sample.tsv` — 52 rows × 20 cols.
- `vizier_verheijen2001_uma_hi_results.tsv` — 43 rows × 24 cols.
- `cds_readme_J_A+A_370_765.txt` — byte-by-byte column description.

**Of the 41 galaxies with per-ring rotation curves, 15 have `Incl` and/or `PA` that
actually vary with radius** — i.e. a warp or a twist recorded in the same table as the
rotation. All 41 (hence all 15) are matched in the photometry table.

| Galaxy | rings | Δi (deg) | ΔPA (deg) | R_max (arcsec) | Phot |
|---|---|---|---|---|---|
| NGC 3718 | 11 | **25** | **82** | 420 | yes |
| NGC 3726 | 12 | **22** | 16 | 373 | yes |
| NGC 4138 | 7 | **18** | 13 | 213 | yes |
| NGC 3893 | 11 | **13** | 36 | 233 | yes |
| NGC 3917 | 17 | 7 | 0 | 170 | yes |
| NGC 4100 | 25 | 6 | 3 | 261 | yes |
| UGC 6917 | 11 | 4 | 1 | 120 | yes |
| NGC 4013 | 36 | 0 | **27** | 367 | yes |
| NGC 3769 | 12 | 0 | **19** | 426 | yes |
| UGC 6446 | 17 | 0 | 13 | 176 | yes |
| UGC 6923 | 6 | 0 | 9 | 61 | yes |
| UGC 6973 | 9 | 0 | 7 | 90 | yes |
| NGC 4088 | 13 | 0 | 6 | 246 | yes |
| NGC 3949 | 8 | 0 | 5 | 81 | yes |
| NGC 4183 | 23 | 0 | 3 | 241 | yes |

The remaining 26 galaxies in the same file have constant `Incl` and `PA` — a flat-disk
tilted-ring solution — and are still usable as unwarped controls.

*Caveat:* `Incl` and `PA` are quoted to whole degrees, so `ΔPA = 3` (NGC 4183) is only
marginally resolved. Treat the top ~9 entries as the credible warped set.

## 1.2 Warp geometry tabulated, rotation only in figures

**Herrmann & Ciardullo 2009, ApJ 705, 1686 (Paper III)** — 5 nearly face-on spirals.

| item | status |
|---|---|
| i(R) | **T** for M83 (5 radial zones, i = 24 → 46 deg) and M94 (2 zones); G for IC 342, M74, M101 |
| PA(R) | **T** for M83 (PA = 226 → 172 deg) and M94 (305 → 295 deg); G for the rest |
| V(R) | **F** — H I rotation curves in Fig. 13; M94's is the THINGS curve of de Blok+ 2008 |
| Phot | **T** — Table 1 gives h_R and mu_0 per galaxy |

Files: `herrmann2009_III_table3_disk_geometry_vs_radius.tsv` (10 rows),
`herrmann2009_III_table1_program_galaxies.tsv` (5 rows). Inside the last three M83 zones
PA and i are stated to vary **linearly** with R, so the geometry is fully reconstructible.
The M83 geometry is inherited from a published H I warp model, not fitted to the PNe.

**Zschaechner et al. 2011 (NGC 4244) and 2012 (NGC 4565), HALOGAS** — the warp is real and
strong, and both papers fit a full 3-D tilted-ring model including a *flare*, but the
per-ring `v_rot`, `PA`, `i` and column density are published as figures ("Key parameters
used in the optimal models…"). Only the model **scale heights** are tabulated (Table 3 in
each). i(R) F · PA(R) F · V(R) F · Phot partial. Sources retained:
`arxiv_zschaechner2011_halogas_ngc4244/`, `arxiv_zschaechner2012_halogas_ngc4565/`.

## 1.3 Premise corrections — sources that do not contain what was assumed

**Garcia-Ruiz, Sancisi & Kuijken 2002, A&A 394, 769** (`arXiv:astro-ph/0207112`).
26 edge-on WHISP spirals. What it actually publishes, in
`garciaruiz2002_table_hi_analysis_warp_angles.tsv` (**26 rows × 14 cols**):
`UGC, Lop_kin, Lop_rho, R_HI, M_HI, V_sys, W20, W50, PA (one global value), warp1 ± e,
warp2 ± e, Env`. The warp is reduced to **one angle per side**, measured from Gaussian
centroids perpendicular to the major axis; 20 of 26 galaxies have at least one measured
warp angle and 13 have both sides.
i(R) — · PA(R) G · V(R) **F** · Phot F.
The warp curve `z(R)`, the rotation curve `V(R)` and `Sigma_HI(R)` do exist for all 26 and
are in the same arXiv tarball as **IDL vector PostScript** atlas figures
(`u<UGC>-plotwl2.ps`, `u<UGC>-plotart30.ps`). Recovering numbers from those is a
vector-graphics extraction job with axis calibration, not a download; it was **not**
attempted here. The tarball is retained so that option stays open.

**de Blok et al. 2008, AJ 136, 2648 (THINGS)** (`arXiv:0810.2100`). Table 2 gives only the
**radial mean** ⟨i⟩ and ⟨PA⟩ per galaxy (`deblok2008_things_table2_tiltedring_means.tsv`,
19 rows). The per-ring solutions and the rotation curves are per-galaxy figures 3–56.
i(R) F · PA(R) F · V(R) F · Phot separate (Leroy+ 2008 / SINGS).

**Józsa 2007, A&A 468, 903 — "Kinematic modelling of disk galaxies II. A case-study of
symmetrically warped galaxy disks."** This is a TiRiFiC tilted-ring fit to three
grand-design warped galaxies and is the single most on-target paper for JOB 1. **It is not
on arXiv**, and aanda.org returns HTTP 403 to scripted requests, so it could not be
retrieved in this lane. Papers I and III of the series were retrieved
(`arxiv_jozsa2007_tirific_paperI/`, `arxiv_jozsa2008_ngc2685_paperIII/`); Paper III
(NGC 2685) tabulates only *global* parameters — systemic velocity, dispersion, scale
height — with the per-ring solution in figures. **Recommended follow-up: obtain Paper II
through an institutional route.**

**WALLABY (Deg et al. 2022, PASA 39, e059; arXiv:2211.07333).** 109 (later 236) public H I
kinematic models built by WKAPP from FAT + 3D-Barolo. The pipeline is configured to fit
**flat-disk models only**: position angle and inclination are held constant with radius by
construction. WALLABY therefore cannot supply warp geometry, whatever its sample size.
Not downloaded.

**Levine, Blitz & Heiles 2006, ApJ 643, 881 (Milky Way H I warp)**
(`arXiv:astro-ph/0601697`). The paper's **only** table is a 3-row least-squares fit to the
warp, `W_m(R) = k0 + k1 (R − R_k) + k2 (R − R_k)^2` for modes m = 0, 1, 2
(`levine2006_table1_mw_warp_mode_fit.tsv`). The quantities a vertical-field test would
want — `Sigma(R,phi)`, mean height `h(R,phi)`, half-thickness `T_h(R,phi)` and the mode
phases `phi_1(R)`, `phi_2(R)` — are contour figures. Underlying data: the LAB H I survey,
which is separately public. The warp *height* field is nonetheless analytically
reconstructible for the m=0 mode from Table 1 alone.

**Poggio et al. 2020 (Nat. Astron. 4, 590; arXiv:1912.10471)** and
**Chrobáková & López-Corredoira 2021 (ApJ 912, 130; arXiv:2105.04348)**: warp *models* and
precession-rate fits with parameter tables, no tabulated `h(R,phi)`. Both retained as
source tarballs. Note these two disagree about whether precession is detected — the second
is titled "A Case against a Significant Detection of Precession in the Galactic Warp".

**O'Brien, Freeman & van der Kruit 2010 (A&A 515, A60–A63, Papers I–IV)** and
**Peters et al. 2017 (MNRAS, "The Shape of Dark Matter Haloes" I–V)**. These are the
strongest *physics* match in the whole search: both model edge-on H I cubes for rotation
curve, surface density, **layer thickness (flaring)** and **velocity dispersion**
simultaneously, i.e. genuinely two-direction. Peters III does this for IC 5052, IC 5249,
ESO 115-G021, ESO 138-G014, ESO 274-G001 and UGC 7321; O'Brien III for 8 southern
edge-ons. **Every one of these quantities is published as a per-galaxy figure.** The only
tables are H I masses and observing logs. Sources retained:
`arxiv_obrien2010_edgeon_halo_{I,II,III,IV}*/`,
`arxiv_peters2017_shapeDMhaloes_{III,V}*/`.

**Allaert, Gentile, Baes et al. 2015, HEROES II (A&A 582, A18; arXiv:1507.03095)**.
Tilted-ring models of 6 edge-on spirals; "Variation of the radially dependent parameters
of the final models: the H I surface density, the rotation velocity, the inclination, the
position angle" is **Figure 12**, and the table holds only the radially *constant*
parameters. Source retained.

**Bosma 1981 (AJ 86, 1791/1825)** and **Briggs 1990 (ApJ 352, 15)** are the classic
tabulated tilted-ring/warp compilations, but both predate arXiv and neither is in VizieR
(`J/AJ/86/1791` and `J/ApJ/352/15` both 404). Recovering them means OCR of ADS scanned
pages, which is exactly the kind of silent-extraction risk the standing brief warns about.
**Not attempted.** If the programme wants them, they should be typed in by hand against
the scans and cross-checked row by row.

## 1.4 Supporting H I catalogues acquired

| file | rows × cols | content |
|---|---|---|
| `vizier_lvhis_table9_hi_kinematics.tsv` | 47 × 17 | LVHIS (Koribalski+ 2018) ATCA H I kinematic properties: **single** global inclination, PA, V_rot per galaxy — no radial dependence |
| `vizier_lvhis_table6_atca_hi_properties.tsv` | 82 × 18 | LVHIS ATCA H I properties |
| `cds_readme_J_MNRAS_478_1611.txt` | — | LVHIS byte-by-byte |

## Query that establishes the negative

Run against `https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync`:

```sql
SELECT t.table_name
FROM TAP_SCHEMA.columns AS t
JOIN TAP_SCHEMA.columns AS u ON t.table_name = u.table_name
JOIN TAP_SCHEMA.columns AS v ON t.table_name = v.table_name
WHERE t.column_name LIKE '%Vrot%' AND u.column_name LIKE '%Incl%'
  AND (v.column_name LIKE '%Rad%' OR v.column_name LIKE '%Dist%')
GROUP BY t.table_name
```

Result over 64,325 VizieR tables: `J/A+A/370/765/table4` and `J/A+A/697/A38/catalog`
(the latter is the CS4G optical/IR disc-structure catalogue, not a kinematic one). A
broader variant allowing `Vc`/`Vcirc` returned 11 tables, all Tully-Fisher or stellar
catalogues on inspection. Script: `_tap_colsearch.py`.

---

# Section 2 — Resolved `sigma_z(R)` and `K_z`

For every entry: **RESOLVED PROFILE** vs **FITTED (central value + scale length)**;
whether the scale height `h_z` is **MEASURED** or **INFERRED**; and the **instrumental
velocity resolution**, with a statement of whether the reported dispersions sit above or
below it. A dispersion quoted below the instrumental resolution is a model extrapolation,
not a measurement.

## 2.1 RESOLVED PROFILES — external galaxies

### NGC 6946 — Aniyan et al. 2021, MNRAS 500, 3579 (arXiv:2010.03991) ✔ acquired as TSV

**RESOLVED PROFILE.** `sigma_z` measured independently at **5 radii**: 1.6, 2.9, 4.3, 7.2,
9.9 kpc. Inner two from VIRUS-W integrated-light spectra, outer three from Planetary
Nebula Spectrograph velocities. Published as a table, with the implied total surface
density beside it.

- `aniyan2021_ngc6946_table6_sigmaz_and_surface_density.tsv` — **5 × 12**:
  `R (kpc), sigma_z_hot (km/s), Sigma_T, Sigma_C_gas, LC/LD, F_C, Sigma_D, Sigma_C_star,
  Upsilon_{B,V,I,3.6um}`.
- `aniyan2021_ngc6946_table3_sigmaz_virusw.tsv` — 2 × 8, cold/hot decomposition, BIC, chi2.
- `aniyan2021_ngc6946_table4_sigmaz_pns.tsv` — 3 × 6, PN.S bins; the cold-component values
  are **90 % confidence upper limits** (preserved as `<=` in the file).

**Scale height: INFERRED, not measured.** `h_z = 376 ± 75 pc` (I band), taken from
statistical `h_R/h_z` relations for edge-on galaxies applied to the measured scale length.
NGC 6946 is at i = 37°, so `h_z` is not directly observable. `Sigma_T` inherits this
linearly — the *same* structural weakness as DiskMass, though the dispersion itself is
resolved.

**Instrumental resolution: VIRUS-W high-resolution mode, R = 8700, Gaussian sigma of the
PSF = 14.7 km/s.**
- Hot-component `sigma_z` = 65.3, 64.6, 32.0, 20.9, 14.8 km/s. The first four are above
  the instrumental sigma; **the outermost (14.8 km/s) sits essentially *at* it** and must
  be treated as a limit, not a measurement.
- Cold-component `sigma_z` = 26.6, 19.3 km/s (VIRUS-W) and ≤12.1, ≤12.9, ≤12.3 km/s
  (PN.S). **The three PN.S cold values are below 14.7 km/s** — the authors themselves quote
  them as upper limits. The two VIRUS-W cold values are above it. The authors state
  explicitly that high S/N lets them "measure velocity dispersions somewhat lower than the
  velocity resolution (sigma) of the instrument" — a deconvolution claim, so anything
  within a factor ~1.5 of 14.7 km/s should be flagged in downstream fits.
- PN.S dispersions come from **discrete** tracer velocities, so the relevant floor there is
  the per-PN velocity error, not the spectrograph sigma; dispersions were additionally
  corrected by quadratically subtracting an H I dispersion of ~6 km/s.

### NGC 628 — Aniyan et al. 2018, MNRAS 476, 1909 (arXiv:1802.00465) ⚠ acquired, NOT transcribed

**RESOLVED PROFILE**, same design: `sigma_z` at **5 radii** (Table 6), with Tables 4 and 5
giving the VIRUS-W (2 bins) and PN.S (3 bins) cold/hot decompositions.
**Scale height INFERRED:** `h_z = 397.6 ± 88.3 pc`, from Möllenhoff (2004) / de Grijs /
Yoachim & Dalcanton scale-height correlations for edge-on discs — NGC 628 is face-on.
**Instrumental resolution: identical VIRUS-W setup, sigma = 14.7 km/s.** Hot-component
values run 55.4 → 17.5 km/s (all above the floor, the outermost close to it);
cold-component values 4.6–16.7 km/s, of which the three PN.S values (4.6, 6.7, 6.2 km/s)
are **far below the instrumental resolution** and are quoted by the authors as upper
limits or deconvolved values.

**Not converted to TSV.** The arXiv source for this paper contains only a wrapper `.tex`
plus the compiled PDF — there are no LaTeX tables — and the two-column PDF interleaves
table cells with body text under `pdftotext`, which is precisely the silent-extraction
failure mode the standing brief forbids. The paper PDF and a `-layout` text dump are kept
at `arxiv_aniyan2018_discHalo_I_ngc628/` with a manifest recording the derivation.
**Follow-up: transcribe by hand from the journal PDF, or obtain the MNRAS
machine-readable table.**

## 2.2 RESOLVED PROFILES — the Milky Way

### Gaia DR3 — Gaia Collaboration / Drimmel et al. 2023, A&A 674, A37 ✔ acquired

**RESOLVED PROFILE of `sigma_vZ(R)`, and it is a direct kinematic measurement** — no
potential model, no dark-matter assumption, no scale height involved at all.

- `vizier_gaiadr3_drimmel2023_sigmaZ_vs_R_rgb.tsv` — **68 rows × 15 cols**, RGB tracers,
  R = 0.1 → 13.5 kpc, `sigvZ` = 14.5 → 87.8 km/s.
- `vizier_gaiadr3_drimmel2023_sigmaZ_vs_R_ob.tsv` — **18 rows × 15 cols**, OB tracers,
  R = 6.5 → 9.9 kpc, `sigvZ` = 4.9 → 7.1 km/s.
- Columns: `Radius (kpc)`, `vphimin/vphi/vphimax`, `sigvRmin/sigvR/sigvRmax`,
  `sigvphimin/sigvphi/sigvphimax`, **`sigvZmin/sigvZ/sigvZmax`** (16th/50th/84th
  percentile, km/s), `npix`.
- `cds_readme_J_A+A_674_A37.txt` — byte-by-byte description.

**Scale height: not applicable** — this is a dispersion profile, not a surface-density
inference. To turn it into a vertical force you must supply your own vertical density
profile, which keeps the assumption explicit and separable rather than baked in.

**Instrumental resolution.** There is no spectrograph dispersion floor here: velocities
come from Gaia astrometry (proper motions × distance) plus RVS radial velocities. The
error budget is distance and proper-motion systematics, not a resolution limit. The
quoted 16th/84th percentiles are the **spread of the velocity distribution across pixels
in each radial bin**, not measurement errors — do not read them as error bars. The OB
sample's `sigvZ` ≈ 5–7 km/s is a genuinely cold-disc measurement well above Gaia's
per-star velocity precision for bright nearby OB stars.

### Bovy & Rix 2013, ApJ 779, 115 (arXiv:1309.0809) ✔ acquired

**RESOLVED `Sigma_1.1(R)` and `K_Z,1.1(R)` from vertical kinematics — 43 points,
R = 4.59 → 8.55 kpc.** This is the two-direction Milky Way measurement the brief asked for.

- `bovyrix2013_table3_Sigma_Kz_vs_R.tsv` — **43 × 8**:
  `FeH (dex), aFe (dex), R (kpc), Sigma_1.1 (Msun/pc2), e_Sigma_1.1, R0_minus_R (kpc),
  K_Z_1.1 (2 pi G Msun/pc2), e_K_Z_1.1`.
- `bovyrix2013_anc_table3_Sigma_Kz_vs_R.tsv` — the authors' arXiv ancillary CSV of the same
  table. **40 of 43 rows agree exactly; 3 differ by up to 4.7 % in `Sigma_1.1`.** Both are
  kept; the typeset `surf.txt` version (what the v3 manuscript renders) is the primary.

**Sampling structure — read this before fitting.** Each row is one mono-abundance
population (MAP) of SEGUE G dwarfs, and `R` is the radius at which *that* population best
constrains the surface density. The 43 rows are a profile assembled *across populations*,
not 43 independent radii of a single population. Rows at similar `R` are not independent
measurements of the same thing.

**Model dependence — this is NOT a raw observable.** `Sigma_1.1` and `K_Z,1.1` come from
fitting a quasi-isothermal distribution function to the vertical and radial motions inside
a **parametrised Newtonian potential family** (disk + bulge + power-law halo + gas), with
`V_c(R_0) = 230 km/s`, `z_h = 400 pc` and `dlnV_c/dlnR = 0` held fixed. Using these numbers
to test a modified gravity law is **circular** unless the underlying SEGUE kinematics are
refitted under the candidate law. Flagged in the manifest as well.

**Instrumental resolution:** not a dispersion measurement in the spectroscopic sense —
SEGUE radial-velocity precision is ~2–5 km/s per star against population vertical
dispersions of tens of km/s, so the measurement is not resolution-limited.

## 2.3 RESOLVED but published only as figures — primary data acquired instead

### Herrmann & Ciardullo 2009, ApJ 705, 1686 (Paper III) — 5 face-on spirals

Correct title: "Planetary Nebulae in Face-On Spiral Galaxies. **III. Planetary Nebula
Kinematics and Disk Mass**". Sample: **IC 342, M74 (NGC 628), M83 (NGC 5236),
M94 (NGC 4736), M101 (NGC 5457)** — not M81/M51/NGC 1068/NGC 3184.

The paper **does** measure `sigma_LOS(R)` and `sigma_z(R)` in radial bins of 15–18 PNe
(≈ 6, 6, 10, 7 and 3 bins respectively), but publishes them **only in Figures 4–8**. Its
five tables are:

| file | rows | what it is |
|---|---|---|
| `herrmann2009_III_table1_program_galaxies.tsv` | 5 | type, i, D, h_R, mu_0, E(B−V), v_max, N_PN, survey radius |
| `herrmann2009_III_table2_scale_heights.tsv` | 5 | h_z from three different h_R/h_z relations + stability and rotation-curve limits |
| `herrmann2009_III_table3_disk_geometry_vs_radius.tsv` | 10 | PA and i vs radial zone (the warp geometry, §1.2) |
| `herrmann2009_III_table4_asymmetric_drift.tsv` | 5 | one v_asd per galaxy, assumed constant with R |
| `herrmann2009_III_table5_disk_mass_models.tsv` | 7 | **FITTED**: `sigma_z(0)` + dynamical `h_R` of an exponential, plus Sigma(0), Upsilon(0) |

So Table 5 is a **FITTED central value plus a scale length**, exactly the DiskMass form.
**Scale height: INFERRED** — every entry in Table 2 comes from an `h_R/h_z` scaling
relation (de Grijs 1998; Kregel+ 2002; Bizyaev & Mitronova 2002) plus Toomre-stability and
rotation-curve limits. The paper says so directly: these are face-on systems, so `h_z` is
not observable.

**But the primary data are machine-readable and were acquired.**
`vizier_herrmann2009_PN_radial_velocities.tsv` — **774 rows × 17 cols** from VizieR
`J/ApJ/703/894/table4` (Paper II): PN ID, RA/Dec, `m5007`, line ratio, `RV (km/s)`,
`e_RV (km/s)`, remarks, spectrum filenames.

Per-galaxy audit of that file against Paper III's Table 1:

| galaxy | PN candidates | with measured RV | with e_RV < 15 km/s | Paper III N_PN |
|---|---|---|---|---|
| IC 342 | 165 | 106 | 100 | 99 |
| M74 | 153 | 112 | 109 | 102 |
| M83 | 241 | 204 | 193 | 162 |
| M94 | 150 | 130 | 129 | 127 |
| M101 | 65 | 63 | 60 | 60 |
| **total** | **774** | **615** | **591** | **550** |

The small excess over Paper III is the H II-region contaminants and probable halo objects
the paper rejects (3, 4, 6, 3 and 1 respectively) plus a magnitude/quality cut. **The
resolved profile is therefore exactly reconstructible**: bin by radius at the paper's own
bin sizes, deproject with Table 3's geometry, subtract Table 4's asymmetric drift, and
apply the paper's epicyclic `sigma_LOS → sigma_z` conversion. That is a re-derivation from
published primary observables, not a proxy.

**Instrumental / measurement floor:** median per-PN velocity error ≈ 6 km/s, all
< 15 km/s (WIYN/Hydra, Blanco/Hydra, HET-MRS). Central fitted `sigma_z(0)` values run
29–91 km/s, i.e. **5–15× the per-tracer error** — the most comfortable margin of any entry
in this document; the per-bin profile values fall below that at large radius but stay well
above 6 km/s. Discrete-tracer dispersions are limited by √(2N) sampling noise, not by
spectrograph resolution; with 15–18 PNe per bin the fractional dispersion error is ~18 %.

## 2.4 FITTED ONLY — confirmed, do not mistake for profiles

### DiskMass Survey (the programme's existing holding)

| paper | what it publishes | resolved? |
|---|---|---|
| DMS I, Bershady+ 2010a (ApJ 716, 198) | VizieR `J/ApJ/716/198/table2`: 231-row sample list only (acquired: `vizier_dms1_table2_sample.tsv`) | no |
| DMS IV, Westfall+ 2011 (ApJ 742, 18), UGC 463 | Tables: observing log, scale lengths/heights, stellar templates, **kinematic geometry**, projected rotation speed, pointing coordinates, enclosed mass, halo properties. **No sigma(R) table and no V(R) table.** Source retained. | no |
| DMS VI, Martinsson+ 2013 (A&A 557, A130) | Table 6 caption verbatim: *"Exponential fits to the sigma_LOS and sigma_z radial profiles"* | **FITTED** |
| DMS VII, Martinsson+ 2013 (A&A 557, A131) | per-galaxy scalars; Sigma_* profiles are Atlas figures | no |
| DMS XI, Swaters+ 2025 (ApJS 276, 59) | see below | **resolved but gas, not stars** |

The DMS VI/VII resolved profiles live in the **PPak Atlas**: 30 galaxies × 2 PostScript
pages of figures (`arxiv_martinsson2013_dms6/PPakAtlas/`, 60 files). No numbers.

**Scale height: INFERRED** in DMS VII (from the `h_R/h_z` relation of Bershady+ 2010b), so
`h_z` and `h_R` are correlated by construction — the programme already knew this and it is
confirmed here.

**Instrumental resolution: PPak, λ/Δλ = 7700 mean, `sigma_inst = 17 km/s`** (DMS VI
abstract; ≈16 km/s in the body text). Fitted central `sigma_z,0` values across the sample
are ~15–40 km/s, so **the outer parts of essentially every DMS dispersion profile fall at
or below the instrumental resolution** — which is precisely why the survey publishes an
exponential fit rather than per-radius points. DMS VI explicitly notes one UGC 8196
measurement at the instrumental resolution limit and that some dispersions "smaller than
the instrumental resolution [were] therefore ignored".

### DMS XI, Swaters et al. 2025 (ApJS 276, 59) ✔ acquired — resolved, but ionized gas

- `vizier_dms11_table7_halpha_fiber_kinematics.tsv` — **18,288 rows × 23 cols**,
  per-fibre H-alpha measurements for 137 low-inclination galaxies (mean kinematic
  inclination 26°). Columns include `oRA, oDE` (fibre offsets in arcsec), `VelN, e_VelN`,
  **`sigmaN, e_sigmaN`** (narrow-component dispersion), and broad-component equivalents.
- `vizier_dms11_sample_velfield_photometry.tsv` — 125 rows × 35 cols, velocity-field
  parameters + photometry + inverse-Tully-Fisher inclinations.

This **is** a spatially resolved `sigma_LOS` map in near-face-on disks — but of **ionized
gas**, not stars. Gas dispersion is set by turbulence and thermal broadening, not by the
vertical stellar potential, so it is **not** a substitute for stellar `sigma_z`. Recorded
as gas kinematics, and useful for disk geometry, not for the vertical stellar field.

**Instrumental resolution: 13 km/s (sigma), Bench Spectrograph on WIYN 3.5 m.** The
H-alpha dispersion distribution peaks at 18 km/s and reaches 20 km/s at high surface
brightness — **only ~1.4× the instrumental sigma**, so individual low-dispersion fibres are
at or below the resolution. Secondary broad components (~50 km/s) are safely above.

### Others confirmed as fits or plots only

- **Bottema 1993, A&A 275, 16**: one dispersion value per galaxy for 12 disks (the
  `sigma ∝ 0.29 V_rot` relation). **Not resolved.** Pre-arXiv, not in VizieR.
- **Gerssen & Shapiro Griffin 2012, MNRAS 423, 2726** (arXiv:1204.3430, acquired):
  Table 3 is a **FITTED** `sigma_R,0`, `sigma_z,0`, kinematic scale length `h_kin` for
  NGC 2280 and NGC 3810; Table 4 is an 8-galaxy compilation of single `sigma_z/sigma_R`
  values. Exactly the form the programme wants to avoid.
- **Kuijken & Gilmore 1989/1991, Holmberg & Flynn 2004, Zhang et al. 2013**: `K_z(z)` is
  reported as fitted analytic forms and figures plus scalar surface densities
  (e.g. 65 ± 6 M☉/pc² to 0.8 kpc, 74 ± 6 to 1.1 kpc). **No tabulated `K_z(z)` or
  `Sigma(z)` profile.**
- **Widmark et al. 2021 phase-space spirals (A&A 650 A124; 653 A86; arXiv:2111.13707)**:
  the only table is the free-parameter list; per-bin results are sky/disc maps with no
  data release located. Treat a per-bin tabulation as unconfirmed.
- **Mackereth et al. 2019 (MNRAS 489, 176)**: VizieR `J/MNRAS/489/176` holds only
  `dr14ages.dat` (74,748 stellar ages). The `sigma_z`, `sigma_R` results are figures.
- **Sharma et al. 2021 (MNRAS 506, 1761)**: Table 2 gives MLE coefficients of a separable
  analytic form for `sigma_vz` and `sigma_vR` — a **formula**, not a table of `sigma(R)`.
- **Falcón-Barroso et al. 2017 CALIFA (VizieR `J/A+A/597/A48`)**: the catalogue is
  `tableb1.dat` only — 300 rows of Name, ID, z, PA, eps, type, M*, R_eff. **No `sigma(R)`.**
  The Voronoi-binned V and sigma **maps** are released as FITS at the CALIFA data-products
  page; combined with the PA/eps in `tableb1.dat` a low-inclination subsample could yield
  genuine `sigma_LOS(R)` profiles. CALIFA's V1200 setup has higher spectral resolution than
  MaNGA and is the better IFU route. **Not pursued in this lane.**
- **MaNGA**: deliberately not pursued — another lane is building resolved `sigma_LOS(R)`
  from the DR17 DAP `STELLAR_SIGMA` / `SPX_ELLCOO` maps. Note for that lane: MaNGA's
  instrumental sigma is ≈ 70 km/s, well **above** the `sigma_z` of most face-on disks
  (15–60 km/s), so essentially every such measurement is a deconvolution below the
  resolution limit — which is exactly why DiskMass built SparsePak/PPak instead.

## 2.5 Vertical structure from H I layer thickness — the untapped route

`sigma_z` of the *gas* plus the measured H I layer thickness gives the vertical force
directly, and in an **edge-on** galaxy the thickness is genuinely measured rather than
inferred. Peters et al. 2017 (III) and O'Brien et al. 2010 (III) both deliver exactly this
— rotation curve, surface density, FWHM thickness **and** dispersion, per galaxy, jointly
fitted to the cube. O'Brien III covers 8 southern edge-ons (ESO 074-G015 / IC 5052,
ESO 109-G021 / IC 5249, ESO 115-G021, ESO 138-G014, ESO 146-G014, ESO 274-G001,
ESO 435-G025 / IC 2531, UGC 7321); Peters III re-analyses a **subset of the same six**
with the Galactus tool, so the union is 8 galaxies, not 14 — the two series are not
independent samples. **All of it is published as figures.** Sources are on disk. This is
the single largest untapped two-direction dataset found in this lane, and it would need
either digitisation or a request to the authors.

---

## Summary counts

| question | answer |
|---|---|
| Warped galaxies with warp geometry **and** rotation both tabulated | **15** (Verheijen & Sancisi 2001; all with photometry) — plus M83 and M94 with tabulated piecewise geometry but figure-only rotation |
| Genuinely resolved `sigma_z(R)` profile published as a table | **Yes**, for NGC 6946 (5 radii, acquired as TSV), NGC 628 (5 radii, PDF only, not transcribed), and the Milky Way (Gaia DR3, 68 bins) |
| Tabulated vertical force / dynamical surface density from vertical kinematics | **Yes**, Bovy & Rix 2013, 43 points, `Sigma_1.1(R)` and `K_Z,1.1(R)` — model-dependent, see §2.2 |
| Any external galaxy with a **measured** (not inferred) scale height alongside a resolved `sigma_z(R)` | **No.** Every face-on `sigma_z` measurement in the literature pairs with an `h_R/h_z`-inferred `h_z`. The only way to measure `h_z` directly is edge-on, where `sigma_z` cannot be measured. |

## Failure modes checked (per the standing brief)

- **Silent extraction failures.** Every LaTeX transcription asserts its row count against
  the sample size stated in the paper *and* asserts a constant cell count per row;
  `\multicolumn` spans are expanded before splitting so collapsed cells cannot shift
  columns. Garcia-Ruiz's table uses `\multicolumn{2}{c}{---------}` for absent warp
  measurements and would otherwise have silently produced 13-cell rows — the assertion
  caught it. Every VizieR pull asserts the response is real TSV (not the generic HTML
  page), echoes the catalogue identifier back, and counts rows and columns.
- **Shared-denominator artefacts.** Flagged where relevant: DMS VII's `h_z` is derived from
  `h_R`, so those two columns are correlated by construction; Aniyan's `Sigma_T` is
  `sigma_z^2 / (2 pi G h_z)` with `h_z` inferred from `h_R`, so `Sigma_T` and any
  photometric scale length share an input. Do not correlate `Sigma_T` against `h_R`.
- **No data presupposing dark matter treated as an observation.** The Gaia DR3 dispersion
  profile and the Herrmann PN velocities are raw kinematics. Bovy & Rix's `K_Z` is
  explicitly labelled model-dependent (Newtonian potential family + qDF) in both this file
  and its manifest; it is a debugging/comparison input, not an observation.
- **Version discrepancy surfaced rather than silently resolved.** Bovy & Rix's typeset
  table and their ancillary CSV disagree in 3 of 43 rows; both are kept.

## Scripts in this directory

| script | purpose |
|---|---|
| `_acquire.py` | VizieR TSV pull with HTML/identifier/row assertions; arXiv e-print fetch and unpack; CDS ReadMe fetch; manifest writer |
| `_extract_tables.py` | LaTeX → TSV transcription with row/column assertions (Bovy & Rix, Garcia-Ruiz, Herrmann III, Levine, de Blok) |
| `_extract_aniyan.py` | LaTeX → TSV for the Aniyan 2021 NGC 6946 `sigma_z(R)` tables |
| `_probe_cds.py` | Test whether a CDS catalogue designation exists (ReadMe fetch, HTML guard) |
| `_search_vizier.py` | VizieR keyword/catalogue search |
| `_tap_colsearch.py` | VizieR TAP `TAP_SCHEMA` column search — the query behind the §1 negative |

---

## ADDENDUM from the main lane — the NGC 628 gap is closed

`NGC 628 — Aniyan et al. 2018` is marked above as "acquired, NOT transcribed"
because the arXiv source is a wrapper `.tex` plus a compiled PDF and Table 6 is
typeset **rotated 90 degrees** inside a two-column page, so `pdftotext` and
`pdfplumber` both return it as reversed character strings interleaved with body
text.

That table has now been transcribed and validated:
`aniyan2018_ngc628_table6_sigma_z_profile.tsv` (5 rows x 26 columns), produced by
`../../code/extract_aniyan2018_ngc628.py`. The fix was to set the page rotation
in PyMuPDF so the text is re-extracted in its own reading direction.

The transcription was **validated, not trusted**. The paper states independently
in its section 8 that fitting `sigma_z(R) = sigma_z(0) exp(-R / 2 h_dyn)` to
these points gives `sigma_z(0) = 73.6 +/- 9.8` km/s and
`h_dyn = 92.7 +/- 13.1` arcsec. Refitting the transcribed points recovers
**74.4 km/s and 92.7 arcsec**, and the script asserts both.

This matters as a worked example of the brief's silent-extraction failure mode:
reversing the rotated strings by eye — the obvious shortcut — returns 223 for
`Sigma_T` at R = 2.6 kpc, which is actually `Sigma_D`. The true value is 286.
A plausible-looking wrong number, produced silently.

| R (kpc) | sigma_z (km/s) | Sigma_T (Msun/pc^2) |
|---|---|---|
| 2.6 | 55.4 +/- 6.4 | 286 +/- 92 |
| 4.5 | 50.9 +/- 8.9 | 241 +/- 100 |
| 5.5 | 33.8 +/- 3.3 | 106 +/- 31 |
| 8.7 | 22.6 +/- 2.1 | 48 +/- 14 |
| 12.2 | 17.5 +/- 2.6 | 29 +/- 11 |

The caveat in section 2.1 still stands in full: `Sigma_T` is **not** an
independent observation. It is `sigma_z^2 / (2 pi G h_z)` with
`h_z = 398 +/- 88` pc inferred from an `h_R`/`h_z` relation — the same
correlated-by-construction problem the programme already recorded for DiskMass
VI/VII. `sigma_z`, `B-I` and `Sigma_C_gas` are measurements; `Sigma_T`,
`Sigma_D`, `Sigma_C_star` and the M/L columns carry that inference.
