# Polar-ring galaxies with kinematics in BOTH the host disk and the polar ring

Lane: `work/wellnet-2026-09/env-data/raw/polar-rings/`
Acquired 2026-09-03/04 UTC. Every downloaded file has a sibling `<name>.manifest.json`.

---

## 0. The headline, stated plainly

**No polar-ring galaxy has a numerically tabulated rotation curve in BOTH planes anywhere in the
59 arXiv e-print sources and 8 VizieR/CDS tables acquired here** — which, between them, cover every
PRG kinematics, HI, CO, photometry and catalogue paper the searches surfaced from 1993 to 2026.
This is a bounded negative, not a proof: it does not exclude a table hiding in a journal-hosted
supplement I could not fetch (A&A returns HTTP 403 to programmatic requests), nor in the two
pre-arXiv classics named in §2.

Two systems have a tabulated rotation curve in *one* plane:

| System | Tabulated curve | Points | Source |
|---|---|---|---|
| NGC 4650A | HOST-disk plane, stellar LOS velocity + dispersion | 23 | Sackett+1994 ApJ 436,629 Table 2 |
| NGC 2685 | one *warped* HI disk, with per-ring inclination, PA and 3-D spin normal | 21 rings | Jozsa+2009 A&A 494,489 Table 5 |

Everything else that exists in two planes exists **as a figure**. That is the central,
inconvenient finding of this lane. The physics is published; the numbers are not.

Nine systems (Tier A below) do have rotation measured **independently in both planes** with at
least one plane resolved in radius. For those, the curves would have to be recovered by digitising
published figures, or by re-reducing archival data (SAO 6-m SCORPIO long-slit and Fabry-Perot,
WSRT/ATCA HI, MUSE, WALLABY cubes), or by asking the authors.

The closest thing to a ready-made two-plane *pair* anywhere in the literature reached here is
**Iodice et al. 2003, ApJ 585, 730**, which puts a host-plane and a ring-plane velocity on the same
diagram for five systems — **NGC 4650A, NGC 660, NGC 2685, UGC 7576 and A0136-0801** — with the
ring value tabulated (`iodice2003_table1_PRG_TF.tsv`) and the host value plotted only. See §6.5.

---

## 1. What was actually acquired

### 1.1 VizieR — five catalogues, all verified by echo-back and row count

A METAcat title search for `*polar ring*` returns exactly four catalogues; a fifth
(`J/A+A/710/A145`) was found under `*polar*`. All were downloaded with
`-out.all=1 -out.max=unlimited` and every row count matches the ReadMe's stated record count.

| VizieR ID | Contents | Rows | What it is NOT |
|---|---|---|---|
| `J/MNRAS/418/244` | SPRC — SDSS Polar Ring Catalogue (Moiseev+2011) | **275** | no kinematics at all; position, r mag, cz, candidate class |
| `J/A+A/554/A11` t1 | CO(1-0)/(2-1) of 21 PRGs (Combes+2013) | **21** | single-dish, unresolved; 5 detections |
| `J/A+A/554/A11` spectra | reduced IRAM spectra index | **10** | — |
| `J/A+A/319/401` t1 | Effelsberg HI of PRC objects (Huchtmeier 1997) | **44** | global HI profile: S_HI, V_HI, dv20 |
| `J/A+A/319/401` t2 | derived global parameters | **38** | — |
| `J/A+A/386/140` | Parkes HI of southern PRGs (van Driel+2002) | **33** | global HI: W50, W20, I_HI, log M_HI, log L_B |
| `J/A+A/710/A145` | COUGS-DESI polar-structure catalogue (Bahr+2026) | **2989** | pure imaging: griz SB26 mags/radii, extinction, z, K-corr, absolute mags |
| `J/A+A/702/A258` | CIPRG updated (Dobrycheva+2025) | **195** | imaging + a `kconf` yes/no flag |

**None of these five catalogues contains two-plane kinematics.** The two HI surveys and the CO
survey contain *global, spatially unresolved* line profiles. That is one plane at best, and in
classical PRGs the HI sits in the polar structure, so the linewidth traces the **polar** plane.

CIPRG's `kconf` column gives **41 of 195** objects flagged `yes` (kinematically confirmed).

### 1.2 Whitmore et al. 1990, AJ 100, 1489 — the PRC — is NOT in VizieR

Searched METAcat titles for `*polar ring*` (4 hits), `*Polar Ring*` (4), `*polar*` (332, all
scanned), `*Whitmore*` (7, none the PRC), `*ring galax*`, `VII/*` + `*ring*`. The PRC is absent.
It is also pre-arXiv. What survives electronically is the PRC's **designations and
classifications**, propagated into Huchtmeier 1997, van Driel 2002, Iodice 2003, CIPRG and
Yu 2026 — not its photographic atlas or its tabulated rotation velocities.

The PRC's structure, as reported consistently across those papers: **157 objects — 6
kinematically confirmed (category A), 27 good candidates (B), 73 possible (C), 51 related (D)**.

### 1.3 arXiv e-print sources — 59 papers

All under `eprints/`, raw `.tar.gz` kept with manifests. Four papers whose e-print "source" is
dvips PostScript rather than LaTeX were additionally pulled as PDFs into `pdfs/`.

---

## 2. Corrections to the brief's premises

The brief named several sources. Three of its identifications were wrong, and one source turned
out to be far more valuable than the brief expected.

1. **Egorov & Moiseev 2019, MNRAS 486, 4186 is NOT a kinematic compilation.**
   The brief called it "probably the single best modern two-component kinematic compilation."
   It is arXiv:1904.02513, *"Metallicity and ionization state of the gas in polar-ring galaxies."*
   Its Table 1 is an **observing log** (cz, R_eff, slit PA, date, exposure, seeing, wavelength
   range) for 15 SPRC galaxies; its Table 2 is **metallicity**. The line-of-sight velocities of
   stars and ionized gas exist only as a figure (the file `SPRC_velocities_bin_addNew.eps` in its
   e-print source). It does
   deliver kinematic *confirmation* of polar structure in **13 of 15** galaxies, and it does
   tabulate the two slit position angles per galaxy, which is the sky-projected angle between the
   components. That is genuinely useful — it is just not rotation curves.

2. **The NGC 4650A HI paper is not "A&A 325, 145".** Every later paper acquired here that cites
   the ATCA HI observations of NGC 4650A cites **Arnaboldi et al. 1997, AJ 113, 585**,
   *"New HI Observations of the prototype Polar Ring Galaxy NGC 4650A"* (Iodice+2015 cites it as
   `Arn97`; Swaters & Rubin 2003 as "Arnaboldi et al. 1997"). It is **not on arXiv** (verified by
   title and author queries against the arXiv API, which returned zero hits) and **not in VizieR**.
   Nearby 1997 A&A volumes carry different PRG papers that WERE acquired: Reshetnikov 1997
   A&A 325, 933 (= arXiv:astro-ph/9704047, *Global structure and formation of polar-ring
   galaxies*) and Hagen-Thorn & Reshetnikov 1997 A&A 319, 430 (IC 1689, the kinematic-confirmation
   reference for PRC B-03 in Yu+2026).

3. **"Iodice et al. 2002, A&A 391, 103" is not an HI paper.** Verified independently: the BibTeX
   key `2002A&A...391..103I` in the bibliographies of Freitas-Lemes et al. 2012 (arXiv:1208.3421)
   and Mosenkov et al. 2022 (arXiv:2208.12943) expands to *"Iodice E., Arnaboldi M., Sparke L.S.,
   Gallagher J.S., Freeman K.C., 2002, A&A, 391, 103"*, and that is exactly the author list of
   arXiv:astro-ph/0206055. So A&A 391, 103 is
   *"Near-Infrared photometry in the J, H and Kn bands for Polar Ring Galaxies I. Data, structural
   parameters"* — CASPIR imaging and 2-D bulge+disk decompositions of A0136-0801,
   ESO 415-G26, Arp 230, AM 2020-504 and ESO 603-G21. Paper II is astro-ph/0206057. The Iodice
   NGC 4650A paper the brief probably meant is arXiv:astro-ph/0110249, *"The puzzle of the polar
   structure in NGC 4650A"* (ApJ 2002) — also photometry, not HI. All three were acquired.

4. **The brief did not know about the two most important modern sources.**
   * **Yu, Zheng, ... & Li 2026, ApJ 999, 199** (arXiv:2601.22222), *"Insights into the Physical
     Nature of Polar Ring Galaxies from HI Observations"* — FAST HI of a **complete compilation
     of 40 kinematically confirmed PRGs**, with the kinematic-confirmation reference for every
     one. This is now the census backbone.
   * **Khoperskov, Moiseev, Khoperskov & Saburova 2014, MNRAS 441, 2650** (arXiv:1404.1247) —
     whose halo-axis-ratio comparison figure has a caption enumerating the **21 PRGs with a
     published dark-halo axis ratio**, i.e. the set for which someone has actually done a
     two-plane dynamical analysis, with the primary reference for each.

5. **NGC 2685 is not a classical polar ring.** Jozsa et al. 2009 show its HI is one extremely
   warped, kinematically **coherent** disk, inclined ~70 deg to the lenticular body inside and
   becoming coplanar with it outside — not two independent orthogonal rotators. It is demoted to
   Tier B here. For the programme's purpose it remains extremely valuable, just under a different
   description: it is a *direction scan of one disk*, with V_rot tabulated against a
   continuously changing orbit-plane orientation.

---

## 3. The census: 40 kinematically confirmed PRGs

`yu2026_table1_confirmed_PRGs.tsv` (40 rows) is the definitive list, with RA/Dec/z and the
kinematic-confirmation reference for each. `yu2026_table2_HI_properties.tsv` (33 rows) adds
D_L, HI flux, V85, asymmetry, profile shape, log M_HI, optical inclination, V_rot, V_rot(i=90),
NUV-r, log M_star and log M_bary.

**Critical scope limit on Yu 2026 Table 2, stated in the paper itself:** these are *single-dish,
spatially unresolved* HI profiles. V_rot is V85/2 de-projected with the **optical inclination of
the host**. It does not separate the planes. The paper's own conclusion:

> "PRGs do not follow a tight TFR or bTFR if the H I resides primarily in the host galaxy. But the
> scatter decreases significantly if we assume the gas is mainly distributed in the polar ring.
> Spatially resolved H I observations are essential to disentangle the gas distribution and
> kinematics in PRGs."

So: 40 systems with M_HI + M_star + a *geometry-ambiguous* velocity. Excellent for baryons,
useless as a two-plane test on its own. Note also that for **SPRC-69** the fitted inclination is
10 +/- 18 deg and the tabulated V_rot = 1127 +/- 1871 km/s is meaningless; use V_rot(i=90) = 214 +/- 13.

Of the 40, **three are MaNGA targets** (SPRC-1 = MaNGA 01-287487, SPRC-10 = MaNGA 01-460660,
SPRC-13 = MaNGA 01-124268). No SAMI or CALIFA PRG was found. No published MaNGA two-plane
decomposition of these was located.

---

## 4. Systems with genuine two-plane kinematics

Full machine-readable version: **`polar_rings_two_plane_inventory.tsv`** (22 rows, 23 columns,
per-cell references). Tiers are this lane's own classification:

* **A** — rotation measured *independently* in both planes, at least one resolved in radius.
* **B** — rotation *detected* in both planes, but at least one plane has no usable V(r) or V_max.
* **C** — kinematically confirmed PRG, rotation published in one plane only.

Position angles quoted below are SLIT or fitted-KINEMATIC angles where the source is a
spectroscopic paper; the inter-plane angles in the "Angle" column are PHOTOMETRIC deprojections
(two-fold degenerate, hence the "or") except for NGC 4650A, whose 93 deg is the difference of two
measured MUSE position angles, and NGC 4632 / NGC 6156, where 90 deg is imposed by the model.
For SPRC-7, SPRC-260, SPRC-10, SPRC-14 and SPRC-69 the component position angles in the
inventory TSV come from Smirnova & Moiseev 2013 and are PHOTOMETRIC, not slit, angles.

### Tier A — 9 systems

| System | PRC/SPRC | Host plane | Polar plane | Angle | Curve in both? |
|---|---|---|---|---|---|
| **NGC 4650A** | A-05 | stars, long slit, PA 62-63; **tabulated, 23 pts, 0-25.6"** | HI (ATCA) + Halpha/[OIII] + MUSE 2-D, PA 152-160 | **93 deg** (MUSE PAs 67 / 160) | yes (host tabulated, polar figures) |
| **NGC 4262** | SPRC-33 | stars, SCORPIO-2 slits at PA 0 and 160 | HI ring (WSRT), to ~15 kpc | 50 +/- 6 **or** 88 +/- 6 | yes, figures only |
| **SPRC-7** | SPRC-7 | stars, SCORPIO slit PA 150, asym.-drift corrected | ionized gas Hbeta, scanning FP field, tilted-ring, to ~23 kpc | 58 +/- 9 **or** 73 +/- 12 | yes, figures only |
| **SPRC-260** | SPRC-260 | stars, SAO 6-m long slit | ionized gas, FP field | 57 **or** 87 | yes, figures only |
| **NGC 4632** | WALLABY | HI main body, 3DBarolo tilted-ring, i_g=62.5 | HI anomalous gas, MCGSuite polar ring | 90 by construction; beta = 335 +/- 5 | host curve + single ring v_rot |
| **NGC 6156** | WALLABY | HI main body, 3DBarolo, i_g=51 | HI anomalous gas | 90 by construction; beta = 153 +/- 5 | host curve + single ring v_rot |
| **A0136-0801** | A-01 | stars, long slit (SWR83, Whitmore+87) | Halpha **2-D Fabry-Perot field**, >2000 pixels | "nearly perpendicular"; no number located | yes, figures only |
| **UGC 7576** | A-04 | stars, long slit, asym.-drift corrected | HI, to ~17 kpc | perpendicular (PRC A) | yes — **but unreachable** |
| **UGC 9796** | A-06 | stars, long slit, asym.-drift corrected | HI, to ~21.4 kpc | perpendicular (PRC A) | yes — **but unreachable** |

UGC 7576 and UGC 9796 are the two systems of **Reshetnikov & Combes 1994, A&A 291, 57** — the
canonical "two perpendicular rotation curves, one from host stellar kinematics and one from ring
HI" measurement. That paper is **not on arXiv and not in VizieR** (verified). Its headline result,
as quoted by later papers: dark mass = 1.6x and 3x the luminous mass inside 17 and 21.4 kpc
respectively.

### Tier B — 9 systems

NGC 2685 (warped coherent disk; **tabulated** tilted-ring model), NGC 7625 / Arp 212 (warped polar
ring, FP tilted-ring), SPRC-10, SPRC-14, SPRC-69, SPRC-178 (two-slit confirmations from
Moiseev+2011: one slit on the host major axis, one on the ring major axis, with the gas/star
velocity gradients swapping between them), NGC 660 (highly inclined rather than truly polar),
AM 2020-504 (ring RC at PA 17 from Freitas-Lemes+2012; host from Arnaboldi+1993), NGC 4111
(SAURON IFU, two-component gas decomposition).

### Tier C — 4 systems with only one plane published

ESO 415-G26, Arp 230, IC 1689, MCG-05-07-001.

### The 21-system halo-shape roster

Khoperskov et al. 2014 (arXiv:1404.1247) carry a figure captioned "The minor-to-major axis ratio
of the DM halo obtained for different well-studied PRGs", whose caption enumerates every system
with a published halo axis ratio — i.e. every system somebody has analysed dynamically using the polar component. This is the
best available answer to "which PRGs have been used as two-plane probes":

A0136-0801, AM 1934-563, AM 2020-504, AM 226-3206, Arp 230, ESO 415-G26, IC 2006, MCG-5-7-1,
NGC 660, NGC 2685, NGC 3718, NGC 4262, NGC 4650A, NGC 4753, NGC 5122, NGC 5907, SPRC-7,
SPRC-260, UGC 4261, UGC 7576, UGC 9796.

Cross-matching against the 40-object confirmed census, **seven** of these 21 are NOT in it:
**AM 226-3206, IC 2006, MCG-5-7-1, NGC 3718, NGC 4753, NGC 5907, UGC 4261** — so their "polar"
status rests on weaker evidence. (UGC 4261 = PRC C-24 was *explicitly excluded* by Yu et al. 2026
"because lack of kinematically evidence for PRG identification", together with SPRC-201.)
The remaining fourteen are all in the census; note that AM 1934-563 = PRC B-18 = PGC 089058,
so it *is* confirmed, kinematically, by Reshetnikov et al. 2006 A&A 446, 447.

---

## 5. Geometry: the angle between the two planes

`smirnova_moiseev2013_table1_PRG_geometry.tsv` — **78 rows** (Smirnova & Moiseev 2013,
Astrophys. Bull. 68, 371). For each system: inner-disk semi-axes a, b and PA; polar-ring semi-axes
a, b and PA; z; **delta1 and delta2** (the two geometric solutions for the angle between the
planes); and D_ring/D_disk.

Two warnings:

* **The published table is split across two `table*` environments**, the second captioned
  "(continue)". This is exactly the silent-extraction failure mode the brief warns about. The
  transcription script parses both and asserts the combined count equals the paper's stated
  sample of 78; a naive single-environment parse would have silently returned 43.
* **delta1/delta2 are photometric, not kinematic.** Deprojecting two ellipses gives a two-fold
  degenerate answer that imaging alone cannot resolve. Most "best" SPRC rings land at
  delta > 70-80 deg; ~6% sit at 40-55 deg.

Egorov & Moiseev 2019 Table 1 gives the *observed slit* PAs for 15 SPRC galaxies (two slits for
SPRC-2, 10, 12, 14, 27, 37, 40) — a directly measured sky-projected angle, not a deprojection.

---

## 6. Baryonic photometry for g_bar

### 6.1 NGC 4650A — complete, in both planes

`iodice2015_table1_NGC4650A_mass_model.tsv` (Iodice+2015 A&A 583, A48 Table 1):

| Component | M (1e9 Msun) | h (kpc) | r (kpc) | r1 (kpc) | r2 (kpc) | plane |
|---|---|---|---|---|---|---|
| HG bulge | 0.2 | | 0.17 | | | host |
| HG disk | 10.3 | 0.5 | 0.948 | | | host |
| Polar disk (stars) | 15. | 0.5 | | 5.95 | 6.8 | polar |
| HI disk | 7.2 | 0.5 | | 3.4 | 15.3 | polar |
| *DM halo* | *15* | *1.2* | *6.0* | | | *fitted model component — NOT an observation* |

Baryons: **10.5e9 Msun in the host plane, 22.2e9 Msun in the polar plane, 32.7e9 total.**
The DM row is a fit and must never be used as data under the programme's constraint 2.

The earlier **Combes & Arnaboldi 1996** decomposition of the same galaxy is reproduced verbatim in
Lüghausen, Famaey & Kroupa 2013 (arXiv:1304.4931, Sect. 3): Plummer bulge 0.2e9 Msun with
r_p = 0.17 kpc; Miyamoto-Nagai host disk 11e9 Msun, h_r = 0.748 kpc, h_z = 0.3 kpc; stellar polar
ring 9.5e9 Msun, h_r1 = 6.8, h_r2 = 5.95 kpc; gaseous polar ring 6.4e9 Msun, h_r1 = 15.3,
h_r2 = 3.4 kpc; total baryons 27.1e9 Msun.

**Compare the two carefully — they are not independent, and they do not agree everywhere.**
The *geometry* is identical (Iodice+2015 reuse the Combes & Arnaboldi radii verbatim: 5.95/6.8 kpc
for the stellar polar disk, 3.4/15.3 kpc for the gas), so the two models are not independent
measurements of shape. The *masses* differ component by component:

| Component | Combes & Arnaboldi 1996 | Iodice+2015 | difference |
|---|---|---|---|
| bulge | 0.2e9 | 0.2e9 | 0% |
| host disk | 11e9 | 10.3e9 | 7% |
| **polar stellar disk** | **9.5e9** | **15e9** | **+58%** |
| polar gas disk | 6.4e9 | 7.2e9 | 12% |
| total baryons | 27.1e9 | 32.7e9 | 21% |

The polar stellar disk mass is uncertain at the ~50% level, and it is the component that
dominates the polar-plane g_bar. **Any two-plane gravity test on NGC 4650A must carry that as an
explicit systematic**, not adopt one number.

### 6.2 NGC 2685 — complete

`josza2009_table1_NGC2685_properties.tsv` (30 rows): D = 15.2 +/- 3.8 Mpc, m_B = 12.05 +/- 0.15,
m_I = 9.90 +/- 0.10, M_B = -19.1 +/- 0.7, M_I = -21.4 +/- 0.6, L_B = 7.0 +/- 3.7 e9 Lsun,
L_I = 15.2 +/- 7.7 e9 Lsun, M_HI = 1.7 +/- 0.9 e9 Msun, R25 = 10.8 kpc, R_HI = 14.8 kpc,
R_t = 31.0 kpc, V_t = 147 +/- 15 km/s. The tilted-ring table additionally gives the **face-on HI
surface density per ring** — i.e. a radially resolved gas profile, not just a total.

### 6.3 SPRC-7 and NGC 4262

`khoperskov2014_table2_SPRC7_NGC4262_photometry.tsv` — distances, both inclinations, delta, disc
and bulge masses and scales, and the outer radii of both components. **The published table has a
defect**: the header carries 11 labels but a disc-mass label was dropped (its footnote entry
survives commented out in the LaTeX), so labels and units are mutually inconsistent and one
reading gives an unphysical bulge (scale 3.4 kpc > size 0.8 kpc). Columns are therefore emitted
with **neutral names carrying only the published units**; the manifest records the physically
coherent reading and the reason. Do not attach the published labels without re-checking.

### 6.4 NIR photometry for five classical southern PRGs

Iodice et al. 2002, Papers I and II (arXiv:astro-ph/0206055 = A&A 391, 103, verified; and
arXiv:astro-ph/0206057, the companion paper, whose page number was not independently verified):
CASPIR J, H, Kn
imaging of **A0136-0801, ESO 415-G26, Arp 230, AM 2020-504, ESO 603-G21**, with 2-D
bulge+disk decompositions of the host (Sersic n, mu_0, r_h in arcsec and kpc, axis ratio q_d,
B/D). This is the closest thing to the 3.6 um / K-band photometry the brief asked for.

### 6.5 Global photometry across the whole population

* `iodice2003_table1_PRG_TF.tsv` — 16 PRGs with M_Kn, M_B, cz, HI dV20 (Iodice+2003 ApJ 585, 730).
  **dV20 in this table is the polar-plane linewidth only.** But the paper also states, in the text,
  that it computed log(dV) for the HOST galaxies of the five best-studied systems —
  **NGC 4650A, NGC 660, NGC 2685, UGC 7576 and A0136-0801** — "using optical absorption-line
  rotation curves along the host galaxy equatorial plane", from Sackett et al. 1994, van Driel
  et al. 1995, Simien & Prugniel 1997, Whitmore et al. 1990 and Schweizer et al. 1983. Those five
  host-plane velocities are plotted as large crosses, with an arrow to the ring value, in the
  paper's S0 Tully-Fisher figure (`iodicee_fig8.ps`) — **and are nowhere tabulated**. For
  NGC 4650A specifically the host circular velocity was derived from the stellar rotation and
  dispersion of Combes & Arnaboldi 1996 by converting to an equivalent gas linewidth with an
  assumed 10 km/s dispersion. So: **five systems have a published host-plane AND polar-plane
  velocity from one paper, and the host half of every pair is a figure.** Their finding is that the
  five host galaxies fall ON the spiral TF relation while the rings sit off it.
* `vizier_vanDriel2002_HI4_table1.tsv` — 33 southern PRGs with B_T, D25, W50, W20, I_HI, distance,
  log L_B, log M_HI, M_HI/L_B.
* `vizier_Huchtmeier1997_HI2_table1.tsv` — 44 northern PRC objects with B_T, D25, S_HI, V_HI, dv20.
* `vizier_COUGS_DESI_Bahr2026.tsv` — 2989 polar-structure candidates with griz SB26 magnitudes and
  isophotal radii, extinctions, redshifts, distances, K-corrections, absolute magnitudes,
  physical semi-major axis. Imaging only, but homogeneous and enormous.
* Combes+2013: molecular gas for the 5 CO-detected SPRC objects, M(H2) = 4.5-21.5 e9 Msun.

---

## 7. What a follow-up would have to do

1. **Digitise figures.** For the 9 Tier-A systems the two-plane curves exist as published plots.
   The relevant figure files are already extracted under `eprints/`, identifiable from their
   captions. Verified present in the extracted trees:
   * `eprints/1404.1247_Khoperskov2014_oblate/fig_obsRC_1.eps`, `fig_obsRC_2.eps` — the OBSERVED
     rotation curves of the stars and gas for SPRC-7 and NGC 4262; plus `fig_obsstar1-3.eps`
     (long-slit stellar V and sigma) and `fig_kin2d_1.eps`, `fig_kin2d_2.eps` (the polar-component
     velocity fields).
   * `eprints/2309.05841_Deg2023_WALLABY_NGC4632_NGC6156/Figures/NGC4632_BestModel.pdf` and
     `NGC6156_BestModel.pdf` — panel E is the rotation curve; the polar ring is the starred last
     radial point in each right-hand panel.
   * `eprints/1509.01112_Iodice2015_NGC4650A_MUSE/kin_fold.jpg` (folded host-plane stellar
     velocity profile at P.A. 67), `kin_maj_conf.jpg`, `kin_min_conf.jpg`, `prof_PR_gas.jpg`
     (polar-disk gas rotation curve at P.A. 160).
   * `eprints/1107.1966_Moiseev2011_SPRC/SPRC-{10,14,39,60,69,178}_res-eps-converted-to.pdf` —
     the two-slit line-of-sight velocity cuts for gas and stars.
2. **Request or re-reduce archival data.** SAO 6-m SCORPIO/SCORPIO-2 long-slit and FP cubes
   (Moiseev's group), WSRT NGC 4262, ATCA NGC 4650A, MUSE NGC 4650A (ESO archive), WALLABY PDR1
   cubes for NGC 4632/NGC 6156 (CASDA — the kinematic models are *not* in VizieR; a METAcat search
   for `*WALLABY*` returns only three unrelated catalogues).
3. **Chase the two unreachable classics.** Reshetnikov & Combes 1994 A&A 291, 57 and
   Arnaboldi et al. 1997 AJ 113, 585 are ADS-scanned only.
4. **Do not use Yu 2026 V_rot as a two-plane quantity.** It is a global linewidth with a host
   inclination applied. The paper says so.

## 8. Failure modes checked, explicitly

* **Shared-denominator artefacts** — not applicable to an acquisition lane; no correlation was
  computed here. Flagged for downstream: Yu 2026's own TFR/bTFR analysis puts M_HI on both axes
  (log M_bary contains 1.33 M_HI, and V_rot is derived from the same HI profile). Any correlation
  built on that table must simulate the null with the actual error covariance.
* **Monotone-invariant statistics** — no statistic computed.
* **Refitting on the held-out set** — no fitting done. KiDS and wide binaries were never touched.
* **Silent extraction failures** — this is the one that bit. The Smirnova & Moiseev 2013 table is
  split across two `table*` environments; both are parsed and the total asserted against the
  paper's stated 78. Every VizieR fetch asserts the body is not HTML, that `#Column` headers are
  present, and that VizieR echoes back the requested catalogue identifier. Every VizieR row count
  matches the ReadMe. The Sackett 1994 table came out of `pdftotext` as three separate column
  runs; they are re-zipped with a length assertion and a monotonic-radius check.
* **Test bugs that look like solver bugs** — no solver run.
* **Non-monotonic M(r) / clipped outer slopes** — no deprojection performed.
* **No data presupposing dark matter.** The only DM quantities recorded anywhere in this lane are
  the *fitted* halo of Iodice+2015 Table 1 and the halo axis ratios of Khoperskov+2014, and both
  are labelled in their manifests as model output, not observation. Nothing in the TSVs is an
  NFW-fitted mass presented as data.
