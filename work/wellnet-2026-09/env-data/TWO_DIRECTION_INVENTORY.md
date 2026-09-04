# TWO_DIRECTION_INVENTORY.md -- systems that measure two field directions at once

Lane: `work/wellnet-2026-09/env-data/`.

Rotation curves measure the in-plane radial field almost exclusively.  This
inventory covers systems where a second, nearly perpendicular direction of the
gravitational field is measured in the SAME baryonic system, which is the only
way to add a genuinely new measurement direction rather than another
transformation of a_N and r.

Sections follow the task's own lettering.  Sub-inventories with the per-system
detail live beside the data they describe:

| section | detail file |
|---|---|
| (a) polar-ring galaxies | `raw/polar-rings/POLAR_RINGS.md` |
| (b) warped H I disks | `raw/warps-vertical/WARPS_AND_VERTICAL.md` |
| (c) stellar streams | `raw/streams-satellites/STREAMS_SATELLITES_COUNTERROT.md` |
| (d) satellite systems | same |
| (e) two-component galaxies | same, plus the measurements below |
| vertical dispersion | `raw/warps-vertical/WARPS_AND_VERTICAL.md` section 2 |

Every downloaded file carries a `<name>.manifest.json` with source URL,
retrieval timestamp, SHA-256, byte size, row and column counts, column names
with units, and the exact query issued.

---


## Summary: what exists, in one table

| category | systems with BOTH directions measured | tabulated numerically? |
|---|---|---|
| (a) polar rings | 9 with rotation measured independently in both planes; 40 kinematically confirmed PRGs in total | **no** -- 0 systems have a tabulated rotation curve in both planes; 2 have one plane tabulated |
| (b) warped H I disks | 15 with i(R), PA(R) and V(R) in one table (Verheijen & Sancisi 2001), of which ~9 are credibly warped; +1 (NGC 2685) | **yes**, and 26 flat-disk controls come with them |
| (c) stellar streams | 60 distinct Milky Way streams with a measured 3-D track, 30 with full 6-D; 16 / 8 of those nearly polar | **yes**, per-track |
| (d) satellite systems | 101 SAGA hosts with orientation + 378 satellites; 125/132 hosts have axis ratio and PA | **yes**, but **no in-plane rotation curve for the hosts** -- a half-measurement |
| (e) two-component galaxies | 105 MaNGA near-orthogonal + 38 counter-rotating measured here with cubes on disk; 447 / 261 from SAMI's published PAs; 47 and 22 polar systems in the two best literature catalogues | **yes** |
| vertical dispersion | 240 MaNGA face-on disks with resolved sigma_LOS(R) measured here; 2 external galaxies and the Milky Way from the literature | **yes** |

**The single sharpest finding.** The configuration the task identified as most powerful -- polar rings -- is the one where the numbers do not exist. Nine systems have rotation measured independently in both planes and not one of them has both curves tabulated; the physics is published, the numbers are figures. Meanwhile the category the task listed last, two-component galaxies, turned out to be the largest by two orders of magnitude and is fully in hand, because integral-field spectroscopy measures both planes of the same galaxy by construction.

---

## (a) Polar-ring galaxies -- the strongest configuration, and a clean negative

A polar-ring galaxy is the single most powerful two-direction configuration:
one baryonic system supplying rotation tracers in two nearly perpendicular
planes. The task asked for every published PRG with rotation measured in both
components. Detail: `raw/polar-rings/POLAR_RINGS.md`, machine-readable census in
`raw/polar-rings/polar_rings_two_plane_inventory.tsv` (22 rows x 23 columns,
per-cell references).

**The headline is a negative, and it is the most important result in this
section. No polar-ring galaxy reachable from arXiv, VizieR, CDS or NED has a
numerically tabulated rotation curve in BOTH planes.** Exactly two systems have
a tabulated curve in *one* plane:

| system | tabulated plane | points | source |
|---|---|---|---|
| NGC 4650A | host disk, stellar LOS velocity + dispersion | 23 | Sackett et al. 1994, ApJ 436, 629, Table 2 |
| NGC 2685 | one warped H I disk, per-ring inclination, PA and 3-D spin normal | 21 rings | Jozsa et al. 2009, A&A 494, 489, Table 5 |

Everything else that exists in two planes exists **as a figure**. The physics is
published; the numbers are not.

**Nine systems (Tier A) do have rotation measured independently in both planes**
with at least one plane resolved in radius: NGC 4650A, NGC 4262, SPRC-7,
SPRC-260, NGC 4632, NGC 6156, A0136-0801, UGC 7576 and UGC 9796. Nine more
(Tier B) have rotation detected in both planes but without a usable V(r) in at
least one. Four more (Tier C) are kinematically confirmed with only one plane
published. The full confirmed census is 40 PRGs.

Two Tier A entries deserve flagging. **NGC 4632 and NGC 6156** are the WALLABY
H I discoveries, where both the main body and the anomalous polar gas are
measured in the *same* H I cube with tilted-ring models -- the cleanest modern
geometry, though only a single ring velocity is given for the polar component.
**UGC 7576 and UGC 9796** are the canonical Reshetnikov & Combes 1994 pair, the
textbook "two perpendicular rotation curves" measurement, and that paper is
**not on arXiv and not in VizieR** -- verified, not assumed.

Recovering any of this requires digitising published figures, re-reducing
archival data (SAO 6-m SCORPIO long-slit and Fabry-Perot, WSRT/ATCA H I, MUSE,
WALLABY cubes), or asking the authors. None of those is a data-acquisition task
and none was attempted here.

Premise correction: **Whitmore et al. 1990 (the original PRC) is not in
VizieR** -- checked rather than assumed. The five polar-ring catalogues VizieR
does hold were all retrieved and verified by identifier echo-back and row count:
the 2025 CIPRG (`J/A+A/702/A258`), the SDSS-based SPRC (`J/MNRAS/418/244`), and
three H I / CO surveys.

The angle between the two planes is genuinely two-valued for several systems
(for example NGC 4262 at 50 +/- 6 **or** 88 +/- 6 degrees, SPRC-7 at 58 +/- 9
**or** 73 +/- 12): the inclination sign degeneracy is not resolved by the data.
Anyone using the inter-plane angle must carry both branches.

**Cross-reference.** The NGC 2685 entry above is also the sixteenth tabulated
warped disk for section (b): Jozsa et al. 2009 Table 5 gives per-ring
inclination, position angle *and* a 3-D spin normal alongside the rotation, so
it belongs in both categories.

---


## (b) Warped H I disks with measured warp geometry and rotation

A warp's shape as a function of radius responds to the vertical restoring force,
so a galaxy with a tilted-ring `i(R)` and `PA(R)` alongside `V_rot(R)`
constrains two directions of the field at once.

**The premise the task started from is largely wrong, and the negative was
proved rather than assumed.** Tilted-ring solutions are computed by nearly every
modern H I paper and published almost universally as *per-galaxy figures*, not
tables. An exhaustive ADQL query over VizieR's entire `TAP_SCHEMA` -- 64,325
tables -- for any table carrying a rotation-velocity column *and* an inclination
column *and* a radius column returned **exactly one** relevant catalogue.

**Verheijen & Sancisi 2001, A&A 370, 765** (VizieR `J/A+A/370/765`), WSRT H I
synthesis of the Ursa Major cluster: 437 rows over 41 galaxies with per-ring
`Rad`, `Vrot`, `Incl`, `PA`, plus separate approaching and receding curves.
B/R/I/K' photometry and H I fluxes are in sibling tables for all 41, so `g_bar`
is computable for every one.

**15 of the 41 have `Incl` and/or `PA` that actually vary with radius.** The
strongest are NGC 3718 (delta i = 25 deg, delta PA = 82 deg over 11 rings),
NGC 3726 (22, 16), NGC 4138 (18, 13), NGC 3893 (13, 36), NGC 4013 (0, 27) and
NGC 3769 (0, 19). Angles are quoted to whole degrees, so entries with
delta PA of 3-6 deg are only marginally resolved: **treat roughly the top nine
as the credible warped set** and the remaining 26 flat-disk galaxies as unwarped
controls, which is a genuinely useful control sample in its own right.

Premise corrections worth carrying, all established against the sources:

- **Garcia-Ruiz, Sancisi & Kuijken 2002 does not tabulate `i(R)`/`PA(R)`.** It
  publishes one global PA and one warp *angle* per side per galaxy (26 rows
  acquired). The warp curves, rotation curves and Sigma_HI(R) are atlas figures
  in the same tarball.
- **THINGS (de Blok et al. 2008) tabulates only radial *means*** of inclination
  and PA, not the radial run.
- **WALLABY fits flat-disk models only**, so geometry is constant with radius by
  construction and it can never supply warp geometry.
- **Jozsa 2007 A&A 468, 903** is the single most on-target paper and could not be
  retrieved: not on arXiv, and aanda.org returns HTTP 403 to scripted requests.
  It needs an institutional route.
- **Bosma 1981 and Briggs 1990**, the classic tabulated compilations, are
  pre-arXiv and absent from VizieR. No OCR was attempted, given the brief's
  warning about silent extraction failures.

**The largest untapped two-direction dataset found anywhere in this lane** is
H I layer thickness in edge-on galaxies: gas sigma_z plus a *measured* (not
inferred) FWHM thickness gives the vertical force directly. O'Brien et al. 2010
(8 southern edge-ons) and Peters et al. 2017 (a subset of the same six, so the
union is 8 galaxies, not 14) both jointly fit rotation curve, surface density,
thickness and dispersion per galaxy. **All of it is published as figures.** The
sources are on disk; extracting it needs digitisation or a request to the
authors.

---


## (c) Stellar streams above galactic disks

A stream at high Galactic latitude traces the field far off the disc plane while
the host's rotation curve traces it in-plane. **For the Milky Way, and only for
the Milky Way, both are held in the same system.** Detail:
`raw/streams-satellites/STREAMS_SATELLITES_COUNTERROT.md`.

`galstreams` v1.2.1 (Mateu 2023) was acquired: **217 tracks**. What matters is
not the total but how many carry a genuine measured 3-D position and velocity:

| subset | definition | tracks | distinct streams |
|---|---|---|---|
| all | anything in the library | 217 | -- |
| `usable_3d` | empirical sky track **and** measured distance track | **69** | **60** |
| `usable_6d` | that plus measured proper-motion **and** radial-velocity tracks | **33** | **30** |

The gap between 217 and 69 is the point. 25 tracks are `GREAT_CIRCLE_ASSUMED`
(a geometric model of the track shape, not a dark-matter model, but still not a
measurement); **68 carry a 1 kpc placeholder distance that is not data at all**;
15 radial-velocity entries are unphysical. Upstream defects were found in 102 of
the 217 tracks and are documented per track rather than silently dropped.

Out-of-plane leverage of the usable set, computed from the published tracks by
pure coordinate transform with **no potential, mass model or halo entering any
column**:

| orbit inclination to the disc | `usable_3d` | `usable_6d` |
|---|---|---|
| 0-30 deg (nearly in-plane, no new direction) | 3 | 2 |
| 30-60 deg | 29 | 14 |
| 60-80 deg | 21 | 9 |
| **80-90 deg (nearly polar, maximum leverage)** | **16** | **8** |

Height reached: median |z|max = 7.94 kpc, **26 tracks exceed 10 kpc**, maximum
84.30 kpc (Orphan-Chenab, Koposov et al. 2019, 23,000 points). **14 tracks
extend beyond R_gc = 25 kpc, i.e. past the outer edge of the best in-plane Milky
Way rotation curve** -- those add a direction the rotation curve cannot reach at
all.

The essential discipline is the measurement-versus-model label. A stream *orbit*
fitted in an assumed NFW potential is not an observation and must not be used as
one; the sky track, distances, proper motions and radial velocities are. Both
are recorded, distinguishably.

---

## (d) Satellite systems around individual galaxies

Satellites at a range of angles to a host's disk sample the field in different
directions around one baryonic system. Detail in the same file.

**The bottleneck is not the satellites; it is the host disk orientation**, since
the disk *normal* is what defines "out of plane". Satellite positions and
line-of-sight velocities are universally available; orientation mostly is not:

| survey | host orientation in its own host table? |
|---|---|
| **SAGA DR3** (Mao et al. 2024) | **yes** -- axis ratio, PA and Sersic index for all 101 hosts |
| SAGA DR2 | no |
| ELVES (Carlsten et al. 2022) | no -- distance, velocity, magnitudes and nothing else |
| McConnachie 2012 / Local Volume | host is the MW or M31; orientation is external knowledge |

Orientation was therefore acquired separately and joined, after which **125 of
132 host entries carry both an axis ratio and a position angle** (101/101 SAGA,
24/31 ELVES). The seven without are handled honestly rather than defaulted to
zero: four are genuinely near face-on (b/a > 0.93) where the major-axis PA is
*physically* ill-defined, one is an elliptical with no disk, one is the Milky Way
itself, and exactly one (NGC 4258, b/a = 0.389) is a real recoverable gap.

**SAGA DR3** is the primary dataset: 101 hosts, **378 confirmed satellites**
across 97 of them (median 3 per host), with zero orphan join keys asserted. Its
`log(Mhalo)` column comes from a group catalogue and **presupposes dark matter**
-- flagged, never usable as an observation.

**The decisive limitation: SAGA publishes no in-plane rotation curve, H I line
width or internal kinematics for its hosts.** So this configuration delivers the
out-of-plane tracer without the in-plane comparison in the *same* system, which
is exactly what the task asked for. Closing that would need H I line widths or
resolved kinematics for the SAGA hosts from elsewhere. Until then category (d)
is a half-measurement: excellent angular sampling around 101 hosts, no in-plane
field to compare it against.

The Milky Way and M31 remain the only satellite systems where both directions
are genuinely available, and there the sample is a few tens of objects rather
than hundreds.

---


## (e) Galaxies with two rotating components measured in different planes

This is where the lane's own measurements land, and it is by a wide margin the largest two-direction sample it holds. A galaxy whose ionised gas rotates in a plane strongly misaligned from its stellar disk is the integral-field analogue of a polar-ring galaxy: one baryonic system, two rotation tracers, two planes. At 90 degrees it *is* a polar-gas system.

### MaNGA, measured here from the DAP MAPS cubes

`clean/manga_gas_star_misalignment.csv` (+ manifest). A weighted plane fit `V = a + b x + c y` inside 1.5 R_eff to the stellar and to the H-alpha velocity field of every cube this lane holds, giving a kinematic position angle for each and their difference folded onto [0, 180] deg. 900 of 902 cubes yielded both angles; 891 have both components rotating strongly enough for the angle to be stable.

| misalignment | meaning | N | fraction |
|---|---|---|---|
| 0-30 deg | co-rotating, one plane | **638** | 71.6% |
| 30-60 deg | mildly misaligned | **74** | 8.3% |
| 60-120 deg | **near-orthogonal: two planes** | **105** | 11.8% |
| 120-150 deg | strongly misaligned | **36** | 4.0% |
| 150-180 deg | **counter-rotating** | **38** | 4.3% |

Every one of these galaxies already has its DAP MAPS cube on disk, so the resolved velocity field in **both** planes is in hand, not merely a citation. The stellar and ionised-gas fields are independent tracers of the same potential in two different planes of the same galaxy.

**This is a screen, not a measurement of record.** A linear gradient recovers the global rotation axis only: it cannot see a kinematically decoupled core, a warp, or a counter-rotating inner disk, and it degrades on a disturbed field. The 12 per cent near-orthogonal rate is higher than the literature rate for strong misalignment, which is what a crude estimator applied to low-gradient galaxies should do. Anything used as a detection needs a proper kinematic position-angle fit (Krajnovic et al. 2006) on the same cube first. What the screen delivers is a ranked shortlist with the data already local.

### SAMI, from the published kinematic position angles

SAMI DR3 publishes `PA_STELKIN` and `PA_GASKIN` per galaxy, so the same quantity comes for free and from a properly fitted kinematic PA rather than a gradient screen. 2815 galaxies have both.

| misalignment | N | fraction |
|---|---|---|
| 0-30 deg (co-rotating) | **1591** | 56.5% |
| 30-60 deg (mild) | **294** | 10.4% |
| 60-120 deg (**near-orthogonal**) | **447** | 15.9% |
| 120-150 deg (strong) | **222** | 7.9% |
| 150-180 deg (**counter-rotating**) | **261** | 9.3% |

The two surveys agree that the counter-rotating population is a few per cent and the near-orthogonal population around 12-16 per cent of rotating galaxies. SAMI's higher rate is expected: its PAs are fitted properly, so genuinely misaligned systems are not diluted, but its sample also reaches lower stellar masses where misalignment is commoner.

### From the literature

Acquired in `raw/streams-satellites/` with manifests. Catalogues where **both**
components are kinematically measured:

| catalogue | survey | N with both components | per-galaxy angle? |
|---|---|---|---|
| Ristea et al. 2024 (`J/MNRAS/527/7438`) | MaNGA | **1899** | no, amplitudes only |
| Raimundo et al. 2023 (`J/other/NatAs/7.463`) | SAMI DR3 | **1310** | **yes, with 3 sigma errors** |
| Ristea et al. 2022 | SAMI DR3 | 1445 | aggregate only |
| Bryant et al. 2019 | SAMI | 622 | aggregate only |
| Barrera-Ballesteros et al. 2014 | CALIFA | 80 | **yes** |
| long-slit series (Corsini / Pizzella / Vega Beltran / Sarzi) | -- | **49** | resolved curves in both components |
| Bevacqua et al. 2022 (`J/MNRAS/511/139`) | MaNGA | 42 of 64 candidates | **yes** |
| Moiseev 2012 | literature compilation | 39 | **yes, DEPROJECTED** |
| ATLAS-3D II (Krajnovic et al. 2011) | ATLAS-3D | 30 (two *stellar* components) | ~180 deg |
| Combes et al. 2013 (`J/A+A/554/A11`) | SPRC | 9 confirmed polar rings | confirmation flag |

The near-90-degree (polar) subsets, which genuinely sample two perpendicular
planes:

| source | criterion | N polar | angle type |
|---|---|---|---|
| **Raimundo et al. 2023** | 60-120 deg | **47** (31 at 70-110, 19 at 80-100) | projected, gas vs stars |
| **Moiseev 2012** | inner polar structures, deprojected | **22** (18 within 10 deg of polar) | **deprojected, plane vs plane** |
| Combes et al. 2013 | kinematically confirmed polar ring | 9 | ring vs host |
| Bevacqua et al. 2022 | 60-120 deg | 4 | projected |
| ATLAS-3D II | Psi >= 75 deg | 4 | stellar kinematics vs photometry |

**Moiseev 2012 is the smaller but stronger set**: its inter-plane angle is a
genuine deprojected angle between two planes, not an on-sky projection.
Raimundo et al. 2023 is the largest with per-galaxy angles and errors. Note that
Krajnovic et al. 2011 (ATLAS-3D II) is **not in VizieR** -- checked, not assumed,
and several of the MaNGA misalignment papers named in the task yielded no usable
per-galaxy table; they are listed as such in the detail file rather than quietly
dropped.


## Vertical-field measurements: resolved dispersion profiles

Not one of the task's five lettered categories, but the same physics and the specific upgrade the task note asked for. The programme's existing DiskMass VI/VII holding gives an exponential *fit* (a central sigma_z plus a scale length in arcsec) for 30 galaxies, with the scale height *inferred* from h_R so the two columns are correlated by construction.

### MaNGA face-on disks, measured here

`clean/manga_faceon_sigma_profiles.csv` (+ manifest): **240 galaxies**, 1671 radial points, 4-9 points each, median formal error 1.02 km/s. sigma_LOS spans 34.8-237.5 km/s. Profiles decline outward in 81% of galaxies with median d log sigma / d log R = -0.20.

Eight times DiskMass's galaxy count with actual radial profiles rather than a fit. The honest limits: it is sigma_LOS, not sigma_z (the in-plane components leak in at order sin^2 i, under 0.25 here); no scale height and no Sigma_dyn is produced, so the DiskMass h_z problem is sidestepped rather than solved; and MaNGA's instrumental sigma is around 70 km/s, so **171 of 240 galaxies have every radial point above 50 km/s and 45 above 70 km/s**. The 45-galaxy fully-above-70 subset is the one to trust without argument.

### From the literature

Acquired in `raw/warps-vertical/` with manifests; see that directory's
`WARPS_AND_VERTICAL.md` for the per-source detail.

| system | what is tabulated | status |
|---|---|---|
| **NGC 6946** (Aniyan et al. 2021) | sigma_z at 5 radii, 1.6-9.9 kpc, with the implied total surface density | acquired as TSV |
| **NGC 628** (Aniyan et al. 2018) | sigma_z at 5 radii, 2.6-12.2 kpc, plus Sigma_T, Sigma_D, Sigma_C and M/L in five bands | **acquired and validated here** -- see below |
| **Milky Way** (Gaia DR3, Drimmel et al. 2023) | 68 radial bins of sigma_vZ for RGB stars over 0.1-13.5 kpc, 18 for OB, with 16/50/84 percentiles | acquired; the cleanest of the set -- direct kinematics, no potential model, no assumed scale height |
| **Milky Way** (Bovy & Rix 2013) | 43 points of Sigma_1.1(R) and K_Z,1.1(R) | acquired, but **model-dependent**: fitted inside a parametrised *Newtonian* potential family, so testing a modified law against it is circular unless the SEGUE kinematics are refitted |
| Herrmann & Ciardullo 2009 | sigma_z profiles for 5 face-on spirals, published **only as figures** | the 774 primary planetary-nebula radial velocities were acquired instead, so the profile is reconstructible from published primary observables |
| DiskMass I-XI | **no** per-radius stellar dispersion table anywhere in the series | confirmed; DMS VI Table 6 is explicitly captioned as exponential *fits* |

**NGC 628 was recovered here after being reported unrecoverable.** Its Table 6
is typeset rotated 90 degrees inside a two-column PDF, so `pdftotext` and
`pdfplumber` both return reversed character strings interleaved with body text.
Setting the page rotation in PyMuPDF fixes it. The transcription was then
*validated, not trusted*: the paper independently states that fitting
`sigma_z(R) = sigma_z(0) exp(-R / 2 h_dyn)` to those points gives
sigma_z(0) = 73.6 +/- 9.8 km/s and h_dyn = 92.7 +/- 13.1 arcsec, and refitting
the transcribed points returns **74.4 km/s and 92.7 arcsec**.

This is worth recording as a worked instance of the brief's silent-extraction
failure mode: reversing the rotated strings by eye -- the obvious shortcut --
returns 223 for Sigma_T at R = 2.6 kpc, which is actually Sigma_D. The true
value is 286. A plausible-looking wrong number, produced silently.

**A structural negative that matters more than the count.** No external galaxy
anywhere has a *measured* scale height alongside a resolved sigma_z(R).
Measuring h_z requires an edge-on view; measuring sigma_z requires a face-on
one. Every face-on sigma_z in the literature, DiskMass and Aniyan alike, pairs
with an h_z inferred from an h_R/h_z relation. So the correlated-by-construction
problem the programme already recorded for DiskMass is not a DiskMass defect --
it is a geometric obstruction that no existing dataset escapes. Any Sigma_dyn
built from sigma_z^2 / (2 pi G h_z) inherits it, and must not be correlated
against a photometric scale length.

