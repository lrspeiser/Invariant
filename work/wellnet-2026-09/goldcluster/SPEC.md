# Gold Cluster Acquisition Specification

Lane `work/wellnet-2026-09/goldcluster/`. Registry `BO-goldcluster`.
Generated 2026-09-07T07:58:04Z from `probes.json` and `ranking.json`; every number below is
rendered from those files.

**INVENTORY ONLY.** This lane opened no pixel, no shear, no spectrum and no
kinematic measurement, and computed no gravity-relevant statistic. That is
enforced, not promised: `guard.check_query` rejects any archive query whose
projection names a shape, an ellipticity, a response or a per-source
redshift, and `guard.assert_no_statistic` refuses to write any result whose
keys collide with a declared observable. The lane asks archives to COUNT
sources; it never asks them to hand one over.

## 0  Why this specification exists

BE.7 recorded that no public cluster satisfies the charter's complete
experiment, and instructed that a specification be written **before a
target is chosen, so candidates rank by how much new telescope time they
need rather than by how much code already exists for them**.

The programme's seven working clusters -- the Hubble Frontier Fields six
plus Abell 2029 -- were selected the second way. They are the clusters with
the most published products. That is a selection for objects other people
have already spent time on, which is not the same as a selection for objects
where the remaining experiment is cheapest to finish.

This specification therefore starts from the whole X-ray-selected sky.

## 1  The six channels

The complete experiment needs ONE object carrying all six together:

| | channel | what counts as having it |
|---|---|---|
| C1 | resolved baryons | member light profiles, fittable from calibrated multi-band imaging |
| C2 | member internal dynamics | **resolved** stellar kinematics per member, not an aperture number |
| C3 | cluster dynamics | member redshifts, radially complete enough for sigma(R) |
| C4 | raw weak lensing | per-source shapes. A mass map is not lensing data |
| C5 | strong lensing | multiple images, ideally a measured time delay |
| C6 | environment | the large-scale field around the cluster |

C2 is the binding one. BE.7 named it the genuinely missing layer, and this
lane's survey does not change that: **C2 is ABSENT for every one of the 1863
candidates**. The only published resolved member kinematics within this
programme's reach is the MUSE/Granata set, which is held in the CONFIRMATION
RESERVE and was not opened. So C2 costs new IFU time everywhere, and that is
what a proposal should ask for.

## 2  The routes, probed live

Not one of these was taken on the strength of a data-availability sentence.
Each was called and its answer recorded.

| route | channel | verdict | what it actually serves |
|---|---|---|---|
| eRASS1 primary cluster catalogue (Bulbul+2024) | C0 | **AVAILABLE** | M500, R500, Mgas500 and Fgas500 are scaling-relation products and are inadmissible as a quantity to score a gravity law against; the count rates, counts, fluxes and kT are direct observables and ar... |
| DECADE metacal shear (DELVE DR3, Astro Data Lab TAP) | C4 | **AVAILABLE** | archival DECam, so the footprint is PATCHY at the degree scale -- coverage must be checked per cluster, never assumed from a survey boundary |
| DES Y3 metacalibration shape catalogue | C4 | **AVAILABLE** | 312 GB in ONE HDF5, stored in coadd_object_id order, NOT sky order. Accept-Ranges is present, the HDF5 magic verifies, and h5py DOES open it remotely over range requests in about one second -- the ... |
| BUFFALO HLSP lensing DR1 (STScI) | C4 | **PARTIAL** | arXiv:2602.06904 states the six-cluster pyRRG catalogues 'will be made available upon acceptance' at this HLSP. Re-checked live: the directory tree is unchanged since 2023-07 and only abell370 carr... |
| DESI DR1 redshift catalogue (Astro Data Lab) | C3 | **AVAILABLE** | DESI's footprint is northern-hemisphere-weighted and its cluster-member completeness is set by fibre assignment, not by radius -- radial completeness must be measured per cluster before any sigma(R... |

### Never opened, recorded by identity only

| route | status | why |
|---|---|---|
| KiDS weak lensing catalogue | PERMANENTLY_SEALED | KiDS is a permanent sealed holdout under the programme brief. It was scored in round 1, so it is validation, not confirmation. |
| SPT cluster lensing / SPTcl | CONFIRMATION_RESERVE | reserved; recorded by identity, never opened |
| MUSE / Granata 2026 member dispersions | CONFIRMATION_RESERVE | the only published RESOLVED member kinematics for HFF clusters. Reserved deliberately: it is the scarcest channel and therefore the most valuable thing left to seal. |
| Gaia DR dynamical products | CONFIRMATION_RESERVE | reserved |
| X-GAP, CLoGS | CONFIRMATION_RESERVE | reserved |

## 3  The candidate pool

Source: eRASS1 primary cluster catalogue (Bulbul+2024), 12,247 clusters,
fetched under `catalogue_validation` v3 -- all three detectors passed.

Cuts, declared before any coverage was probed: Dec <= +40 (DECam's reach),
0.05 <= z <= 0.7 (the lensing kernel), >= 100 X-ray counts inside R500 (a
profile rather than a detection), contamination probability <= 0.3.
**1863 candidates** survive.

eRASS1's `M500`, `R500`, `Mgas500` and `Fgas500` are scaling-relation
products calibrated on weak lensing. They are **not** carried forward as
observables: scoring a gravity law against a weak-lensing-calibrated mass
using weak lensing would be circular. What is carried forward is count
rates, counts, fluxes and kT.

## 4  What the sky actually holds

Coverage was asked per cluster, one query each, because DECADE is a
reprocessing of **archival** DECam pointings and its footprint is patchy at
the degree scale. This is not a theoretical worry: the single best candidate
in the pool by X-ray counts has about a million DECADE sources within 5
degrees and **exactly zero** within 0.5 -- a hole several degrees across.
A survey boundary would have called that cluster covered.

Measured over 1863 probed candidates:

- **605 clusters have a public raw weak-lensing channel** (>= 2000 usable
  background sources within 0.5 deg, passing the DECADE cosmology selection
  and behind the cluster in redshift).
- Spectroscopy was probed for **all 605** of them, and **186 carry four
  public channels at once** -- C1 resolved baryons, C3 cluster dynamics,
  C4 raw weak lensing and C6 environment. Only C2 and C5 are missing, and
  C2 is missing everywhere.

For scale: before this lane, the programme had a public per-source shear
catalogue for **one** cluster, Abell 370, and its cluster-scale correlation
rested on the twelve X-COP systems.

## 5  The ranking

Rank key: channels already public, then weak-lensing source count, because
C4 is the channel that cannot be substituted. Every candidate needs C2 and
almost every candidate needs C5, so the ranking is really *how much of the
cheap half is already done*.

Top 20 by that key:

| cluster | RA | Dec | z | z type | X-ray counts | kT | shear sources | spec members | public channels |
|---|---|---|---|---|---|---|---|---|---|
| 1eRASS J104432.9-070406 | 161.137 | -7.069 | 0.1322 | spec_z_boot | 509 | 3.72 | 29920 | 237 | 4 |
| 1eRASS J113251.1+142739 | 173.213 | +14.461 | 0.0811 | spec_z_boot | 811 | 4.12 | 29132 | 145 | 4 |
| 1eRASS J134345.3+040548 | 205.939 | +4.097 | 0.1181 | spec_z_boot | 155 | - | 26458 | 205 | 4 |
| 1eRASS J113538.8+133907 | 173.912 | +13.652 | 0.0794 | spec_z_boot | 201 | 1.62 | 25741 | 83 | 4 |
| 1eRASS J101348.5-000718 | 153.452 | -0.122 | 0.0936 | spec_z_boot | 153 | 2.17 | 25516 | 167 | 4 |
| 1eRASS J100901.5-051719 | 152.257 | -5.289 | 0.1478 | spec_z_boot | 106 | 2.03 | 25408 | 153 | 4 |
| 1eRASS J123625.2+163246 | 189.105 | +16.546 | 0.0685 | cg_spec_z | 337 | - | 25199 | 131 | 4 |
| 1eRASS J102024.0-063110 | 155.100 | -6.519 | 0.0540 | spec_z_boot | 183 | - | 25136 | 277 | 4 |
| 1eRASS J120921.0-021336 | 182.338 | -2.227 | 0.1746 | spec_z_boot | 149 | 1.39 | 24454 | 333 | 4 |
| 1eRASS J123933.6+153004 | 189.890 | +15.501 | 0.0709 | spec_z_boot | 146 | - | 23968 | 61 | 4 |
| 1eRASS J114341.5-014429 | 175.923 | -1.742 | 0.1064 | spec_z_boot | 143 | 3.45 | 23562 | 419 | 4 |
| 1eRASS J130118.6-032909 | 195.328 | -3.486 | 0.0843 | spec_z_boot | 104 | 1.51 | 23402 | 395 | 4 |
| 1eRASS J141736.6+020342 | 214.403 | +2.062 | 0.0537 | spec_z_boot | 361 | 1.22 | 23269 | 242 | 4 |
| 1eRASS J120143.7-001110 | 180.432 | -0.186 | 0.1677 | spec_z_boot | 136 | 2.63 | 23031 | 1100 | 4 |
| 1eRASS J085751.0+031014 | 134.463 | +3.171 | 0.2025 | spec_z_boot | 164 | - | 23009 | 473 | 4 |
| 1eRASS J132852.4-025642 | 202.218 | -2.945 | 0.1840 | spec_z_boot | 158 | 4.20 | 22784 | 363 | 4 |
| 1eRASS J143736.1-001740 | 219.401 | -0.295 | 0.1379 | spec_z_boot | 175 | - | 22595 | 626 | 4 |
| 1eRASS J033308.8-014206 | 53.287 | -1.702 | 0.1407 | cg_spec_z | 286 | - | 22497 | 375 | 4 |
| 1eRASS J131417.1-065942 | 198.571 | -6.995 | 0.1946 | spec_z_boot | 244 | 3.28 | 22165 | 201 | 4 |
| 1eRASS J114604.4-081556 | 176.518 | -8.266 | 0.0981 | photo_z | 194 | - | 21834 | 99 | 4 |

## 6  What to ask for

The specification's operative conclusion is that the expensive channel is
the same one for every candidate, so the target should be chosen by the
cheap channels and the proposal written for C2.

1. **Choose from the top of section 5**, not from the HFF six. Those
   clusters already carry C1, C3, C4 and C6 from public survey data.
2. **Ask for C2**: resolved stellar kinematics for member galaxies in one
   cluster that already has public shear. That is an IFU programme, and it
   is the only thing standing between this programme and a complete scene.
3. **Do not ask for C4.** It is already public for 605 clusters.
4. **Seal before scoring.** See section 7.

## 7  The sealing opportunity, which is the real prize here

The programme has no sealed confirmation set. KiDS and the wide binaries
were both scored in round 1, so they are validation, not confirmation, and
nothing else was ever held back.

`confirmation_status` v2 is explicit that **Mentioned, Acquired and
Transformed are NOT spent** -- only Scored, Inspected and Decision-used are.
This lane has therefore Acquired nothing but identities and counts, and has
Scored nothing. Every one of the 1863 clusters is still pristine.

That makes a genuine confirmation set available for the first time, and it
is independent of the spent data on all four of v2's axes:

- **untouched outcome** -- no shear behind any of these clusters has been read;
- **untouched objects** -- none of the seven worked clusters is in this pool;
- **untouched survey** -- DECam/DELVE, not KiDS, not HSC, not HST;
- **untouched reduction pipeline** -- DECADE metacalibration, not lenstool, not pyRRG.

**The recommendation is to split the 605 now, before anything is scored,
and seal one half at loader level under `holdout_seal` v2.** Splitting after
a first look is how the last confirmation set was lost.

## 8  Limits of this specification

- **"Four public channels" is two independent measurements, not four.**
  C1 is inferred from C4 (a cluster with DECADE shapes necessarily has the
  DECam imaging those shapes were measured from -- but band coverage and
  depth were not verified per cluster), and C6 is scored identically to C3
  (the spectroscopy that gives cluster dynamics also samples the field --
  but no independent environment statistic was computed). So the honest
  statement is: 605 clusters have public shear, and 186 of those also have
  usable public spectroscopy. Do not read the channel count as four
  separate archive products.
- C5 (strong lensing) was **not** probed per cluster. It is recorded as
  UNMEASURED rather than absent.
- **The 0.5 deg box is a coverage test, not the measurement aperture.**
  Computed for the covered clusters in a flat LCDM (H0=70, Om=0.3), its
  physical half-width is 1.76 Mpc at worst and 4.60 Mpc at the median, so
  no cluster was called covered on a box too small to hold a profile. But
  for the 186 lowest-redshift covered clusters it is under 3.5 Mpc, i.e.
  smaller than the outer bin Chiu+2022 and Umetsu+2020 use. For those the
  extraction needs a wider angular aperture, and the source counts here
  UNDERSTATE what is available.
- Spectroscopic completeness is counted, not characterised. DESI fibre
  assignment is not radially uniform, and a sigma(R) field built from it
  without a selection function would inherit that.
- eRASS1 photo-z clusters carry redshift error into every physical radius.
  809 of the pool have spectroscopic redshifts and should be preferred.

