# Public-data acquisition audit: lanes 2, 6, 7, 8, and 9

Observed 2026-08-31. This is an acquisition ledger, not a scientific result. Downloads are confined to `work/private` except for pre-existing frozen packages. Every newly acquired scientific payload was treated as opaque and SHA-256 hashed; no HDF5/FITS/XLSX/TXT scientific array or row was decoded. Public event-metadata JSON was used only to bind GWOSC product filenames. Lane-7 SLACS confirmation responses and all Lane-9 scientific rows remain unopened.

## Lane 2 — coherent BNS propagation

Status: **source acquisition materially advanced, executor still blocked**.

Official sources: [LALSuite GitLab](https://git.ligo.org/lscsoft/lalsuite), [LALSimulation waveform API](https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_inspiral__h.html), [GWTC-1 calibration release P1900040](https://dcc-llo.ligo.org/LIGO-P1900040/public), [GW170817 source-properties release P1800061](https://dcc.ligo.org/LIGO-P1800061/public), [GW170817 GWOSC page](https://gwosc.org/events/GW170817/), [GW190425 official event page](https://gwosc.org/eventapi/html/O3_Discovery_Papers/GW190425/v1/).

Acquired:

| Product / revision | Bytes | SHA-256 | Local path |
|---|---:|---|---|
| LALSuite `lalsuite-v7.26.15`, tag commit `2c00d8c200308422036d0a23b7c0394f7c73faad` | 240,331,185 | `f2cdd7aa4676f7b1cccb5e4de8513022e4004136e60577fbae8874437d56c042` | `work/private/open-gravity-lane2-waveform-source-v1/lalsuite-lalsuite-v7.26.15.tar.gz` |
| Tag metadata | 1,885 | `b9e87372e449ee85c159c584ff5eae53612e19d14d8ef88a5dc47a1a00ec28d1` | same directory |
| `GWTC1_CalEnv.tar.gz`, DCC P1900040/001 | 1,363,426 | `0fb552271b43d149f62c8972c59d2d9500471b057247bcae7bc8d070a33574ad` | `work/private/open-gravity-lane2-calibration-source-v1` |
| `GW170817_PSDs.dat`, DCC P1800061/011 | 14,515,336 | `eedbaf2fbcaaa6b0cf5b5314f6469a745a8b8388dda302fe1b627a8bf717fbae` | same directory |

The repository already has the exact three C00 4096-second GW170817 products: H1 125,217,658 bytes / `9e3f8a3adb966f6d70eeade0bc44bea2344f85b2af5233a3cba34a723984c9e2`; L1 124,266,501 / `48eedcf12e5c6d5fea68c8c66facae657b72ad65c5d405d543dd87fcdaef5e0b`; V1 129,470,892 / `ed723e67105551a051b2758484b5b80a127f6b57f28174b460b17d3ee601b4cf`. The 4096-second files provide bounded off-source data around the event. The separately acquired R1 products in Lane 8 are a different revision and must not be silently mixed with C00.

Holdout: GW190425 (DOI `10.7935/ggb8-1v94`) is the clear independent public BNS candidate. Keep its strain, glitch model (DCC T1900685), and parameter-estimation samples (DCC P2000026) sealed until a BNS holdout selection rule and event role are frozen. No second comparably unambiguous public BNS holdout was verified; do not cherry-pick lower-probability catalog events.

Still required before coherent inference: one frozen data revision; exact `IMRPhenomPv2_NRTidalv2`/alternative-waveform call and version; coherent detector response and timing convention; calibration-envelope marginalization; PSD construction from preregistered off-source windows; glitch/gating rule; priors; sampler and convergence criteria; injection recovery; and a response-blind development/confirmation split. LALSuite is GPL-2.0-or-later. GWOSC identifies its open data as CC BY 4.0; DCC records should be cited under their release terms.

## Lane 6 — Solar-System ephemeris

Status: **DE440 evaluator source complete; independent refit source blocked**.

Official sources: [NAIF generic kernels](https://naif.jpl.nasa.gov/naif/data_generic.html), [DE440 binary](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp), [DE440/DE441 paper](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440_and_de441.pdf), [INPOP21a documentation](https://www.imcce.fr/content/medias/recherche/equipes/asd/inpop/inpop21a.pdf), [INPOP binary format](https://www.imcce.fr/content/medias/recherche/equipes/asd/inpop/inpop_file_format_2_0.pdf).

| Product | Bytes | SHA-256 | Published MD5 | Local path |
|---|---:|---|---|---|
| NAIF/JPL `de440.bsp` | 119,799,808 | `a4ce9bf9b3282becc9f4b2ac3cebe03a2ae7599981aabd7265fd8482fff7c4b5` | `c9d581bfd84209dbeee8b1583939b148` | `work/private/open-gravity-lane6-ephemeris-source-v1/de440.bsp` |
| DE440/DE441 paper PDF | 5,501,569 | `96c3945deb12e0f0843c2bf0703d3542d950cf85ff903b1fe11a501634f68ee0` | `de9c54d0bd6f1fdf593237c85951d3cb` | same directory |

DE440 is sufficient for a fixed-ephemeris evaluator, but not for an independent reproduction/refit. The exact DE440 observation rows, weights, data corrections, spacecraft/VLBI/ranging likelihoods, and the fitted 343-asteroid plus 30-KBO/ring state-and-mass manifest were not found as one public, revision-bound packet. NAIF's older `codes_300ast_20100725.bsp` is not DE440-equivalent. INPOP21a is an eligible independent-ephemeris control only after its exact binary revision and comparison protocol are frozen; its public documentation is not a replacement for DE440's fit inputs. NAIF/JPL files are public U.S. government archive products; cite NAIF/JPL.

## Lane 7 — ESO 325-G004 and SLACS

Status: **raw public imaging acquired; registered likelihood payload still blocked; SLACS confirmation sealed**.

Official sources: [MAST program/product service](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html), [ESO Science Archive](https://archive.eso.org/cms.html), [ESO 325-G004 science release](https://www.eso.org/public/news/eso1819/), [SLACS CDS catalog ReadMe](https://cdsarc.cds.unistra.fr/ftp/J/ApJ/682/964/ReadMe).

| Product ID | Bytes | SHA-256 | Local path |
|---|---:|---|---|
| MAST HST 10429, `hst_10429_09_acs_wfc_f814w_j95t09_drc.fits` | 369,486,720 | `f2a711874a38cf6364d7222d17cb210e8800b054fb5e2f01bf3f8aa061ad484a` | `work/private/open-gravity-lane7-eso325-source-v1` |
| MAST HST 10429, `hst_10429_10_acs_wfc_f475w_j95t10_drc.fits` | 367,663,680 | `7e77aa1ca44a26f491fe0ac8d6bfd8614ff6d036a460967f39c7a4bfdf2d0d17` | same directory |
| ESO MUSE white-light `ADP.2016-09-07T12:23:32.516` | 1,108,800 | `453ac7da20c8d8e8b2bb845870983603ce62e86dee45ce4e949b8583b81ffa73` | same directory |

The corresponding MUSE cube is exact product `ADP.2016-09-07T12:23:32.515`, 7,378,352,640 bytes, at `https://dataportal.eso.org/dataPortal/file/ADP.2016-09-07T12:23:32.515`; it was not downloaded because a 7.38-GB cube is not a bounded substitute for the unavailable registered paper products.

Still missing: exact HST science masks, correlated-noise map/model, HST PSFs, lens-light subtraction recipe, MUSE kinematic Voronoi bins, per-bin 2x2 velocity/dispersion covariance, MUSE PSF/LSF, registered HST/MUSE intersection, and the frozen cosmology/distance receipt. These likely require an author-supplied reproduction bundle or a verified supplement. MAST/ESO products are publicly accessible after archive release, but no file-level permissive-license assertion is made; apply archive acknowledgments.

SLACS retains the pre-existing 45-development/12-confirmation split in `runs/gravity/roadmap/item-17-slacs-running-strength-v1-source/sample-manifest.json`. No confirmation Einstein-radius or velocity-dispersion response was opened or acquired. Even public HST/SDSS payload acquisition should be role-filtered from the frozen manifest and restricted to development objects until the ESO likelihood is complete.

## Lane 8 — dispersion, polarization, collapse, and gravity-entanglement tests

Status: **bounded multi-detector GWOSC and Gran Sasso source files acquired; four executors remain method-limited**.

Official sources: [GWOSC API](https://gwosc.org/api/), [GWTC-1 documentation](https://gwosc.org/GWTC-1/), [GWTC-3 tests-of-GR Zenodo record](https://zenodo.org/records/7007370), [Gran Sasso Nature Physics article and source data](https://www.nature.com/articles/s41567-020-1008-4), [Holometer data-management plan](https://holometer.fnal.gov/Holometer_DMP.pdf), [original BMV proposal](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.240402), [2026 BMV feasibility discussion](https://journals.aps.org/prd/abstract/10.1103/87dc-qt73).

GWOSC v3 metadata JSON was saved for GW150914 (13,903 bytes / `1ce014e3277cc6ef491daa48250465eefda026f2b974d2a15bb8c6ea5596e37e`), GW170814 (17,388 / `01d25d55e607d16ef32c95cb8672dc5c99fd452d06e61c0e48d17594f5809d20`), and GW170817 (20,094 / `2cd7e903b895f5d3a406383362172a411b127efdcf9be65590a8234b9352ddcb`). Exact 4096-Hz, 4096-second R1 HDF5 payloads were acquired without opening datasets:

| Event/detector | Bytes | SHA-256 |
|---|---:|---|
| GW150914 H1 | 130,158,262 | `8ef8b762cff53b422429c474b1e35a6d9bfa86f3ed0c0d3e855f599781390a27` |
| GW150914 L1 | 134,516,622 | `b01e19ead54aa08d82dc3feb98722a36870eab87ce3599a2923b1823e4a158f8` |
| GW170814 H1 | 150,355,479 | `a06161fabc396b057838f5e80d5bff83a8b39955b0985033308a5eeb5bd36232` |
| GW170814 L1 | 161,252,898 | `2a29b2391e3f925f14166b3fd565364a72314c8ccf85dbd1680e607bcab5ee71` |
| GW170814 V1 | 153,331,630 | `445e9139126836011c6b4a51a60f774e536cfe9129e9a0f459d3e1b7bfc154b3` |
| GW170817 H1 | 144,890,745 | `4f0be5c6f38f37f44df55b1bf504498fe84f5ff7ec98342ac451e55e78b187e2` |
| GW170817 L1 | 161,963,412 | `e051d3de42bab57095e23c4c4345ce84f6e1c1dc23596abdf0d637c8d5ba0899` |
| GW170817 V1 | 144,677,558 | `58d8cab3c352229e5c3b7f6831a634226cd094243df301f7de60127ab1d352e2` |

These are in `work/private/open-gravity-lane8-gwosc-source-v1`. GW170814 supplies the three-detector geometry needed for a polarization development test; the three events provide a bounded dispersion development set. Still required: a frozen waveform family, calibration model, PSD/off-source rule, sky/time marginalization, tensor/vector/scalar antenna convention, dispersion parameterization, priors, injections, and a held-out event split. The official GWTC-3 tests-of-GR archive is CC BY 4.0 but its LIV posterior ZIP is about 7.4 GB; it was not downloaded because it is an aggregated posterior product, not a substitute for the frozen raw likelihood.

Gran Sasso publisher files, retained opaque in `work/private/open-gravity-lane8-gran-sasso-source-v1`:

| File | Bytes | SHA-256 |
|---|---:|---|
| `MOESM1_ESM.pdf` | 940,600 | `69114bff9d34b90661a2f0c1eff8acd6eac0ea75fbc9516b5dfbc78f307d5be3` |
| Figure 3 source data `MOESM2_ESM.xlsx` | 50,100 | `ef86fe9fac36fcee81c0a762571737950c5cccbc7343178593743199ef6e5e23` |
| Figure 4 source data `MOESM3_ESM.xlsx` | 8,889 | `29a1f3ad10ee59e00bc5feb3a81453b9728ab6113151cb30f1b92cad78d3d6d7` |

The spreadsheets resolve exact public figure data but not the full collapse-spectrum likelihood. A frozen workbook sheet/range/schema, detector response/background covariance, Geant4 implementation, and experimental details identified by the paper as on-request or NDA-limited are still missing. Publisher copyright/terms apply; no CC claim is made.

Holometer: the official plan releases processed power/cross spectra and averaged spectra, not raw time streams. It therefore cannot test the proposed time-domain third/fourth cumulants without a different public raw stream. BMV: no observed gravity-mediated entanglement response dataset was found; current public work remains proposal/theory/feasibility. No empirical download is eligible.

## Lane 9 — CF4/VAST void correlation and Pantheon+

Status: **source bytes are substantially complete; executor remains BLOCKED and may not decode development rows**.

Official sources: [CF4 CDS record J/ApJ/944/94](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJ/944/94), [VAST CDS record J/ApJS/265/7](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/265/7), [VAST Zenodo v1.3.1](https://zenodo.org/records/11043278), [VAST code](https://github.com/desi-ur/vast), [Pantheon+SH0ES DataRelease](https://github.com/PantheonPlusSH0ES/DataRelease).

The pre-existing opaque receipt already binds seven exact CF4/VAST files: CF4 ReadMe 19,918 / `2cfed1418147d5a626dee1fa37c47252124477c9828490508d9bbe511d34edb4`; table3 1,617,747 / `1c02e2b3829b0b323524a5f3671a5f530cbdccae3c10b432db0c2e2fe09672fe`; table4 2,528,953 / `be91d4fae6fa01552ab3bc85db695411fca3249eeae08b566a712e6ea790bd99`; VAST ReadMe 12,888 / `787a0476455c7593e4c84cf8f002f8e81e086b20822c752d7a89210b332b68bb`; table1 426,486 / `fe430c4e31f879e5678479b6735d84e2e6ac6c26f5bd71115b56d8d2c1bdbeff`; table2 3,360,828 / `8e2a919f70883a5160b668660a06738d9ac0d091e751ee397431f64d4a77c9bf`; table3 373,744 / `b503649b9cbb19f1a9255d00e950ffee9782a49952cdd4886fe70d5fe33230a1`.

Added from official Zenodo release 1.3.1 (published 2024-04-22, CC BY 4.0), without decoding:

| Product | Bytes | SHA-256 | Published MD5 |
|---|---:|---|---|
| `VoidFinder-nsa_v1_0_1_Planck2018_comoving_holes.txt` | 3,150,288 | `9465f913d9efd0c7df88ad0ec039339e021b1ce2f3ca9aad3702dfbeefa98e21` | `d49a52273fd3818141dfc8b63f6ba2ca` |
| `VoidFinder-nsa_v1_0_1_Planck2018_comoving_maximal.txt` | 181,659 | `98082b9973e491757ca854a519a3ecd98689d064b928c522ae31954d58604afc` | `1c770ccb0441f5c3cc7e9209feda12f8` |

Added Pantheon+SH0ES at exact repository commit `c447f0fea703fcd0fff57de5000947b5ca81286b`, without decoding: `Pantheon+SH0ES.dat` 579,283 / `1cb0fc379ef066afdc2ffd1857681cc478024570d8a3eba284fb645775198cf8`; `Pantheon+SH0ES_STAT+SYS.cov` 33,284,960 / `abf806d966485e64afdb359c87bffc0ecc00d05eff0a31ced66f247385df0fdc`. The repository exposes the files publicly but declares no GitHub-detectable license; rights are `NOASSERTION`, so do not claim CC redistribution.

Missing: the exact 64 HR4 mocks used for VAST RSD/calibration were not found in the official CDS/Zenodo/code releases; request a revision-bound author archive. The public VAST code is BSD-3-Clause, but its small synthetic example is not a paper mock. Before any decode, repair/freeze the Lane-9 executor contract: define the STF shear basis and normalization; make the velocity nuisance units explicit including division by `c`; specify permutation statistic, tie rule, tail correction, and profile solver; resolve minimum-count staging without opening validation/confirmation; and define duplicate-row semantics. Maintain canonical mask parsing and exact row/count aborts. The executor may not decode even development rows until that contract passes.

## Gate summary

| Lane | Safe next action | Current prohibition |
|---|---|---|
| 2 | Freeze coherent likelihood and choose C00 versus R1; then decode development strain only | Do not open GW190425 holdout |
| 6 | Build fixed-DE440 evaluator; separately negotiate/refit-source availability | Do not claim an independent DE440 refit from the BSP |
| 7 | Obtain registered masks/PSFs/Voronoi covariance; then open ESO development | Do not open 12 SLACS confirmation responses |
| 8 | Freeze event roles and four distinct likelihood contracts | Do not treat processed Holometer spectra or BMV theory as missing positive data |
| 9 | Repair and re-audit executor, then authorize development decode | No CF4/VAST/Pantheon+/Planck row decode now |

