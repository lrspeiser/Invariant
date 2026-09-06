# SLACS source-provenance pilot 001

The bounded ingest is complete for three previously exposed development systems.
The current executable receipt is `replay-002/summary.json`; the first receipt
in `ingest-001/` is preserved. This package remains **SOURCE_BLOCKED** for an
ordinary-matter motion/lensing test. There are **zero new gravity fits, lensing
scores, or admitted joint likelihoods**. No GPU was used.

| System | Lens z | Source z | SDSS aperture dispersion (km/s) | Grillo Chabrier/BC03 stellar mass (10^11 solar masses) |
|---|---:|---:|---:|---:|
| J0216-0813 | 0.3317 | 0.5235 | 333 ± 23 | 7.0 +1.1 / -2.2 |
| J0737+3216 | 0.3223 | 0.5812 | 338 ± 17 | 5.2 +0.1 / -1.2 |
| J0912+0029 | 0.1642 | 0.3239 | 326 ± 16 | 4.3 +0.8 / -0.8 |

The redshifts and dispersions are exact selected rows of
[Bolton et al. 2008, Table 4](https://arxiv.org/abs/0805.1931), published through
[CDS J/ApJ/682/964](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJ/682/964).
Dispersion is uncorrected for the 3-arcsec diameter SDSS aperture; errors are the
published RMS values, with the paper's 5% floor subject to catalog rounding.
Earlier SLACS releases report different dispersions and errors. No object-level
redshift errors are present in the selected table, and decimal precision is not
an uncertainty. This is an aperture measurement, not a resolved velocity field.

The stellar masses are **photometric inferences**, with the published asymmetric
intervals and integer catalog rounding retained. The pilot includes all four
Grillo IMF/template alternatives and all five SDSS ugriz magnitudes and RMS
errors. Their inputs are total photometry and redshift, with solar metallicity,
dust-free templates and an assumed star-formation history. The separate lensing
mass analysis in that paper is excluded: no Einstein-aperture factors, total
lensing masses, or response-selected preferred IMF enter the source packet.
See [Grillo et al. 2009](https://arxiv.org/abs/0904.3282) and
[CDS J/A+A/501/461](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A%2BA/501/461).

The packet also contains HST B/V/I/H profile photometry and ancillary stellar
estimates from [Auger et al. 2009](https://arxiv.org/abs/0911.2471). Its section
4.2 conditions the metallicity prior on **the lens velocity dispersion**. Those
SPS masses cannot be treated as independent input mass for a dispersion test.
The HST magnitude/radius table does not supply object-level errors. No invented
error floor is substituted, and the later photometry is not attributed to the
acquired early single exposures.

## Actual image holdings

Three native MAST calibrated ACS/WFC F814W exposures were downloaded:

| System | Native product | Observation date | Exposure | SHA-256 |
|---|---|---|---:|---|
| J0216-0813 | j93i27kuq_flt.fits | 2005-01-05 | 420 s | `3cde15da19ba2e9597e3e7302e72acf3b06901c453a762a8c48ef251c3c756db` |
| J0737+3216 | j93i40gzq_flt.fits | 2004-09-21 | 420 s | `75ba3dc0d0f3cc93e939e69d792cae08c33c00bc257f57fad0be6ad9edad9971` |
| J0912+0029 | j93i43ilq_flt.fits | 2004-10-12 | 420 s | `44ec97bfb1c1333298ec0adf8a32bffb9d533549ac34105c0eb373cdfb4f8cab` |

Each product has two 2048×4096 science chips and matching ERR/DQ arrays, plus
astrometric support extensions. Target identity is joined through the exact
SDSS plate/MJD/fiber, HST TARGNAME and sky coordinates. Pointing/catalog
separations are 0.2041, 0.1403 and 0.0200 arcsec. A 201×201 target region on
SCI EXTVER=2 is saved privately with exact native SCI, ERR and DQ values. No
resampling, foreground subtraction, acceptance mask or flux conversion is
applied. The pixel-to-sky-to-pixel check has maximum error 1.4911e-9 pixel.

SCI and ERR have verified ELECTRONS units. DQ remains an integer bit field;
750, 685 and 710 target-region pixels respectively have nonzero bits. These
counts are not an adopted rejection rule. Native header calibration version is
10.4.1 (08-Aug-2025), and the retrieved release is distinguished from the
historical paper processing. FLT is neither a geometrically corrected image nor
the pixel-CTE-corrected FLC product. CRCORR=OMIT and PCTECORR=PERFORM are retained.
Visual inspection shows cosmic rays. ERR is a pipeline error estimate, not an
independently validated complete pixel covariance. No embedded FITS checksums
were present; byte integrity is supplied by the download hashes. Instrument
definitions are bound to the downloaded
[ACS file types](https://hst-docs.stsci.edu/acsdhb/chapter-2-acs-data-structure/2-1-types-of-acs-files),
[file structure](https://hst-docs.stsci.edu/acsdhb/chapter-2-acs-data-structure/2-2-acs-file-structure)
and [pipeline documentation](https://hst-docs.stsci.edu/acsdhb/chapter-3-acs-calibration-pipeline/3-2-pipeline-overview).

The [author-hosted Paper I archive](https://web.mit.edu/~burles/www/SLACS/table.htm)
was also acquired: `driz_20050714.tar.gz`, 141,187,558 bytes,
SHA-256 `fbb4e9d6d928e477c03e6da0afadbc3ab4edd8a47f568aa57bd1e33612519d6e`.
Only the three selected systems' 24 FITS arrays were extracted and parsed: two
bands per system, each with `subdrz`, `drzerr`, `drzcrm`, and `drzftm` files.
These are 601×601 arrays. Their headers retain only RAZERO/DECZERO positional
information and lack BUNIT, full celestial WCS and calibration metadata. The
filename-suggested error/mask roles remain uncertified; some mask-like values
are fractional. They are not used as a calibrated likelihood. Native F814W and
legacy F814W versions refer to the same underlying exposures and are not
independent observations. No pixel-equivalence claim is made between reductions.

## Readiness and exposure limits

Native image pixels provide a concrete future image-response input. No measured
multiple-image position/error table, background-source reconstruction, PSF
kernel, focus/chromatic PSF uncertainty, or foreground/arc analysis mask was
acquired. Native DQ and candidate legacy masks do not supply those missing
objects. Light-profile separation, noise/covariance validation, calibrated
photometry, independent stellar M/L, gas and dust constraints, source depth,
orbital/seeing uncertainty and line-of-sight environment remain necessary.

Lensing-inferred mass, convergence maps, halo fits, dynamical mass and SIE
Einstein radii are not used as baryonic truth. The ingestion only validates
files, identities, units, array integrity and bookkeeping; it implements no
physical source builder or gravity/light operator. A future operator requires
independent benchmarks, dimensions/limits/convergence/boundary checks and an
explicit **relativistic light-propagation closure**. A nonrelativistic MOND
acceleration rule alone does not define the deflection of light.

These SLACS galaxies are not automatically matched to the atlas's twelve nearby
HI pilots. No such coordinate crossmatch or matched sample has been established.
The convenience selection of three existing Item 17 exploration systems is not
a held-out or representative scientific sample.

**Exposure disclosure:** discovery opened the complete MIT overview and
extracted text from the Bolton 2006/2008, Auger 2009 and Grillo 2009 PDFs,
including table content with legacy reserved systems. Those primary PDFs and
the original multi-system archive are cached privately. Do not certify any of
the twelve legacy confirmations entirely unseen after this session. Production
exact-row queries and FITS-array extraction/parsing were restricted to the
three exploration systems. No reserved exact-row query, gravity fit, lensing
score or mutation of historical receipts occurred. The coordinator was notified.

## Files and reproduction

Repository root:
`C:\Users\henry\Documents\Codex\2026-09-04\pu-2\work\Invariant`.

New implementation files are `scripts/mond_atlas_lensing_pilot.py`,
`scripts/run_mond_atlas_lensing_pilot.py`,
`configs/mond_atlas_lensing_pilot_v1.json` and
`tests/test_mond_atlas_lensing_pilot.py`. All raw downloads, original papers,
legacy arrays and ROI pixels are under the Git-ignored
`work/private/mond-atlas-lensing-pilot-001/`. Conservative cumulative download
body accounting is **650,224,529 bytes**, below the 1,000,000,000-byte cap;
the principal 29-asset manifest contains 650,187,552 bytes. Offline replay made
no new network calls. Download headers, timestamps, original/resolved URLs,
release descriptions and SHA-256 values are retained.

`replay-002/` contains `summary.json`, `systems.json`, `measurements.csv`,
`selected-source-tables.json`, `source-manifest.json`,
`legacy-archive-inventory.json`, `exposure-and-local-audit.json`,
`configuration-snapshot.json`, `input-bindings.json` and `output-manifest.json`.
The root `verification.json` and `deliverable-manifest.json` record final checks
and the exact publication set. The first run is preserved and predates the
expanded exposure wording and explicit field-role labels.

From the repository root, the executed commands were:

```powershell
& 'C:\Users\henry\AppData\Local\Programs\Python\Python313\python.exe' -B scripts/run_mond_atlas_lensing_pilot.py --output work/gravity-first-principles/mond-atlas-lensing-pilot-001/ingest-001
& 'C:\Users\henry\AppData\Local\Programs\Python\Python313\python.exe' -B scripts/run_mond_atlas_lensing_pilot.py --offline --output work/gravity-first-principles/mond-atlas-lensing-pilot-001/replay-002
& 'C:\Users\henry\AppData\Local\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests -p test_mond_atlas_lensing_pilot.py -v
```

Use a new directory name under the assigned output root for another run.
Existing receipts cannot be overwritten. A fresh online cache can acquire the
configured sources; primary documentation hashes are pinned, while current
catalog/MAST bytes are recorded per acquisition. The exact historical bytes are
replayed from the verified cache. Python 3.13.5, NumPy 2.2.6 and Astropy 7.1.1
were used on CPU. **21 tests pass**, covering malformed/extra/duplicate rows,
unit/identity handling, reserved-target exclusion, forbidden source columns,
archive traversal/links, immutable receipts, cache tampering, offline misses,
download limits, independent historical serialization comparisons and exact
native ROI equality. These are ingest checks, not scientific solver admission.

Only assigned new files were edited. Common modules, handoff documents,
historical outputs and Git metadata were not edited; no commit or push was made.
The coordinating task owns integration and publication.
