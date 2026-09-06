# Source metadata recovery for the twelve resolved seeds

The two missing S4G originals were recovered from their exact CDS URLs. Their
raw and decompressed SHA-256 hashes match the old source receipt. The new audit
reproduces all 65 previously derived geometry fields for five galaxies and
independently checks 37,632 source fields with the Astropy CDS reader. This is a
completed source-metadata milestone; observational admission remains
`SOURCE_BLOCKED`.

The recovered tables each have 2,352 unique names. Nine atlas seeds occur in
both. DDO154 (also searched as UGC08024), NGC6946 and NGC7331 do not. Absence here
does not establish absence of geometry information elsewhere.

| Seed | S4G distance +/- catalog SD (Mpc) | P4 PA (deg) | P4 ellipticity | P4 flag | Stellar branch | Published CO status |
|---|---:|---:|---:|---|---|---|
| DDO154 | absent | absent | absent | absent | SINGS flux | upper limit |
| IC2574 | 3.827 +/- 0.624 | 44.6 | 0.750 | uncertain | SINGS flux | upper limit |
| NGC2841 | 16.778 +/- 5.279 | 149.5 | 0.543 | uncertain | SINGS flux | detection |
| NGC2903 | 9.058 +/- 1.584 | 19.6 | 0.513 | reliable | P5 cleaned flux | detection |
| NGC2976 | 3.611 +/- 0.707 | 144.0 | 0.401 | reliable | P5 cleaned flux | detection |
| NGC3198 | 13.987 +/- 1.632 | 33.6 | 0.666 | reliable | P5 cleaned flux | detection |
| NGC3521 | 12.078 +/- 2.835 | 162.3 | 0.547 | uncertain | P5 cleaned flux | detection |
| NGC4214 | 3.401 +/- 1.007 | 132.6 | 0.077 | uncertain | P5 cleaned flux | detection |
| NGC5055 | 8.333 +/- 1.867 | 101.4 | 0.444 | uncertain | SINGS flux | detection |
| NGC6946 | absent | absent | absent | absent | SINGS flux | detection |
| NGC7331 | absent | absent | absent | absent | SINGS flux | detection |
| UGC04305 / HoII | 3.340 +/- 0.900 | 30.5 | 0.239 | uncertain | SINGS flux | upper limit |

Distance SD is the catalog's scatter, not an error on its mean or a calibrated
distance posterior. P4 centers belong to its input image coordinates, not P5
cutouts. Geometry definitions and nullable fields are preserved in
[geometry.json](geometry.json), with source links and hashes in
[source-manifest.json](source-manifest.json). Outer-isophote interpretation
requires assumptions about disk shape and thickness; the
[P4 measurement paper](https://arxiv.org/html/1503.06550) describes that procedure.

All 53 pre-existing stellar/CO files (249,250,218 bytes) were rehashed and their
headers checked. No image arrays were loaded. The five `STELLAR_MASS_MAP`
filenames remain cleaned flux in MJy/sr. Header unit overrides are needed for
five color maps, five categorical masks and seven coverage-weight maps. P5
quality is excellent for NGC2976 and acceptable for the other four cleaned
seeds. These quality labels do not certify an absolute mass calibration.

The source audit found a historical applicability error: the old five-object
builder's `COLOR_PUBLISHED_WITH_FALLBACK` branch uses a relation calibrated for
integrated uncleaned light on cleaned P5 colors. It is distinct from Meidt's
old-population relation. The old SINGS conversion contract also mislabels
arXiv 1402.5210 as McGaugh & Schombert; it is Meidt et al. See
[conversion-metadata.json](conversion-metadata.json) for exact formulas,
applicability and citations. Frozen files were preserved.

**The current common-basis and mixed-source atlas fields use fixed M/L=0.6.**
Thirteen recorded source/configuration/code bindings were rehashed without
mismatch. The traced source builder first stores luminosity; both field loaders
apply the fixed scalar. The old color-branch issue does not apply to these
current runs. This conclusion is limited to the recorded chain in
[field-provenance.json](field-provenance.json); no fields were recomputed.

Nine unit/algebra checks pass, including 12 distance/inclination conservation
cases. Definitions using the paper's stated zero point and solar magnitude give
703.79838 Lsun/pc2 per MJy/sr, 0.03432% below its published 704.04. This small
discrepancy is retained and passes the prospectively declared 0.1% tolerance;
exact numerical agreement is not claimed. CO column-density units give
alpha_CO10=4.35704 including helium, within the declared 2% check of the rounded
4.35 and 4.4 conventions. The atlas coefficient 4.35/0.65=6.69231 is a conditional
conversion. No observed mass, motion score or unique 3D source was inferred.

Twenty IRAC2 files declared by the older ten-object inventory are absent at
their exact recorded local paths. No global colors were recomputed. Remaining
requirements include those originals, matched photometry and aperture
calibration, source covariance, uncertain/missing geometry, P1/P5 registration
outside NGC2903, release-specific CO calibration and selection, spatially
variable conversion factors, and unmeasured baryon phases.

Reproduce from the Invariant repository using the existing CPU environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\henry\AppData\Local\Programs\Python\Python313\python.exe' scripts/run_mond_atlas_baryon_recovery.py acquire
& 'C:\Users\henry\AppData\Local\Programs\Python\Python313\python.exe' scripts/run_mond_atlas_baryon_recovery.py validate --output work/gravity-first-principles/mond-atlas-baryon-recovery-001/replay-002
& 'C:\Users\henry\AppData\Local\Programs\Python\Python313\python.exe' -m unittest discover -s tests -p test_mond_atlas_baryon_recovery.py -v
```

Use an unused validation subdirectory; the runner refuses to replace an existing
output. `acquire` reuses cached payloads, with 22 download receipts totaling
6,372,480 bytes, below the 1 GB cap. Raw files remain under
`work/private/mond-atlas-baryon-recovery-001`, already excluded by `.gitignore`.
The final checks and implementation-development failures are retained in
[verification.json](verification.json) and [development-notes.json](development-notes.json).
No common modules, handoff documents or Git metadata were edited. No GPU,
commit, push or new motion-residual access occurred.
