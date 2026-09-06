# Active goal: MOND observation atlas

The user authorized execution, not just design. The goal remains active and
unfinished. Read `work/gravity-first-principles/mond-atlas-execution-009/README.md`,
`execution-status.json`, `verification.json` and `publication-manifest.json`.
This supersedes execution-008 readiness; earlier findings and failures remain.

Current scale: 13,525 identity groups (not certified distinct), 175 radial baseline
galaxies, 126 descriptive-cut galaxies, 12 resolved seeds, 22 source-image fits,
29 conditional field runs for one galaxy, 67 passing atlas unit tests and ZERO
admitted full-field galaxy cube likelihoods. Target remains 10–20 development
pilots then 100–300 eligible resolved systems and thousands of population records
where coverage permits. Population rows do not count as resolved predictions.

Latest finding: 11/12 galaxies have detected HI support in some old background
patches. NGC5055 and NGC6946 passed old noise gates despite extensive overlap.
Background-mean covariance accounting alone leaves the same three sensitive
galaxies. The fixed HI-plus-smoothing exclusion leaves 49/192 evaluable splits;
all 49 pass, but those SAME 49 already passed before exclusion. Do not claim a
noise failure was cured. Only NGC2976 and NGC7331 retain enough background and
pass all 16 splits. No pure-noise, mask-independence or inner-galaxy transfer
claim is established. 143 other splits have insufficient support, not a gravity
failure. Fixed channel-band ends are NOT certified line-free.

New immutable runs: mond-atlas-noise-mean-001, mond-atlas-background-support-001,
mond-atlas-emission-excluded-noise-001. Configurations declare SOURCE_BLOCKED
before implementation. Read docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md
before additional source/solver/scoring work. No new motion scores were computed.
All existing seed galaxies are development-exposed; previous motion comparisons
remain exploratory and nonadmitted. Do not invent retrospective preregistration.

Earlier numerical work remains in execution-008: common bilinear image/source
basis, conditional thin/mixed NGC2903 fields, separate lateral/vertical/box
perturbations and above-plane checks. At R=5 kpc, model QUMOND force-equivalent
speed differs by 1.7%, downward force at z=.25 kpc by 32%, total model mass by
.50%. This is joint deprojection sensitivity, not a measured gravity anomaly.

Next requirements: native selection and spectral-response reconstruction with
validated line-free support and injection recovery; source-image noise and
beam/pixel/absolute-flux likelihood; stellar and HI/H2 conversions and missing
components; independently constrained 3D and exterior-field ensembles; distinct
AQUAL controls and proper warp/streaming/pressure cube prediction; additional
pilots and galaxy/group/survey holdouts. Keep raw observations outside Git.

Five historical STELLAR_MASS_MAP files are P5 cleaned stellar FLUX in MJy/sr.
Only NGC2903 has validated P1-to-P5 relative transfer. The two original S4G
geometry tables remain missing. Do not treat the bound derived geometry record
as revalidated raw metadata. Nondetection is not missing coverage or empty space.

Publication: linked local Git metadata is outside the writable root. The prior
connected GitHub create_blob was rejected because it requires approval while
policy is never. Reads worked and last verified main was
afc721a13782acec4ebc94ad8f6d97ed71be7152; fresh remote review is needed before any
eventual write. Do not bypass either restriction. Nothing has been published.
The previous handoff is archived in this report. Preserve unrelated local work.

Working runtime: bundled Python 3.12/NumPy on CPU; old CUDA environment cannot
start. Shell downloads have been denied. Goal is active because useful local
scientific work continues, despite blocked publication and incomplete sources.
