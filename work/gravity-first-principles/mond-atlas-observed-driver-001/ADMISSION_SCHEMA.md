# Parent-authored admission contract v1

The real admission file **does not exist** in this milestone. Its expected location is `parent-admission.json` in this directory. The driver never writes it. A pass string alone does not admit anything.

Required top-level fields:

- `schema`: `observed-driver-admission-v1`; `reviewer`: `parent`; `phase`: `fit_and_evaluation`.
- `cube`, `cache`: repository-relative or absolute paths. Cube must equal the source named by the bound NGC2976 native-spectral instrument document.
- `cube_header_sha256`: SHA256 of `fits.getheader(cube).tostring().encode()`.
- `bindings`: map **absolute resolved paths** to SHA256. Include all entries returned by `code_closure()`, frozen aperture CSV, covariance JSON, cache NPZ, raw cube, instrument NGC2976 JSON, every evidence JSON and all source assets/provenance dependencies. Byte hashing may read held bytes, but does not interpret response values.
- `source_bindings`: nonempty list of source/provenance paths already in bindings. Review must include actual source conversion, projection and pressure/force dependencies, not merely file existence.
- `cases`: nonempty unique objects `{id, kwargs}`. kwargs must explicitly specify `height`, `model`, `pressure_reference`, `spin`, `branch`, `material`; cache path is supplied separately. Both spins -1 and +1 must occur exactly once for each otherwise identical case. `case_sha256=canonical(cases)` using the exported helper.
- `evidence`: exactly three keys `spectral_convergence`, `source_assumptions`, `background_injection`.

Each evidence entry has `path`, `sha256`, `covered_case_ids` (exactly all declared cases), `required_row_ids` (unique nonempty), and `assertions` (every required row represented). Each assertion is `{row_id, pointer, operator, expected}`. `pointer` is a JSON path array, e.g. `["comparisons", 0, "relative_L1"]`; operators `less_equal`, `greater_equal` require finite numeric values; `equals` requires equal types and values. Multiple assertions per row are permitted. Every assertion is evaluated against the bound evidence document. Include count/completeness assertions and all numerical criteria, not just a summary status. Failed cases cannot be erased by listing fewer rows. The parent reviewer must independently verify that row IDs and covered cases exhaust the predeclared experimental design; this is scientific review, not a security signature. Hashes prevent accidental changes, not malicious rewriting of both an artifact and its receipt.

The driver checks each declared model can be constructed from the bound cache before opening training pixels. All planned physical source alternatives and three instrumental branches must be accounted for in the parent's admission review. A smaller case list cannot silently redefine the first-score goal. The present cache API admits one cache per run; different physical source caches require separate predeclared, equally bound runs. Evaluating a subset and selecting the winning law/spin/source from it is not supported.

Restrictions to document explicitly in source_assumptions evidence: historically exposed galaxy, photometric deprojection and fixed inclination/PA, source-conditioned HI/CO dependence, raw-column denominator with smoothed pressure numerator, spin ambiguity, fixed spectral response branches, shared covariance approximation, absolute HI flux convention, unknown-emitter treatment, finite-source alternatives not yet reconstructed. Any acknowledged approximation must have its declared numerical budget checked. Acknowledgement alone does not convert a failed required numerical check to pass.

The full-phase gate is deliberately stricter than a provisional training-only diagnostic. This version implements **no provisional training-only bypass**. That option would need a separate prospective declaration; evaluation remains forbidden without the complete evidence contract.
