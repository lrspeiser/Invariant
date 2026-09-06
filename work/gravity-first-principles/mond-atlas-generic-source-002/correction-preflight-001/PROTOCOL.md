# Frozen v2 discretization correction

SOURCE_BLOCKED. V1 is preserved, including its failed actual annular-fill D²
check. Parent explicitly authorized separate append-only v2 modules, tests and
fresh immutable corrected NGC3198 packets. No response access is allowed.

Evidence before implementation: source images and coverage are unchanged to
roundoff across distance cases; trusted masks are identical. floor of the
physical radius/physical annulus width relabels 248 and 304 exact ring-boundary
cells after lower/higher distance scaling. This changes annular fill, not signed
measured or zero-fill integrals. V1 total-mass scaling error reaches 0.03584%.

Minimal correction, frozen before v2 implementation:
1. Keep all sky vectors, registration, projected areas, pixel integration,
   source conversions, observed arrays, coverage rules, masks and fills unchanged.
2. Require explicit integer index geometry: half-width 256 cells, annulus width
   2 cells, taper start 192 cells, cutoff 224 cells. Check agreement with physical
   lengths within 1e-12 relative metadata tolerance (not an observation tolerance).
3. Cell index coordinates are exact signed integers j,k about the center. Use
   integer r²=j²+k² to assign annuli. Ring n contains (n*A)² <= r² < ((n+1)*A)².
   The inner boundary is inclusive and outer boundary exclusive. Use exact integer
   squared thresholds and searchsorted(side=right), no rounding of observations.
4. Compute cutoff and linear taper from the same dimensionless radius sqrt(r²).
   Physical lengths remain h times index lengths. Use physical radius only in
   report labels, not annular classification. Source pixel deposition is unchanged.
5. Reuse v1 physical functions and copy its rebin implementation with only the
   specified index-geometry substitution. A new runner calls unchanged execute
   through a process-local callable substitution, never modifying v1 files.
   It binds v2 code/config/tests in a new receipt before execute. Legacy and current
   v1 freeze bindings still verify. Every fresh run path must be unused.

Independent controls frozen before implementation:
- Nine existing registered-source tests remain unchanged and must pass.
- Exact and one-ULP-near squared annulus boundaries at several integer widths;
  expected IDs from Decimal/integer comparisons, not the v2 implementation.
- Integer-grid classification compared cell by cell to math.isqrt(r²)//A.
- Multiple distance scales including 12.355/13.987, 15.619/13.987, .1, .7, pi,
  and 1000; annular IDs and taper arrays must be identical in index space.
- Retain a target-free demonstration that v1 physical indexing changes exact
  boundary memberships under these rescalings.
- Manufactured signed images with holes and negatives, known independent pixel
  coordinates/areas and annular reference; verify signed, zero and annular masses
  scale as D² at 1e-10. Independent replay of conserved sums at 1e-12. Source
  fixtures avoid coverage-threshold equality to isolate annular classification.
- Reject noninteger/negative index metadata and inconsistent physical lengths.
- Before actual corrected reconstruction: repeat original nine tests, all v2
  tests and actual-header WCS/area/translation checks; retain all failures.
- After construction: all 12 corrected packets, 72 mass rows and actual measured,
  zero/annular D² scaling must pass unchanged tolerances; preserve before/after
  values and pixel-quadrature flags. Failure stops response admission, never retunes.

No new photometric aperture, source conversion, geometry, support or benchmark
threshold is selected. V1 packets/config/code snapshots are immutable. One CPU
numerical thread, no GPU, no new downloads. Only the same 12-case NGC3198 family
is rerun after all controls pass. There is no gravity or velocity operation.
