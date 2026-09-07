# Centroid cancellation diagnosis and full-design test

The original42 failures are primarily a property of the **signed continuum-subtracted moment**, not a large change in the predicted positive line profile. This diagnosis does not waive the old gate or admit observations.

For those42 comparisons, the signed-centroid discrepancy reaches3.57484km/s, while the independently calculated positive precontinuum centroid discrepancy is at most0.009499km/s. Original signed-profile relativeL1 remains at most0.002799. On all60 source-quadrature versions of the15 affected case/aperture combinations, the signed centroid lies outside the stored velocity window: approximately−270 to566km/s. Positive centroids lie within it, approximately−83.11 to83.45km/s. These signed values are ratios involving negative line-profile weights, not a physical mean gas velocity.

Define c(s)=sum(v*s)/sum(s). For two signed profiles a,b,

`c(a)-c(b) = sum((a-b)*(v-c(b))) / sum(a)`.

Continuum subtraction can make sum(a) small while leaving large positive and negative terms. The cancellation condition number sum(abs(s))/abs(sum(s)) is4.53–12.72 for failed rows. The exact difference identity agrees within3.95e-13km/s, and recomputed spectra agree with saved032 outputs within5.33e-15mJy/beam. This reproduces the failure mechanism; it does not show that continuum subtraction itself is incorrect.

The positive alternative is `c_positive=sum(v*p)/sum(p)`, where p is the predicted signal **before** continuum subtraction, after the same spectral response and stored-channel selection. This is not clipping negative pixels. Positive weights make it a convex average inside the channel velocity range. It describes only the captured window. Affected rows capture81.02–88.30% of valid-emitter line flux in the42stored channels; their63parent channels capture99.966–99.991%. A roughly12–19% missing window fraction prevents treating the stored centroid as the whole intrinsic line mean.

After the42-row diagnosis, a separate prospective preflight expanded the check to every192 originally frozen nuisance case, everyfour source quadrature and allfive comparison pairs. Across11520profiles and14400comparisons, the positive centroid criterion0.5km/s has zero failures; maximum difference0.0180661km/s. Signed relativeL1 stays below0.01 everywhere (maximum0.00482768). All42 original combined-gate failures remain present. This establishes a useful full-design numerical diagnostic without selecting only successful cases. CPU/GPU checks cover allthree response branches; all saved signed spectra replay within1.78e-14mJy/beam. Capture values up to1+1.1e-14 were retained explicitly as floating-point roundoff, within the declared1e-12 allowance.

Future protocols can separately require convergence of the positive captured-window centroid and the actual signed profile used for scoring, while reporting capture fractions. The current calculation does not retroactively replace the032 protocol, fit any gravity law, open observed spectra, solve unknown-emitter motion, or validate missing source alternatives. It remains conditional on the same source geometry, pressure closure and modeled valid-emitter subset. Unknown-motion emission is still bounded by the existing model and is not physically declared absent.

Implementation: independent identity/fullHanning/decimatedHanning matrices with nonnegative unit row sums; scalar Gaussian-bin quadrature control; CPU calculation of all original failures; float64CUDA allcase extension cross-checked against CPU. Initial run001 retained a dependency-path typo after manufactured tests but before source arrays. The exact failed script/error are preserved, and run002 changes only that operational path construction. No execution032 file changed; prior-integrity.json verifies every published entry.

See run002/retained-failure-comparisons.csv for the42 original failures and paired diagnostics, run002/all-four-cache-diagnostics.csv for all660 affected-case profiles, and all-cases001/{all-comparisons.csv,all-moments.csv,summary.json} for the complete design.
