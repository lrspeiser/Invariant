# Stellar source geometry requires correction or verification

**The existing spherical cluster scores cannot yet support physical rankings or exclusions.** Their stellar input mapping has not been established: the source paper describes projected cumulative mass, while our calculation uses it as spherical enclosed mass.

The [X-COP release page](https://dominiqueeckert.wixsite.com/xcop/data) attributes the stellar profiles to Ghizzardi et al. The paper's [Section 4.1](https://arxiv.org/abs/2007.01084) describes projected profiles, with a model-dependent 0.75 conversion at R500. This is not a universal conversion at every radius. It also describes statistical and background-variance uncertainties and excludes intracluster light from the stellar estimate.

We checked all five already accessed development stellar FITS files. Their hashes match the original access records. Our derived packet exactly copies radius and mass from the MSTAR_SMOOTHED extension. Its comments identify the published profile and total uncertainties; they do not establish a deprojection. The raw extension contains separate statistical and cosmic-variance columns. Both extensions have lower/upper uncertainty columns omitted from our derived packet. Their numerical conventions and correlations still need verification.

Consequences for current results:

- Preserve all prior scores and force failures as calculations under the recorded source assumption. They are not established predictions from a verified three-dimensional stellar source.
- Do not fix the geometry by multiplying every radius by 0.75, or fix derivative failures by selecting a wider smoother.
- The source-fidelity tolerance previously used checks fidelity to the retained curve, not correct geometry or observational uncertainty.
- Earlier gradient-information statements remain true for the reduced packet; the original files do contain additional uncertainty information that must be recovered.

Next implementation: retain source geometry and uncertainty columns explicitly; verify the uncertainty convention; fit or reconstruct a positive three-dimensional stellar distribution by forward-projecting it to the observed cumulative profile, with justified outer continuation and uncertainty treatment. Validate projection/deprojection on analytic sources before transferring the existing global gravity cards. This source work must not optimize against pressure residuals.

No pressure, temperature, lensing or reserved-cluster response files were accessed in this audit. Evidence and reproducible header/identity checks are saved under `stellar-projection-audit-001` in the research worktree.
