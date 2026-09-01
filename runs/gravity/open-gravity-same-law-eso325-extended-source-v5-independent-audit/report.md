# Lane 7 V5 independent strict audit

**Status: BLOCK**

This was a read-only, source-free audit. It opened no HST or MUSE scientific payload and no SLACS row or response value. The V5 config, module, test, receipt, and six receipt artifacts match their frozen hashes; the V5 receipt self-hash also verifies. Eighteen explicitly source-free V5 tests passed, and the synthetic gate reproduced its stored numbers exactly.

That reproducibility does not validate the claimed target-free pass:

1. **Holdout leakage:** `g` is fit on indices 0:30 before the holdout is defined. Six of those fitted points—4, 9, 14, 19, 24, and 29—are later scored as held out. The stored held-out LPD is therefore contaminated.
2. **Amplitude-blind convergence:** every run's matter, lensing, and image channel is divided by its own RMS before pad/cell comparisons. A synthetic 25% amplitude change becomes a normalized error of `1.0145e-16` instead of `0.25`. The actual unnormalized channel errors were also calculated (padding 0.00114–0.00433; resolution 0.04150–0.05271), but they are not the gate.
3. **Declared Helmholtz law not solved:** the implementation forcibly sets the Yukawa `k=0` kernel to zero. For the declared `(nabla^2-mu^2)Y=4*pi*G*rho`, the required zero-mode kernel in the frozen synthetic case is `-0.000486422283572764`. The independent integrated-equation probe has relative residual `1.0`.
4. **Pseudo-NFW is not executable:** the module has no pseudo-NFW function or runtime identifier. Tests only confirm that descriptive strings exist. The claimed primary density comparator has not been executed.
5. **Scientific predictive likelihood is not executable:** the only runtime LPD function evaluates one diagonal-Gaussian prediction. There is no `logsumexp`, 4096-draw mixture, posterior sampler, full covariance, or conditional HST/MUSE predictive implementation required by the frozen contract.
6. **Mutation closure is incomplete:** `validate_config` accepted in-memory replacement of the access ledger with an unrelated zero key, repointing the SLACS sample-manifest binding, repointing V4 preservation, and changing declared output paths. The already-written no-clobber receipt helps detect later drift when fully validated, but the claimed config-level fail-closed binding is incomplete.

The package remains correctly explicit that no ESO score or SLACS result exists and that four external source dependencies are missing. No empirical or modified-gravity claim is authorized. A repaired append-only version must split before fitting, score raw dimensional convergence per channel, implement a boundary-consistent Helmholtz solver, execute the pseudo-NFW comparator and full posterior-predictive likelihood, and exact-bind the complete ledger/seal/path schema.
