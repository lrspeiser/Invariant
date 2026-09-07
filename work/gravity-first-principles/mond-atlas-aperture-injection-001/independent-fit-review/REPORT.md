# Independent injection fit review

All 51 saved trials and all 153 start losses replayed successfully. Maximum absolute numerical discrepancy was 1.11e-15. The independent replay uses a dense covariance solve, separately from the fitter's Cholesky whitening. It verifies all saved prediction hashes, parameter and aperture bindings, minimum successful training-loss selection, declared rank threshold, parameter errors, unknown-emission bounds, held-out discrepancies and aggregate medians. Source predictions reuse the GPU callback already compared with an independent CPU spectral calculation; this is not a new independent source reconstruction.

The original 10 training and 5 evaluation apertures remain disjoint. Code inspection finds that only training spectra enter nuisance optimization and selection. Numerical held predictions are saved and hashed before evaluation. This ordering is verified from code; no independent tamper-proof timing ledger exists. Loading the mock background packet does not make it a secure held-data vault, but held spectra are not used by the fit callback.

The fitter verifies the actual-spectrum review's headline gate and binds that receipt, but does not recursively verify its nested input hashes. This review independently checked all five nested bindings after the run; all match. Thus no changed-input discrepancy was found, while the prospective enforcement gap should remain explicit for a future runner.

All three noiseless recoveries pass. The median held discrepancy is 0.98747 per channel for Gaussian backgrounds and 1.23337 for the previously exposed real backgrounds. These are descriptive transfer results, without a significance or acceptance threshold. The same real background cores recur between trials and response branches, so 51 trials are not 51 independent experiments.

The signals are conditional valid-emitter templates: global steady balance is false, and unknown edge emission is retained separately. This neither scores observed source-region spectra nor compares gravity laws. One of the three initial guesses equals the planted truth, so noiseless success is a consistency check rather than proof that arbitrary initial guesses will recover an unknown solution. No observed source-region spectra were read during this review.

Reproduce with `python work/gravity-first-principles/mond-atlas-aperture-injection-001/independent-fit-review/replay.py`. The accompanying receipt records checked file hashes and arithmetic results. No parent implementation or completed fit output was modified.
