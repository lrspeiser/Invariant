# Actual-template aperture injection and fitting transfer

Freeze before opening this experiment's background arrays or generating trials.
Use the new actual-source callback after its required numerical source-spectrum
checks. Primary Newton, stellar height.1kpc, pressure-reference10km/s, positive
spin. Repeat all three native spectral-response branches. Inject nuisance truth
(systemic0km/s, intrinsic line width10km/s, global emission1). The signal is the
declared valid-emitter regional prediction plus a separately retained bound for
unknown edge emission; this does not assert a globally steady galaxy.

Use the original15 geometric apertures and their exact10training/5evaluation
partition. Freeze the existing western mean and central12x12 covariance. Eight
conditional Gaussian trials use protocol seeds9063001..9063008; draw independent
aperture spectra under that working covariance, without claiming actual patches
independent. Also use eight fixed cyclic assignments of fifteen of the27
previously exposed eastern background cores: trialt gets indices(t..t+14)mod27.
Each core contributes its central12x12 mean in native42channel order andmJy/beam.
No normalization by measured eastern residuals, no re-selection of cores or
noise recipe. These trials share real backgrounds and are not independent.

Fit the existing three nuisance parameters with all three frozen starts using
only ten training spectra. Freeze the five model predictions before forming
the corresponding evaluation residuals. Report every start, bound contact,
rank failure, parameter error and held residual discrepancy. There is no new
gravity parameter or source fit. Preserve all failures and do not calibrate a
noise correction from them. The real source-region cube remains unopened.

First export and verify the central-aperture background means independently
by144scalar pixel additions versus vectorized means (absolute1e-12). Validate
Gaussian sample construction against Cholesky arithmetic and preserve seeds.
The actual-template noiseless parameter/held-prediction recovery gate is1e-5
absolute per parameter and1e-6 relative profileL2, matching scorer controls.
No arbitrary small-sample q-pass interval or discovery significance is defined.
Gaussian and real-background trial results are descriptive transfer diagnostics.

Public scripts, receipts and compact parameters/scores; private noise and
predicted-spectrum arrays. No actual galaxy velocities, source-region spectra
or reserved galaxies read; background exposure history remains explicit.
