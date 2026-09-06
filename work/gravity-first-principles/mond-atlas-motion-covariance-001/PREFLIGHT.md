# Correlated-channel motion benchmark preflight

**THEORY_BENCHMARK_ONLY**, frozen before implementation and response generation.
This extends the published synthetic motion operator by read-only import. No
observational source, mask or covariance is admitted. The source geometry, center,
profile, flux, asymmetry phase and instrument remain supplied. Pressure support,
force balance, dynamics, vertical structure and self absorption remain missing.

Primary statistical references are the author-hosted published mathematical
reference Rasmussen & Williams (2006): [Appendix A, equations A.4-A.6 and A.18](https://gaussianprocess.org/gpml/chapters/RWA.pdf)
and [chapter 2, equation 2.30 and Algorithm 2.1](https://gaussianprocess.org/gpml/chapters/RW2.pdf).
The independent library check uses [SciPy 1.16.1 multivariate_normal](https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.stats.multivariate_normal.html).
The former gives the Gaussian density and conditioning identity; the latter
chapter supplies its Cholesky implementation. We use a parametric emitting-ring
mean, with no Gaussian-process prior on the physical signal. Aitken's historical
least-squares publication metadata was consulted, but its inaccessible full text
is not used as an equation/benchmark source.

For each pixel p, channel noise has covariance s_p^2 K with
K_cd = sigma_c sigma_d rho^|c-d|, rho=0.75. Distinct pixels and independent
realizations have exactly zero covariance. Stationary AR(1) innovations generate
the noise independently of the Cholesky inference implementation. All noise is
added after the declared instrument; this does not model native interferometric
noise or spatial beam-correlated measurement errors.

Let T denote training cells and H held-out cells. At any fixed mean parameter
theta, training uses the marginal C_TT, not the T block of the full precision.
For a residual r_T=y_T-m_T(theta), the same-realization conditional forecast is
m_H(theta)+A r_T, A=C_HT C_TT^-1, with S=C_HH-A C_TH. Cholesky solves replace
explicit inversion. Independent held-out pixels have A=0 and S=C_HH. Negative,
nonfinite, asymmetric or non-positive-definite covariance is rejected, with no
jitter or threshold repair. The deliberately diagonal method uses diag(C) for
both fitting and forecasting and is explicitly a misspecified approximation.

For independent fresh noise, C_H'T=0: the forecast is m_H(theta) and its noise
covariance is the marginal C_HH. Transferring A r_T to fresh data is saved only
as a negative-control diagnostic, scored with C_HH, never with S. Noiseless
mean errors independently measure signal recovery. An oracle using the true
mean shows how much same-realization improvement can come solely from noise
interpolation. All reported fitted-mean forecasts are plug-in diagnostics:
the noise covariance formulas are exact at fixed theta, while uncertainty in
the fitted nonlinear mean is not integrated out. No posterior coverage or
unqualified predictive calibration is claimed.

The JSON config binds the covariance, PCG64 seeds, three cases, four independent
noise realizations per case, two fresh replicates per realization, deterministic
measurement masks, two overlapping folds, four methods, fit budgets and all
comparison criteria before responses. The methods share draws for paired
comparison. Folds are averaged within each realization before summary; they do
not double the independent sample size. Missing measurement cells are excluded
and marginalized. No mask depends on flux or noise, so this cannot validate a
response-selected native-gas mask. The same-noise conditional forecast score
cannot satisfy the signal-recovery criterion by itself.

Pre-response controls include dense small-matrix likelihood/whitening,
independent joint-precision conditioning, factorization of the joint density,
positive definiteness, AR innovations covariance, diagonal and independent-pixel
limits, fresh-noise behavior, oracle Monte Carlo moments, response-independent
covariance and two-method held-out perturbation tests. The previous 25 mechanics
controls replay without editing their source or receipts. All failed gates stop
study response generation and remain recorded. Study recovery is an outcome,
not an implementation gate to be tuned. No actual observational reads or scores
are needed, and none will be performed by this package.
