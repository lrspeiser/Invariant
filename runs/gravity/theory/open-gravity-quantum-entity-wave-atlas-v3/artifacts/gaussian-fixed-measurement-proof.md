# Fixed finite-dimensional Gaussian pushforward

Let `h` be an arbitrary finite-dimensional Gaussian random vector with mean `m` and
covariance `C`.  For a fixed linear readout `R` and an independent Gaussian
measurement-noise vector `n` with mean `b` and covariance `N`, define the
jointly commuting output record `y = R h + n`.

For every real test vector `t`, independence gives the characteristic function

`E exp(i t^T y) = exp(i t^T(Rm+b) - t^T(R C R^T+N)t/2)`.

The characteristic function uniquely fixes the pushforward distribution:
`y` is Gaussian with mean `Rm+b` and covariance `R C R^T+N`.  Thus two models
sharing `m`, `C`, `R`, `b`, and `N` are equivalent only for this specified
measurement class.

## Required assumptions

- commuting joint output
- fixed model-independent Gaussian POVM or kernel
- independent Gaussian measurement noise
- linear readout
- no unmodeled back-action
- same state preparation and conditioning

## Excluded cases

- adaptive POVMs
- noncommuting sequential records
- model-dependent back-action
- higher connected cumulants
- entanglement generation
- number-resolved exchange

This is a standard finite-dimensional probability result and an audit gate,
not a novelty claim or evidence that gravity is classical or quantum.  Q04's
stored AST evaluates the mean and covariance formulas directly; the proof does
not extend Q04 beyond its declared reduced fixed-measurement diagnostic.
