# Fixed Gaussian measurement pushforward

Let `h` be an arbitrary finite-dimensional Gaussian random vector with mean `m` and covariance `C`. For a fixed linear readout `R` and independent Gaussian measurement noise `n` with mean `b` and covariance `N`, define the jointly commuting output `y=R h+n`.

For every real test vector `t`, independence gives

`E exp(i t^T y) = exp(i t^T(Rm+b) - t^T(R C R^T+N)t/2)`.

This characteristic function uniquely defines a Gaussian output with mean `Rm+b` and covariance `R C R^T+N`. Two models sharing these five objects are therefore equivalent only within this measurement class.

## Required assumptions

- all reported outputs commute and are represented by one joint classical random vector
- the Gaussian POVM or measurement kernel is fixed independently of the model label
- the readout is linear in the mediator coordinates
- measurement noise is Gaussian and independent of the mediator
- no unmodeled measurement back-action changes later outputs
- state preparation and detector conditioning are identical

## Excluded cases

- noncommuting sequential measurements
- adaptive or model-dependent POVMs
- unmodeled back-action
- connected cumulants above second order
- entanglement generation or number-resolved exchange

This is a general finite-dimensional pushforward proof and an audit gate. It is not a novelty claim and does not establish that gravity is classical or quantum.
