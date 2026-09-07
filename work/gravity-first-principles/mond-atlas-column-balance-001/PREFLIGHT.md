# Conditional steady-balance profiles on the resolved aperture-radius range

Freeze before calculating this product. Use the actual f4 HI source and the
run003 column-force table on0.75<=R<=2.5kpc only, where the separate force and
radial-refinement checks pass. This does not establish emission support outside
that interval, telescope-aperture prediction or an observed gravity score.

Use both stellar-height source alternatives and all separate stellar,atomicHe,
CO source forces. Models: Newton and Newton plus fixed eta1,L4kpc,b.05kpc log.
Use pressure-normalization speeds5/10/15km/s, not fitted spectral line widths.
Consistent revised pressure closure: Pi=s_reference^2 Sigma_smoothed,
dPi/dR=s_reference^2 dSigma_smoothed/dR, denominator=raw HI column mean from
the same angular quadrature used in the force table. Gaussian radial smoothing
standard deviation.25kpc is the source-only packet's declared regularization.
Thus Pi/Sigma_raw varies radially; do not call s_reference a constant local
dispersion. No observed response determines smoothing or branch choice.

Compute vphi^2=R*gbar+R*Pi'/Sigma_raw using the existing SurfaceColumn balance.
Keep signed values; negative values mean no steady circular solution. Never
clip a negative value into zero speed or exclude it from a claimed valid model.
Record total and component inward forces, pressure term and implied speeds.

One-at-a-time material factors: baseline stellar M/L.6, lower.4, upper.8;
HI.8/1.2; CO.5/2. Changing HI mass scales both pressure estimate and its raw
denominator as well as HI gravity, so its pressure acceleration stays unchanged.
No new stellar/gas reconstruction or 30arcsec smoothing sensitivity is claimed.
Use full gravity source support even though this balance table evaluates only
the numerically resolved radius interval. No spherical substitution.

Bindings and source gates must precede execution. Preserve force-table and
pressure-source differences; reject missing, inconsistent, nonfinite or zero
HI denominators. This is a conditional source-based physical model calculation,
not a determination of observed orbital speeds or a general nonaxisymmetric
fluid solution. Primary source references and independent force/pressure
benchmarks are inherited through bound source/column-force/pressure packages.
