# Actual gas-column predictions and the route to measured spectra

This milestone joins more of the real-source pipeline and produces a concrete
prediction ambiguity: the fixed logarithmic force and a predefined increase
in stellar mass produce similar orbital-speed changes. It also finds where
the assumed steady gas-pressure model fails. No observed spectra were fitted,
and no gravity mechanism has been observationally selected. The broader goal
remains active.

## What changed

| Product | Evidence now available | Limit retained |
|---|---|---|
| HI-weighted column force | Actual maps; separate stellar, HI+He and CO forces; analytic vertical averaging; final 16,104-row table covering 0.05–6.025 kpc | Individual gas-component relative gates still fail near zeros. Prescribed complete-model checks pass over repaired emitter support; arbitrary new factors are not admitted. |
| HI pressure and emission | Primary 134.0 million solar masses of HI, 43.61265 Jy km/s; 902,592 projected 3D emitter nodes; conserved cell masses and checked derivatives | Conditional reconstruction, uncertain depth/missingness, and retained coarse-node placement failure. Finer actual-source spectral convergence remains. |
| Native spectrum renderer | Fifteen 12×12 apertures, all 42 stored channels; correct 63-parent history, sampled elliptical restoring beam; all 180 final synthetic refinement cases pass | Actual-source quadrature, dirty/CLEAN and primary-beam approximations remain distinct from these synthetic controls. |
| Aperture covariance | Exact frozen-model marginal; independent dense contraction; 29 fixed-recipe western leave-one-core-out variants | Working weights, not a calibrated joint or source-region likelihood. |
| Three-nuisance scorer | Training-only fit, identifiability gate and frozen evaluation predictions; all manufactured controls pass | Actual source/instrument callback and observational access driver not yet executed. |
| Real-source balance | 84 fixed source/height/pressure/force profiles evaluated; full-emitter check also executed | Inner profiles are feasible, but every tested global steady profile fails in faint outer gas. |

## The useful physical patterns

Within 0.75–2.5 kpc, the fixed log response raises median predicted speed by
7.41–8.01 km/s. Increasing stellar mass-to-light ratio from 0.6 to 0.8 under
Newton raises it by 9.18–9.81 km/s. Their radial-change vectors align closely
(uncentered cosine 0.983–0.988); the two fixed total predictions differ by
2.05–2.11 km/s RMS. No parameter was fitted to mimic the other or to observations.
These source choices are sensitivity tests, not independently established
stellar-mass confidence bounds. Faster predicted rotation alone is unlikely
to identify the extra-force mechanism uniquely in this example.

Raising the pressure normalization from 5 to 15 km/s typically lowers median
speed by approximately 4 km/s, with an opposite sign at some radii. The gradient
matters. The explicit closure is Pi=s_reference² Sigma_smoothed, divided by raw
HI density in the Euler equation; the local pressure variance varies with radius.
The fitted tracer line width is a separate quantity.

All 84 inner profiles have nonnegative rotation squared. Extending to every
emitting radius gives zero globally feasible profiles: invalid regions start
as far inward as 5.288 kpc in the most affected case. Smoothing the pressure
estimate while raw density becomes tiny near the boundary gives excessive
outward pressure support. This tests the declared pressure/source closure, not
Newtonian physics generally or a measured instability. No invalid speed was
clipped or assigned a substitute value.

The affected outer gas contributes very little to the chosen apertures. The
conservative bound for any nonnegative normalized line is at most 5.525e-6
mJy/beam at emission multiplier 1, or 1.105e-5 at the permitted maximum 2.
The working noise standard deviation is 0.102–0.174 mJy/beam. A restricted
prediction can carry this explicit unknown-contribution envelope; it must not
be presented as a globally valid steady galaxy model. No response score has
yet been computed from that bound.

The noise model itself is dominated by broad spatial fluctuations: DC and
low-frequency modes contribute 96.83% of aperture covariance trace. Leaving
one western calibration core out changes the covariance by at most 3.45%
in relative matrix norm. Averaging 144 pixels therefore cannot be treated as
144 independent noise measurements.

## Repairs and checks retained

The force calculation now integrates narrow HI arcs by splitting angular
quadrature at the original bilinear grid crossings. Its final table covers
the repaired emitter radii without force extrapolation. The original zero-HI
center, coarse radial failures and near-zero component failures remain in their
receipts. Complete-model convergence was checked directly, not inferred from
components that failed. The final field is still a conditional static source
response, not a causal, relativistic or time-history mechanism validation.

Emitter mass centroids replace the earlier geometric cell-center approximation
in a new packet; the old packet is preserved. Fine placement passes, while
16 coarse centroids still land in subcell holes. Source mass is not altered.
The pressure closure now explicitly matches its stated derivative and raw-HI
denominator; old ambiguous labels are preserved and superseded in the addendum.

The native renderer uses sampled pixel centers; the earlier continuous-aperture
integration introduced an unestablished pixel top-hat and is preserved as a
separate, unusable native convention. It also retained 15 failed synthetic
profiles before a separately frozen higher quadrature passed. The actual
history has 63 parent channels, not the earlier protocol's 64. VELO-HEL with
VELREF=258 is independently interpreted as native radio velocity by WCSLIB;
no unannounced frame shift is applied.

The first balance-admission join required superseded coarse radial checks and
failed before balance calculation. Its replacement uses the actually refined
table. The initial full-emitter interpolation used linear interpolation; the
replacement uses the PCHIP convention verified by the force-table checks. Both
outputs and the independent unknown-spectrum bounds remain visible.

## Next executable work

1. Validate a finer actual HI emission quadrature through the sampled renderer,
   including force/pressure interpolation and the explicit invalid-edge envelope.
2. Freeze both rotational signs before training: photometric position angle
   alone does not determine the receding side. Preserve the planned source
   alternatives, including missing-region and common-resolution effects; these
   are not supplied by merely changing scalar masses or smoothing pressure.
3. Assemble the actual model callback and training/evaluation driver. Use the
   existing aperture covariance and 29 calibration variants, three spectral
   alternatives, material/pressure sensitivities and motion adequacy controls.
   Report conditional spectral discrepancies with every failed branch retained.
4. Carry resulting evidence back to the broader distributed, density/refraction,
   arrangement/external-field, current and time/memory hypotheses. Their earlier
   completed tests are unchanged in this increment; none becomes fully validated
   by the new static source calculation. Lensing, clusters and Solar-System
   constraints remain part of the full goal, not completed here.

No raw/private source arrays are published. The explicitly identified gzip is
a finite derived force table, verified by its schema and decompressed hash.
Individual packages retain equations, source papers, numerical failures, source
assumptions, private input hashes and reproducible scripts.
