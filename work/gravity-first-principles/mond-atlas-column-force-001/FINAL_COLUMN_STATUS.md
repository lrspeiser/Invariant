# Final Execution031 column-force status

Use **run004/column-force-table.csv.gz**, not the earlier radial interpolation
tables, for the current conditional source-force adapter. Earlier runs and all
their failures remain unchanged. This final table uses the actual HI density
inside the azimuthal integral and analytically averages the two vertical
exponential distributions. Neither observed spectra nor fitted gravity strengths
were used.

The table has16,104 rows,11columns,2source cases,4component labels and2,013radii
per case/component. It spans .05..6.025kpc, with spacing.00025 in the central
.05.. .25 and outer5.95..6.025 intervals and.00625 between them. The angular
integration splits each circle at every original HI bilinear-grid crossing,
using Gauss8 per segment; Gauss4 provides an independent refinement comparison.
Thus narrow emitting arcs at the central hole and outer boundary are integrated,
not missed by a fixed angular grid. Every retained ring has positive HI column.

The repaired primary-emitter centroid support supplied by the independent source
audit is .0736569564..6.0121585431kpc. It lies entirely within this table. The
force table does not supply a value for the zero-HI R=.025 ring and must not be
extrapolated outside its support. The source audit's separate tiny outer-beam
flux bound is not counted as a force-convergence pass.

## Interface and units

Keys: `case`, `component`, `r_kpc`. Cases are `f4-stars-h0p1` and
`f4-stars-h0p4`. Components are `stellar_luminosity`, `atomic_helium`, `co21`,
and `total`. `sigma_hi_msun_pc2` is the raw angular-mean pure-HI column, with
the original atomic-plus-helium map divided by1.36. The same constant cancels
in the force weighting. The atomic force itself includes its original helium.

`newton_gbar`, `log_extra_gbar`, `newton_plus_log_gbar` use (km/s)^2/kpc,
positive inward; the third is exactly the sum of the first two. Signed component
values are retained. `newton_grid_delta` and `log_grid_delta` are fine minus
coarse; `newton_angular_delta` and `log_angular_delta` are Gauss8 minus Gauss4.
All numeric values are finite. To apply the protocol's one-at-a-time source
factors, multiply component forces and their error differences by those same
factors, then sum. Do not count components and `total` together.

The previous tables retain nonradial diagnostics. This radial table is the
restricted HI-weighted closure supplied to the separately defined pressure
adapter; it is not a full vector motion prediction or vertical-equilibrium
solution. The pressure adapter must retain its prescribed stress gradient,
raw-HI denominator and signed negative-rotation-squared failures.

## What passed and what remains flagged

The piecewise run completed in52.5seconds. Of208 component/total grid, angular
and piecewise radial checks,196 pass. All angular and radial-interpolation
checks pass, including center and outer extension. Both baseline total fields
(Newton and log extra) pass the unchanged1%RMS/3%point grid checks over the
entire .05..6.025 interval. Earlier box/vertical controls remain inherited.

The12failed checks are duplicated full/central and shared-gas cases:

| Component/model | Retained relative problem | Absolute difference within actual emitter interval |
|---|---|---:|
| Atomic gas, Newton | Grid point gate, force crosses near zero | <=2.2361 (km/s)^2/kpc |
| Atomic gas, log extra | Grid point gate near zero | <=0.0002839 (km/s)^2/kpc |
| CO, Newton | Grid point gate near zero | <=0.4145 (km/s)^2/kpc |

These failures are also present within the actual emitter-radius extent and
must not be reclassified as passing components. They do not imply a large
error in the complete predicted gravity: the physical prediction sums all
components, and the sum has its own independently replayed numerical checks.
No relative-error floor or relaxed threshold was introduced.

The final independent review explicitly reconstructs every protocol variant:
primary baseline; stellar M/L.4 and.8 relative to.6; HI factors.8 and1.2;
CO factors.5 and2; plus the separate stellar-height.4 baseline. Thus there
are7primary mass settings and1height setting. Each is checked for Newton and
Newton-plus-log. **Every complete-model grid and angular check over the actual
emitter interval passes**, with worst RMS0.3044% and worst point1.0767%.
All complete-model piecewise radial checks pass too. Across112full/emitter and
radial checks,111pass: the sole failure is M/L.4 Newton over the full table,
worst point3.0286%, occurring below the repaired emitter minimum. It remains
flagged; it is not a passing full-table model.

This distinction is deliberate: components near a zero crossing retain failed
relative gates, while the fixed complete models have directly checked errors
on the emitting interval. It does not grant general component-level admission,
validate arbitrary new mass factors, or establish spectral convergence. The
observed-response workflow still needs all source, pressure, beam, spectral
and adequacy checks specified by its protocol before scoring.

## Independent checks and artifact identity

Adaptive vertical integrals, Gaussian/Hankel forces, compact-source direct
integration and source mass controls preceded actual source force calculation.
A separate RTX5090 direct sum over the actual fine stellar cells reproduces
three log-column vectors within0.00543%. The final review checks input hashes,
linear component sums and complete-model gates directly from the saved table.
The inherited conditional 3D/source assumptions and selection history remain.

Compressed size1,268,737bytes; uncompressed CSV3,170,632bytes. The compressed
SHA256 is `1023fb8ad4d0ee80e54a05bca4c1fb4fcd4e928fc342abb1d7c733aa89dd5b40`.
`run004/artifact-receipt.json` records both compressed and CSV hashes, schema,
finite-value audit and units. This gzip contains only a derived conditional
force table, not raw observational spectra. `run004/independent-final-review.json`
contains the retained component flags and all112complete-model checks.
