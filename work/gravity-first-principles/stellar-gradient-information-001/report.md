# What the stellar inputs constrain

The retained development packet supplies stellar enclosed mass and radius, but no stellar derivative uncertainty or covariance. This is a statement about the packet used in our calculations, not a claim that the original observations contain no further information.

| Cluster | Input intervals | Flat intervals after monotonic correction | Corrected mass values |
| --- | ---: | ---: | ---: |
| A1795 | 605 | 253 | 35 |
| A2142 | 739 | 302 | 0 |
| A2319 | 403 | 66 | 0 |
| A85 | 495 | 265 | 81 |
| ZW1215 | 605 | 321 | 197 |

The inherited reconstruction linearly interpolates nondecreasing enclosed mass and then smooths its radial mass distribution. Flat segments and changes in slope affect the reconstructed density gradient. The largest inherited mass correction is about 1.83%, in ZW1215. Small mass corrections do not bound derivative errors.

## Explicit information-limit example

Consider dimensionless measurements of enclosed mass at radii 1 and 2, both matched by M(r)=r. Within an unmeasured interval of width w centered at 1.5, add

    delta M = epsilon*w*b(t)/16,
    b(t) = 256*t^4*(1-t)^4,
    t = (r - (1.5-w/2))/w,

and set the addition to zero outside that interval. The addition and its first three derivatives vanish at its boundaries. Since |b'| <= 16, choosing epsilon=0.5 guarantees M' >= 0.5: the enclosed mass remains increasing and its associated spherical density remains positive. Both measured masses remain exactly unchanged.

Yet the added second mass derivative at the interval midpoint is -2*epsilon/w. It becomes arbitrarily large in magnitude as w shrinks, while the maximum mass perturbation epsilon*w/16 becomes arbitrarily small. Since spherical density is M'/(4*pi*r²), its gradient depends on M'' and is not bounded by these mass measurements alone. The same construction fits inside a gap in a finite set of mass samples, provided the baseline mass slope there is positive.

This is an information-limit example, not a fitted cluster source or proof that any actual cluster reverses gravity. Physical minimum scales or independent smoothness constraints could restrict these perturbations. Their justification must come from source physics or additional information rather than a preference for passing gravity predictions.

The next source assessment must inspect the original stellar measurement construction, uncertainties and spatial resolution, then propagate justified reconstruction choices through the full force. The nine one-megaparsec candidates remain unresolved. No law is excluded or validated by this audit.

Saved evidence: `stellar-gradient-information-001` in the research worktree, including the source-packet SHA-256, counts and dimensionless witness values. Symbolic checks verified the compact joins, derivative factorization and midpoint second derivative.
