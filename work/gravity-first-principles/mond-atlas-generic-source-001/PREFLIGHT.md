# Registered tracer source preflight

Disposition before implementation: SOURCE_BLOCKED. Source-only conditional flat-disk
maps are permitted; no observational response score or gravity operator is admitted.
The exact data, papers, geometry alternatives, exclusions and gates are in the config.
A second source pilot NGC2976 is the bounded goal of this package.

Pixel coordinates use zero-based x/y. Plain TAN or SIN WCS supplies unit sky vector v.
For the stellar correction, P1 tangent CD times its measured pixel offset supplies
angular east/north offset a. The exact composed direction is normalize(M v), where
M = I + a c1^T and c1 is the P1 tangent-center unit vector (a dot c1 = 0).
This is equivalent to P5 pixel -> sky -> P1 pixel -> plus shift -> sky; its sign is
independently checked against Astropy core-WCS composition. SIP is explicitly omitted
under the prior calibrated plain-TAN choice, not implicitly applied by a library.

A sky vector maps to a fixed-distance galaxy tangent plane, with position angle from
north through east. Major/minor coordinates then stretch minor by 1/cos(i), conditional
on a flat oblate source. Tangent-plane area comes from the analytic spherical WCS
Jacobian times D^2/(v dot galaxy_center)^3. Stellar correction multiplies the spherical
Jacobian by abs(det(M))/norm(M v)^3. Projected area divided by cos(i) is disk area.
Surface intensity times conversion times projected area is the signed tracer integral.
The small-galaxy, parallel-ray cos(i) intensity deprojection is explicit and retained.

Subdividing each native source pixel evaluates midpoint Jacobians and deposits its
signed flux and covered area into fixed coarse disk cells. Summed integrals and outside
field losses must close, and independent finite-difference areas, Astropy positions,
distance/inclination/PA invariances, analytic uniform fields, refinement and boundaries
can falsify the implementation. Neither physical source uncertainty nor instrument noise
is inferred from these numerical tests. Source fit fields retain signed data before
nonnegative conditional projection. Missing coverage is never measured zero.

Independent source benchmarks run before actual source construction. The source maps
have prior development exposure. No target velocities select cases or gates; all cases
and failed convergence flags remain in the report. No new 3D density lift is asserted.
