# Undefined optional center extension retained before force evaluation

Run001 passed manufactured controls but stopped at the actual HI support check,
before any actual source force evaluation. R=.025kpc has exactly zero HI column
at all2048 azimuths, so its density-weighted force is undefined. Every requested
radius R=.05..6 is positive. Run001 bindings/controls are retained.

Run002 uses exactly the requested support R=.05..6 with .025spacing,239radii;
the .05spacing subset is120radii. No zero-emission central force is interpolated,
filled or counted as observed zero. This removes only the explicitly optional
center extension, not any requested radius or force failure. All source maps,
physical parameters, quadrature rules and numerical gates remain frozen. V2
runner binds this addendum before its controls and actual force calculation.
