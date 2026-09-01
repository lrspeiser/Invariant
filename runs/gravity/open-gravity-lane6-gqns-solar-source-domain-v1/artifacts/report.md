# Lane 6 GQNS Solar-System/source-domain falsifier

## Result

**DECISIVELY_EXCLUDED_AS_UNCHANGED_GLOBAL_SOLAR_SOURCE_LAW** for the unchanged Lane-6 global GQNS source functional. This is a response-independent analytic and published-bound preflight, not a precision DE440 or INPOP refit.

The frozen JPL Table-1 approximation was evaluated at 601 epochs from J2000 through 2050 for 8 explicit source domains. No ephemeris binary or observational response row was downloaded or opened, and no parameter was tuned.

## Decisive mechanism

For a point source, the GQNS effective enclosed-mass fraction is

`A_Q [1-(1+r/L) exp(-r/L)]`.

It is strictly increasing with `r/L` whenever `A_Q>0`. Consequently, one fitted inverse-square mass scale cannot absorb the same global GQNS field at two unequal radii. For the Sun plus eight planetary barycenters, the median values are `A_Q=0.6483` and `L=0.337966 au`. After removing the best common inverse-square scale from all frozen planet vectors, the maximum Neptune residual is 21632.2 times the conservative published outer-planet constant-acceleration bound.

## Source-domain failure

Exact spherical shutoff is not Solar recovery. Named boundaries through the inner planets, Jupiter, Saturn, all eight planets, a resolved Earth-Moon pair, and an asteroid-ring sensitivity give different `A_Q`, `L`, and forces. The package also retains the non-additivity between solving the Sun-plus-planets globally and solving Sun and planets as separate source components. An arbitrarily selected host-only Sun domain can suppress the effect, but that is a new source-localization law, not the original Lane-6 rule.

## Controls

- 64 arbitrary SO(3) rotations test both moment invariance and force covariance.
- Translation and co-located mass splitting are checked.
- Solar oblateness, Moon splitting, a 36-point main-belt mass ring, distant low-mass sources, and named source boundaries are retained.
- The normalized Helmholtz enclosed-mass and density implications are explicit.
- A common inverse-square nuisance and a more permissive per-planet inverse-square nuisance are both reported; neither is tuned to an observational response.

## Claim boundary

The result excludes the unchanged **global Solar-System application** at pre-fit margins far larger than the published bound. It does not substitute for a full planetary-ephemeris refit and does not prove that every possible new localization theory fails. Any localization repair needs explicit covariant dynamics, conservation, cluster decomposition, and a new version.
