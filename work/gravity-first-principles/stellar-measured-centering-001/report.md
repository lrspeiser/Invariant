# Published central-galaxy positions constrain the offset test

Compared the brightest-cluster-galaxy coordinates in [van der Burg et al. (2015), Table 1](https://arxiv.org/abs/1412.2137) with the centers in the existing stellar FITS headers. Angular separations were converted using the header redshifts and the reference flat cosmology H0=70 km/s/Mpc, Omega_m=0.3.

| Cluster | Separation | Projected offset | Offset / inherited R500 |
| --- | ---: | ---: | ---: |
| A1795 | 13.34 arcsec | 15.98 kpc | 0.01386 |
| A2142 | 17.11 arcsec | 28.72 kpc | 0.02017 |
| A2319 | 26.65 arcsec | 28.83 kpc | 0.02107 |
| A85 | 1.28 arcsec | 1.38 kpc | 0.00112 |
| ZW1215 | 2.19 arcsec | 3.18 kpc | 0.00234 |

These offsets are derived from catalog coordinates, not selected against gravity or source residuals. The A2319 value lies slightly outside the earlier hypothetical range ending at 0.02R500. The next source test should use these coordinate-derived positions, rather than broadening the offset range until a fit succeeds.

Positional errors, coordinate-definition differences, central light distributions and line-of-sight structure remain unresolved. The table does not establish a complete three-dimensional source or justify the previous spherical interpretation. No gravity law is scored or excluded.

Evidence: `stellar-measured-centering-001/result.json`, retaining the transcribed coordinates, FITS centers, redshifts, conversion cosmology and separations. The publication table also contains dynamical columns; none were used in this coordinate calculation or scored as new validation data.
