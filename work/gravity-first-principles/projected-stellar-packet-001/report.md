# Recovered stellar bounds and checked forward projection

Recovered RADIUS, MSTAR, MSTAR_LO and MSTAR_HI from the five previously accessed development files. Every original file hash matches, every mass matches the old packet, and every lower/upper pair brackets its mass. No pressure or reserved-cluster responses were read.

| Cluster | Median half-width of bracket / mass |
| --- | ---: |
| A1795 | 8.33% |
| A2142 | 8.67% |
| A2319 | 6.96% |
| A85 | 18.88% |
| ZW1215 | 12.13% |

These are descriptive bracket widths, not newly established confidence intervals. Their joint covariance and precise confidence convention remain unresolved. They must not be treated as independent measurements at hundreds of cumulative radii. The previous 2% source-fidelity tolerance was a numerical reconstruction criterion, not an observational uncertainty estimate.

The file extensions also use different radius conventions: the raw extension uses R/R500; the retained smoothed extension uses kpc. The recovered packet preserves units and provenance explicitly. Its geometry remains flagged as projected according to the associated publication, pending any file-specific transformation evidence.

For a spherical density, the forward projection sums complete shells inside projected radius R and the polar caps of shells outside R:

    M_projected(<R) = M_spherical(<R)
        + integral_R^infinity 4*pi*r²*rho(r)*[1-sqrt(1-R²/r²)] dr.

An independently integrated implementation passed 30 Plummer-source checks across mass, length and radius scales. With 512 outer quadrature nodes, the worst relative error against the analytic projected mass was 1.15e-9, below the declared 1e-7 threshold. This tests the forward operator on these sources, not arbitrary deprojection stability.

Next: explore positive three-dimensional source models constrained by forward projection into the recovered brackets, retaining source-model and outer-continuation alternatives. Bracket feasibility can be tested without inventing a likelihood, but will not establish statistical coverage or resolve substructure. Source selection must not use gravity residuals.

The five recovered profiles, all bounds, units, access hashes, 30 numerical controls and executable calculation are saved in `projected-stellar-packet-001` in the research worktree. No gravity candidate was scored or admitted.
