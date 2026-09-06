# Pressure-support admission before implementation

Disposition: **THEORY_BENCHMARK_ONLY**. This is a restricted gas-mechanics
benchmark, with no real source admitted and no observational scoring allowed.
The config is frozen and hashed before either new Python implementation exists.
All supplied force fields, density/pressure laws, independent controls, numeric
thresholds, study cases, seeds, fit bounds and evaluation radii are fixed there.

The implementation will distinguish local volume Euler balance from vertically
integrated balance. A surface-pressure gradient requires a density-weighted
radial force, unless extra assumptions justify a midplane substitution. A
separately manufactured flaring three-dimensional equilibrium will test this.
General anisotropic stellar Jeans dynamics is outside the admitted model.

Thermal pressure uses the mean gas-particle mass, whereas thermal broadening
uses the tracer mass. Only isotropic one-component turbulent variance is added
to the admitted scalar stress. Instrumental and unresolved line broadening do
not automatically provide supporting pressure. An explicit same-linewidth
counterexample preserves this degeneracy.

Rotation is derived from supplied force and pressure through Euler balance;
it is not freely prescribed independently of those inputs. Negative rotation
squared is an inadmissible steady circular equilibrium and is retained with its
signed values. Nonzero radial or vertical mean flow is rejected. Profiles have
a regular center and continue beyond the outer sampling window; no artificial
vacuum truncation, self-consistent Poisson solve or energy equation is implied.

Primary equation sources are Wang et al. (2010), arXiv:1004.5593v1, equations
(1), (2), (6), (9), Appendix A; and Iorio et al. (2017), arXiv:1611.03865,
doi:10.1093/mnras/stw3285, section 4.3 equations (6)-(7). Public equation text
was consulted before this freeze. No galaxy measurement tables, source images,
velocity files or previous observational configuration were opened for this
increment. These primary equations admit only theory controls, not real data.

Independent controls use exact harmonic and cored-log scalar potentials,
Gaussian density/pressure laws, finite differences, boundary/resolution tests,
and direct vertical quadrature of a manufactured flaring equilibrium. Numerical
admission must pass before study noise is generated. The harmonic example also
tests an exact force-pressure degeneracy: perfect speeds need not recover force.

The study stays in radial mean-speed space and does not add a cube path. Earlier
motion and covariance packages remain immutable. The new module will use no
observational source/velocity inputs. One CPU thread, no GPU. Raw synthetic
arrays stay in the owned private directory. All unsuccessful controls or fits
remain visible. No Git operations; stop after this bounded increment for parent
review. The broader gravity goal remains unresolved.
