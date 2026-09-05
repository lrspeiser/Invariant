# Differentiable shell source prototype

Constructed a smooth shell by averaging positive Plummer mass distributions over a sphere of centers. With total mass M, shell radius s and smoothing size a>0, its Newtonian potential is

    Phi(r) = -2 GM / [sqrt((r-s)^2+a^2) + sqrt((r+s)^2+a^2)].

This is a positive, finite-mass source with a smooth potential. Its derivatives are generated from that single expression, rather than interpolated independently. It approaches a thin shell as a tends to zero; we do not use that singular limit for the derivative-sensitive gravity action.

All 42 controls passed across two shell radii, three relative smoothing sizes and seven probe radii. Potential derivatives through third order agree with 70-digit calculations to a worst relative error of 2.52e-10. The Poisson relation agrees with an independent angular average of positive Plummer density to a worst scaled residual of 1.68e-10. Both targets were 1e-8. The Poisson residual is scaled by the largest contributing term, not by a potentially tiny density alone.

These tests establish the sampled numerical behavior of a source primitive. They do not establish a physical smoothing width, stable deprojection, a cluster fit, or a new gravity law. The next step is to project smooth-shell mixtures into the retained stellar bounds and assess both source representation and smoothing sensitivity, while retaining measured centering constraints and unresolved cases.

Evidence: `smooth-shell-controls-001`, with the registered cases and targets, executable symbolic/high-precision checks, all numerical residuals and zero observational scores or physical exclusions. The ongoing galaxy angular-refinement job continues independently.
