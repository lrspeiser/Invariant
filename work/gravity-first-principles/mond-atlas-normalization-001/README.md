# Local gravity calibration matters for a smoothed-density law

The benchmark passes 36 manufactured environment pairs and their inverse-square/flux checks, with maximum relative arithmetic error 2.22e-16. These are analytic controls, not galaxy or Solar-System measurements.

Our proposed equation contains a constant G_bare. Experiments instead determine a local effective strength. In the restricted limit where the response epsilon is locally constant and the apparatus does not appreciably change it:

`G_measured = G_bare / epsilon_lab`

For another constant environment:

`gravity / Newton_using_measured_G = epsilon_lab / epsilon_environment`.

This means a low response coefficient everywhere does not automatically produce extra gravity relative to local measurements: the same coefficient in both places cancels. For example epsilon_lab=epsilon_environment=0.2 gives exactly the measured Newtonian strength, not five times that strength. If epsilon_lab=1 and epsilon_environment=0.2 the ratio is five. The environment difference, gradients and geometry matter.

For our Gaussian averaging lengths of 250 and 500 parsecs, an isolated one-solar-mass object changes the dimensionless averaged density by at most 4.06e-7 and 5.08e-8, respectively. Its maximum change in epsilon is only 3.25e-7 and 4.06e-8. Consequently the Sun's high internal density by itself does not guarantee epsilon near one under this particular smoothed prescription. Surrounding Galactic matter may dominate the average and must be included. These numbers do not assert that the real laboratory environment has epsilon=0.2 or predict a Solar-System anomaly.

The earlier galaxy force ratios remain conditional calculations that use the same numerical G for Newton and the candidate equation. Their numerical comparison is valid under that convention. Converting them into observed enhancements requires a consistent calibration of the candidate's G_bare and its local environment. At fixed source/epsilon and consistently scaled self-field boundary conditions, the self-field is linear in G_bare; arbitrary imposed external fields cannot simply be rescaled in this way.

This audit concerns our added Gaussian-neighborhood model. It is not a falsification of all [refracted-gravity models](https://arxiv.org/abs/1603.04943), whose local-density prescription differs. It also does not supply a covariant completion, lensing prediction, conservation proof, or response-data fit.

The next requirement is an independently constrained local source/environment and a multiscale prediction using one calibration across laboratory, Solar-System and galaxy regimes. Choosing epsilon_lab to improve galaxy fits would not satisfy that requirement. No prior results were changed and no new observed velocities were accessed.
