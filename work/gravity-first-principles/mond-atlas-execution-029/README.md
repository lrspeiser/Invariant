# Round 029: static response, calibration sensitivity and transfer

This round advances the memory mechanism and puts the density-response predictions on a more explicit local-calibration footing. Numerical, manufactured and observational-background evidence remain distinct. No new galaxy-motion or lensing fit is claimed.

## A conservative memory modification produces static attraction

The previous oscillator term was `U_old=omega^2*(q-h)^2/2`; at equilibrium q=h it supplied no extra static potential. Removing its cancelling term gives `U=omega^2*q^2/2-omega^2*q*h`. Eliminating the equilibrium oscillator yields `U_effective=-omega^2*h^2/2`. With the inherited Gaussian distance dependence this gives an inward force.

Sixteen two-body integrations pass the declared conservation, trajectory-refinement and circular-orbit checks. A separate radial-coordinate Radau integration reproduces all 3,208 fine samples within 2.06e-11, independently of the production Cartesian DOP853 implementation. Source splitting preserves the tested acceleration and energy after accounting for internal constant energies.

At the inherited fast setting, the added force reaches about 3.70% of the softened Newton force in the sampled static test. This is progress from the earlier zero-equilibrium-force model, but it remains a manufactured mechanical example. The Gaussian tail fades too quickly for an extended flat halo; increasing source mass does not suppress its relative contribution; an undamped oscillator does not automatically settle to equilibrium. No observed history fixes the oscillator states. Time itself has not been identified as an energy source.

## A logarithmic coupling extends the force profile

One subsequent, separately frozen kernel replaces the Gaussian target with `h=eta*sqrt(mi*mj)*sqrt(log(1+L/s))`, where `s=sqrt(r^2+b^2)`. Its eliminated potential is `-C*mi*mj*log(1+L/s)`, with `C=omega^2*eta^2/2`. The central softening makes the fixed-total-mass energy bound finite. The spherical effective density is positive; the unsoftened static potential has the Jaffe form, with finite total extra mass.

Unlike the Gaussian tail, its extra force has an intermediate 1/r regime when `b << r << L`, followed by a 1/r^2 outer tail. The inherited b/L=0.05 offers only a limited intermediate interval; this is not a demonstrated galaxy-scale flat rotation curve. The inherited slow/fast settings give asymptotic extra/source mass ratios of 0.045% and 4.5%, with no fit to data.

Another 16 integrations pass the mechanical gates, and independent radial replay reproduces 3,208 fine samples within 1.53e-11. The initial attempt stopped at a JSON serialization error before trajectories; it remains preserved, followed by a serialization-only correction and the completed run002. There were no equation or parameter changes between those attempts.

This repairs the Gaussian kernel's short tail in a precise mathematical way. It does not derive that kernel from microscopic physics, calibrate its amplitude or prove reflection. At equilibrium it is another static pair potential; static force data alone cannot distinguish that representation from an oscillator mechanism. Undetermined oscillator histories, absence of automatic equilibration, mass-scaling behavior and relativistic/lensing completion remain open.

## Galaxy predictions depend strongly on local calibration

At fixed source and response geometry, calibrating `G_bare=epsilon_lab*G_measured` rescales the self-generated force vectors. Six frozen manufactured local environments yield 4,608 point scenarios from the two conditional NGC2976 source models. At u_lab<=0.1 neither retains any sampled force-norm enhancement over Newton. At u_lab=1, enhancement remains at 179/384 positions for ell=0.25 kpc and 143/384 for ell=0.5 kpc; the median ratios are 0.934 and 0.818.

These are not measured local densities. They show why a numerical 1.36x or 1.56x galaxy enhancement under an uncalibrated convention is insufficient. Directions and signed inward projections are retained: two positions point outward in both Newton and the candidate, so a more inward difference there can mean less outward. Scalar normalization cannot repair force direction or source geometry.

A separate published gas-profile calculation gives gas-only coefficients 0.6928 and 0.5824 for the two averaging lengths, independently reproduced by quadrature. This was not added to the frozen sweep or treated as actual laboratory calibration. It assumes a horizontally uniform gas model and omits stellar matter and local structures. The compiled stellar scale heights have gravity-model dependencies, so the full table cannot simply be used as an independent local calibration.

## Descriptive noise flags are not significance tests

Under an idealized reference with 27 independent Gaussian cores and known correct parameters, 22.67 of 42 channels are expected inside the 20% descriptive range. The observed 21 is ordinary under that reference: count-tail probability 0.357, and no channel rejects after a two-sided Bonferroni correction. This retrospective check leaves every original flag unchanged. It does not validate actual core independence, fitted-parameter uncertainty or the full source likelihood.

## The frozen noise recipe transfers broadly to NGC3198

Using the full released emission footprint and a 120-arcsecond guard yields 20 western training cores and 18 eastern evaluation cores. All 72 native channels are retained. The recipe is fixed from NGC2976; only western noise parameters are estimated, with no model selection in the new galaxy.

The eastern joint q/N is 1.028 and all six aperture scores lie between 0.948 and 1.034. All 576 spatial modes meet the frozen descriptive range. One low-spatial-frequency channel-eigenvalue group remains outside it at 1.262. The individual-channel count is descriptive, not a significance test. This is useful cross-galaxy recipe transfer on development-exposed observations, not a new independent galaxy confirmation or source-region likelihood validation. The emission map itself comes from the same observation and cannot certify completely emission-free background.

## Evidence

- [Static susceptibility mechanism and retained limitations](../mond-atlas-static-susceptibility-001/README.md)
- [Independent radial trajectory replay](../mond-atlas-static-susceptibility-001/independent-review/receipt.json)
- [Longer-range logarithmic susceptibility](../mond-atlas-log-susceptibility-001/README.md)
- [Actual-source normalization sensitivity](../mond-atlas-local-calibration-001/REPORT.md)
- [Conditional local gas contribution](../mond-atlas-local-gas-calibration-001/README.md)
- [Finite-sample noise reference](../mond-atlas-noise-reference-001/README.md)
- [Second-galaxy noise transfer](../mond-atlas-noise-transfer-001/README.md)

The prior distributed spatial kernel, density/refraction, external-field and motion-coupling results remain in force with their original limits. Cluster, Solar-System, relativistic light propagation, source geometry and independently predicted motion remain outstanding. This goal is not complete.
