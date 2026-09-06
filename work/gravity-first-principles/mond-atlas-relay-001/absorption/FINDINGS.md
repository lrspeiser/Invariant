# Absorption and pass-through: first results

The strongest useful pattern in this branch is that **arrangement changes transmission, but passive loss alone goes in the wrong direction for an extra attractive halo**. These are manufactured experiments, not measurements of galaxies. All seven checks passed before 560 relay cases were evaluated; no observed response data were read.

## Formulas and results

1. Direct attenuation: T=exp(-tau), tau=kappa*Sigma. Kappa has units of area per mass and Sigma is intervening mass per area. Assigning g=g_Newton*T always weakens the assumed attraction. In a uniform medium with dimensionless kappa*rho=1, g/g_Newton is 0.90484 at radius 0.1, 0.36788 at radius 1, and 0.04979 at radius 3. The circular-speed logarithmic slope is -(1+r)/2, so the speed falls even more steeply than a point-mass Newtonian orbit. This particular model cannot flatten that rotation curve.

2. Partial relay: one unit arrives, exp(-tau) passes through, and eta times the interacting part is re-emitted. A fraction f of that re-emission goes forward. The forward multiplier is exp(-tau)+eta*f*(1-exp(-tau)). For eta<=1 it never exceeds one in this unfocused model. At tau=1, eta=1 and f=0.5, ten successive forward relays retain only 2.23964% of the original forward amount. Fully forward, lossless re-emission preserves the original amount exactly. The identity of an individual carrier is irrelevant to this bookkeeping.

3. Clumpy screens: at fixed mean optical depth, covering a fraction c of the sky gives T_area=1-c+c*exp(-tau/c). With mean tau=1 and c=0.1, transmission is 0.90000454 instead of 0.36787944 for a uniform screen: 2.44647 times as much gets through, but still less than the unobscured amount. This is a useful geometry-dependent prediction. Moving a fixed column along the same ray changes nothing in the direct exponential law; changing coverage across rays does. These are distinct meanings of arrangement.

4. Active re-emission, eta>1, can increase the outgoing amount, but the extra amount is an input, not a free consequence of relaying. The eta=1.2 rows explicitly count that input. Packet accounting closes to 2.22e-16. These quantities are dimensionless packet-budget units: treating them as physical energy would require a specified carrier energy and coupling. A larger steady field does not automatically mean a larger outgoing energy flux.

5. A naive direction-dependent multiplier has a mechanics problem: g=T(x)*g_Newton need not be the gradient of a potential. A manufactured Gaussian transverse screen produces curl_z=0.05343027387 in dimensionless units. Independent analytic and finite-difference curls agree to 2.27e-10; halving the derivative step reduces error approximately fourfold. Thus this is a property of the proposed force ansatz, not a numerical glitch. Energy can be exchanged with an active screen or field, but an isolated, time-independent conservative force cannot simply ignore this term.

## What looks promising

- A relay can be described without tracking which carrier is original. Conservation and directional redistribution can be tested separately.
- Clumping predicts a clear arrangement effect even with the same total intervening material. A future empirical test must specify actual columns and sky coverage, rather than calling all low-density regions equivalent.
- A conservative joint field-and-matter model might allow redistribution to concentrate effects in some places. This benchmark does not include focusing from other rays or return-scattering; its forward bound is not a theorem against all local enhancement.

## What must change or be supplied

- An outward-moving absorbed particle normally transfers outward momentum. We assigned attraction by hand here. A theory must explain the sign of the force, not infer attraction from the phrase "sending back."
- Exponential attenuation of Newtonian gravity alone does not supply the missing inward acceleration. Additional redistributed influence, a different relation between the field and force, or additional dynamics is needed.
- Adding a spatial opacity multiplier creates curl in general. An action, a joint energy budget, or explicit dynamical degrees of freedom must account for it before applying the ansatz to stable orbits.
- Composition-sensitive coupling must confront composition tests. MICROSCOPE reported the titanium/platinum differential-acceleration parameter [-1.5 +/- 2.3(stat) +/- 1.5(syst)] x 10^-15. This is not a universal numerical limit on kappa; a concrete coupling and experiment geometry are needed to translate it. [Touboul et al. 2022](https://arxiv.org/abs/2209.15487)
- Eclipse anomalies are not established evidence for absorption: a primary analysis explains why the 1997 eclipse observations do not support that inference. We do not take a claimed anomaly as a calibration target. [Unnikrishnan, Mohapatra and Gillies 2001](https://doi.org/10.1103/PhysRevD.63.062002)

## Next falsifiable direction

Use a conservative interaction model first, then vary a shield's column and transverse coverage while explicitly including its own ordinary gravity. Compare uniform and clumpy arrangements with equal total mass, and add off-axis probes to distinguish redirected influence from simple attenuation. A candidate that gives an apparent enhancement only relative to an already attenuated baseline must not be reported as stronger than Newtonian vacuum gravity. Delay and repeated scattering require separate dynamical tests.

## Audit and reproduction

Run `python scripts/mond_atlas_absorption_experiment.py` only into a fresh result package; the current runner refuses to overwrite its completed receipt. Run the seven tests independently with `python -m unittest discover -s tests -p test_mond_atlas_absorption_experiment.py -v`. Tests cover an independent transport ODE, packet budgets, limits, column additivity and second-order quadrature, clumping inequalities, rotational covariance, and independently derived curl with convergence. At very large cumulative absorption the forward value underflows to zero in double precision; that is not a physical claim of exact zero. The present calculation is one-pass hemispheric bookkeeping, not a three-dimensional radiative-transfer or gravitational solver. No GPU was necessary.
