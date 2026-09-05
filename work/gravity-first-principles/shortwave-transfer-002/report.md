# Short-wavelength response of the retained length action

Derived the static linear response around a constant auxiliary field, with a sinusoidal perturbation perpendicular to that field. For P=x+xK(x+h), where E=xK and h is the squared Hessian invariant, the perturbation transfer is

    delta Phi = T(k) delta psi,
    T(k) = 1 + E'(x) + ell² x K'(x) k².

Here k is wave number and ell is the action's length. Because the registered excess E is increasing and strictly concave, with E(0)=0, xK'(x)=E'(x)-E(x)/x is negative for x>0. Thus every nonzero ell eventually gives negative transverse transfer at sufficiently large k. This concerns the perturbation response, not reversal of the whole background force.

At background auxiliary acceleration equal to a0 (x=1), the transition wavelength is approximately 3.14, 3.88 and 4.28 times ell for shapes 0.5, 1 and 2 respectively. Wavelengths shorter than the threshold have transfer opposite in sign to the Newtonian potential perturbation. The zero-length case lacks this k² term.

Checked 27 shape/background combinations using the full implemented variational flux, with two wave numbers and two perturbation amplitudes each. An initial amplitude rule controlled the gradient but not the curvature at large k; its largest small-amplitude discrepancy was about 0.12%. That run is retained. Bounding both perturbations reduced the maximum discrepancy to 4.01e-12.

This is a formal static result. The sinusoidal density perturbation is signed, and the background is a constant external field. It is not yet a complete everywhere-nonnegative source example, a dynamical instability or ghost calculation, an observational exclusion, or evidence that the entire force becomes repulsive. The tested flux comparison checks the local linearization; it is not an independent full-domain Poisson solve.

The next theoretical checks are whether the sign-changing response persists in physically admissible positive-source backgrounds, what it implies for matter dynamics, and whether any proposed short-distance limit follows from the underlying theory. An arbitrary cutoff chosen after seeing failures would not establish a first-principles solution. No original candidate or negative result is removed.

Evidence: `shortwave-transfer-001` and `shortwave-transfer-002`, with frozen action-module copies, all parameters, analytical thresholds and direct flux comparisons. No observational score or physical exclusion is claimed.
