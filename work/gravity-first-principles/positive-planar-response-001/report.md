# Sign-changing response persists with positive matter density

Used the exact auxiliary potential

    psi(x,z) = z + b*x²/2 - A*cos(k*x)/k²,
    A = c*b,  b>0,  0<c<1,

in units a0=ell=G=1. Its matter density is (b+A*cos(k*x))/(4*pi), positive everywhere. The full variational flux was evaluated for this source and its unperturbed counterpart. In planar symmetry the physical x-gradient is Jx plus a constant; comparing the same constant, or projecting the difference onto the sine mode, determines the physical response.

All 54 cases agree in sign with the preceding linear calculation: 18 longer-wavelength cases have positive transfer and 36 shorter-wavelength cases negative transfer. Doubling the phase samples from 1024 to 2048 changes the measured coefficient by at most 1.51e-10 under the recorded normalization. The test spans all three action shapes, three positive background curvatures and two density contrasts.

This removes the earlier concern that the effect required negative total matter density. It remains a planar, infinite-mass source, not a finite isolated cluster or galaxy. The result concerns the density-mode response rather than reversal of the entire background gravitational field.

## What it does and does not imply for stability

A negative static transfer is not itself proof of a dynamical instability or a ghost. If one additionally assumes standard pressureless-fluid dynamics in a frozen local background, the density perturbation obeys schematically

    delta_rho_ddot = 4*pi*G*rho_background*T(k)*delta_rho.

Negative T gives an oscillatory/restoring contribution in that approximation; positive T permits ordinary gravitational growth. This is an interpretation under additional local assumptions, not a stability proof for the inhomogeneous background or a derived relativistic matter sector.

Next checks must use finite positive sources, establish boundary sensitivity and determine the theory's dynamical and short-distance behavior. No observational score, physical exclusion or completed gravity law is claimed.

Evidence: `positive-planar-response-001`, with the frozen action code, registered source, all density minima, physical response coefficients and resolution comparisons.
