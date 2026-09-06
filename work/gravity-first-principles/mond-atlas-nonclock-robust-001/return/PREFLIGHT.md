# Continuous return-formula development: frozen protocol

POST_HOC_DEVELOPMENT_ONLY, DATA_AND_PAPER_ADMITTED for radial empirical comparison
only. SPARC source paper https://arxiv.org/abs/1606.09251, original metadata and
rotation-component archive, registered 139 historical identities, original loader
and cuts, expected 102 galaxies. No reserved archive member bodies. Original
clock-relay source audit hashes are required before access. This is a response
to earlier model outcomes, not pristine confirmation or exhaustive theory search.

Same three seeds 9062301/2/3 and five whole-galaxy folds. Equal-galaxy mean squared
log10-speed residuals: every radius gets weight 1/(Ntrain galaxies * its galaxy
radius count). Fit global parameters only on training galaxies. Preserve every
start, status, failure, bound and held prediction; no selection by held loss.

Newtonian component uses the original signed gas V^2 and mf times stellar V^2;
mass proxy is 1e9*(0.5*mf*L3.6+1.33*MHI) Msun. Same known bulge/molecular/geometry
limitations. mf continuous [0.8,2]. A0=1.2e-10 m/s^2, rM=sqrt(GM/A0).

Families:
- fixed MOND, mf=1 and A0 factor=1;
- adjusted MOND with mf [0.8,2], A0 factor [0.1,3];
- finite_mix: extra=A GM/L^2*((1-q)/(1+x)^2+q*x/(1+x)^3), x=r/L;
  A [0,100], L=lambda*Rd^(1-t)*rM^t, lambda [0.1,30], t [0,1], q [0,1];
- truncated_point_kernel: extra=A GM/r^2*m(min(r/L,C)), same A,L,t,
  m(x)=log(1+x)-x/(1+x), C [3,100];
- finite_flat_bridge: extra=eta*sqrt(GM*A0)*r/((r+delta*Rd)^2*(1+r/(C*rM))),
  eta [0,10], delta [0.1,10], C [1,100]. This is a cored finite outer response.

All fits use scipy least_squares, 3 deterministic interior starts, normalized
parameters [0,1] mapped linearly to physical bounds, max_nfev500, ftol/xtol/gtol
1e-10. Start normalized vectors all0.15, all0.5, all0.85. Retain only successful
finite fits for selection; failed starts remain visible; fail closed if all fail.
Explicit zero-amplitude candidates optimized over mf alone are also retained for
each extra family. Boundary flag: normalized distance <=1e-4 from either edge.
No refits after inspecting held response, no subsequent grid or bound expansion.

Target-free gates before source loading: positive finite output, zero-amplitude
Newton baseline, independent NFW density quadrature, positive spherical mass
derivative, finite outer enclosed mass, potential gradient via independent
integration (relative1e-5), planted-formula fitting (training MSE<1e-8), and
training-only fit invariant to altered held responses (params abs1e-10).
Tests use manufactured radial sources. Full 3D convolution, energy exchange,
clock/storage physics, lensing, clusters and Solar System predictions not claimed.

Outputs: all optimizer starts, selected parameters and bounds, held radial
predictions, per-galaxy metrics, inner r/Rd<1 and outer r/Rd>=3 signed log-speed
bias with equal galaxy weights. More flexibility can improve a development fit
without identifying a causal mechanism; report complexity and all families.
