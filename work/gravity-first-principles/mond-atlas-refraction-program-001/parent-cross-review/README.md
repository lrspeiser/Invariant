# Independent refraction cross-review

The initial and finer programs implement the declared elliptic equation consistently: div(epsilon grad Phi)=4 pi G rho, with acceleration -grad Phi. The linear operator assembled for CG is -div(epsilon grad), so its right-hand side is -4 pi G rho with Dirichlet contributions moved to the right. Positive harmonic face coefficients are symmetric and produce a positive discrete energy. G in kpc (km/s)^2/Msun and rho in Msun/kpc^3 give Phi in (km/s)^2 and acceleration in (km/s)^2/kpc.

Independent sparse assembly on anisotropic meshes, using epsilon=1+.2x+.1y and Phi=x^2+2y^2+3z^2, gives analytic RHS 12+2.8x+1.6y. Direct sparse solutions agree with the program's CG solutions within **6.70e-12 relative**. The operator is exactly symmetric in these tests. Analytic potential errors decrease from 5.14e-5 to 1.59e-5 on refinement, passing the independently checked convergence condition. The original manufactured RHS 6+1.6x is also correct for its simpler epsilon and potential.

Replaying saved conditional force samples preserves the distinction between a solved equation and a resolved field:

| Fine-to-finer comparison | Overall vector RMS difference | Height-group differences | Frozen field gate |
|---|---:|---:|---|
| Newton | 4.14% | 0.82–6.26% | Pass |
| Point-density epsilon refraction | 13.17% | 8.62–16.23% | **Fail** |

Therefore the point-density refraction field remains numerically unresolved under the declared tolerance, even though its physical PDE residual and flux balance pass. None of its detailed force differences should be promoted to observed predictive success.

The monopole Dirichlet boundary -GM/(epsilon_v r) is a conditional isolated exterior approximation; source flattening, multipoles and material outside the domain are not exactly represented. Boundary enlargement must be checked for each modified constitutive law, not assumed transferable automatically. Original source conversion and X-major/Y-minor orientation are preserved.

Mass bookkeeping sums cell averages at all grid nodes, but the Dirichlet-node layers do not enter the interior PDE right-hand side. Their mass fractions are **1.25e-5 at base, 6.24e-6 at fine and 8.44e-8 at enlarged box**. This small reporting distinction cannot explain a 13% field discrepancy; active interior RHS mass is reported separately in receipt.json. Source adapter is shared for this mass audit, while the manufactured operator is independently assembled.

For any upcoming smoothed-density epsilon law, the required distinction is epsilon=rho-smoothed response while RHS remains the original physical rho. Smoothing epsilon's input introduces a new physical response length; it is not simply a numerical convergence fix for the original point-density law. Gaussian boundary treatment and finite-box loss must be reported, and original failures retained. This review opens no observed velocities or lensing targets.
