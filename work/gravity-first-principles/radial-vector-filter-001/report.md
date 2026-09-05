# Radial vector filter: the directional response

For a spherical vector V=v(r) rhat, the Laplacian is (v''+2v'/r-2v/r²) rhat. The last term distinguishes it from the scalar operator. The inverse uses the angular l=1 Green function, with a regular origin and no growing homogeneous term at infinity.

In dimensionless radii q=r/L and t=s/L, the kernel is t² i1(min(q,t)) k1(max(q,t)). Here i1(z)=(z cosh(z)-sinh(z))/z² and k1(z)=exp(-z)(z+1)/z², explicitly without the alternative pi/2 normalization. Scaled functions avoid overflow at large radius.

We manufactured smooth Gaussian-gradient vector solutions and applied the inverse to their exact Helmholtz sources. These are mathematical test fields, not proposed matter densities. All 24 controls passed the predeclared 1e-9 absolute tolerance; worst discrepancy 8.88178e-16. Twenty kernel pairs also satisfy symmetry in the physical r² dr measure, with worst relative discrepancy 3.15725e-16.

As a negative control, the scalar inverse was applied to the same radial component. Its maximum absolute error was 23.1852, including nonzero origin values where a smooth radial vector must vanish. This demonstrates why componentwise scalar radial filtering would change the theory.

For the gravitational action, filter the scalar potential before taking its Hessian. The divergence of the resulting nonlinear tensor product is a radial vector and requires this l=1 inverse. The calculation has not yet verified the full isolated action variation, its nonlinear stability, or any galaxy, cluster, Solar System or lensing prediction.

Evidence: radial-vector-filter-001/result.json. SHA-256 04772fef046103402b1910508102fbaf8efad3a3e1f92b6bcc980320c5524773.
