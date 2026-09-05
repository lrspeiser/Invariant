# Isolated three-dimensional scalar filter

The radial Green function of S=(1-L² Laplacian)^(-1) contains an image subtraction:

    Sf(r) = integral_0^infinity [exp(-|r-s|/L)-exp(-(r+s)/L)] s f(s) ds / (2 L r).

This follows by setting u=r Sf, imposing u(0)=0 and excluding the exponentially growing solution at infinity. At the origin the limiting kernel is s exp(-s/L)/L². The underlying three-dimensional Green function is exp(-r/L)/(4 pi L² r).

For f=-GM/r the exact result is -GM(1-exp(-r/L))/r, with finite origin value -GM/L. Its radial cusp at the origin reflects the singular input; this is not a smooth-source gravity admission.

An independent smooth manufactured solution uses Sf=exp(-r²/(2 a²)), with f=(1-L² Laplacian)Sf. That input is signed and is an operator control, not a proposed mass distribution. Widths a/L=0.1, 1, 10 test narrow and broad structure. Quadrature over the full half-line tests radii r/L=0 through 100 without a finite outer box.

All 32 controls passed the predeclared 1e-9 absolute tolerance for normalized fields. Worst absolute discrepancy: 4.44089e-16. These are scalar inverse checks, not a verification of the full nonlinear gravitational action.

Implementation constraint: a radial vector has an additional -2/r² term in its radial Laplacian. Applying this scalar kernel separately to radial vector or tensor components would implement the wrong operator. For the action's filtered Hessian, first filter the scalar potential and then differentiate it; the outer vector response needs its appropriate angular sector or an equivalent Cartesian treatment.

Next: verify that outer vector filter and the isolated action variation before making spherical-source or astronomical predictions. No observations were scored and no candidate was admitted.

Evidence: radial-filter-002/result.json; SHA-256 8fa829a30ff067aa94ecde7bf74f2a88e175c00797503765a42d83de4a8bb0ab. The predecessor radial-filter-001 stopped during JSON serialization of a NumPy boolean; its registration and runner are preserved. The successor changes only that serialization and output directory.
