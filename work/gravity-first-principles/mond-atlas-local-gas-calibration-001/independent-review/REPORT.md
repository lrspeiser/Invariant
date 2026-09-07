# Independent gas-profile review

PASS for the restricted convolution calculation. No implementation error was found. The five density profiles match the bound McKee, Parravano and Hollenbach (2015), Table 2 HTML. The column densities and table footnotes support using hydrogen nuclei, including for H2, and multiplying by 1.4 for helium and heavier elements. Adding another factor of two for H2 would be incorrect. The proton-mass approximation is explicitly declared; it neglects the small hydrogen electron/binding correction.

For a plane-parallel source and a normalized isotropic Gaussian of standard deviation ell, the horizontal kernel integrates to one. The remaining one-dimensional Gaussian product gives n0*h/sqrt(h^2+2*ell^2). Completing the square in the exponential-profile integral gives n0*exp(ell^2/(2*h^2))*erfc(ell/(sqrt(2)*h)), evaluated stably by erfcx. Both formulas are correct for the table's scale-height conventions. In particular, its Gaussian is exp[-(z/h)^2], not exp[-z^2/(2*h^2)].

An independent 512-point Gauss-Legendre integration in physical z coordinates, over both sides of the plane through |z|=12ell, reproduced all ten smoothed component densities to maximum relative error 2.06e-13. The omitted normalized Gaussian tail is below 4e-33. A separate cm-to-pc volume conversion agrees with the script's SI calculation. The resulting totals are:

| ell | Gas-model density (solar masses/pc^3) | Gas-only response coefficient |
|---:|---:|---:|
| 250 pc | 0.01604283664 | 0.69281380090 |
| 500 pc | 0.009157002476 | 0.58239813301 |

The column comparison correctly retains rounding differences. The HII profile deliberately excludes the Gum Nebula while the table's integrated columns include it; the footnote supports that distinction. The parent code's positivity, smoothing monotonicity and small/large smoothing limits are consistent with the analytic expressions.

These numbers are not actual epsilon_lab values or an independently established calibration of laboratory G. They assume the paper's horizontally uniform mean gas profiles, evaluation at the Galactic midplane, the specified smoothing law and the adopted density-response parameters. They do not reconstruct the local bubble, the Sun's height, radial structure, stellar mass or source-dependent local departures. The published gas conversion and opacity assumptions also remain. The word 'lower' applies only to this fixed model with nonnegative omitted matter and the monotonic response law; it is not an empirical lower confidence bound on the true local environment.

Even a complete smoothed density would still require justification of the approximately constant coefficient over the laboratory experiment and of the restricted measured-G relation. This calculation neither predicts a Solar-System anomaly nor establishes a causal or relativistic gravity theory. No target-motion values were used. It was reviewed separately and was not appended to the already-frozen manufactured sensitivity sweep.

Verified the bound paper bytes/hash and both frozen code/preflight hashes. The initial review's text-matching check rejected the equivalent table spelling '0.80' versus Python '0.8'; the review parser was corrected to compare numeric values before the successful replay. No scientific calculation or parent file changed.
