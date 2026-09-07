# Conditional local gas contribution to the response coefficient

SOURCE_BLOCKED for complete independent laboratory calibration. DATA_AND_PAPER_ADMITTED only for convolution of the published plane-parallel gas model, not measured 3D local truth. No observed galaxy motion or local dynamical-mass estimate enters the calculation.

Source: McKee, Parravano & Hollenbach2015, arXiv1509.05334v1, Table2. Transcribed (hydrogen nuclei density cm^-3, scale pc): H2(.15,105 Gaussian exp[-(z/h)^2]), CNM(.80,127 Gaussian), WNM1(.13,318 Gaussian), WNM2(.077,403 exponential exp[-abs(z)/h]), HII(.0154,1590 exponential). Apply1.4*m_H to nuclei, without an extra factor2 for H2. HII analytic profile excludes Gum Nebula. Table rows are model summaries of emission/absorption/dispersion observations, not independent 3D voxels. Planar translation invariance is assumed; no local bubble, Sun height or radial structure is claimed reconstructed.

Freeze ell250/500pc and evaluate at z=0. Analytic Gaussian convolution: nbar=n0*h/sqrt(h*h+2*ell*ell). Exponential: nbar=n0*erfcx(ell/(sqrt(2)*h)). Independently integrate each product against a normalized Gaussian with relative agreement1e-10. Check positivity, nbar<=n0, units, monotonic decrease with smoothing, unsmoothed limit and large-ell limit Sigma/(sqrt(2*pi)*ell). Test ell=1e-5 and1e7pc for limits (relative1e-7). Verify integrated columns, retaining rounding differences againstTable2 rather than forcing them away.

Response law is our prior epsilon=.2+.8*u/(1+u), u=rho_bar/(.01 Msun/pc^3). Its gas-only value is a lower contribution bound only under this adopted gas model, nonnegative omitted matter and the local constant-medium approximation. It is not a confidence bound on actual epsilon_lab, not a complete G calibration and not a prediction of Solar-System anomalies.

The full published stellar table is not used as an independent calibration: sectionIV.1 describes gravity-equation-derived relative scale heights among its dependencies. Acquisition will bind exact source HTML bytes privately; publish URL/hash and numeric rows only. Raw response scores0. No fits, exclusions or adaptive formula modifications.
