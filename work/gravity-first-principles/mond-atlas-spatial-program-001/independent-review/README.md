# Independent spatial-source review

Full readback of every array in all eight unique source packets completed, with expected SHA256 hashes checked. Major X is array axis zero and deprojected minor Y is axis one in the original registered-source builder; the refinement operator and spatial quadrature preserve this convention.

The finite NFW-shaped secondary kernel has the correct potential, derivative and exterior inverse-square limit. Its potential and force are continuous at the cutoff. Independent radial quadrature checks the enclosed kernel weight. A coincident point-source force has an undefined cusp direction; the implementation assigns zero there, but the evaluated targets do not coincide with these cell-center/vertical quadrature sources.

Independent integration of piecewise-linear tent bases reproduces source masses at all three planar spacings without mass renormalization. Independent CPU fields use separately integrated cell masses, SciPy Laguerre nodes and an enclosed-plus-exterior-shell potential expression. Every unique component has two coarse checks; the first stellar component also has one fine check. Maximum relative acceleration discrepancy from GPU output is **2.77e-15**.

All 16 case/component refinement comparisons pass the fixed numerical gates. Largest fine-versus-middle force RMS difference is **0.0193%**, and largest single-point difference is **0.0826%**. This is strong numerical agreement for the paired quadratures; it does not isolate planar from vertical error or validate the source's physical reconstruction.

For total fields, azimuthal RMS departures from the mean cylindrical field span **1.00–16.39%** across the declared R,z groups; tangential RMS contribution reaches **13.96%** of total force RMS. These quantify source-arrangement sensitivity of the specified kernel with eta=1, not observed noncircular motion or a missing-gravity fit.

Changing stellar assumed height from 0.1 to 0.4 kpc changes the total field by **6.99–7.03% RMS**, with individual-point changes up to **15.27%**. The planar inverse reconstruction also changes, so this is explicitly not a pure thickness experiment. Refining the conditional planar source representation from f1 to f4 changes total fields by **0.476–0.490% RMS**, with maxima **1.16–1.28%**. These effects exceed the paired numerical quadrature differences but remain conditional source-model differences.

Initial review incorrectly required every archival array to be finite. Unobserved source_mean cells contain NaNs with both fitting and evaluation weights zero; this is expected missing-data representation. The initial failure is retained. Corrected readback records all such counts, checks zero exclusion weights and still requires every actual source/operator input finite. No parent source or field result was changed.

Artifacts: source-readback.json, independent-masses.json, cpu-subset.json, kernel-controls.json, receipt.json and arrangement-summary.json. No observed velocity/lensing targets were opened. Source uncertainty, CO reconstruction floor, missing beam matching and boundary assumptions still block observational source admission.
