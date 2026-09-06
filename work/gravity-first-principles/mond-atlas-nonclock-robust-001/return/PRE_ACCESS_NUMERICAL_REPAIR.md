# Numerical check repair, before observational access

First test run: 3 tests passed and potential-gradient test failed for truncated
kernel at r=2, relative error 0.00015136 against fixed 1e-5 threshold. Independent
diagnostic isolated unsplit infinite-interval quadrature across the sharp cutoff;
other smooth families were below 1e-8. Repair explicitly integrates to the known
cutoff and adds the analytic Keplerian tail. Formula, tolerance and source/sample
protocol remain unchanged. No observed response had been opened in this branch.
