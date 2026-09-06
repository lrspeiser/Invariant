# Before-east diagnostic subdivision

Freeze before reading new arrays: for selected model report q separately for each spatial band; split each non-DC band at median kx²+ky² of its modes (lower<=median, upper>median); and report channel-eigenvalue quartiles within each band. Eigenvalues ordered ascending; np.array_split42indices into4 groups. Report every nonempty group with counts. These diagnostics do not select models or change thresholds. Aggregate q passing cannot establish cross-mode independence, source-region calibration or causality of previous failures.
