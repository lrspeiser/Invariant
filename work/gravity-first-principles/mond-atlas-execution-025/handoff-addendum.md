## Execution-025: larger spatial and dynamic program executed

This is the latest status and supersedes the older T2/T3 progress notes above.
The active research goal remains unfinished; no new observed full-cube, lensing,
cluster or Solar-System likelihood is admitted by this milestone.

- T2: NGC3198 recovery completed: 12 real-source packets and 72 mass alternatives,
  fixed integer-annulus distance scaling, independent replay, measured maps
  unchanged. Missing source coverage/conversions/depth remain.
- T3: Western-selected channel covariance now passes six aperture checks, but
  the new explicit spatial/channel joint-core covariance fails joint calibration
  (q/N=0.480). Aperture success is not whole-cube likelihood admission.
- T6 distributed response: NGC2976 stellar+HI+CO source integration on RTX5090,
  four source alternatives, three quadratures, 72 positions, 3456 field records.
  All component and total refinement gates pass; independently replayed. Inner
  angular structure is much larger than farther out, conditional on assumed
  geometry, and is not by itself evidence beyond nonspherical Newton gravity.
- T6 density/refraction: 18 actual-source PDE solves. Point-density law retains
  a 13.17% finer-grid failure; new fixed 0.25/0.5-kpc neighborhood-density laws
  pass declared field refinement (3.10/3.40%) and their own boundary checks.
  These are new nonlocal prescriptions, not an observational fit or proof of
  coherence. Source assumptions and observed-motion admission remain.
- Motion/history: 72 dimensionless integrations, including conservative memory,
  motion coupling, source-partition repair and a fixed strength sweep. The
  partition defect is fixed. Weak coupling preserves tested regular orbits but
  initially weakens inward acceleration; strong coupling disperses configurations.
  Long clustered and intermediate co-rotation convergence failures are retained.
  Equilibrated memory does not supply a static extra halo in this model.

The complete evidence, formula changes, failures and next dependencies are in
`work/gravity-first-principles/mond-atlas-execution-025/README.md` and its linked
branch reports. Next work should establish joint noise/mask transfer and source
uncertainty before using observed motion to rank the numerically eligible fields.
Physical reflection, causal transfer, independent source histories and broad
environment/cluster/Solar-System/lensing coverage remain open.
