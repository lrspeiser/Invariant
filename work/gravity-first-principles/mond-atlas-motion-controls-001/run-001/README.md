# Executed motion-forward benchmark

**THEORY_BENCHMARK_ONLY.** Prescribed thin emitting rings; no observed galaxy
motion, source/covariance admission, gravity score or mass inference.

Run `run-001` completed 2026-09-06T17:49:07.357822+00:00 on CPU.
All 24 frozen numerical controls passed before synthetic response generation.

The table reports joint held-out channels AND pixels. q/N uses the supplied
independent Gaussian noise covariance. Truth error/N compares to the noiseless
independent quadrature truth. Recovery criteria were frozen before the run and
require all three held-out subsets to pass. No result is discarded.

| Injection | Circular q/N | Expanded q/N | Circular truth error/N | Expanded truth error/N | Predictive recovery | Parameter recovery |
|---|---:|---:|---:|---:|---|---|
| amplitude_zero | 0.9675 | 0.9684 | 0.0012 | 0.0015 | True | True |
| warp_only | 1.0241 | 0.9928 | 0.0775 | 0.0028 | True | True |
| radial_only | 1.0774 | 0.9702 | 0.1179 | 0.0013 | True | True |
| asymmetry_only | 1.6497 | 1.0225 | 0.5662 | 0.0010 | True | True |
| combined | 1.3585 | 0.9154 | 0.4601 | 0.0016 | True | True |
| face_on_radial_unidentifiable | 1.0202 | 1.0195 | 0.0001 | 0.0018 | True | False |

Full channel-only, pixel-only, joint, and training values, both optimizer starts,
parameter errors, bound contacts and sensitivity degeneracies are in summary.json.
One random noise draw per case is an illustration, not a coverage study.

## Physical and statistical limits

The coordinate convention, equations, primary references, flux accounting,
source restrictions and all thresholds are in ../PREFLIGHT.md and the frozen config.
Spectral channels use the existing Gaussian integration primitive; the beam uses
the existing zero-padded convolution. A beam-support halo preserves outside-field
in-scatter. The declared spatial response is a linear tent, not a top-hat detector.
Independent truth uses different radial/azimuthal quadrature, rotation matrices,
SciPy Gaussian CDF and direct convolution; it shares the declared physical model.

The expanded model is conditional on known center, emission radial profile,
total intrinsic flux, asymmetry phase, channel response and beam. The diagonal
covariance is known by construction and does not validate observed gas noise.
The noiseless face-on control is exactly insensitive to planar speed and radial
flow. An unresolved axisymmetric line profile also has an exact rotation/radial
amplitude degeneracy for the shared radial profile. Local Jacobian sensitivities
and parameter errors show remaining confusion even when pixel predictions pass.

Gaussian line width does not implement pressure support. No force balance,
continuity/time evolution, finite thickness, optical depth, self absorption,
vertical motions, mass-to-light conversion, gravity or lensing closure is solved.

## Reproduce

From the Invariant repository with the existing NumPy/SciPy/threadpoolctl environment:

```powershell
& 'C:/Users/henry/AppData/Local/Programs/Python/Python313/python.exe' scripts/run_mond_atlas_motion_controls.py
& 'C:/Users/henry/AppData/Local/Programs/Python/Python313/python.exe' -B -m unittest discover -s tests -p test_mond_atlas_motion_controls.py -v
```

Every execution creates a fresh run directory and retains synthetic arrays in
work/private/mond-atlas-motion-controls-001/. Existing receipts are never overwritten.
