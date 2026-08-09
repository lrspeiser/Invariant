# GR, Solar-System, and galaxy-exterior reference run

Golden GR checks: 5/5 passed.

| Check | Status | Key result |
|---|---|---|
| schwarzschild_vacuum | pass | 16 Ricci components |
| gr_ppn_recovery | pass | 1 |
| mercury_perihelion | pass | 42.98197549744704 |
| solar_limb_light_deflection | pass | 1.7512432813674266 |
| shapiro_delay_geometry_control | pass | 110.12540895862828 |

## Galaxy exterior control

Status: **expected_mismatch**.

Outside a finite baryonic mass, weak-field GR predicts v_c proportional to r^-1/2, not a flat rotation curve.

At angular-radius ratios 1:2:4, the baryons-only exterior prediction gives velocity ratios 1.000:0.707:0.500. Distance and absolute mass cancel.

This is a failure of the **GR + measured-baryons-only hypothesis** to produce a flat asymptotic curve. Unobserved source components are not an allowed rescue in this project.

## Still deferred

- candidate-specific background solution and static dictionary
- an independently audited direct-observation dataset manifest
- measured extended-source likelihood using only policy-permitted observables
