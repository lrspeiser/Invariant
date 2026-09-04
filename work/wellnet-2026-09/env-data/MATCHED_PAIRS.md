# MATCHED_PAIRS.md — field/cluster matched galaxy pairs (Test 1)

Lane: `work/wellnet-2026-09/env-data/`.  Machine-readable version: `clean/matched_pairs.csv` (one row per pair, cluster member prefixed `cl_`, field member `fi_`).  Per-tier statistics: `clean/matched_pairs_summary.json`.

## What a pair is

Each pair is one MaNGA DR17 galaxy sitting inside the virial radius of a Tempel+2014 friends-of-friends group, matched one-to-one against one MaNGA galaxy that the same friends-of-friends run left as a singleton.  The assignment is a global optimum (`scipy.optimize.linear_sum_assignment`) over the normalised distance in the matching space, restricted to pairs that lie inside a hard tolerance box on **every** matching variable simultaneously.  No galaxy is used twice within a tier.

**Blind protection.**  The matching space, the quality gate and the field/cluster split contain no kinematic quantity of any kind.  This is asserted at run time in `code/build_matched_pairs.py` (`MATCH_FORBIDDEN`).  The cluster-versus-field kinematic contrast — the thing this sample exists to measure — has not been evaluated in this lane.

## Declared tolerances

Declared in code before any residual was inspected:

| variable | symbol | declared tolerance |
|---|---|---|
| log10 M_b (stellar) | `logMstar_nsa` | 0.1 dex |
| log10 R_d | `logRd` | 0.1 dex |
| log10 Sigma_b | `logSigma_b` | 0.15 dex |
| log10 g_bar(2.2 R_d) | `log_gbar_2p2Rd` | 0.1 dex |
| inclination | `incl_deg` | 10 deg |
| B/T | `pym_r_BT_SE` | 0.15 abs |
| redshift | `z` | 0.01 abs |

The first five are the five matching variables the brief asks for; the last three are nuisance controls (projection, bulge fraction, and physical resolution via redshift).

## Tiers, sample sizes, and the tolerances actually achieved

`B1_primary` is the primary sample.  The remaining tiers were declared after seeing that the primary cluster arm held only 48 galaxies — a sample-size observation, not a residual observation — and are reported separately.  They are **not** to be merged: they trade environmental contrast or morphological purity for sample size, and each answers a slightly different question.

### B1_primary — 23 pairs

- morphology gate: deep-learning late type: TType>0 and P_LTG>0.5, 25<i<75 deg
- environment gate: host sigma_v>=400 km/s, Ngal>=10, R_proj/R_vir<=1.0
- cluster arm 48 galaxies, field arm 1089 galaxies

| matching variable | declared tol | max abs(delta) | median abs(delta) | rms delta |
|---|---|---|---|---|
| log10 M_b (stellar) | 0.1 | **0.0992** | 0.0381 | 0.0451 |
| log10 R_d | 0.1 | **0.0943** | 0.0250 | 0.0403 |
| log10 Sigma_b | 0.15 | **0.0984** | 0.0637 | 0.0647 |
| log10 g_bar(2.2 R_d) | 0.1 | **0.0989** | 0.0609 | 0.0639 |
| inclination | 10 | **9.3346** | 4.1476 | 5.5907 |
| B/T | 0.15 | **0.1424** | 0.0929 | 0.0888 |
| redshift | 0.01 | **0.0088** | 0.0016 | 0.0041 |

Cluster-side environment of these pairs:

| quantity | min | median | max |
|---|---|---|---|
| host sigma_v [km/s] | 406.900 | 604.700 | 840.200 |
| R_proj / R_vir | 0.235 | 0.754 | 0.975 |
| |g_ext| / a_0 | 0.076 | 0.171 | 0.770 |
| angle(disk normal, host dir) [deg] | 16.419 | 64.391 | 88.773 |

### A1_gas_matched — 0 pairs

- morphology gate: deep-learning late type: TType>0 and P_LTG>0.5, 25<i<75 deg
- environment gate: host sigma_v>=400 km/s, Ngal>=10, R_proj/R_vir<=1.0
- cluster arm 48 galaxies, field arm 1089 galaxies

**Zero pairs.**  See the note on gas matching below.

### B2_disk_strict — 121 pairs

- morphology gate: disk-bearing (S0 or later): TType>-3 and B/T<=0.80, 20<i<80 deg
- environment gate: host sigma_v>=400 km/s, Ngal>=10, R_proj/R_vir<=1.0
- cluster arm 248 galaxies, field arm 1850 galaxies

| matching variable | declared tol | max abs(delta) | median abs(delta) | rms delta |
|---|---|---|---|---|
| log10 M_b (stellar) | 0.1 | **0.0992** | 0.0472 | 0.0539 |
| log10 R_d | 0.1 | **0.0943** | 0.0273 | 0.0394 |
| log10 Sigma_b | 0.15 | **0.1172** | 0.0505 | 0.0571 |
| log10 g_bar(2.2 R_d) | 0.1 | **0.0989** | 0.0460 | 0.0547 |
| inclination | 10 | **9.8418** | 4.1715 | 5.2985 |
| B/T | 0.15 | **0.1492** | 0.0675 | 0.0808 |
| redshift | 0.01 | **0.0099** | 0.0028 | 0.0044 |

Cluster-side environment of these pairs:

| quantity | min | median | max |
|---|---|---|---|
| host sigma_v [km/s] | 406.900 | 840.200 | 840.200 |
| R_proj / R_vir | 0.099 | 0.670 | 0.998 |
| |g_ext| / a_0 | 0.063 | 0.242 | 2.002 |
| angle(disk normal, host dir) [deg] | 12.127 | 60.763 | 89.964 |

### B3_late_wide — 80 pairs

- morphology gate: deep-learning late type: TType>0 and P_LTG>0.5, 25<i<75 deg
- environment gate: host sigma_v>=300 km/s, Ngal>=5, R_proj/R_vir<=1.5
- cluster arm 154 galaxies, field arm 1089 galaxies

| matching variable | declared tol | max abs(delta) | median abs(delta) | rms delta |
|---|---|---|---|---|
| log10 M_b (stellar) | 0.1 | **0.0992** | 0.0383 | 0.0474 |
| log10 R_d | 0.1 | **0.0943** | 0.0269 | 0.0407 |
| log10 Sigma_b | 0.15 | **0.1130** | 0.0379 | 0.0527 |
| log10 g_bar(2.2 R_d) | 0.1 | **0.0989** | 0.0419 | 0.0516 |
| inclination | 10 | **9.5559** | 4.0457 | 5.2722 |
| B/T | 0.15 | **0.1491** | 0.0606 | 0.0793 |
| redshift | 0.01 | **0.0098** | 0.0026 | 0.0044 |

Cluster-side environment of these pairs:

| quantity | min | median | max |
|---|---|---|---|
| host sigma_v [km/s] | 300.500 | 432.450 | 840.200 |
| R_proj / R_vir | 0.062 | 0.953 | 1.490 |
| |g_ext| / a_0 | 0.028 | 0.103 | 2.161 |
| angle(disk normal, host dir) [deg] | 16.419 | 58.706 | 89.880 |

### B4_disk_wide — 281 pairs

- morphology gate: disk-bearing (S0 or later): TType>-3 and B/T<=0.80, 20<i<80 deg
- environment gate: host sigma_v>=300 km/s, Ngal>=5, R_proj/R_vir<=1.5
- cluster arm 614 galaxies, field arm 1850 galaxies

| matching variable | declared tol | max abs(delta) | median abs(delta) | rms delta |
|---|---|---|---|---|
| log10 M_b (stellar) | 0.1 | **0.0992** | 0.0450 | 0.0543 |
| log10 R_d | 0.1 | **0.0960** | 0.0274 | 0.0405 |
| log10 Sigma_b | 0.15 | **0.1138** | 0.0415 | 0.0531 |
| log10 g_bar(2.2 R_d) | 0.1 | **0.0998** | 0.0424 | 0.0526 |
| inclination | 10 | **9.8739** | 4.1872 | 5.2715 |
| B/T | 0.15 | **0.1492** | 0.0621 | 0.0779 |
| redshift | 0.01 | **0.0100** | 0.0030 | 0.0045 |

Cluster-side environment of these pairs:

| quantity | min | median | max |
|---|---|---|---|
| host sigma_v [km/s] | 300.200 | 590.900 | 840.200 |
| R_proj / R_vir | 0.036 | 0.870 | 1.498 |
| |g_ext| / a_0 | 0.028 | 0.117 | 2.161 |
| angle(disk normal, host dir) [deg] | 12.127 | 59.964 | 89.964 |

### C1_xray_late — 61 pairs

- morphology gate: deep-learning late type: TType>0 and P_LTG>0.5, 25<i<75 deg
- environment gate: galaxy lies within 2 Mpc projected of an MCXC X-ray peak at |dz|<0.01 (L500 is a direct observable), and R_proj/R_vir<=1.5 of its Tempel host
- cluster arm 110 galaxies, field arm 1089 galaxies

| matching variable | declared tol | max abs(delta) | median abs(delta) | rms delta |
|---|---|---|---|---|
| log10 M_b (stellar) | 0.1 | **0.0996** | 0.0391 | 0.0505 |
| log10 R_d | 0.1 | **0.0943** | 0.0282 | 0.0423 |
| log10 Sigma_b | 0.15 | **0.0989** | 0.0400 | 0.0515 |
| log10 g_bar(2.2 R_d) | 0.1 | **0.0989** | 0.0408 | 0.0506 |
| inclination | 10 | **9.3346** | 4.1027 | 5.0974 |
| B/T | 0.15 | **0.1491** | 0.0722 | 0.0845 |
| redshift | 0.01 | **0.0098** | 0.0032 | 0.0050 |

Cluster-side environment of these pairs:

| quantity | min | median | max |
|---|---|---|---|
| host sigma_v [km/s] | 41.800 | 604.700 | 840.200 |
| R_proj / R_vir | 0.170 | 0.805 | 1.471 |
| |g_ext| / a_0 | 0.002 | 0.106 | 0.609 |
| angle(disk normal, host dir) [deg] | 16.419 | 53.120 | 89.880 |

### C2_xray_disk — 218 pairs

- morphology gate: disk-bearing (S0 or later): TType>-3 and B/T<=0.80, 20<i<80 deg
- environment gate: galaxy lies within 2 Mpc projected of an MCXC X-ray peak at |dz|<0.01 (L500 is a direct observable), and R_proj/R_vir<=1.5 of its Tempel host
- cluster arm 413 galaxies, field arm 1850 galaxies

| matching variable | declared tol | max abs(delta) | median abs(delta) | rms delta |
|---|---|---|---|---|
| log10 M_b (stellar) | 0.1 | **0.0996** | 0.0464 | 0.0554 |
| log10 R_d | 0.1 | **0.0977** | 0.0283 | 0.0420 |
| log10 Sigma_b | 0.15 | **0.1172** | 0.0395 | 0.0513 |
| log10 g_bar(2.2 R_d) | 0.1 | **0.0984** | 0.0397 | 0.0508 |
| inclination | 10 | **9.9313** | 4.2763 | 5.2172 |
| B/T | 0.15 | **0.1492** | 0.0663 | 0.0814 |
| redshift | 0.01 | **0.0099** | 0.0031 | 0.0047 |

Cluster-side environment of these pairs:

| quantity | min | median | max |
|---|---|---|---|
| host sigma_v [km/s] | 28.100 | 604.700 | 840.200 |
| R_proj / R_vir | 0.099 | 0.788 | 1.474 |
| |g_ext| / a_0 | 0.002 | 0.131 | 2.002 |
| angle(disk normal, host dir) [deg] | 12.127 | 59.246 | 89.964 |

## The gas-matched tier returns zero pairs, and that is a result

`A1_gas_matched` adds f_gas to the matching box and yields **0 pairs**.  The reason is not a catalogue gap.  Of the MaNGA galaxies covered by HI-MaNGA that sit in hosts with sigma_v >= 400 km/s, **17 of 494 are HI detections**; among late types, 14 of 156.  In the field arm the late-type detection rate is 572 of 1603.  Neutral hydrogen has been stripped out of the cluster galaxies, so the gas fraction cannot be matched between the two arms.

This is a structural obstacle to the experiment as the brief frames it: the environment whose effect is under test has removed one of the five variables that were supposed to be held fixed.  Every tier other than A1 therefore drops f_gas and matches on stellar baryons only, carrying the HI detection flag and the HI upper limit for each galaxy so the residual gas contribution can be bounded downstream (`cl_logMHI_use`, `cl_logMHI_limit`, `cl_hi_detected` and the `fi_` equivalents).

## The five matching variables are not five independent directions

Correlation matrix of the matching variables across the quality-passing parent sample:

| | `logMstar_nsa` | `logRd` | `logSigma_b` | `log_gbar_2p2Rd` | `f_gas_or_nan` |
|---|---|---|---|---|---|
| `logMstar_nsa` | 1.000 | 0.730 | 0.213 | 0.216 | -0.778 |
| `logRd` | 0.730 | 1.000 | -0.484 | -0.478 | -0.255 |
| `logSigma_b` | 0.213 | -0.484 | 1.000 | 0.995 | -0.275 |
| `log_gbar_2p2Rd` | 0.216 | -0.478 | 0.995 | 1.000 | -0.277 |
| `f_gas_or_nan` | -0.778 | -0.255 | -0.275 | -0.277 | 1.000 |

`log Sigma_b` and `log g_bar(2.2 R_d)` correlate at **r = 0.9955**.  They are one matching direction, not two: g_bar of an exponential disk is Sigma_b times a shape factor that barely varies across the sample.  `f_gas` correlates with `log M_star` at r = -0.778.  The effective number of independent matching directions is about **three** (M_star, R_d, and a partially independent f_gas), not five.

This matters for how the result is stated.  Reporting five tight tolerances would overstate how completely the internal structure has been controlled.  Two of the five are near-deterministic functions of the other three.

## How the host mass was derived, and what may be used as an observation

| quantity | column | provenance | admissible as |
|---|---|---|---|
| member rms velocity | `cl_t14_grp_sigma_v` | Tempel+2014, rms radial velocity deviation of the FoF members | **observation** |
| projected clustercentric radius | `cl_t14_Rproj_kpc` | angular separation from the group luminosity centre times the angular-diameter distance | **observation** (geometry + redshift) |
| X-ray luminosity of the host | `cl_t14_mcxc_L500_1e44` | MCXC, [0.1-2.4] keV luminosity inside R500 | **observation** |
| potential-depth proxy | `log_Phi_proxy` = log sigma_v^2 | member kinematics only; no mass model | **observation up to the assumption that the galaxies trace the potential** |
| external field proxy | `cl_gext_over_a0` = (sigma_v^2 / R_proj)/a_0 | same | same |
| R_vir | `t14_grp_Rvir_Mpc` | projected harmonic mean radius of the members | geometric, but its interpretation as a virial radius is model-laden |
| M_NFW | `cl_t14_grp_MNFW_rank_only` | Tempel+2014, assumed NFW profile | **ranking only — dark-matter dependent** |
| M200 | `cl_t17_grp_M200_rank_only` | Tempel+2017, assumed NFW profile | **ranking only — dark-matter dependent** |
| M500, R500 | `*_mcxc_M500_rank_only`, `*_mcxc_R500_rank_only` | MCXC L-M scaling relation calibrated on hydrostatic masses | **ranking only — dark-matter dependent** |

Every dark-matter-dependent column in this lane carries the literal suffix `_rank_only` in its name, in the master table and in the pair table, so it cannot be picked up as an observable by accident.

## Disk orientation relative to the host

`cl_t14_psi_norm_host_deg` is the angle between the galaxy's disk normal and the direction to its host centre, and `cl_t14_theta_sky_deg` is the sky-plane angle between the disk major axis and that direction.

What is genuinely observable is the sky-projected geometry plus the inclination.  The line-of-sight offset between a galaxy and its host centre is not measurable (redshift differences are dominated by peculiar velocity, not distance), and the near side of a disk is not determined by the photometry.  psi is therefore computed as

```
psi = arccos( sin(i) * |sin(PA_host - PA_disk)| )
```

which assumes the galaxy-to-host offset lies in the plane of the sky, and is folded onto [0, 90] deg to absorb the unknown near side.  It is a projected lower bound on the true 3-D angle, not the 3-D angle.  Anyone using it to test a tidal-eigenvector alignment must propagate that.

## Primary tier: the 19 pairs in full

| # | cluster plateifu | field plateifu | log M* (cl/fi) | R_d kpc (cl/fi) | i deg (cl/fi) | sigma_v | R/Rvir | g_ext/a0 | psi deg | X-ray host |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 12506-6102 | 11018-9101 | 10.88 / 10.90 | 4.90 / 5.42 | 70 / 74 | 576 | 0.23 | 0.77 | 61 |  |
| 2 | 8933-9101 | 11009-1901 | 9.23 / 9.27 | 0.94 / 0.99 | 67 / 68 | 840 | 0.32 | 0.61 | 71 |  |
| 3 | 9863-12704 | 10513-9101 | 9.89 / 9.99 | 1.37 / 1.70 | 73 / 73 | 840 | 0.39 | 0.51 | 49 |  |
| 4 | 11746-12701 | 9894-12702 | 10.64 / 10.59 | 2.57 / 2.64 | 49 / 44 | 836 | 0.81 | 0.44 | 49 | J0828.6+3025 |
| 5 | 8935-6101 | 8934-9102 | 9.18 / 9.16 | 1.53 / 1.64 | 67 / 61 | 840 | 0.77 | 0.25 | 87 |  |
| 6 | 9874-12703 | 8446-3703 | 9.84 / 9.79 | 1.21 / 1.18 | 66 / 59 | 840 | 0.78 | 0.25 | 65 |  |
| 7 | 8990-3703 | 11948-9102 | 9.65 / 9.64 | 1.92 / 2.08 | 50 / 60 | 638 | 0.83 | 0.24 | 86 | J1134.8+4903 |
| 8 | 8990-12705 | 8589-12704 | 10.04 / 10.01 | 2.20 / 2.18 | 43 / 43 | 638 | 0.87 | 0.23 | 84 | J1134.8+4903 |
| 9 | 9873-12702 | 11759-6104 | 9.39 / 9.35 | 1.14 / 1.22 | 65 / 69 | 840 | 0.87 | 0.22 | 25 |  |
| 10 | 8951-12703 | 11963-12702 | 9.34 / 9.29 | 2.47 / 2.60 | 64 / 65 | 840 | 0.94 | 0.21 | 59 |  |
| 11 | 8602-12701 | 9037-12703 | 10.99 / 11.04 | 7.71 / 8.90 | 44 / 44 | 605 | 0.36 | 0.20 | 73 |  |
| 12 | 8550-12704 | 10842-3703 | 10.72 / 10.68 | 4.63 / 4.72 | 37 / 46 | 605 | 0.43 | 0.17 | 64 |  |
| 13 | 8312-3703 | 8341-3702 | 9.88 / 9.88 | 1.79 / 1.60 | 73 / 72 | 605 | 0.44 | 0.17 | 53 |  |
| 14 | 8322-9102 | 11957-12705 | 10.20 / 10.15 | 2.55 / 2.64 | 49 / 58 | 407 | 0.87 | 0.12 | 64 |  |
| 15 | 10216-9102 | 11867-12704 | 10.64 / 10.60 | 2.75 / 2.91 | 51 / 42 | 448 | 0.87 | 0.12 | 65 |  |
| 16 | 8550-3701 | 8461-6103 | 10.24 / 10.21 | 2.29 / 2.30 | 73 / 69 | 605 | 0.64 | 0.12 | 31 |  |
| 17 | 8312-12705 | 8139-12705 | 10.39 / 10.35 | 3.48 / 3.74 | 51 / 47 | 605 | 0.64 | 0.12 | 89 |  |
| 18 | 8550-6103 | 12495-6101 | 10.47 / 10.44 | 2.21 / 2.64 | 41 / 38 | 605 | 0.68 | 0.11 | 63 |  |
| 19 | 11944-12704 | 11871-12703 | 11.17 / 11.19 | 5.40 / 5.72 | 31 / 26 | 423 | 0.75 | 0.11 | 71 |  |
| 20 | 12673-3702 | 10507-3701 | 10.21 / 10.24 | 1.68 / 1.87 | 64 / 70 | 605 | 0.70 | 0.11 | 77 |  |
| 21 | 9869-9102 | 11017-12701 | 10.70 / 10.66 | 4.59 / 4.49 | 68 / 59 | 605 | 0.74 | 0.10 | 32 |  |
| 22 | 8602-6102 | 9036-6103 | 9.50 / 9.41 | 1.88 / 1.58 | 74 / 70 | 605 | 0.91 | 0.08 | 16 |  |
| 23 | 8604-12703 | 8444-12703 | 10.87 / 10.82 | 5.62 / 5.43 | 45 / 38 | 605 | 0.97 | 0.08 | 66 |  |

The other tiers are in `clean/matched_pairs.csv`; filter on the `tier` column.

## Resolved kinematics for every galaxy in every pair

`raw/manga/maps/` holds the DAP `MAPS-HYB10-MILESHC-MASTARSSP` file for each of the 645 distinct galaxies appearing in any tier: stellar velocity field, stellar velocity dispersion, H-alpha velocity field, H-alpha dispersion, emission-line fluxes, and the elliptical polar radius map, with the matching inverse-variance and mask extensions.  See `raw/manga/maps/maps.manifest.json` for the per-file SHA-256 list.

