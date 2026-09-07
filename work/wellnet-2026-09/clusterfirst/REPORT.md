# Run BT — gravity from where the baryons actually are

Lane `work/wellnet-2026-09/clusterfirst/`.

## What was run

Eleven clusters carry all three channels at once, which is what makes this
possible and why it has waited:

- **ACCEPT** deprojected, calibrated electron density `n_e(r)` from Chandra
- deep **Chandra** event lists (Run BR)
- public per-source **DECADE** weak-lensing shear, from the OPEN half of the holdout

One of them is 1E0657-56, the Bullet Cluster.

The baryons are turned into a gravitational field with no fitted halo, no mass
model and no scaling relation:

```
rho_gas(r)        = 1.15 m_p n_e(r)
M_gas(<r)         = ∫ 4π r² rho_gas dr
g_bar(r)          = G M_gas(<r) / r²
Sigma_gas(R)      = Abel projection of rho_gas
DeltaSigma_bar(R) = mean Sigma inside R − Sigma at R
```

and `DeltaSigma_bar` is compared directly against `DeltaSigma_obs` from the
shear. The comparison happens **in projection**, where lensing actually
measures, so no new deprojection assumption enters — the only spherical
assumption is the one ACCEPT already made, and re-projecting it is
self-consistent with that.

## The result

| law | median predicted / observed | scatter |
|---|---|---|
| Newton (baryons only) | **0.148** | 0.69 dex |
| RAR (McGaugh+2016) | **0.456** | 0.65 dex |
| MOND simple ν | 0.459 | 0.65 dex |
| MOND standard ν | 0.393 | 0.65 dex |

Read directly: **the observed lensing is about 7× what the observed baryons
produce under Newtonian gravity, and about 2.2× what the RAR predicts.**

That is the textbook cluster discrepancy, and it is not new — MOND's residual
factor of roughly two in clusters has been known since the 1990s. What is new
here is only that it reproduces with a lensing dataset that has never been used
for this: DECADE metacalibration shapes, an independent survey and an
independent reduction pipeline from anything in the existing literature on these
objects.

Stars are missing from `g_bar` — hot gas is 85–90% of cluster baryons and no
stellar profile exists for these eleven. That biases `g_bar` low by 10–15%,
which makes the discrepancy slightly *worse* than quoted. The bias is
conservative with respect to the headline.

## Why this configuration cannot measure it precisely

The honest limitation, and it is severe.

**The calibrated gas and the lensing barely overlap in radius.** ACCEPT profiles
stop between 0.34 and 1.18 Mpc; the shear profile runs from 0.29 to 4.4 Mpc. So
of ten lensing bins per cluster, **one to three** fall inside the measured gas.
Two clusters have **none**.

| cluster | gas measured to | lensing bins inside | obs / baryon |
|---|---|---|---|
| 1eRASS J065829.9−555637 (Bullet) | 1.18 Mpc | 3 | 1.2 |
| 1eRASS J134730.8−114510 | 0.99 Mpc | 2 | 9.7 |
| 1eRASS J052042.5−132848 | 1.00 Mpc | 2 | 13.0 |
| 1eRASS J063846.8−535831 | 0.49 Mpc | 2 | 4.2 |
| 1eRASS J043900.7+071603 | 0.77 Mpc | 2 | 11.8 |
| 1eRASS J123625.2+163246 | 0.42 Mpc | 1 | 20.5 |
| 1eRASS J102339.7+041108 | 0.38 Mpc | 1 | 9.1 |
| 1eRASS J111320.3+173542 | 0.34 Mpc | 1 | 1.9 |
| 1eRASS J140102.0+025240 | 0.59 Mpc | 1 | 1.5 |
| 1eRASS J045410.6−030056 | 0.77 Mpc | **0** | — |
| 1eRASS J044309.7+021017 | 0.35 Mpc | **0** | — |

**About fifteen usable data points in total**, and the per-cluster ratios span a
factor of seventeen. That is the 0.65 dex scatter, and it is dominated by shape
noise in single lensing bins at the smallest radii, where the shear
signal-to-noise is at its worst. The Bullet's 1.2 is not a measurement that its
lensing matches its baryons; it is one of the noisiest entries in the table,
with the fewest background sources of any cluster in the set.

## The extrapolation that was refused

A first version of this ran without a radial cut, letting the gas density
continue as a fitted power law out to the lensing radii. The mass supplied by
that extrapolation was **3× to 140× larger than anything ACCEPT measured** —
2887%, 1043%, 1726%, 13813% of the measured mass, cluster by cluster.

The prediction would then have been a statement about the fit, not about the
baryons, and it would have looked entirely reasonable: the un-cut run gave
Newton at 0.145 and the RAR at 0.765, with a *tighter* 0.48 dex scatter than the
honest version. Better-looking numbers, produced by extrapolating a density
profile a decade in radius past its data.

`temperature_support` v2 already forbids exactly this — fail closed, never clamp
silently, always print the extrapolated fraction. The rule was written for
temperature and applies unchanged to density. Every bin outside the measured
range is now dropped, and the surviving count is reported per cluster above.

## What would make this a measurement

In order of leverage:

1. **Gas profiles that reach the lensing radii.** ACCEPT is an inner-region
   product. X-COP, CHEX-MATE or a dedicated re-analysis of the Chandra events
   already in `clusterxray/raw` would extend `n_e(r)` outward and turn one bin
   per cluster into six.
2. **More clusters with both channels.** Eleven is what the current archive
   overlap allows; the number grows with any new public shear release.
3. **Strong lensing at small radii**, which measures where the gas actually is
   rather than where the weak lensing is best.

None of those is compute. All three are acquisition.

## Provenance

`eRASS1` `M500`, `R500` and `Mgas500` are **excluded throughout**: they come
from eROCOP, which is weak-lensing calibrated, and the aperture is circular even
where the measurement is not — the identity that killed "organised by r/R500"
(`failures/artefacts/08-r500-is-r.md`). Only fixed-physical-aperture X-ray
quantities and the deprojected profile are admissible.

Sealed half never queried: 304 sealed, 0 tokens spent.
