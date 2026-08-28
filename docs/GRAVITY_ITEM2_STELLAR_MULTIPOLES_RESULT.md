# Gravity roadmap Item 2: comparable stellar multipoles (attempt 3)

## Decision

**`INCONCLUSIVE_ITEM2_STELLAR_MULTIPOLES`**

This attempt fixed an important defect in attempt 2: galaxy shape had been measured from
3.4-micron stellar light while cluster shape had been measured from X-ray gas. Here both
populations use stellar tracers. The new CLASH cluster representation passes an independent,
target-blind morphology test, but the resulting stellar shape variables still fail to predict
the cross-scale response within galaxies and within clusters. Item 2 therefore remains open.

The immutable receipt is
`runs/gravity/roadmap/item-02-stellar-multipoles-v3.json`:

- file SHA-256: `018656dd24b667b4b06fc13808216e17780286e26372d23139ac062dcc250465`
- content SHA-256: `7683167d0cbaa2a0adcff5ac61de503196c5facc7a7ad76fb0533c004344ec6e`

## What was tested

The extractor downloaded and hash-bound the public Molino et al. CLASH HST photometric
catalog for each of the same 20 clusters used by Item 1. It selected extended sources using
a rule frozen without the gravity-response labels: acceptable SExtractor flags, 16-band
coverage, signal-to-noise at least 5, a valid non-lensing-corrected stellar mass, and either
secure spectroscopy or a sufficiently concentrated photometric-redshift interval containing
the cluster redshift.

The least complete angular sector across all clusters reaches 184.1 kpc, so the common
aperture was frozen at 150 kpc. Central catalog detections inside 5 kpc were replaced by the
independently tabulated BCG stellar mass. The primary map weights every member by its raw
stellar mass and measures the same dimensionless grammar used on the 68 quality-passing
unWISE galaxy images:

- concentration inside 0.2 aperture;
- multiscale centroid shift;
- BCG-centered quadrupole;
- third and fourth aperture multipoles;
- their combined multipole energy.

Equal and square-root-mass weightings were retained as representation-robustness controls.
No lensing-corrected stellar masses, gravity-response values, direct lensing likelihoods,
SPARC confirmation objects, or paid model calls entered extraction.

## Representation validation passed

Before joining the Item 1 development labels, the primary stellar features were compared
with independent Chandra X-ray morphology at 500 kpc. All five preregistered directions were
positive:

| Matched statistic | Spearman rho |
|---|---:|
| Stellar concentration versus X-ray concentration | 0.239 |
| Stellar centroid shift versus X-ray centroid shift | 0.268 |
| Stellar quadrupole versus X-ray ellipticity | 0.318 |
| Stellar third multipole versus log X-ray `P3/P0` | 0.234 |
| Stellar fourth multipole versus log X-ray `P4/P0` | 0.317 |

Their mean Spearman statistic was 0.275. A deterministic 100,000-trial whole-cluster
permutation test gave one-sided `p = 0.02769`. Half of the stellar/X-ray axes agree within
30 degrees, with a median absolute axis difference of 29.5 degrees. The joint association
also survives equal weighting (`p = 0.02007`) and square-root-mass weighting
(`p = 0.01681`). This validates the representation well enough to test; it does not validate
a gravity mechanism.

## Gravity-response result failed

Fourteen frozen model families were evaluated with five whole-object folds over 68 galaxies
and 20 clusters. Four folds selected concentration plus multipole energy and one selected all
multipoles. This is a reproducible cross-scale discriminator, but not a causal law:

| Gate metric | Galaxies | Clusters |
|---|---:|---:|
| Held-out coefficient `R^2` | -0.432 | -0.740 |
| Held-out coefficient MSE | 0.1128 | 0.2911 |
| Disk/cluster proxy MSE | 0.0814 | 0.1740 |
| Selected observational chi-square per point | 141.4 | 2.565 |
| Constant observational chi-square per point | 898.3 | 6.670 |

The selected shape model beats the constant observational prediction in both populations,
but it has negative coefficient `R^2` within each population and loses to the forbidden
disk/cluster proxy in each. In the shared multipole-energy interval, all 20 clusters and 41
galaxies are present; the stellar model still loses to the proxy in both groups. Population
separation therefore explains its apparent global success.

## Scientific meaning

This is material progress even though it is not a gravity solution:

1. It demonstrates that the third cluster representation is measuring real morphology, so
   the negative gravity result cannot be dismissed as the raw-W1 foreground contamination
   seen during the source audit.
2. It strengthens the exclusion result: global stellar concentration, centroid shift, and
   low-order multipoles at these apertures do not generate the Item 1 coefficient variation.
3. It shows that replacing mismatched gas/stellar tracers improves the observational fit but
   does not recover within-population causality.

The next Item 2 attempt must add a genuinely intermediate or filamentary population, or a
radially resolved/nonlocal shape operator. Retuning the same 88 development labels or adding
more regressions over these five global summaries is not justified.

## Replay

```powershell
python -m sigma_theory_compiler.gravity_item2_clash_stellar_multipoles validate --root . --cache-dir work/item2-common-w1-v3-audit/molino-raw
python -m sigma_theory_compiler.gravity_item2_stellar_multipole_experiment --root . --check
python -m pytest tests/test_gravity_item2_stellar_multipoles.py -q
```

The raw catalog cache is optional for ordinary receipt replay. Passing `--cache-dir` adds a
byte-for-byte replay of all 20 downloaded source catalogs.
