# Gravity roadmap Item 2: MaNGA nonlocal stellar shape (attempt 4)

## Decision

**`INCONCLUSIVE_ITEM2_MANGA_NONLOCAL_SHAPE_QUALITY_GATE`**

This was a genuinely different real-data test. It moved from rotating disks and rich
clusters to pressure-supported ellipticals and S0 galaxies, replaced model-derived halo
mass with direct two-dimensional stellar kinematics, and tested radially resolved and
nonlocal stellar-mass geometry rather than retuning the five rejected global summaries.
It did not pass. Five of 60 preregistered exploration galaxies failed the frozen
representation-quality gate, and the 55 valid galaxies independently give a negative
shape result.

The immutable receipt is
`runs/gravity/roadmap/item-02-manga-nonlocal-shape-v4.json`:

- file SHA-256: `fea2bd9461242e3ea14a393f97250e1972b41c84ad0c09d4900275e131a76818`
- content SHA-256: `3046daec435b42fcd8381c7664f642f58a2905c3c3531dced8472b552430d521`

## Frozen design and target boundary

The complete selection and model contract was committed and pushed as
`c49ba6ba2f3b2658c43b2e8bb6a27e7a851d598f` before any selected kinematic map was
opened. Public SDSS DR17 morphology and i-band PyMorph catalogs selected one observation
per galaxy, with certain visual classifications, acceptable photometric fits, redshift
`0.01–0.10`, half-light major axis at least 3 arcsec, and two visual classes (elliptical
and S0) crossed with three projected-axis-ratio bins. SHA-256 ordering froze:

- 60 exploration galaxies: 10 per class/axis-ratio stratum;
- 30 reserved confirmation galaxies: 5 per stratum;
- zero PCA or DAP endpoint queries during selection.

Only the 60 exploration source pairs were subsequently acquired: 120 files and
320,892,601 bytes. The 30 confirmation targets were not queried, downloaded, or opened.
There were no paid model calls, SPARC confirmation accesses, or direct-lensing accesses.

The baryonic maps are the public MaNGA PCA resolved stellar-mass products. The response
uses the public MaNGA DAP `STELLAR_VEL`, `STELLAR_SIGMA`, and the officially required first
`STELLAR_SIGMACORR` channel. No DynPop/JAM total mass, dark-matter fraction, NFW parameter,
or lensing-derived mass was read.

## Direct observable tested

For each galaxy, the target-blind PCA map defined the center, principal axis, nested
apertures, stellar mass, and all shape features. Only after that vector was finalized did
the response extractor open the kinematic map. The primary dimensionless aperture response
was

```text
eta_ap = R_e,circ <(v_los-v0)^2 + sigma_corr^2>_L / (G M_star(<R_e))
sigma_corr^2 = max(sigma_DAP^2 - sigma_corr,DAP^2, 0).
```

The same frozen models were also checked with the major-axis radius, uncorrected
dispersion, and stellar-mass rather than i-band-luminosity weighting. `eta_ap` is a direct
aperture virial summary, not a field equation or a direct measurement of the Item 1
coefficient. It remains sensitive to projection, orbital anisotropy, the stellar IMF,
distance, and omitted gas.

The qualifying features included radial quadrupole changes, quadrupole-profile variance,
inner/outer axis twist, odd/even multipole changes, outer multipole energy, profile
roughness, and inner/outer centroid alignment. All are dimensionless and invariant under
translation after recentering, rigid rotation, reflection, and uniform mass rescaling.

## Representation quality result

Five objects failed exactly the preregistered source-quality rules and were not replaced:

| Plate-IFU | Frozen failure |
|---|---|
| `10518-3702` | insufficient PCA aperture coverage |
| `12506-3702` | insufficient PCA aperture coverage |
| `8274-3704` | insufficient spaxels in the innermost nested aperture |
| `8979-12704` | insufficient PCA aperture coverage |
| `9487-3701` | insufficient PCA aperture coverage |

The remaining 55 contain 29 ellipticals and 26 S0s and retain every class/axis-ratio
stratum. They have at least 57 unique stellar-kinematic bins; the median is 178.

## Held-out result

Ten model families and 46 distinct model/ridge cells were selected with nested,
whole-galaxy five-fold validation. The selected family varied by fold: radial odd/even
shape once, boundary/twist shape once, the already rejected global multipoles once, and
the nonqualifying morphology-class control twice.

| Frozen prediction | Held-out MSE | Held-out `R^2` |
|---|---:|---:|
| Nested selected model | 0.07625 | **-0.166** |
| Mass/size nuisance | 0.05165 | 0.210 |
| Morphology-class nuisance | 0.04991 | 0.236 |
| PyMorph Sersic/axis-ratio control | 0.05593 | 0.144 |
| Prior global multipoles control | 0.05238 | 0.199 |

The selected result is also negative within each population: `R^2=-0.542` for ellipticals
and `-0.086` for S0s. Removing the selected shape terms while retaining mass and size
improves overall `R^2` to 0.213. Every response robustness variant loses to the
morphology-class nuisance. No qualifying shape family is positive within both classes;
the closest, radial quadrupole, has `R^2=0.210` overall but `R^2=-0.001` in S0s and adds no
reliable benefit over mass/size.

## Scientific meaning

This attempt gives a useful exclusion, not a gravity law:

1. Low-order global shape failure was not merely caused by mixing rotating disks with
   spherical cluster data or by the model-dependent CLASH coefficient target.
2. Adding radial changes, twists, boundary energy, and nonlocal profile roughness does not
   robustly predict direct stellar dynamical response in this pressure-supported bridge
   population.
3. Mass, size, and morphology class contain reproducible information, but the tested
   projected stellar-shape summaries do not explain that information as a universal cause.
4. The 30 confirmation galaxies stay sealed because exploration did not justify opening
   them.
5. Item 2 remains open, but another attempt should not add more regressions over these
   MaNGA summaries. A useful next attempt needs a different physical variable—most
   plausibly spectroscopic group geometry plus member dynamics, a direct pressure-support
   operator, or a truly filamentary response—not a weaker quality threshold chosen after
   seeing this result.

## Replay

Receipt replay does not require the 321 MB local raw cache:

```powershell
python -m sigma_theory_compiler.gravity_item2_manga_nonlocal_shape check-sample --root .
python -m sigma_theory_compiler.gravity_item2_manga_nonlocal_shape_experiment --root . --check
python -m pytest tests/test_gravity_item2_manga_nonlocal_shape.py -q
```

When the exact exploration cache is present, source extraction can also be replayed:

```powershell
python -m sigma_theory_compiler.gravity_item2_manga_nonlocal_shape extract-exploration --root . --cache-dir work/item2-manga-v4-raw
```
