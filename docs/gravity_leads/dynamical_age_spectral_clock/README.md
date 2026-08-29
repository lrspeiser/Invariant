# Dynamical-age spectral clock

## Status

This is the strongest replicated empirical association in the roadmap, but it is not
currently evidence for modified gravity. It is an active phenomenon/publication lead.

Primary records:

- [Item 12 dynamical-age result](../../GRAVITY_ITEM12_MANGA_DYNAMICAL_AGE_RESULT.md)
- [Item 13 relaxation and mergers](../../GRAVITY_ITEM13_MANGA_RELAXATION_MERGERS_RESULT.md)

## Core idea

Several stellar-spectrum age and star-formation indicators are combined into a
consensus clock. Its interaction with a stellar surface-density proxy predicts stellar
velocity dispersion beyond ordinary structural measurements.

Schematic selected family:

    log10(sigma_star)
      = structural baseline
      + b standardize[clock_consensus * tanh(surface_density_proxy)]

## What was learned

- Item 12 froze 262,144 age/settling cells before response access.
- Of 750 exploration galaxies, 585 passed all quality rules.
- The selected family reduced held-out MSE by 18.33% and improved R2 from 0.7949 to
  0.8325.
- All five folds selected spectral-clock consensus times stellar surface density with
  a positive coefficient.
- The gain was positive in both halves of age proxy, mass, Sersic index, and redshift.
- Item 13 tested 300 disjoint MaNGA identities; 243 passed quality.
- The fixed age consolidation improved over structure by 23.06% and remained a 22.69%
  improvement after visible disturbance controls.
- Visible tidal debris, asymmetry, clumpiness, and merger indicators did not explain the
  association in that test.
- Both datasets belong to the MaNGA/SDSS ecosystem, so this is not cross-source
  confirmation.
- The response was integrated stellar dispersion, not a rotation curve, lensing map, or
  cluster observable.

## Relationship to known work

Age, Dn4000, mass-to-light ratio, size, velocity dispersion, and assembly history are
known to correlate. The exact multi-clock interaction may be a useful compression, but
historical novelty is doubtful without specialist comparison.

Starting literature:

- Zahid and Geller, Dn4000 and velocity dispersion:
  https://arxiv.org/abs/1701.01350
- Lu et al., MaNGA DynPop stellar population and dynamics:
  https://arxiv.org/abs/2304.11712

## Likely ordinary first-principles explanation

Virial reasoning gives sigma squared of order GM/R. Stellar population affects the
inferred stellar mass-to-light ratio and traces formation, contraction, accretion,
quenching, and orbital structure. The clock may therefore measure missing assembly or
mass-calibration information rather than a change in gravity.

It does not show that old stars retained extra primordial speed or that gravitational
signals accumulated slowly.

## Suggested next steps

1. Freeze the existing family and test it unchanged in a non-MaNGA spectroscopic survey.
2. Replace angular surface density with calibrated physical surface density.
3. Include modern stellar-population mass-to-light estimates and IMF uncertainty.
4. Predict a fresh resolved dynamical observable rather than integrated dispersion.
5. Test whether simulation assembly histories reproduce the same residual relation.
6. Keep this as an astronomy-paper track independent of the alternative-gravity track.

