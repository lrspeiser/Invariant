# Foreground hypothesis: plausible context, unresolved frame match

**The exact NGC2976 cube uses radio velocity, not FELO-HEL optical velocity.**
Astropy/WCSLIB interprets its legacy `CTYPE3=VELO-HEL`, `VELREF=258` as
`VRAD`, `SPECSYS=BARYCENT`, with units m/s. `CUNIT3` and `SPECSYS` are absent
from the original header; the interpretation and warnings are retained.
The header has four axes, including singleton Stokes. The successful computation
extracts `WCS.spectral`; an initial three-coordinate call to the four-axis WCS
failed without producing coordinates or accessing science arrays.

| Stored index, zero based | FITS channel, one based | Native radio velocity, km/s | Same-frame optical parameter, km/s |
|---|---:|---:|---:|
|10|11|+54.52666992|+54.53658911|
|20|21|+3.00000000|+3.00003002|
|21|22|−2.15266699|−2.15265153|
|30|31|−48.52666992|−48.51881630|

Thus an injection centered at stored index 20 is centered at +3 km/s, not
−2.153 km/s. The latter belongs to index 21. The original native-selection
report's index-21 label is consistent with this mapping.

The independent expression
`CRVAL3 + (index + 1 - CRPIX3) * CDELT3` agrees exactly with Astropy for this
actual radio axis; its parameters are 3000 m/s, 21, and −5152.666992 m/s.
The world-to-pixel roundtrip also passes. This linear expression is not being
generalized to FELO-HEL or another nonlinear spectral type.

Using the header rest frequency 1420405750 Hz, the calculated same-frame
conversion is `v_opt = v_radio / (1 - v_radio/c)`. It changes the four values
by 0.00991919, 0.00003002, 0.00001546 and 0.00785362 km/s respectively: at most
0.193% of the 5.152667 km/s channel spacing. This is a change of Doppler
parameterization, **not** a heliocentric/barycentric/LSR frame correction.
No numerical size for an unresolved frame correction is assumed here.

## Primary-source context

[THINGS, Walter et al. 2008](https://arxiv.org/html/0810.2125), Table 2, gives
NGC2976's observing center as 3.0 km/s and describes the setup velocities as
barycentric/heliocentric with the optical convention. Table 5 and Figure 30
give a systemic velocity of 2.6 km/s. The delivered cube's radio header must
therefore govern channel coordinates rather than blindly inheriting the
observing-table Doppler label. These published values supply low-systemic-velocity
context; they were not used as fitting targets.

[Sorgho et al. 2019](https://arxiv.org/html/1903.03767), §2.3, reports Galactic
HI contamination in a different DRAO/EBHIS M81-group observation. Its reported
intervals are −76.3 to −40.0 and −23.5 to +19.3 km/s, with peaks −56.5 and
−2.1 km/s. Inspection of the author HTML, observing table and Figure 3 did not
establish the velocity reference frame or Doppler convention; the figure labels
only velocity in km/s. Its larger beam and short-spacing correction differ
from THINGS. The published version is
[MNRAS 486, 504](https://academic.oup.com/mnras/article/486/1/504/5374532);
the publisher fetch failed, so the author preprint was used.

**No numerical interval-to-channel membership is admitted**, because the DRAO
frame is unresolved. The similar near-zero numbers alone are not a validated
match, and the other reported negative-velocity component prevents a simplistic
“only the central channel is foreground” interpretation. No DRAO mask was copied,
and no THINGS channel was removed or reclassified.

## Testable hypothesis and alternatives

Hypothesis: spatially structured Galactic HI or its imaging residuals contribute
to the channel-dependent background that reduced injection recovery. A useful
test would first obtain an independently documented spectral frame and foreground
map at the exact sky positions, then predict its response through the appropriate
beam, spectral kernel and interferometric spatial filtering. The test should
predict background structure on separately chosen positions before changing any
mask or scoring galaxy motion. Both published foreground components should be
considered rather than choosing a velocity interval from the recovery outcome.

Competing explanations include bandpass/calibration residuals, visibility flags
and weights, dirty-beam sidelobes/CLEAN artifacts, continuum residuals in the
delivered background, and undetected galaxy/group emission. The prior source-only
continuum ablation does not remove these effects from real noise. Foreground and
instrumental explanations can coexist. Velocity coincidence without spatial and
instrument-response evidence cannot distinguish them.

## Files and access

The script is `scripts/mond_atlas_foreground_metadata.py`.
The exact existing cube is
`work/private/things-observable-12gal-003/NGC_2976_NA_CUBE_THINGS.FITS`, SHA256
`e8ce711a354fcf9c76e7ba9afd9e55fa97df6169863d5473140f20f9c8250169`.

[header-and-coordinates.json](run001/header-and-coordinates.json) records the
header, normalized WCS, all coordinates, arithmetic controls and warnings.
[paper-source-evidence.json](run001/paper-source-evidence.json) records exact
remote paper/figure URLs, byte counts and hashes; downloaded paper bytes were
read in memory only. [literature-metadata.json](run001/literature-metadata.json)
keeps the two papers' metadata separate.

All four file bindings were reverified. No science-array values, new observed
motion scores, raw observational downloads, masks, source edits or commits.
`SOURCE_BLOCKED` remains for foreground attribution and observational gravity
likelihoods. Reproduction uses the same private cube and Astropy environment:
`python -B scripts/mond_atlas_foreground_metadata.py`; immutable run001 must be
preserved and reproduction performed in a separate checkout/output copy.
