# Independent conditional selection-ablation audit

PASS_WITH_SCOPE_LIMITS. Independently recomputed all432 aggregate cells,324
paired factorial effects and144 channel-placement gaps from the2,808 saved
trials. Every table value and material-effect gate reproduces exactly.
The702 original-factor trials also reproduce exactly, checking six numerical
metrics and peak-selection flags. All14 bound input hashes are verified.

All54 stage ledgers reconcile with their2,268 channel-level rows to3.54e-16
relative. The source continuum stage loses at most4.37031e-9 of the signed
integrated source flux in these templates. Disabling that source stage changes
retained-flux fractions by zero at the recorded precision. This is not the
cause of the observed channel-placement retention gap in this experiment.

## Threshold attribution, with exact numbers

The fixed-global threshold is defined consistently: every stored channel uses
the median of the same42 western calibration MAD values,6.218646e-5 in the
stored intensity units. The rule remains strictly above2sigma for three
consecutive channels. It does not use eastern patch values to adjust thresholds.
This is a diagnostic counterfactual, not a validated replacement mask.

Averaging the frozen branch/morphology/amplitude combinations equally, the
channel20 retained-fraction gap relative to the mean of channels10 and30 is:

| Background | Per-channel MAD threshold | Fixed global MAD threshold |
|---|---:|---:|
| Empirical delivered patches | -18.4644 percentage points | -8.5052 points |
| Noiseless | -9.0879 points | approximately zero |

The empirical gap magnitude falls by53.9377%, appropriately rounded to53.9%.
This is the reduction in this conditional mean gap, not the fraction of all
missing flux or an observational causal confidence level.

## What the remaining gap can and cannot identify

The observed background is unchanged in every factorial cell. Replacing source
continuum subtraction with channel selection changes injected emission only;
it cannot undo the telescope's continuum operation on the actual delivered
noise and sky signal. The diagnostic S covariance is saved but is not substituted
into these empirical trials. The code preserves that distinction.

With fixed global thresholds, the noiseless placement gap disappears while
an empirical gap remains. This supports a residual interaction with the
delivered background under the specified injection/mask procedure. It does not
identify a particular instrument defect, sky contaminant, foreground, noise
covariance or calibration error. Overlapping, previously screened patches are
not independent empty-sky realizations. Source construction and detector/mask
assumptions remain conditional.

No new raw background pixels or observed velocities were opened by this review,
and no expensive production trials were rerun. The audit confirms saved-trial
arithmetic, stage accounting, matched-factor definitions and implementation
scope; observed-motion likelihood admission remains SOURCE_BLOCKED.
