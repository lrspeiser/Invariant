# Frozen radial-only numerical refinement

Run002's aperture-range force box/grid/vertical/angular checks all passed,
but .05→.025radial interpolation failed for individual HI/CO components.
All failures remain in run002. This is numerical interpolation, not a physical
parameter adjustment or observed-response-dependent change.

Run003 re-evaluates only the same finest Newton96/.03125 and log32/.015625/order128
field settings at R=.05..6 step.00625, with angular2048; retain512/1024 angular
subsets as controls. Compare PCHIP .025→.0125 and .0125→.00625 at intervening
radii; same1%RMS/3%point gates, full versus aperture support separately. Output
the .00625 component tables regardless of failures. No source/support/strength
changes. The previous full-support force-grid failures are not cured by this
radial-only check. Estimated RAM<4GB, source computation<300seconds. Total public
output kept<20MB by saving finest angular table plus compact angular convergence
receipt rather than repeating every angular row if necessary.
