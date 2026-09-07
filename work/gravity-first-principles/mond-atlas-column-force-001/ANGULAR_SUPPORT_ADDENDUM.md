# Piecewise angular and central/outer radial refinement

The source-audit exact-mass centroid repair gives actual primaryf4 emitter radii
.0736569564..6.0121585431kpc and positive underlying HI. Run004 extends the
conditional force table to6.025 without extrapolation and resolves the central
weighted-force structure. Retain run001-003 and all their failures.

Freeze radii .05.. .25 at .00025 spacing, .25..5.95 at .00625 spacing,
and5.95..6.025 at .00025 spacing, removing duplicate endpoints. For each ring,
split azimuth at every original HI bilinear-grid crossing and integrate each
segment with Gauss-Legendre4 and8. Zero column remains undefined; save any zero
ring in a separate support failure receipt, not a filled force. Original HI
weight remains inside the integral. The target/source height laws are unchanged.

Re-evaluate only existing Newton96/.0625 and96/.03125, log32/.03125/order128
and32/.015625/order128. Source coefficient and force laws remain identical.
Save fine/order8 components and differences against coarse fields/order4.
Separate central(.05.. .25), main(.25..5.95), outer(5.95..6.025), full and
aperture(.75..2.5) gates: same1%RMS/3%point; interpolation comparison uses
every-other radial node within each piece. Retain failures, absolute errors
and signed forces; do not relax a gate around zero. Earlier box/vertical
checks remain inherited, with their limitations. No observed spectra.

Estimate working RAM<8GB and source-only calculation<300seconds. Store new
large tables compressed losslessly so public package remains<20MB. This is a
numerical integration refinement over the same physical source, not a new
source model or a response-dependent change.
