# Real-source distributed response: geometry survives the numerical checks

The finite secondary kernel has now been integrated over NGC2976's actual
conditional stellar, atomic-gas and molecular-gas reconstructions, rather than
replacing the galaxy with a central point. Four source alternatives, three
numerical resolutions, three components plus their sum, and 72 positions produce
3,456 field records. The RTX 5090 run took 67 seconds.

The clearest pattern is that the secondary field retains the inner source's
asymmetry but becomes rounder farther out. Across the four conditional source
models, its azimuthal vector variation is about 10–16% at R=1 kpc in the plane,
falling to about 1–2% at R=6 kpc. Tangential components reach about 14% RMS of
the vector strength. This is the kind of directional prediction a central
halo profile cannot reproduce by changing its amplitude alone. Ordinary Newtonian
gravity also responds to nonspherical matter. Directional structure by itself is
therefore not a distinctive discovery; a same-source Newton comparison and
independent motion predictions are needed to distinguish the mechanisms.

These are calculated secondary-force components, not measured gravity anomalies
or percentages of the total observed gravitational field. The kernel strength
was fixed at eta=1 and L=4 kpc without fitting galaxy speeds. The comparison
uses a 12-rotation average of the same source, preserving its mass and radial
profile; that average is a counterfactual, not another observation.

![Conditional source geometry predictions](geometry-patterns.png)

Source assumptions matter much more than the remaining numerical error.
Changing the assumed stellar height from 0.1 to 0.4 kpc while independently
refitting its planar source changes the total field by about 7% RMS and up to
15.3% at an individual position. It is not a pure thickness experiment.
Refining the inverse source representation from f1 to f4 changes the total field
by about 0.48% RMS, up to 1.28% locally.

All four total-field numerical comparisons pass the frozen gates: fine-versus-
middle RMS changes are 0.0069–0.0171%, and the largest point change is 0.0745%.
All sixteen component-plus-total comparisons pass too, with worst individual
component point change 0.0826%. Independent CPU calculations reproduce selected
GPU forces within 2.8e-15 relative. Eight unique source packets, all their arrays,
and 24 source-mass integrations were independently checked. The initial overly
broad audit rejection of intentionally masked source-mean NaNs is retained in
independent-review; the actual integrated source arrays are finite.

The finite kernel is conservative as a specified static pair potential, has
reciprocal forces, and returns to an inverse-square force outside its 10L cutoff.
That establishes a mathematical model, not an energy-transfer mechanism,
recursive relay, time theory, relativistic light law or physical reflecting
material. The measured tracer maps do not establish a unique 3D mass distribution.
Source beams, conversions, missing material, depth and surrounding matter remain
uncertain. No observed motion, cube or lensing likelihood was opened or scored.

The source papers are [Querejeta et al.](https://arxiv.org/abs/1410.0009),
[Walter et al.](https://arxiv.org/abs/0810.2125), and
[Leroy et al.](https://arxiv.org/abs/0905.4742). Exact inherited data hashes are
in source-bindings.json and the source-resolution/generic-source packages.
PREFLIGHT.md defines the model and gates. The next discriminating observation
would compare predicted directional motions against an admitted motion model
including ordinary geometry, streaming, pressure, selection and noise.

Reproduction: scripts/mond_atlas_spatial_program.py; independent implementation
and audits: independent-review/review.py. Existing outputs are frozen evidence;
use a fresh copied output directory when replaying the original runner, which
does not itself enforce immutable output paths.
