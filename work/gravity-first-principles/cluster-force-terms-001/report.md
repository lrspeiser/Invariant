# Stellar density gradients drive the retained force reversals

We decomposed the full spherical force for all nine one-megaparsec candidates, eight development clusters, and three already registered source widths: 216 cases. No model was fitted or rescored.

The force was split algebraically into the local response, remaining derivative reaction, geometric terms, gas-density-gradient contribution, and stellar-density-gradient contribution. Their sum reproduces the implemented full force to a maximum relative error of 4.91e-16, normalized by the larger of the force magnitude and the sum of component magnitudes. All 216 negative-force node counts reproduce the parent fine-grid results.

At the primary width, 3,567 sampled model–radius points have nonpositive force; at width 0.005, 1,153 do. Every one becomes positive when the stellar-density-gradient contribution is subtracted for diagnosis. These are correlated computational samples, not independent observations. The widest source has no such points, but fails the preset A85 source-fidelity limit.

The most negative force occurs for the shape-0.5, a0=2e-10 m/s², one-megaparsec candidate in ZW1215 near 749.66 kpc. The full acceleration is -7.79e-11 m/s²; the stellar-gradient contribution alone is -1.03e-10 m/s². The remaining terms sum to an inward acceleration.

This identifies an algebraic driver, not whether the source gradient is physically accurate. The gradient contribution is required by the action: simply removing it would change the law. A small discrepancy in enclosed stellar mass can coexist with a large discrepancy in its higher spatial derivatives. We need independently justified source reconstructions and derivative uncertainty before treating the reversals as robust physical exclusions. A positive force alone would also leave pressure, lensing, stability, galaxy and Solar System requirements open.

For clusters without supplied stellar profiles, the component called stellar includes the campaign's inherited gas-proportional missing-stellar prescription. All primary failed clusters have supplied stellar profiles.

The saved diagnostic and complete input snapshots are under `cluster-force-terms-001` in the research worktree. Original files were hashed before calculation and verified unchanged afterward. The audit retains all candidates and makes zero new observational scores or family exclusions.
