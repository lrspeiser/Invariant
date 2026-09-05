# Projected agreement leaves outer stellar mass poorly constrained

For A2142's narrow smooth-shell dictionary, minimized and maximized total stellar mass while retaining every original projected bracket and the measured central offset. Both solutions satisfy the constraints numerically.

The allowed total mass spans **0.911 to 161.47 times the last measured projected mass** within this finite dictionary. This is a deterministic range, not a confidence interval. Its upper end depends on the allowed outer components; it is not evidence for that much stellar mass. The original feasibility solver happened to return an intermediate, very large outer mass because its objective did not penalize unmeasured tails.

Consequently, projected feasibility cannot validate a complete source. Outer continuation needs observational or physical justification and sensitivity checks. Choosing the minimum-mass solution would itself impose an additional source preference; it is not automatically the correct reconstruction. The current derivatives and full-field gravity predictions remain unadmitted.

The source campaign also terminated during A2319's narrow-width solve when its strict direct-constraint check failed. Six preceding cases and the failure record are preserved in `smooth-stellar-feasibility-001`. A successor run reuses those six cases by hash and records constraint inconsistencies for later cases instead of stopping without diagnostics. No failed check is converted to a pass.

Evidence for the mass-range test is `smooth-source-tail-range-001`, including both positive mass vectors, their predicted profiles, constraint residuals and hashes of the inputs. No gravity observations were scored and no candidate was excluded.
