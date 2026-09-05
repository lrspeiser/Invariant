# Smooth-shell projection: failed quadrature retained, revised method checked

Compared two calculations of projected smooth-shell mass: averaging offset Plummer aperture masses over shell orientations, and integrating the positive spherical density through a cylinder. Tested three shell widths and ten aperture radii with 64, 128 and 256 quadrature nodes.

The first orientation rule, uniform in cosine of polar angle, passed 28 of 30 cases at 256 nodes. At the smallest aperture and narrowest shell its relative discrepancy was 2.06e-5, exceeding the 1e-7 target. The failure is retained in `smooth-shell-projection-001`.

Changing the integration variable to polar angle, with its required sine Jacobian, better resolves contributions near the viewing axis. All 30 cases then passed at 256 nodes; the worst discrepancy against the independently integrated density was 6.25e-9. The revised result is retained in `smooth-shell-projection-002`. Coarser-node errors remain available in both runs.

This verifies sampled projection behavior, not arbitrary source widths, universal error bounds, or a cluster reconstruction. The next source-mixture calculation must check its actual projection matrices and fitted predictions under refinement. Neither smoothing nor quadrature may be chosen to improve gravity residuals.

The two runs retain their executable calculations, declared targets, complete errors and zero new gravity scores or physical exclusions. The galaxy angular-refinement process remains a separate ongoing calculation.
