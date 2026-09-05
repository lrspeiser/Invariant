# P0701 spent barycentric radial-alignment joint screen

Frozen before candidate scores: 2026-08-02

Verdict: **fails jointly and closes the spent DDO154/RX J2129 controller loop**

## Exact candidate

P0701 gates the coherent and local equation sources with the alignment of the
summed Newtonian field and the global inward barycentric direction:

\[
\mathcal A_r={\max[0,-\mathbf g_N\cdot\hat{\mathbf r}]\over|\mathbf g_N|},
\qquad
S_{\rm base}=\mathcal A_rS_{\rm coh}+(1-\mathcal A_r)S_{\rm local}.
\]

The projected routing correction, constants, photon closure, nuisance bounds,
and rejection thresholds are unchanged.  No per-object gravity setting or
alignment transform is fitted.

## Joint result

| System | Metric | Candidate | Comparator or gate | Verdict |
|---|---|---:|---:|---|
| DDO154 | ordinary RMSE | `2.347 km/s` | algebraic MOND `2.916` | pass (`0.805x`) |
| DDO154 | weighted RMSE | `2.898 km/s` | algebraic MOND `1.226` | fail (`2.365x`) |
| DDO154 | ordinary RMSE | `2.347 km/s` | 3D QUMOND `3.936` | pass (`0.596x`) |
| DDO154 | mean bias | `-0.692 km/s` | absolute `<=3` | pass |
| RX J2129 | median physical deflection | `7.037 arcsec` | `1-20` | pass |
| RX J2129 | training / heldout roots | `14/15`, `6/7` | exact coverage | fail |
| RX J2129 | missing-multiplicity families | `4/7` | `0` | fail |
| RX J2129 | parity-diverse / critical families | `4/7`, `6/7` | `7/7`, `7/7` | fail |
| RX J2129 | nuisance parameters near bounds | `2` | `0` | fail |

All numerical, source-identity, boundary, curl, finite-field, parameter, and
ordinary galaxy gates pass.  The weighted galaxy gate and ten cluster
fit/topology gates fail.  The exact controller is retired.

## Failure anatomy

The measured score-region alignments are:

| Region | Median alignment |
|---|---:|
| DDO154 midplane over the registered score radii | `0.9839` |
| RX J2129 strong-lens annulus over all line-of-sight cells | `0.9948` |

Both fields are almost perfectly inward relative to their global barycentric
centers.  The controller therefore selects the coherent base in both objects
and closely reproduces P0697's cluster failure.  A net-field/global-center
angle cannot detect the multi-center cluster structure relevant to lens
topology.

Small departures from one also change the DDO154 inner source enough to worsen
its high-weight points even though its ordinary RMSE remains strong.  Applying
an exponent, cutoff, or remapping to force those values back to one would be a
post-hoc fit to spent outcomes and is prohibited.

## Decision: stop formula selection on these two objects

DDO154 and RX J2129 have now generated and rejected multiple controller
families.  Further formula invention from their residuals would make any
subsequent score increasingly circular.  The next research stage is therefore
not P0702 on the same outcomes.  It is to:

1. lock the formula ledger and preserve all positive and negative mechanisms;
2. identify and register genuinely untouched resolved galaxies and raw-lensing
   clusters without running a candidate;
3. obtain their baryonic maps, velocity/image constraints, licenses, hashes,
   coordinate conventions, and uncertainty models;
4. preregister one universal scoring and rejection protocol; and
5. evaluate only a candidate chosen from non-outcome criteria and synthetic
   audits.

P0699 remains the most useful cluster mechanism result (`2.481 arcsec`
heldout, `0.978x` compact halo, six exact families), while P0697 remains the
best spent DDO154 mechanism result (`1.887 km/s` ordinary RMSE).  They are
different endpoints, not a surviving switch law.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/run_p0701_spent_barycentric_radial_alignment_joint_screen.py
python -m pytest tests/test_barycentric_radial_alignment.py tests/test_local_vector_coherence.py tests/test_coherent_monopole.py tests/test_p0635_ddo154_map_commissioning.py
```

Artifacts are in
`results/p0701_spent_barycentric_radial_alignment_joint_screen/`.

## Claim boundary

P0701 is spent mechanism evidence and uses a diagnostic zero-slip photon
closure.  P0633 and P0640 remain sealed.
