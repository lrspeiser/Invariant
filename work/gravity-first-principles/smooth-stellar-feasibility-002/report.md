# Smooth stellar-source feasibility completed

All five profiles were tested at three preset smooth-shell widths, retaining every mass bracket and the coordinate-derived central offset. No gravity residual was used.

| Cluster | Width/radius 0.02 | 0.10 | 0.20 |
| --- | ---: | ---: | ---: |
| A1795 | 1.51% | 4.38% | 6.40% |
| A2142 | 0% | 0% | 0% |
| A2319 | 7.57% | 8.43% | 8.97% |
| A85 | 0.305% | 2.63% | 4.51% |
| ZW1215 | 0% | 0% | 0% |

Values are additional allowances outside the quoted projected-mass bounds, divided by measured mass. They are not uncertainties or statistical significances. Smoothing changes feasibility, so a fit obtained with singular shells cannot automatically be transferred to a differentiable source.

The actual source projection matrices passed the declared quadrature-refinement target. Two A2319 cases failed a stricter optimization-consistency check: the dual-simplex solution's directly evaluated violation differed from its reported objective by 3.37e-7 and 7.31e-8. Disabling presolve did not repair those failures. The interior-point method, on the same matrices and bounds, passed both checks with discrepancies below 9e-14 and essentially unchanged objectives. Original failed solutions remain retained.

The successor campaign reused six completed parent cases by hash; the remaining nine were calculated and retained. These results establish only feasibility in a finite smooth-source dictionary. A2142 and ZW1215 are not validated full sources: outer mass, source uncertainty correlations, three-dimensional structure, missing light and physical smoothing scales remain unresolved. A2142's previously demonstrated outer-mass ambiguity still applies.

No source is admitted for a physical cross-regime gravity verdict. No gravity observations were scored and no candidate law was excluded. Next steps require independently justified source constraints rather than increasing flexibility until every profile fits.

Evidence: `smooth-stellar-feasibility-002` and `smooth-source-solver-replay-001`, preserving the completed case table, reused-case hashes, projection checks and solver alternatives.
