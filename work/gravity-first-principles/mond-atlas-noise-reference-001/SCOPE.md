# Retrospective finite-sample reference

This diagnostic follows inspection of the exposed42DCchannel scores and their21/42descriptive pass count. It is not a new holdout, a changed gate, or a likelihood admission. Original scores and failed descriptive checks remain unchanged.

Null reference:27independent zero-mean Gaussian background-core residuals per channel with known correctly specified variance. Then q=mean(residual^2/variance) has chi-square27/27 distribution. The model's fitted mean and variance are treated as true predictive values ONLY for this conditional reference. This does not represent uncertainty from selecting/fitting those parameters or spatial correlation among cores.

Calculate probability of q in[.8,1.2], expected count among42, independent-channel binomial lower-tail count probability, and all42two-sided marginal chi-square p values with Bonferroni adjustment. The marginal reference assumes independent cores. Binomial count additionally assumes independent channels; Bonferroni does not. No p value establishes independence or correctness of the actual data-generating process. Alpha=.05 fixed for the family diagnostic. Independently integrate the analytic chi-square density and check total normalization and distribution function within1e-10.

Purpose: distinguish expected finite-sample scatter from a rigid descriptive range. No model changes, exclusions, source-region access or additional observed-motion scores.
