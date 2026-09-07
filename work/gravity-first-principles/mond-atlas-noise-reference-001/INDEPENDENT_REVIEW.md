# Independent conditional-reference review

The arithmetic and stated caveats are correct. A separate implementation
integrated the gamma density of q directly and summed the finite binomial
probabilities, without calling the production chi-square or binomial routines.
All original input hashes were reverified; no old outputs or raw arrays changed.

| Quantity | Independent result |
|---|---:|
| Descriptive count in [0.8,1.2] |21/42|
| Conditional probability in that band |0.53982148175|
| Expected count |22.67250223|
| Count SD, additionally assuming independent channels |3.23007716|
| Probability of 21 or fewer, independent-channel reference |0.35719934665|
| Raw two-sided p-value range |0.01292952–0.99294373|
| Bonferroni-adjusted range |0.54304002–1|
| Family-alpha 0.05 rejections |0|

Maximum probability replay discrepancy is 3.45e-15. The executable check and
full per-channel values are in [independent-review.py](independent-review.py)
and [independent-review.json](independent-review.json).

The chi-square law has **27 degrees of freedom divided by 27** only if the
twenty-seven core residuals are independent Gaussian variables with the known,
correct reference mean and variance. Its standard deviation for q is 0.27217.
Thus a fixed ±20% interval includes only about 54% of draws under that reference;
21 flags outside it are not automatically statistically exceptional failures.
There is no reason to substitute 26 degrees of freedom merely because a mean
exists: the reference treats it as known and does not estimate it from the same
twenty-seven validation cores.

In reality, the pipeline's mean and variance were fitted using western cores
and the regularization was selected. Conditioning on that fitted estimate does
not make it the true sky mean/variance. Training uncertainty, model selection,
spatial core dependence and earlier development exposure are omitted. Therefore
these reference p values are not calibrated p values for the actual fitted
pipeline. A training-derived variance adjustment does not by itself establish
the assumed chi-square distribution.

The expected count uses linearity of expectation and **does not require channel
independence**. The quoted binomial tail and count SD do require it. Bonferroni
does not require independence between channels, but it does require valid
marginal p values; the core/parameter assumptions remain necessary. The count
probability is a lower tail, not a two-sided or prospective significance test.

No numerical or interpretive defect was found within the stated conditional
scope. Zero reference rejections does not validate the model, show channel/core
independence, or admit the source likelihood. The original descriptive flags
remain useful diagnostics and remain unchanged.
