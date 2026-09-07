# Resource revision before any new source solve

The coordinating agent clarified that the8GB ceiling was its conservative
operational estimate, not a user constraint, and authorized up to16GB sampled
working memory with the same300second time limit. No21kpc calculation or new
source solve had started. Preserve the original assessment but execute the
requested24kpc halfwidth at fine spacing(0.125,0.125,0.0625),385^3 cells, only.

The9.13GB conservative20-array allowance fits within the revised16GB ceiling;
require at least32GB currently available physical memory before allocating.
Monitor sampled process RSS at0.1seconds, stop at callback/stage boundaries if
it exceeds16,000,000,000bytes. Same caveat: sampled memory is not a continuous
peak guarantee. CPU1thread and300seconds remain unchanged.

Reuse exact fine18 andbase24 ell0.5 outputs with hashes. Compare fine18->fine24
(domain at identical fine spacing) andbase24->fine24 (mesh at identical domain),
using original raw/center-relative global5% and height8% gates. This supplies the
requested joint endpoint if both comparisons pass; it is still a conditional
numerical test and no observational validation. No21kpc result is claimed.
