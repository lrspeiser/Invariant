# Local calibration can remove a predicted enhancement

The extra pull in these conditional NGC2976 source models depends materially on how the model is calibrated to laboratory Newton's constant. This experiment changes that normalization only. It does not determine the laboratory environment or compare predictions with observed motion.

For a locally constant coefficient, G_bare = epsilon_lab G_measured. At fixed source density and response geometry, all self-generated candidate force vectors therefore multiply by epsilon_lab. Imposed external fields are excluded: their amplitudes do not obey this rescaling rule. A manufactured variable-coefficient PDE verified the linear scaling, including the consistently scaled self-field boundary, before the saved source fields were read.

We froze u_lab = 0, 0.01, 0.1, 1, 10, 100 and epsilon_lab = (u_lab+0.2)/(u_lab+1). These are manufactured scenarios, not Solar-neighborhood measurements. The source geometry, smoothing scales, Newton comparison and all 384 sampled points are unchanged.

| Manufactured u_lab | epsilon_lab | ell=0.25 kpc median force-norm ratio | Points above Newton | ell=0.5 kpc median ratio | Points above Newton |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.2000 | 0.3113 | 0/384 | 0.2727 | 0/384 |
| 0.01 | 0.2079 | 0.3237 | 0/384 | 0.2835 | 0/384 |
| 0.1 | 0.2727 | 0.4246 | 0/384 | 0.3719 | 0/384 |
| 1 | 0.6000 | 0.9340 | 179/384 | 0.8182 | 143/384 |
| 10 | 0.9273 | 1.4435 | 325/384 | 1.2645 | 312/384 |
| 100 | 0.9921 | 1.5444 | 375/384 | 1.3529 | 384/384 |

At u_lab=1, the counts above Newton by height z=0, 0.2, 0.5 and 1 kpc (96 points each) are 41, 41, 42, 55 for ell=0.25, and 33, 34, 34, 42 for ell=0.5. Thus enhancement survives more often at the largest sampled height in this scenario; this is a conditional model pattern, not observed evidence for the response law.

The full vector and signed projection matter. We define inward as -g dot rhat toward the adopted source center, with three-dimensional rhat. Both candidate models and Newton point outward at two sample locations; each candidate has 382 inward points. Positive rescaling preserves those directions and signs. At u_lab=1, 186/384 and 150/384 candidate projections are more inward than Newton, respectively. A positive difference at an outward location can mean less outward, not actual inward attraction. The two exceptional locations are excluded from inward-ratio thresholds but retained in all vector outputs.

For an uncalibrated positive ratio R, enhancement requires epsilon_lab > 1/R. When 1<R<=5, this means u_lab > (1/R-0.2)/(1-1/R). These are strict thresholds derived from model predictions, not measured parameters. For force norms, the first sampled enhancement appears above u_lab=0.10498 (ell=0.25) or 0.14368 (ell=0.5). Two ell=0.25 norm ratios are already below Newton before this rescaling, so no allowed calibration can enhance all its points. The ell=0.5 model exceeds Newton in norm everywhere only above u_lab=29.136. These extrema depend on the frozen sample and numerical field accuracy; they are not precision physical constraints.

The favorable feature is that the normalization question is measurable in principle and has large consequences. The challenge is that an independent, sufficiently resolved local matter environment and a justified constant-medium laboratory approximation are still missing. A galaxy fit cannot provide that independent calibration. Scalar rescaling also cannot repair incorrect force directions or source geometry.

All inherited selected smoothed-source and Newton convergence gates passed; the earlier unsmoothed failure remains excluded and unchanged. There were zero new real-source 3D solves and zero observed-response scores. An independent implementation verified all 15 input hashes, 4,608 point rows, 60 summaries and 768 threshold rows. Maximum arithmetic difference was 1.82e-12 in the saved field units; threshold identities agreed to 6.67e-16. Counts near ratio one retain the inherited numerical uncertainty and are not independent statistical observations. The separate gas-only environment calculation was not appended to this frozen sweep.
