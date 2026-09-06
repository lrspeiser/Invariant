# Frozen selection transfer experiment

Developmental extension, frozen before new cube-array access. SOURCE_BLOCKED for
observational gravity likelihoods. No measured source velocities or reserved
galaxies will be read. Real data are the previously exposed NGC2976 standard
THINGS cube backgrounds; injected source brightness and motion are synthetic.
Sources and instrument evidence inherit the hash-bound native-selection-001
package, Walter et al. 2008 (https://arxiv.org/html/0810.2125), archive
https://things.www3.mpia.de/Data.html and NRAO VLAOS_0302 manual. No download.

Keep the existing western calibration medians/MAD, 12 eastern patch centers,
30 arcsec detector, >2 sigma in three consecutive stored channels, restoring
beam and three spectral/continuum response branches. These backgrounds are
held away from calibration but already development-exposed, overlap, and are
MOM0-screened from the same observation, not certified empty or independent.

New synthetic templates have identical fixed elliptical Gaussian brightness
(major FWHM 30 arcsec, projected axial ratio .5). At each pixel integrate a
Gaussian local line FWHM 2 stored channels. Symmetric velocity offset is
4 tanh(R/15 arcsec) X/R. Warp substitutes X cos(theta)+Y sin(theta),
theta=30 degrees R/(R+15), in that velocity field only; this is a kinematic
twist with fixed brightness, not a self-consistent warped density disk.
Streaming adds 2 tanh(R/15) Y/R. R=sqrt(X²+Y²), Y=projected y/.5.
Offsets are synthetic line-of-sight channel offsets, not fitted orbital speeds.
Centers are stored channels 10,20,30; amplitudes 5 and10 times western median
detector MAD. Normalize all morphologies to the symmetric template peak at
the same center/branch, preserving near-equal intrinsic integrated flux.
All 54 cases retained. Twelve actual patches yield648 trials; 16 independent
Gaussian realizations per branch yield864 conditional trials. Seed9062607.
Gaussian noise retains the existing beam-filtered spatial surrogate and full
A H H^T A^T covariance, not a measured dirty-beam covariance.

Before arrays: existing independent quadrature/convolution/continuum/covariance
controls must pass; new checks require zero-velocity morphology identity,
antisymmetric velocity to1e-12 and summed intrinsic spectral flux invariance
to1e-5; independent loop run-mask matches exactly. Positive signals must remain
nonnegative before continuum subtraction. Failure stops source trials and is
retained. No code/gate retuning from real outcomes.

Report true source flux retained, peak selection, paired selected noisy flux,
and selected noisy flux separately. Uncertainty: Gaussian Monte Carlo SD and
SE across16 draws; empirical min/max/SD only, no binomial or iid confidence
claim. Paired morphology differences use same backgrounds. A conditional
adequate-recovery case requires mean true retained fraction >=.9 and absolute
mean paired flux bias <=.1. A transferable morphology pair requires absolute
mean retention difference <=.05. These are diagnostic gates, not observational
admission. All failures retained, no source correction inferred from them.

New private output zero bytes; memory-resident cubes only, free-disk reserve8GB.
Missing exact correlator response, dirty beam/CLEAN/residual scaling/primary
beam, independent line-free support and publisher mask remain explicit.
