"""Build the bounded cube-pilot findings from frozen evidence."""
import csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False))
D=ROOT/'work/gravity-first-principles/cube-pilot-findings-003';D.mkdir(exist_ok=False)
OUT=ROOT.parents[1]/'outputs';OUT.mkdir(exist_ok=True)
shutil.copy2(__file__,D/'runner.py')
cube=read('work/gravity-first-principles/conditional-cube-pilot-001/result.json')
audits=read('work/gravity-first-principles/conditional-cube-pilot-001/data-audit.json')
gas=read('work/gravity-first-principles/cube-gas-coverage-002/result.json')
numerical=read('work/gravity-first-principles/cube-numerical-validation-001/result.json')
rows=[]
for o in cube['objects']:
 baseline=o['fits'][0]['test_loss']
 for f in o['fits']:
  rows.append(dict(name=o['name'],mode=f['mode'],train_loss=f['train_loss'],test_loss=f['test_loss'],
   improvement_percent=100*(1-f['test_loss']/baseline),converged=f['optimizer_success'],iterations=f['iterations']))
with (D/'scores.csv').open('w',newline='') as file:
 w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
full=[r for r in rows if r['mode']=='full'];converged=[r for r in full if r['converged']]
bound_audit=[]
for obj in cube['objects']:
 p=np.array(obj['fits'][-1]['params'])
 bound_audit.append(dict(name=obj['name'],warp_or_stream_at_bound=bool(np.any(np.abs(p[12:16])>.999)),
  rotation_at_bound=bool(np.any(np.abs(p[:5])>2.499)),rotation_changes_sign=bool(np.min(p[:5])<0<np.max(p[:5]))))
save(D/'parameter-bound-audit.json',bound_audit)
assert len(cube['objects'])==12 and len(rows)==60
assert len(gas['predictions'])==8 and all(p['beta']==0 for p in gas['predictions'])
assert all(a['mask_overlap']==0 and not a['mask_uses_response'] for a in audits)
assert all(a['covariance_pass'] for a in audits)
summary=dict(status='COARSE_CONDITIONAL_CUBE_PILOT_NOT_GRAVITY_VALIDATION',objects=12,model_fits=60,
 numerical_injections=4,converged_fits=sum(r['converged'] for r in rows),
 combined_converged=len(converged),combined_improved=sum(r['improvement_percent']>0 for r in converged),
 gas_eligible=8,gas_selected_nonzero=0,gas_mean_improvement=gas['mean_fractional_improvement'],
 original_gas_diagnostic_admissible=False,total_density_available=False,independent_confirmation=False,
 admitted_gravity_laws=0,
 baseline_physical_adequacy_validated=False,combined_warp_or_stream_boundary_objects=sum(a['warp_or_stream_at_bound'] for a in bound_audit),
 evidence=['conditional-cube-pilot-001','cube-numerical-validation-001','stellar-co-acquisition-001','cube-gas-coverage-002'])
save(D/'summary.json',summary)
fig,(ax,bx)=plt.subplots(1,2,figsize=(13,6.6),gridspec_kw={'width_ratios':[1.2,1]})
labels=[r['name']+(' *' if not r['converged'] else '') for r in full]
values=[r['improvement_percent'] for r in full]
ax.barh(labels,values,color=['#218a79' if v>=0 else '#cb6862' for v in values]);ax.invert_yaxis();ax.axvline(0,color='#333333',lw=.8)
ax.set_xlabel('Change in withheld spectral prediction loss (%)\nPositive = improvement over circular rotation')
ax.set_title('Extra motion components help some galaxies');ax.grid(axis='x',alpha=.15)
coverage=gas['matter_coverage'];pos=np.arange(len(coverage))
bx.barh(pos-.17,[100*r['stellar_covered_fraction'] for r in coverage],height=.3,label='Stellar tracer footprint',color='#7598cf')
bx.barh(pos+.17,[100*r['co_with_positive_error_fraction'] for r in coverage],height=.3,label='CO with reported error',color='#b38ac8')
bx.set_yticks(pos,[r['name'] for r in coverage]);bx.invert_yaxis();bx.set_xlim(0,105)
bx.set_xlabel('Coverage of geometric test + training positions (%)');bx.set_title('HI does not measure all the matter');bx.legend(loc='lower right',fontsize=8)
fig.suptitle('12-galaxy cube pilot: motion first, gravity interpretation later',fontsize=15)
fig.text(.02,.015,'* Combined fit did not converge. Coverage is not detection or total density. Gas correction selected zero in 8 eligible galaxies.',fontsize=9)
fig.tight_layout(rect=[0,.045,1,.94]);fig.savefig(D/'comparison.png',dpi=170);plt.close(fig)
table='\n'.join(f"| {r['name']} | {r['test_loss']:.2f} | {r['improvement_percent']:+.1f}% | {'yes' if r['converged'] else 'no'} |" for r in full)
covtable='\n'.join(f"| {r['name']} | {100*r['stellar_covered_fraction']:.0f}% | {100*r['co_with_positive_error_fraction']:.0f}% | {r['co_covered_nondetection_pixels']} |" for r in coverage)
report=f'''# Motion and gas structure: first conditional cube pilot

The extra motion components explain some previously unexplained spectra, but they do not supply a consistent explanation across all 12 galaxies. After accounting for those components, the covered-gas comparison selected **no additional gas-structure correction in all eight eligible galaxies**. This does not establish a gravity law or reject small corrections, other formulas or a low-density coherence mechanism.

## What was actually tested

The 5090 fitted the intensity in every velocity channel of a coarsened HI cube, comparing five models: regular circular rotation; rotation plus an outer orientation warp; rotation plus radial streaming; rotation plus a broad lagging spectral component; and all components together. Sixty fits were run; 57 converged. The source brightness was supplied from the same observations. These are conditional predictions of withheld spatial spectra, **not independent observing data and not a full physical 3D disk reconstruction**.

Rotation is allowed five radial coefficients. Warps can change outer position angle by 15 degrees and inclination by 8 degrees. Streaming has two radial coefficients bounded by 30% of the initial speed scale. The asymmetric component has a flux fraction up to 0.4, lag up to 0.6, and width 1.7 times the main profile. These bounded phenomenological choices are not an exhaustive inventory of conventional gas dynamics. Their source is frozen in the registered runner.

The scoring mask is geometric: an ellipse between deprojected radii 48 and 450 arcsec, with fixed alternating 192-arcsec spatial blocks and approximately 120-arcsec guard gaps. All channels are retained, so a surprising measured speed cannot cause a position to be selected. This validates response-independent selection for this aperture; it does not recover the survey's original detection mask or make this a representative sample of all gas.

Channel-noise covariance uses channel-dependent variances and six tapered correlation lags, estimated in a separate outer sky annulus. Disjoint background pixels passed the predeclared broad variance/correlation gate in all 12 objects. Their median whitened variances range from {min(a['whitened_validation_median_variance'] for a in audits):.2f} to {max(a['whitened_validation_median_variance'] for a in audits):.2f}; NGC5055 and NGC6946 are near the lower gate, showing important background nonstationarity. Spatial covariance is not fully modeled, so the losses are comparative scores, **not calibrated chi-square statistics or significance levels**.

## Where the motion models helped

Of the 11 converged combined fits, seven improved withheld spectra and four worsened them. The largest apparent gains occur in DDO154, NGC3198 and IC2574. These are reductions in a channel-intensity prediction error, not percentages of missing gravity explained.

| Galaxy | Combined withheld loss | Change from rotation | Combined converged |
|---|---:|---:|---|
{table}

For DDO154 and IC2574, streaming alone predicts withheld spectra better than the combined model. That is a warning about extra flexibility and geometry/motion degeneracy. It does not prove real radial flows. NGC3198 benefits strongly from the combined model. NGC5055 and NGC6946 worsen with extra components. Many final losses remain far above the nominal whitened-noise floor, so substantial model deficiencies remain.

The parameter audit finds **{sum(a['warp_or_stream_at_bound'] for a in bound_audit)} of 12 combined fits at a warp or streaming bound**. NGC3521 and NGC7331 also reach rotation-coefficient bounds; NGC3521 develops an implausible reversal between radial coefficients. Optimizer convergence therefore does not establish physical adequacy. These features can reflect local optimization, source geometry errors, inadequate radial resolution or an inadequate model family. The baseline needs physically constrained rotation, multiple initializations and instrumental-response validation before either a positive or null gravity interpretation. The limits cannot be expanded merely to make a preferred gas term look successful.

NGC2976's combined fit reached its iteration limit; NGC6946's asymmetric fit and UGC04305's warp fit encountered optimizer failures. These three galaxies were excluded from the stricter gas-term comparison. NGC7331 additionally had no positions meeting the broad gas-coverage rule. Failed fits are retained rather than interpreted as physical exclusions.

## The corrected gas test

The first pilot gas diagnostic incorrectly allowed blanked MOM0 zeros to enter the surroundings descriptor. Those scores are **inadmissible for the gas hypothesis** and retained only for audit. The corrected experiment normalizes smoothing by measured support and requires at least 98% support at both 48- and 96-arcsec widths. It describes the detected-emission domain, not voids. A failed stellar WCS coverage attempt was also retained; the successful replacement marks nonconvergent map coordinates uncovered.

The tested term is `v_rotation -> v_rotation * (1 + beta*C_HI)`. `C_HI` is a bounded broad/local HI contrast with its radial mean removed. That focuses this test on differences around a ring; the flexible baseline rotation curve can already absorb a purely radial enhancement. Candidate beta values run from -0.30 to +0.30 in steps of 0.05. Other galaxies' training spectra select beta, and the target galaxy's separate test spectra score it. All baseline nuisance parameters are frozen.

All eight eligible galaxies selected beta=0. Thus this particular descriptor supplied no transferable addition at the tested grid spacing. This is a diagnostic result against a physically unvalidated baseline, not a decisive null test of coherence. It does not exclude smaller coefficients, improvements after joint refitting, a different measured source descriptor, or a genuinely derived field law. The target galaxy still supplies its baseline kinematic calibration, so this is not prediction of an entirely unseen galaxy without local fitting. All galaxies have been exposed during earlier development.

## Total matter remains unresolved

53 stellar and CO assets were acquired with source URLs, file sizes and hashes. Stellar light, molecular gas and atomic gas must all enter a mass model; neither HI intensity nor its holes is total volume density. The table is a footprint audit at selected sky positions, not a matched-beam mass reconstruction. CO nondetections use the supplied interpolated error map only as a screening indicator; correlated uncertainties have not been propagated into a significance claim.

| Galaxy | Stellar coverage | CO with positive error | Covered CO nondetection positions |
|---|---:|---:|---:|
{covtable}

CO coverage ranges from about 15% to 100% of these positions. Covered nondetections remain upper-limit information, not zero molecular mass. Stellar foreground masks, mass-to-light uncertainty, the CO-to-H2 conversion, beam matching and line-of-sight depth still require validation. We have no calibrated total 3D density map from this pilot.

The stellar FITS headers also contain SIP distortion coefficients without a matching CTYPE suffix. A separate comparison with and without those coefficients changes finite stellar coverage by up to 7.5 percentage points (NGC3198). This does not validate either astrometric interpretation. The plotted stellar footprints are provisional until alignment is checked against source astrometry. The separate HI gas-score calculation is unaffected by this stellar-header ambiguity. Details are retained in `cube-matter-wcs-audit-001/result.json`.

## What would make the next result convincing

1. Replace the coarse projected approximation with a validated tilted-ring cube forward model, including native beam velocity mixing, disk thickness and the actual spectral response. Current additional smoothing is about 48 arcsec and native source-beam effects remain approximate.
2. Test nuisance-model adequacy and identifiability with mismatched simulations, not only injections generated by the fitting model. Four matching-model controls recovered withheld synthetic spectra near the noise floor, but that establishes numerical behavior rather than unique physical causes.
3. Build a common-resolution stellar + HI + molecular mass model with propagated conversion and nondetection uncertainties. A freely fitted rotation curve is useful for separating motion, but a constrained mass model is necessary to test gravity.
4. Lock the observable descriptor and compare predictions on new galaxies or independently acquired data. Preserve the geometric mask and evaluate both spectra and well-defined motion diagnostics without tuning to the test set.

Primary methodological references: [3D-Barolo cube fitting](https://arxiv.org/abs/1505.07834), [THINGS processing and kinematics](https://arxiv.org/abs/0810.2125), and [asymmetric HI profiles in NGC3521](https://arxiv.org/abs/1312.2399). Asset-level observational URLs and hashes are in the acquisition receipts.

## Reproduction and evidence

Use Python 3.13 with NumPy, SciPy, Astropy and PyTorch 2.7.1+cu128. The private CUDA environment is `work/private/torch-cuda-env`; raw data and prepared cubes are deliberately excluded from Git. Immutable directories retain registrations, exact runner copies, numerical controls, every galaxy fit, audit results and failures. The old pilot gas scores must not be promoted. The current reusable pilot script removes that legacy scoring; use the separate coverage audit.

The active first-principles gravity goal remains open. No new law is admitted by these results.
'''
(D/'report.md').write_text(report,encoding='utf-8')
registry=read('configs/gravity_sigma_directions_v33.json');registry['predecessor']='configs/gravity_sigma_directions_v33.json'
registry['conditional_cube_campaign']={'status':summary['status'],'evidence':str((D/'summary.json').relative_to(ROOT)),
 'report':str((D/'report.md').relative_to(ROOT)),
 'findings':['12 galaxies, 60 conditional cube fits; extra kinematic components help some spatial predictions and harm others.',
 'Eight coverage/convergence-qualified gas comparisons select beta=0 on the registered grid.',
 'HI is not total density; acquired stellar and CO maps have incomplete coverage and unresolved conversion/foreground uncertainty.'],
 'limitations':['Coarse projected conditional model, not a fully validated 3D disk fit.','Same-observation brightness and previous development exposure.','No gravity law admitted.'],
 'next':'Validate full instrumental/tilted-ring response and a multi-component mass model before a gravity-law interpretation.'}
save(ROOT/'configs/gravity_sigma_directions_v34.json',registry)
for source,target in [('report.md','Gravity-cube-pilot-findings.md'),('comparison.png','Gravity-cube-pilot-comparison.png'),('scores.csv','Gravity-cube-pilot-scores.csv'),('summary.json','Gravity-cube-pilot-summary.json')]:
 shutil.copy2(D/source,OUT/target)
shutil.copy2(ROOT/'configs/gravity_sigma_directions_v34.json',OUT/'Sigma-gravity-directions-v34.json')
verification={'checks':['12 objects, 60 models','8 corrected gas predictions all beta=0','disjoint response-independent masks','all registered background covariance gates passed'],
 'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.iterdir() if p.is_file()}}
save(D/'verification.json',verification)
print(json.dumps(summary,indent=2))
