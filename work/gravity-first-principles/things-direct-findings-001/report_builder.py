import ast
import copy
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from bs4 import BeautifulSoup
BASE=Path(__file__).parent;ROOT=BASE/'Invariant';W=ROOT/'work/gravity-first-principles';OUT=BASE.parent/'outputs'
D=W/'things-direct-findings-001';D.mkdir(exist_ok=False)
P=W/'things-direct-patterns-003';A=W/'things-observable-acquisition-003';C=W/'things-cube-acquisition-001';N=W/'things-cube-noise-audit-001'
data=json.loads((P/'result.json').read_text());models=data['models'];audit=json.loads((P/'data_audit.json').read_text())
noise=json.loads((N/'result.json').read_text())['objects'];maps=json.loads((A/'receipt.json').read_text());cubes=json.loads((C/'receipt.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
rows=[];rng=np.random.default_rng(6723);resamples=rng.integers(0,10,size=(5000,10))
for m in models:
    local=next(x for x in models if all(x[k]==m[k] for k in ['resolution','scenario','algorithm']) and x['group']=='local')
    row={k:m[k] for k in ['resolution','scenario','algorithm','group','pairs','fractional_gain','kms_gain','galaxies_improved']}
    row['incremental_gain_over_local']=float(1-np.mean(m['after'])/np.mean(local['after']))
    draws=1-np.array(m['after'])[resamples].mean(axis=1)/np.array(local['after'])[resamples].mean(axis=1)
    row['fixed_prediction_bootstrap_025']=float(np.quantile(draws,.025));row['fixed_prediction_bootstrap_975']=float(np.quantile(draws,.975));rows.append(row)
with (D/'scores.csv').open('w',newline='',encoding='utf-8') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
nom=[r for r in rows if r['scenario']=='nominal' and r['group']=='gas_surroundings']
wins=sum(r['incremental_gain_over_local']>0 for r in rows if r['group']=='gas_surroundings')

# Noise values are read as numerical table data, not guessed error bars.
table=json.loads((A/'published_noise_tables.json').read_text())['tables'][0]
published={}
for row in table:
    if len(row)==10 and row[1]=='NA':
        name=''.join(row[0].split())
        if name=='HoII':name='UGC04305'
        published[name]=float(row[5])
for n in noise:n['published_channel_noise_mjy']=published[n['name']]
noise_ratios=[n['median_channel_rms_mjy']/n['published_channel_noise_mjy'] for n in noise]
save(D/'noise_comparison.json',noise)

fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
labels={'ridge':'Linear ridge','trees':'Boosted trees','gpu_rbf_features':'GPU kernel model'}
for i,algorithm in enumerate(labels):
    for resolution,offset,color in [('NA',-.17,'#287caa'),('RO',.17,'#b8752b')]:
        r=next(r for r in nom if r['algorithm']==algorithm and r['resolution']==resolution)
        axs[0].bar(i+offset,100*r['incremental_gain_over_local'],width=.31,color=color,label=resolution if i==0 else None)
axs[0].axhline(0,color='black',lw=1);axs[0].set_xticks(range(3),list(labels.values()))
axs[0].set_ylabel('Extra reduction in fractional squared error (%)');axs[0].set_title('Does measured gas context improve predictions?')
axs[0].legend(title='Processing version');axs[0].grid(axis='y',alpha=.2)
for n in noise:axs[1].scatter(n['published_channel_noise_mjy'],n['median_channel_rms_mjy'],s=45,color='#287caa')
axs[1].plot([.25,1.05],[.25,1.05],color='black',lw=1,ls='--');axs[1].set_xlim(.25,1.05);axs[1].set_ylim(.25,1.05)
axs[1].set_xlabel('Published channel noise (mJy/beam)');axs[1].set_ylabel('Measured cube noise (mJy/beam)')
axs[1].set_title('Independent numerical check of cube noise');axs[1].grid(alpha=.2)
fig.suptitle('Direct gas and motion data: noise check passes broadly; predictive rule does not transfer',fontsize=12)
fig.supxlabel('Left: ten development galaxies, each held out in turn. NA and RO share the same observations. Positive is better.',fontsize=9)
fig.savefig(D/'comparison.png',dpi=160);plt.close(fig)

# Verify source projection cannot read SPARC observed speeds or their error column.
tree=ast.parse((ROOT/'scripts/run_gravity_things_direct_patterns.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='source_fields')
raw={g['name']:g for g in json.loads((ROOT/'configs/sparc_rotation_curves_full_v1.json').read_text())['galaxies'] if g['name'] in data['registration']['names']}
env={'np':np,'WCS':WCS,'raw':raw};exec(compile(ast.Module(body=[node],type_ignores=[]),'<source-fields>','exec'),env)
asset=next(a for a in maps['files'] if a['name']=='NGC3198' and a['resolution']=='NA' and a['moment']==0)
h=fits.getheader(ROOT/asset['file']);g=data['registration']['geometry']['NGC3198']
before=env['source_fields']('NGC3198',g,(1024,1024),h,[.5,.7],0,0)
modified=copy.deepcopy(raw)
for item in modified.values():
    for row in item['rows']:row[1]=str(float(row[1])+123);row[2]=str(float(row[2])+17)
env['raw']=modified;after=env['source_fields']('NGC3198',g,(1024,1024),h,[.5,.7],0,0)
assert all(np.array_equal(before[k],after[k]) for k in before)
assert len(models)==84 and all(len(m['names'])==10 and len(m['folds'])==10 for m in models)
assert all(m['nonpositive']==0 for m in models)
for m in models:assert set(f['test'] for f in m['folds'])==set(m['names'])
verification=dict(status='PASS',models=84,galaxies=10,pairs=868,positive_predictions=True,
    all_galaxies_held_out_once_per_model=True,source_projection_unchanged_by_poisoned_sparc_targets=True,
    numerical_controls=json.loads((P/'controls.json').read_text()),
    source_feature_dependency='Map contrasts depend on MOM0 only; velocity masks affect admission and targets, not the contrast values.',
    calibrated_moment1_noise=False,independent_confirmation=False)
save(D/'verification.json',verification)

report='''# Direct gas-to-motion test: first multi-galaxy result

The new direct gas measurements do not reproduce the earlier gas-force-proxy lead reliably. We now have substantially better observational inputs, including unblanked spectral cubes, and a tested way to compare real projected gas structure with measured motions. This particular gas-surroundings predictor is not ready to become a coherence formula.

## Data acquired and checked

All 12 galaxies in the pre-existing source selection now have natural and robust integrated HI brightness, velocity and velocity-dispersion products: 72 FITS maps. All 12 natural-weighting spectral cubes were also acquired, totaling 4.52 GB of cube files plus about 0.31 GB of maps. The exact download URLs, file hashes, dimensions, units, beams and blanking records are retained locally. Large FITS inputs remain outside Git; acquisition manifests and analysis evidence are versioned.

The current publisher index omits IC2574. Its exact historical official map URLs remain accessible and were used with their provenance recorded. Initial index-validation failures were retained before any analysis; subsequent successful acquisition contains all 72 maps.

The standard cubes contain the noise outside detected emission. The published survey distinguishes these from flux-rescaled products used for moment maps. We use the standard cubes to audit noise and detection support, and retain the official moment maps for brightness and motion measurements. Standard-cube flux is not silently substituted for rescaled flux. [THINGS measurement paper](https://arxiv.org/html/0810.2125v1)

## What we can trust more now

'''
report+=f'Measured per-channel noise ranges from {min(n["median_channel_rms_mjy"] for n in noise):.3f} to {max(n["median_channel_rms_mjy"] for n in noise):.3f} mJy/beam. Across the twelve cubes, measured noise is {min(noise_ratios):.2f}-{max(noise_ratios):.2f} times the published survey value. This is a useful check that these are usable noise-bearing cubes, not zero-filled images masquerading as measurements. It does not validate every individual velocity estimate.\n\n'
report+='''We reconstructed a detection mask using the documented recipe: smooth to 30 arcsec FWHM and require emission above twice the measured smoothed noise in three consecutive channels. The resulting sky support overlaps the official support imperfectly. Intersection-over-union ranges from about 0.50 to 0.88; NGC7331 has the largest mismatch. The reconstruction is a diagnostic, not a replacement for the original channel mask or calibrated upper limits.

Neighboring channel noise is correlated, with measured lag-one correlations roughly 0.20-0.40 in clipped background samples. Channels and neighboring sky pixels must not be treated as independent measurements. The released velocity-dispersion map measures gas line width, not uncertainty in its mean velocity. Consequently this experiment does not report noise-calibrated chi-square or discovery significance.

The cubes have two sky coordinates plus a velocity axis. They do not provide a unique spatial depth coordinate or a complete 3D mass distribution.

## Exactly what was predicted

At matched locations on opposite sides of a galaxy, take half the absolute difference in the measured line-of-sight velocities. This removes the systemic recession velocity without fitting a target-galaxy offset or choosing its approaching side. It measures the antisymmetric component of gas motion. Warps, radial motions, asymmetric line profiles and other noncircular flows can affect it; it is not automatically a circular speed or gravitational acceleration.

Both observations and the projected RAR reference are intensity-weighted and smoothed to a common 20 arcsec Gaussian width. The source-only baseline uses published SPARC stellar/gas force components; no SPARC observed rotation speeds are used as predictor values. For five objects, geometry comes from photometry with an assumed intrinsic thickness ratio. Other published metadata include kinematically inferred inclinations and position angles, so those predictions are conditional on previously inferred geometry.

Added gas features come directly from HI brightness: broad/local contrasts at Gaussian widths 40/20 and 80/20 arcsec, plus brightness asymmetry between opposite locations. These are overlapping projected averages rather than physical shell masses or counts of voids. Local comparator inputs include local HI brightness, source-model quantities, radius, projection angle and aperture coverage at all scales. A complete 12-object radial surface-brightness supplement is unavailable in the development package, so the comparator uses a labeled stellar force-component scale rather than inventing missing brightness measurements.

## Sample and validation

Quality cuts were frozen before scoring: both sides must have sufficient detected-emission support, adequate source-profile radial coverage, a projected reference speed above 10 km/s, and be away from the projected minor axis. Samples are taken on a fixed grid and must remain usable across both processing products and all seven source/geometry scenarios.

Ten galaxies retain 868 opposite-side pairs. NGC2976 has only three usable pairs and NGC7331 none under the common-coverage rule; both remain in the acquisition/noise audit but fail the predeclared minimum of ten pairs for prediction. They were not removed for poor fit scores. The admitted sample is DDO154, IC2574, NGC2841, NGC2903, NGC3198, NGC3521, NGC4214, NGC5055, NGC6946 and UGC04305.

Three algorithms were tested: ridge regression, shallow boosted trees and GPU random-feature kernel regression. Each holds out one entire galaxy in turn. Three inner galaxy folds select regularization or tree complexity from fixed grids. Seven scenarios cover nominal/lighter/heavier stars, inclination +/-5 degrees and position angle +/-5 degrees. These are sensitivity brackets, not probability distributions. With two processing versions and two input groups, the experiment contains 84 model runs and 42 added-gas comparisons.

The same 868 locations are reused for all comparisons. Pixels, pairs, processing versions and scenario variations are not independent new galaxies. All are project development tests, not pristine confirmation data.

## Main result

'''
report+=f'Adding measured gas context improves fractional squared error over the otherwise identical local model in **{wins}/42** comparisons. Under nominal source and geometry assumptions, it improves only **one of six** algorithm/processing comparisons.\n\n'
report+='''| Algorithm | Natural weighting: extra gain | Robust weighting: extra gain |
|---|---:|---:|
'''
for alg,label in labels.items():
    a=next(r for r in nom if r['algorithm']==alg and r['resolution']=='NA');b=next(r for r in nom if r['algorithm']==alg and r['resolution']=='RO')
    report+=f'| {label} | {100*a["incremental_gain_over_local"]:+.2f}% | {100*b["incremental_gain_over_local"]:+.2f}% |\n'
report+='''
Positive means lower error after adding gas surroundings. The GPU natural-weighting improvement is relative to a poorly performing local kernel model: even the improved version remains worse than the projected RAR reference. Its improvement does not survive switching to the robust-weighting product. In all six nominal added-gas models, squared km/s error is worse than the projected RAR reference.

This is a more direct test than the previous 139-galaxy gas-force-descriptor experiment. The sample, motion target and spatial descriptors differ, so it does not mathematically refute that older statistical finding. It does prevent treating the old finding as established evidence that measured diffuse surroundings strengthen gravity.

The score CSV includes 5,000 paired whole-galaxy bootstrap resamples of fixed outer predictions. They do not repeat model selection or earlier project choices and should not be treated as full uncertainty intervals. With only ten galaxies, a favorable isolated score is weak evidence.

## Verification

Synthetic controls verify constant preservation, systemic-velocity cancellation, linear velocity-field convolution and WCS round trips. Every model predicts each admitted galaxy only from the other galaxies and uses the identical admitted location set. All predicted pair speeds are positive. Deliberately changing SPARC observed velocities and uncertainties leaves the source projection fields exactly unchanged.

Initial pipeline attempts failed before scoring on incomplete brightness-supplement coverage and a string-valued distance field. The completed version uses a labeled stellar source-component descriptor and explicitly parses distances as numbers. These failures and their frozen source snapshots are retained. The successful run is things-direct-patterns-003.

## What this changes for the coherence idea

There is no reliable rule here of the form "the gas surroundings are more diffuse, therefore this region has extra gravitational pull." The data support a useful projected gas measurement, but that measurement alone has not delivered a transferable motion correction.

Before adding formula terms, the most informative next analysis is to distinguish regular rotational motion from warps, asymmetric profiles and streaming. A full cube model can test whether apparently unusual speeds are predicted by those effects. It should use channel noise covariance and a validated selection mask, then check whether a gas-structure term improves independent motion predictions beyond that model. Total stellar and molecular matter remains relevant; HI alone is not total density.

This campaign establishes a usable source archive and reports a negative result for one directly measured gas-context predictor. It does not reject all coherence mechanisms, validate 3D void structure, or produce a new gravity law.

## Evidence

- Map acquisition: things-observable-acquisition-003/receipt.json
- Cube acquisition: things-cube-acquisition-001/receipt.json
- Noise and support reconstruction: things-cube-noise-audit-001/result.json
- Registration, source audit, folds and predictions: things-direct-patterns-003/
- Numerical checks and target-poison check: things-direct-findings-001/verification.json
- Survey methodology: https://arxiv.org/html/0810.2125v1
- Source mass models: https://arxiv.org/abs/1606.09251
'''
(D/'report.md').write_text(report,encoding='utf-8')
summary=dict(status='DIRECT_GAS_CONTEXT_NOT_ROBUSTLY_PREDICTIVE',files=72,cubes=12,galaxies_admitted=10,pairs=868,
    incremental_wins=wins,comparisons=42,nominal_incremental_wins=sum(r['incremental_gain_over_local']>0 for r in nom),
    models=rows,verification=verification,evidence_hashes={str(p.relative_to(ROOT)):sha(p) for p in [A/'receipt.json',C/'receipt.json',N/'result.json',P/'result.json',P/'data_audit.json']},
    independent_confirmation=False,new_laws=0)
save(D/'result.json',summary);(D/'report_builder.py').write_bytes(Path(__file__).read_bytes())
for path,name in [(D/'report.md','Gravity-direct-gas-motion-findings.md'),(D/'comparison.png','Gravity-direct-gas-motion-comparison.png'),
    (D/'scores.csv','Gravity-direct-gas-motion-scores.csv'),(D/'result.json','Gravity-direct-gas-motion-summary.json')]:shutil.copyfile(path,OUT/name)
registry=json.loads((ROOT/'configs/gravity_sigma_directions_v32.json').read_text())
registry['predecessor']='configs/gravity_sigma_directions_v32.json'
registry['direct_gas_motion_campaign']={'status':summary['status'],'evidence':str((D/'result.json').relative_to(ROOT)),
    'report':str((D/'report.md').relative_to(ROOT)),'findings':['72 maps and 12 unblanked cubes acquired and noise audited.',
    'Ten source-qualified galaxies, 868 paired LOS motion measurements, whole-galaxy CV.',
    'Measured HI surroundings improve fractional error in only 12/42 comparisons and 1/6 nominal comparisons.',
    'No unique 3D structure, calibrated moment-velocity covariance or new gravity law established.'],
    'next':'Model noncircular kinematics and channel covariance before interpreting gas context as a gravitational enhancement.'}
save(ROOT/'configs/gravity_sigma_directions_v33.json',registry)
shutil.copyfile(ROOT/'configs/gravity_sigma_directions_v33.json',OUT/'Sigma-gravity-directions-v33.json')
print(json.dumps({'report':str(OUT/'Gravity-direct-gas-motion-findings.md'),'wins':wins,'noise_ratios':noise_ratios}))
