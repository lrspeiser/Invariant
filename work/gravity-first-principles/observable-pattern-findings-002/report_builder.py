import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).parent;ROOT=BASE/'Invariant';OUT=BASE.parent/'outputs'
WORK=ROOT/'work/gravity-first-principles'
P=WORK/'observable-pattern-campaign-002';R=WORK/'observable-pattern-robustness-001';M=WORK/'observable-map-pilot-001'
D=WORK/'observable-pattern-findings-002';D.mkdir(exist_ok=False)
data=json.loads((P/'result.json').read_text());models=data['models']
boot=json.loads((R/'bootstrap.json').read_text());geo=json.loads((R/'geometry.json').read_text())
perms=json.loads((R/'permutations.json').read_text());ma=json.loads((M/'map_audit.json').read_text())
labels={'ridge':'Linear ridge','trees':'Boosted trees','gpu_rbf_features':'GPU kernel model'}
rows=[]
for model in models:
    local=next(m for m in models if m['scenario']==model['scenario'] and m['salt']==model['salt'] and m['algorithm']==model['algorithm'] and m['group']=='local_coverage')
    rows.append({k:model[k] for k in ['scenario','salt','algorithm','group','fractional_mse_gain','kms_mse_gain','chi_square_gain','galaxies_improved','nonpositive_predictions']}|
                {'incremental_fractional_gain_over_local':1-np.mean(model['after'])/np.mean(local['after'])})
with (D/'model_scores.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True,sharey=True)
for ax,group,title,color in zip(axs,['photometry_multiscale','gas_force_proxy'],
        ['Measured starlight profile at several scales','Calculated gas-force profile (comparison only)'],['#177d9a','#b56f20']):
    subset=[b for b in boot if b['group']==group]
    for i,algorithm in enumerate(labels):
        for salt,offset,marker in [('A',-.13,'o'),('B',.13,'s')]:
            b=next(x for x in subset if x['algorithm']==algorithm and x['salt']==salt)
            lo,_,hi=np.array(b['fixed_prediction_bootstrap_quantiles'])*100
            mid=b['incremental_gain']*100
            ax.errorbar(i+offset,mid,yerr=[[mid-lo],[hi-mid]],fmt=marker,color=color,capsize=4,label=f'Partition {salt}' if i==0 else None)
    ax.axhline(0,color='black',lw=1);ax.set_xticks(range(3),list(labels.values()));ax.set_title(title,fontsize=11)
    ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9)
axs[0].set_ylabel('Extra reduction in fractional squared error (%)\nrelative to the same algorithm with local inputs')
fig.suptitle('139 development galaxies: added surroundings information remains uncertain',fontsize=13)
fig.supxlabel('Bars: 95% paired galaxy bootstrap of fixed predictions; model selection uncertainty omitted. Positive is better.',fontsize=9)
fig.savefig(D/'prediction_comparison.png',dpi=170);plt.close(fig)

sbwins=int(sum(r['incremental_fractional_gain_over_local']>0 for r in rows if r['group']=='photometry_multiscale'))
gaswins=int(sum(r['incremental_fractional_gain_over_local']>0 for r in rows if r['group']=='gas_force_proxy'))
allnom=[r for r in rows if r['scenario']=='nominal']
nomgas=[r for r in allnom if r['group']=='gas_force_proxy']
with (D/'report.md').open('w',encoding='utf-8') as f:
    f.write('''# First observable-pattern campaign: completed results

We can measure a stable projected gas-surroundings contrast in NGC3198. We have not yet established that this directly measured feature predicts anomalous gravity across galaxies. The population prediction experiment uses existing radial starlight and calculated gas-force profiles; it is separate from the single-galaxy gas-map pilot.

## What actually ran

- CuPy 13.5.1 executed CUDA array arithmetic, image filtering, Gaussian beam matching and random-feature kernel regression on the RTX 5090. The installed PyTorch 2.7.1 build is CPU-only; no CUDA PyTorch installation was required or made.
- The population run evaluated 54 combinations: three algorithms, three input groups, two whole-galaxy partitions and three stellar mass assumptions. Each uses five outer galaxy folds and three inner folds for parameter selection. All 139 galaxies and 2,720 radii are previously exposed development data.
- Algorithms: regularized linear regression; small histogram gradient-boosted trees; GPU regression using 256 fixed random Fourier features approximating an RBF kernel. The last is not an exact Gaussian process and supplies no calibrated posterior uncertainty.
- Follow-up diagnostics: 2,000 paired galaxy bootstrap samples of fixed predictions; 64 shuffled-feature population runs with inner selection repeated; nine projected aperture geometries with restoring-beam covariance matched between two gas-map products.

## Finding 1: an apparently empty pixel is often a processing mask

About 79.9% and 80.2% of the full natural/robust HI image arrays are exactly zero. FITS HISTORY explicitly records blanked pixels being replaced with zero. These percentages refer to the rectangular images, not the physical fraction of a galaxy that is empty. We exclude zero-filled pixels from the pilot instead of interpreting them as low-density or coherent regions.

The HI intensity unit is Jy/beam m/s. The beam is recorded in HISTORY rather than ordinary BMAJ/BMIN keywords. We recover that beam and convert to Jy/arcsec² km/s before comparisons. The natural and robust beams are about 11.43 by 9.36 and 6.55 by 5.24 arcsec respectively.

The local CO maps include an error product, but HI noise/censoring information is not recovered by this pilot. A conservative nonzero support mask is not a calibrated selection function. CO and stellar mass images were inventoried, not combined into a total matter-density map.

## Finding 2: a reproducible projected gas descriptor

At each usable location we compare a broad Gaussian average of HI brightness with a narrower local average. Positive means the surroundings are brighter; negative means the location is brighter than its surroundings. These are overlapping averages, not counts of physical spheres, a pure exterior annulus, or measurements of spatial volume density.

The initial circular filters use Gaussian widths of 15, 30 and 60 arcsec. At the pilot's adopted distance 13.987 Mpc these correspond approximately to 1.02, 2.03 and 4.07 kpc in the sky plane; no distance is needed to compute the angular descriptors. Both processing products agree in sign at about 97-98% of sampled well-supported locations, with rank correlations above 0.99. They use the same underlying observations and related masking: this is processing robustness, not independent measurement replication.

The more careful follow-up subtracts each native Gaussian beam covariance from the desired filter covariance before convolution. It varies projected axis ratio 0.284/0.334/0.384 and position angle 28.6/33.6/38.6 degrees, around the existing photometric geometry. Local and broad major-axis Gaussian widths are 30 and 60 arcsec. These ranges are sensitivity brackets, not measured probability intervals.

''')
    f.write(f'Across the {geo["common_locations"]} locations supported in every geometry, {geo["all_geometry_sign_agreement_fraction"]:.1%} keep the same contrast sign throughout. Among the {geo["material_locations"]} locations whose reference contrast exceeds 5% in magnitude, **{geo["material_sign_agreement_fraction"]:.0%} keep their sign**. The median range of contrast across geometries is {geo["median_geometry_contrast_range"]:.3f}, roughly three percentage points. Locations are correlated; this is not a significance calculation.\n\n')
    f.write('''This establishes a useful candidate observable. It does not establish whether a low-brightness region is gas-poor in three dimensions, whether total matter is sparse, or whether gravity there is stronger. Disk thickness and line-of-sight rearrangements remain unresolved.

## Finding 3: prediction from the measured starlight surroundings is weak

The population baseline includes local baryonic acceleration, gas/bulge force descriptors, local starlight, radial slope, and observational radial coverage. Added photometric features are interior/exterior annular-area-weighted brightness contrasts at fixed radial reaches 0.5, 2 and 8 kpc. They are derived from published radial brightness profiles, not the NGC3198 image filters or complete 2D maps.

''')
    f.write(f'Adding multiscale photometry improves fractional error in **{sbwins}/18** algorithm/partition/mass combinations. Trees improve in all six, the linear model worsens in all six, and the GPU kernel model is mixed. Every nominal photometry incremental-gain bootstrap interval includes zero. This is not an algorithm-independent predictive relationship.\n\n')
    f.write('''## Finding 4: the gas-force profile is the stronger statistical lead, with important failures

''')
    f.write(f'The calculated gas-force descriptor improves fractional error over local inputs in **{gaswins}/18** combinations. In the nominal runs it reduces fractional squared error by {min(r["fractional_mse_gain"] for r in nomgas):.1%} to {max(r["fractional_mse_gain"] for r in nomgas):.1%} relative to RAR. However, **all six nominal gas-profile runs worsen squared km/s error relative to RAR**, and five of six worsen measurement-error-weighted chi-square. Five of six incremental-gain bootstrap intervals include zero.\n\n')
    f.write('''This descriptor is computed from a mass-model force component. It is not directly observed gas concentration and cannot yet identify a density/coherence mechanism. Different metrics reveal which galaxies carry the gains and losses; there is no universally improved fit here.

| Algorithm | Input addition | Fractional error gain vs RAR, A / B | km/s squared error gain vs RAR, A / B |
|---|---|---:|---:|
''')
    for algorithm in labels:
        for group,name in [('photometry_multiscale','Starlight surroundings'),('gas_force_proxy','Gas-force surroundings')]:
            a=next(r for r in allnom if r['algorithm']==algorithm and r['group']==group and r['salt']=='A')
            b=next(r for r in allnom if r['algorithm']==algorithm and r['group']==group and r['salt']=='B')
            f.write(f'| {labels[algorithm]} | {name} | {100*a["fractional_mse_gain"]:.2f}% / {100*b["fractional_mse_gain"]:.2f}% | {100*a["kms_mse_gain"]:.2f}% / {100*b["kms_mse_gain"]:.2f}% |\n')
    f.write('''
Negative gains mean worse predictions. The model was selected on fractional error; km/s and chi-square are additional checks, not alternate tuning targets after seeing the result.

## Does correct spatial ordering matter?

Shuffling only the added descriptors among radii within each galaxy preserves their galaxy-wide distributions but breaks their radial alignment. Eight such shuffles were run for each of two algorithms, two added-feature groups and two partitions, repeating inner parameter selection.

The real photometric ordering beats all eight shuffles in both algorithms and both partitions. The real gas-force ordering also does so except for GPU kernel partition A, where two shuffles perform at least as well. Alignment therefore carries information in this diagnostic, but that information need not provide a reliable improvement over omitting the features. The small, post-result shuffle experiment is not a discovery p-value and does not eliminate all coverage or source-reconstruction effects.

## Verification and limits

GPU/CPU filter and regression calculations agree to better than 1e-9. A masked constant field is preserved. Manufactured Gaussian beam convolution reproduces covariance addition to within 5e-7 relative error. Changing all observed speeds and errors leaves source feature matrices exactly unchanged. The original nominal RAR chi-square is reproduced. No candidate produced nonpositive predicted speeds.

The first prediction launch stopped before any model scoring because of a CuPy error-state API name. It is retained as campaign-001; campaign-002 fixes that call and completes. Map pilot-001 predates exact beam matching; the follow-up retains both results instead of overwriting the pilot.

The bootstrap resamples fixed outer predictions; it omits uncertainty from repeated model selection and earlier research choices. Published brightness zeros may involve censoring. Physical-radius descriptors inherit distance assumptions. Stellar mass brackets are not a full uncertainty analysis. We have not marginalized over inclination, distance, thickness, molecular-gas conversion, correlated noise, pressure support or noncircular motions.

## Decision

Keep projected gas-surroundings contrast as the next measurement target. Do not promote a coherence formula or run an unconstrained symbolic-equation hunt on these scores. The current data do not yet establish an observable-to-gravity relationship worth compressing into a new law.

Next acquisition should bind gas intensity/cube, mask, noise/beam information, stellar light and an independently usable kinematic product for multiple source-selected galaxies. Then compute the same angular/projected descriptors, propagate source/geometry uncertainties, and test predictions on genuinely unexposed galaxies. One NGC3198 map pilot cannot provide that population test. Recovering calibrated upper limits is essential before selecting physical voids.

## Sources and reproducibility

- THINGS measurement and processing paper: https://arxiv.org/abs/0810.2125
- SPARC source paper: https://arxiv.org/abs/1606.09251
- Local input hashes and masks: observable-map-pilot-001/map_audit.json
- Population registration, selected fold parameters and predictions: observable-pattern-campaign-002/
- Beam/aperture, shuffle and bootstrap diagnostics: observable-pattern-robustness-001/
- Geometry anchor: configs/gravity_map_axisymmetric_source_v3.json (sourced photometric center, angle and ellipticity)

No new gravitational law, independent astronomical confirmation, cluster prediction, solar-system prediction or 3D density reconstruction is claimed.
''')

evidence={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
    P/'result.json',P/'controls.json',M/'map_audit.json',R/'geometry.json',R/'permutations.json',R/'bootstrap.json',
    ROOT/'configs/gravity_map_axisymmetric_source_v3.json']}
summary={'status':'OBSERVABLE_DESCRIPTOR_ROBUST_PREDICTIVE_GRAVITY_RELATION_UNCONFIRMED','evidence_hashes':evidence,
    'photometry_incremental_wins_of_18':sbwins,'gas_force_proxy_incremental_wins_of_18':gaswins,
    'geometry':geo,'models':rows,'independent_confirmation':False,'admitted_laws':0}
(D/'result.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(D/'report_builder.py').write_bytes(Path(__file__).read_bytes())
for source,name in [(D/'report.md','Gravity-observable-pattern-findings.md'),
    (D/'model_scores.csv','Gravity-observable-pattern-scores.csv'),
    (D/'prediction_comparison.png','Gravity-observable-pattern-comparison.png'),
    (M/'NGC3198-multiscale.png','Gravity-NGC3198-observable-structure.png'),
    (D/'result.json','Gravity-observable-pattern-summary.json')]:shutil.copyfile(source,OUT/name)
registry=json.loads((ROOT/'configs/gravity_sigma_directions_v31.json').read_text())
registry['predecessor']='configs/gravity_sigma_directions_v31.json'
registry['observable_first_campaign']={
    'status':summary['status'],'evidence':str((D/'result.json').relative_to(ROOT)),
    'report':str((D/'report.md').relative_to(ROOT)),
    'findings':['Zero-filled HI blanking is not measured empty space.',
        'Projected gas contrast is robust to processing and specified aperture variations in one galaxy.',
        'Photometry adds inconsistent predictive value; calculated gas-force context helps fractional scores but fails other metrics.',
        'Next need is calibrated multi-galaxy source and kinematic products, not unconstrained equation search.'],
    'independent_confirmation':False,'admitted_laws':0}
path=ROOT/'configs/gravity_sigma_directions_v32.json';path.write_text(json.dumps(registry,indent=2),encoding='utf-8')
shutil.copyfile(path,OUT/'Sigma-gravity-directions-v32.json')
print(json.dumps({'output':str(OUT/'Gravity-observable-pattern-findings.md'),'photometry_wins':sbwins,'gas_proxy_wins':gaswins}))
