"""Report the constrained-cube, selection, astrometry and matter checkpoints."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parents[1]/'outputs'
def read(p):return json.loads((ROOT/p).read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-report',action='store_true');args=parser.parse_args()
    result=read('work/gravity-first-principles/constrained-cube-root-001/result.json')
    gas=read('work/gravity-first-principles/root-cube-gas-001/result.json')
    matter=read('work/gravity-first-principles/ngc2903-matter-002/result.json')
    mask=read('work/gravity-first-principles/constrained-cube-balanced-001/mask-audit.json')
    alignment=read('work/gravity-first-principles/stellar-gaia-alignment-001/NGC2903.json')
    relative=read('work/gravity-first-principles/ngc2903-matter-002/p1-p5-registration.json')
    assert len(result['objects'])==12 and not result['failures']
    D=ROOT/'work/gravity-first-principles/constrained-matter-findings-001';D.mkdir(exist_ok=args.refresh_report)
    shutil.copy2(__file__,D/'runner.py')
    rows=[];full=[]
    for obj in result['objects']:
        base=obj['fits'][0]['test_loss'];diagnostic=obj['geometry_diagnostic']
        for f in obj['fits']:
            row=dict(name=obj['name'],mode=f['mode'],train_loss=f['train_loss'],test_loss=f['test_loss'],
                ratio_to_rotation=f['test_loss']/base,converged=f['optimizer_success'])
            rows.append(row)
            if f['mode']=='full':full.append(dict(**row,**diagnostic))
    with (D/'scores.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    saved=list(csv.DictReader((ROOT/'work/gravity-first-principles/ngc2903-matter-002/selected-surface-matter.csv').open()))
    data={key:np.array([float(r[key]) for r in saved]) for key in saved[0]}
    fraction=data['sigma_atomic_with_helium']/data['sigma_total_nominal']
    lower_share=float(np.median(data['sigma_atomic_with_helium']/data['conditional_upper']))
    upper_share=float(np.median(data['sigma_atomic_with_helium']/data['conditional_lower']))
    summary=dict(status='VALIDATION_REPAIRS_AND_PROJECTED_SOURCE_FINDINGS_NOT_GRAVITY_ADMISSION',
        galaxies=12,selected_models=60,converged_selected_models=sum(r['converged'] for r in rows),
        combined_models_improving=sum(r['ratio_to_rotation']<1 for r in full),
        combined_possible_fold_objects=[r['name'] for r in full if r['possible_fold_fraction']>0],
        maximum_root_residual_arcsec=max(r['max_root_residual_arcsec'] for r in full),
        maximum_36_52_iteration_whitened_rms=max(r['root_36_vs_52_whitened_rms'] for r in full),
        gas_eligible=len(gas['predictions']),gas_nonzero_selected=sum(r['beta']!=0 for r in gas['predictions']),
        stellar_astrometric_pass=['NGC2903'],ngc2903_joint_positions=27,ngc2903_nominal_median_atomic_share=float(np.median(fraction)),
        atomic_share_sensitivity_medians=[lower_share,upper_share],independent_gravity_confirmation=False,
        total_3d_density_available=False,admitted_gravity_laws=0)
    save(D/'summary.json',summary)
    # Show the actual geometric extrapolation problem without comparing errors
    # across changed test samples.
    old=dict(np.load(ROOT/'work/private/conditional-cube-pilot-001/NGC3198.npz'))
    new=dict(np.load(ROOT/'work/private/constrained-cube-balanced-001/NGC3198.npz'))
    fig,axes=plt.subplots(2,2,figsize=(11,7.5))
    edges=[48,100,160,240,330,450]
    for j,(packet,label) in enumerate([(old,'Original split: inner/outer imbalance'),(new,'Repaired split: shared radial coverage')]):
        for key,color,title in [('train_mask','#3578b5','Training'),('test_mask','#dd8b35','Test')]:
            selected=packet[key]
            axes[0,j].scatter(packet['east'][selected],packet['north'][selected],s=10,c=color,label=title)
            counts=np.histogram(packet['radius'][selected],edges)[0]
            centers=np.arange(5)+(-.18 if key=='train_mask' else .18)
            axes[1,j].bar(centers,counts,width=.35,color=color,label=title)
        axes[0,j].set_title(label);axes[0,j].set_aspect('equal');axes[0,j].set_xlabel('East offset (arcsec)');axes[0,j].set_ylabel('North offset (arcsec)')
        axes[0,j].legend(fontsize=8)
        axes[1,j].set_xticks(range(5),['48–100','100–160','160–240','240–330','330–450']);axes[1,j].set_xlabel('Radius using reference geometry (arcsec)');axes[1,j].set_ylabel('Selected positions')
    for ax in axes[1]:ax.set_ylim(0,100)
    fig.suptitle('NGC3198: validating at radii the model has actually sampled',fontsize=14)
    fig.tight_layout();fig.savefig(D/'Gravity-next-step-mask.png',dpi=160);plt.close(fig)
    # Nominal fractions at every qualified point, with censored CO marked.
    g=next(a['geometry'] for a in read('work/gravity-first-principles/conditional-cube-pilot-001/data-audit.json') if a['name']=='NGC2903')
    pa=np.deg2rad(g['pa']);inc=np.deg2rad(g['inc']);e=data['east_arcsec'];n=data['north_arcsec']
    radius=np.hypot(e*np.sin(pa)+n*np.cos(pa),(e*np.cos(pa)-n*np.sin(pa))/np.cos(inc));order=np.argsort(radius)
    total=data['sigma_total_nominal'];stars=data['sigma_star_nominal']/total;hi=data['sigma_atomic_with_helium']/total;h2=data['sigma_h2_nominal']/total
    fig,ax=plt.subplots(figsize=(11,4.5));x=np.arange(len(order))
    ax.bar(x,100*stars[order],color='#6687bf',label='Stars')
    ax.bar(x,100*hi[order],bottom=100*stars[order],color='#3d9e84',label='Atomic gas + helium')
    molecular=ax.bar(x,100*h2[order],bottom=100*(stars+hi)[order],color='#db9b55',label='Molecular gas + helium (nominal)')
    nondetection=data['co21_signed']<3*data['co21_error_bound']
    for bar,censored in zip(molecular,nondetection[order]):
        if censored:bar.set_hatch('///')
    ax.set_ylim(0,100);ticks=np.arange(0,len(order),4);ax.set_xticks(ticks,[f'{radius[order[i]]:.0f}' for i in ticks])
    ax.set_xlabel('27 covered positions, ordered by reference radius (arcsec)');ax.set_ylabel('Share of nominal projected mass (%)')
    ax.set_title('NGC2903: atomic gas is a small part of the modeled mass here');ax.legend(loc='lower right',fontsize=8)
    fig.text(.02,.015,'Selected region only. Hatched CO contributions have upper-limit status. Conversion ranges are assumptions, not confidence intervals.',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(D/'Gravity-next-step-matter.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,6));labels=[r['name']+(' *' if r['possible_fold_fraction']>0 else '') for r in full]
    ax.scatter([r['ratio_to_rotation'] for r in full],labels,c=['#c67753' if r['possible_fold_fraction']>0 else '#46858d' for r in full],s=60)
    ax.set_xscale('log');ax.set_xticks([.5,1,2,4,6],['0.5×','1×','2×','4×','6×']);ax.axvline(1,color='#555',ls='--');ax.invert_yaxis();ax.set_xlabel('Combined-model error / circular-model error (same test positions)')
    ax.set_title('Additional components still do not consistently predict better')
    fig.text(.03,.015,'Below 1 = better; above 1 = worse. * Possible folded projection: not admitted as a unique physical disk.',fontsize=8)
    fig.tight_layout(rect=[0,.035,1,1]);fig.savefig(D/'Gravity-next-step-motion.png',dpi=160);plt.close(fig)
    table='\n'.join(f"| {r['name']} | {r['test_loss']:.2f} | {r['ratio_to_rotation']:.2f}× | {'flagged' if r['possible_fold_fraction']>0 else 'not flagged'} |" for r in full)
    gastext='; '.join(f"{r['name']}: beta={r['beta']:+.2f}" for r in gas['predictions'])
    report=f'''# Next diagnostic round: trustworthy geometry before gravity formulas

The most useful finding is that **a weak HI signal cannot stand in for low total matter density in our checked region of NGC2903**. The source work now combines stars, atomic gas and molecular-gas measurements, after correcting a real coordinate offset. The kinematic work removed artificial rotation reversals and repaired an important training/test design error. It still does not validate a new gravity formula.

## What changed in the motion model

The single-disk rotation curve now retains one rotation direction, starts at zero at the center, and has finer inner rings. Gas dispersion can vary with radius. Several fixed starting guesses are ranked only by training loss. Warps, radial streaming and broad lagging profiles remain separate candidates and are also combined. This is a conditional projected cube model; a realistic finite-thickness emitting disk with complete instrumental response remains necessary for a full physical interpretation.

The first constrained run revealed a selection problem. In NGC3198, the original training set had no positions below 240 arcsec, while most test points were inside that radius. A flexible inner rotation curve was therefore unconstrained. New block labels balance radius and projected rotation-side geometry, retain at least eight training and eight test points in each admitted radial bin, and preserve at least 120 arcsec between training and test centers. Mutation tests confirm that changing the velocity cube or intensity template cannot change these labels. Initial motion estimates were recomputed from the new training spectra; old fitted parameters were not reused across that change of split.

![Radial support repair](Gravity-next-step-mask.png)

The old and new test sets differ. Their absolute error scores must not be interpreted as a pure improvement in the physical model. Within the repaired split, all five candidate models use the same test positions.

## A numerical correction and a remaining physical ambiguity

The six-step approximation to warped-ring radius was insufficient in four galaxies. Even 20 steps remained insufficient in two. A bracketed root solver now determines radius and uses implicit differentiation during fitting. It agrees with an independent scalar Brent solver to about 0.00013 arcsec in the control and passes finite-difference gradient checks. All final objects passed root residual and 36-versus-52-step precision checks; the largest residual was {summary['maximum_root_residual_arcsec']:.6f} arcsec.

However, accurately solving one radius does not mean there is only one physically relevant intersection of the line of sight with a warped disk. A conservative scan flags possible folded projections in **{', '.join(summary['combined_possible_fold_objects']) or 'none of these final fits'}**. These cases require a model that sums emission through the disk rather than interpreting a single projected sheet. They are excluded from the final gas-term comparison.

| Galaxy | Combined test loss | Error relative to rotation | Projection-fold scan |
|---|---:|---:|---|
{table}

{summary['converged_selected_models']} of 60 selected model fits converged. The combined model improves the same-test-position score in {summary['combined_models_improving']} of 12 objects, but greater flexibility often worsens predictions. Do not choose a physical explanation just because its test score happens to win. Most scores retain substantial excess residuals. Channel covariance is included, but spatial covariance and background nonstationarity remain limitations; these are comparative losses, not calibrated chi-square significance. NGC5055 and NGC6946 had especially uneven background-noise validation in the earlier audit.

![Motion model comparison](Gravity-next-step-motion.png)

## Independent simulation checks

An independent NumPy generator produced analytic rotation and varying dispersion at twice the fitted spatial resolution. It also produced warped, streaming, and vertically layered lagging cases; the latter included Hanning channel smoothing. The streaming model reduced the known-streaming case from about 7.72 to 1.05 in withheld whitened loss. The warped case exposed radial-shape mismatch and optimizer limitations. The thick lagging case could be matched almost as well by a thin rotating model at the simulated resolution and noise. That is an identifiability warning: a good spectral fit alone cannot establish depth, thickness, or a unique flow mechanism.

## Stellar and molecular matter: an actual source-data advance

Gaia DR3 foreground stars supplied an independent positional check. Each field used separate calibration and validation stars to compare the ambiguous FITS SIP distortion interpretation with linear TAN coordinates. All 12 preferred TAN on calibration stars, but only NGC2903 passed the strict full gate. Its 67 catalog stars yielded a validation median offset of {alignment['validation_median_arcsec']:.2f} arcsec and a 90th percentile of {alignment['validation_p90_arcsec']:.2f} arcsec. Applying the ambiguous SIP coefficients gave a median validation offset of {alignment['modes']['header_sip']['validation_median_arcsec']:.2f} arcsec instead. Failed gates elsewhere can reflect undetected infrared counterparts, so they do not prove every other map is astrometrically wrong. Proper motions were propagated to an approximate source epoch; this remains part of the alignment uncertainty. [Gaia programmatic data access](https://www.cosmos.esa.int/web/gaia-users/archive/programmatic-access).

The NGC2903 cleaned stellar cutout also had a relative shift of (-3,-1) original-image pixels, about 2.37 arcsec. After applying that coordinate correction in memory, stellar plus nonstellar emission reconstructs the Gaia-checked image at {100*relative['relative_flux_rms']:.2f}% RMS on validation spatial blocks, after calibration-only scale/background adjustment. The original FITS files are unchanged. The failed first transfer and the source-only development diagnostic are retained.

All nonzero ICA mask labels, including negative labels, were excluded according to the publisher's documentation. HI, stars and CO were brought to a common nominal 48-arcsec Gaussian beam, accounting for their native beams. Only **27 of 242** original geometric positions pass the joint 98% support requirement. Blank or missing regions remain unsupported, never measured empty space. [S4G product and mask definitions](https://irsa.ipac.caltech.edu/data/SPITZER/S4G/docs/P5_README.html).

Within these 27 selected positions, HI plus associated helium contributes a median **{100*np.median(fraction):.1f}%** of the nominal modeled projected baryonic mass. Illustrative conversion/error extremes move that median to roughly **{100*lower_share:.1f}–{100*upper_share:.1f}%**. Stars dominate this selected region. This is not a whole-galaxy fraction or a calibrated confidence interval.

![Projected matter fractions](Gravity-next-step-matter.png)

The stellar conversion uses a nominal 3.6-micron mass-to-light ratio of 0.6, with 0.4 and 0.8 sensitivity cases. CO uses nominal alpha_CO=4.35 including helium and R21=0.65, with broader illustrative alternatives. These are source-model assumptions, not parameters fitted to the gas-motion test. [Stellar light-to-mass calibration](https://arxiv.org/abs/1402.5210), [CO conversion-factor review](https://arxiv.org/abs/1301.3498).

Twelve of the 27 positions have CO below the conservative three-error threshold. Their upper-limit information is retained, along with signed CO intensity. Smoothing the error map supplies a fully correlated-noise upper bound rather than assuming independent pixels. The HERACLES integration window partly uses the local HI mean velocity, so these tracer products are not completely independent of HI kinematics. [HERACLES release definitions](https://www.iram.fr/ILPA/LP001/README).

This is a **projected surface-matter pilot**, not a complete galactic mass distribution or a 3D volume-density measurement. Stellar and HI measurement errors, CO-dark gas and conversion uncertainty, source coverage, and line-of-sight thickness still limit a gravity calculation. In particular, a low HI patch is not enough to classify a region as a total-density void.

## The unchanged gas descriptor after these repairs

The same covered-HI descriptor and beta grid were repeated against the corrected baseline. Eligibility now also requires no flagged projection fold and passing numerical geometry checks. {len(gas['predictions'])} galaxies qualified; {summary['gas_nonzero_selected']} selected a nonzero beta when the coefficient was chosen using other galaxies' training spectra. {gastext}.

This is a limited development check of `v -> v*(1+beta*C_HI)`, with beta spacing 0.05 and frozen nuisance parameters. It is not a test of total-density coherence, and it does not rule out smaller coefficients, other source descriptors, or different physical field laws. Earlier zero-beta results from the flawed split and approximate geometry should not be promoted as decisive null evidence.

## What this means for the gravity search

We have repaired identifiable numerical and selection errors and obtained a checked multi-tracer source region. We have not established a universal anomalous-gravity pattern. The next physical requirement is a forward model that sums emission through a finite-thickness warped disk, including possible multiple intersections and the full instrumental response, then a source mass model with explicit coverage and uncertainty. A free rotation curve can absorb a smooth radial gravity enhancement; testing the gravity law itself requires that mass-constrained comparison. [3D tilted-ring fitting methodology](https://arxiv.org/abs/1505.07834).

Evidence directories preserve the original and repaired partitions, all optimizer starts, failed checks, source queries and hashes, root controls, and final descriptor predictions. Large raw observations and prepared arrays remain outside Git. The first-principles gravity goal remains unfinished; no law is admitted.
'''
    (D/'report.md').write_text(report,encoding='utf-8')
    registry=read('configs/gravity_sigma_directions_v34.json');registry['predecessor']='configs/gravity_sigma_directions_v34.json'
    registry['constrained_cube_and_matter_campaign']=dict(status=summary['status'],report=str((D/'report.md').relative_to(ROOT)),summary=str((D/'summary.json').relative_to(ROOT)),
        findings=['Radial extrapolation in the original spatial split was identified and repaired without using response values.',
        'Sign-preserving rotation, multistarts and accurate root geometry improve computational validity; possible folded projections still require full emitting-disk models.',
        'Gaia alignment and a P5 offset correction support a common-beam NGC2903 projected-source pilot; HI is a small fraction of nominal mass in its qualified region.'],
        independent_confirmation=False,admitted_laws=0)
    save(ROOT/'configs/gravity_sigma_directions_v35.json',registry)
    OUT.mkdir(exist_ok=True)
    for file,target in [('report.md','Gravity-next-step-findings.md'),('summary.json','Gravity-next-step-summary.json'),('scores.csv','Gravity-next-step-scores.csv')]:shutil.copy2(D/file,OUT/target)
    for file in D.glob('*.png'):shutil.copy2(file,OUT/file.name)
    shutil.copy2(ROOT/'configs/gravity_sigma_directions_v35.json',OUT/'Sigma-gravity-directions-v35.json')
    save(D/'verification.json',dict(status='REPORTED_FROM_FROZEN_EVIDENCE',files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.iterdir() if p.is_file() and p.name!='verification.json'}))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
