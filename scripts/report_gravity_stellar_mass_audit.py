"""Render the measured patterns and the limits of the zero-dark-matter mass audit."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_gravity_stellar_mass_audit import ROOT, OUT, save


def main(out,export):
    read=lambda name:json.loads((out/name).read_text(encoding='utf-8'))
    s=read('summary.json');v=read('verification.json');a=read('binary-anchor.json');g=read('galaxy-mass-only.json');m=read('manga-light-only.json')
    assert v['status']=='PASS_TARGETED_SOURCE_REPLAY_AND_ROBUSTNESS_CHECKS'
    primary=g['primary'];optical=v['optical_interpolation']
    ext=list(csv.DictReader((out/'binary-external-predictions.csv').open(encoding='utf-8')))
    galaxies=list(csv.DictReader((out/'galaxy-mass-requirements.csv').open(encoding='utf-8')))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','svg.hashsalt':'stellar-mass-audit-20260905'})
    fig,axes=plt.subplots(2,2,figsize=(13.5,10.2));fig.subplots_adjust(left=.085,right=.97,bottom=.145,top=.85,wspace=.28,hspace=.52)
    fig.suptitle('Stellar mass errors and the galaxy-outskirts problem',x=.085,ha='left',y=.97,fontsize=19,fontweight='bold')
    fig.text(.085,.92,'Newtonian mass audit with the dark-matter contribution fixed at zero',fontsize=12,color='#555555')
    ax=axes[0,0];keys=['V','V_color','K','K_color'];vals=[optical['metrics'][k]['median_absolute_fractional_error_percent'] for k in keys]
    bars=ax.bar(range(4),vals,color=['#8496ab','#5681ad','#257d79','#599995'],width=.65)
    for bar,val in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,val+.25,f'{val:.1f}%',ha='center',fontsize=12)
    ax.set(xticks=range(4),xticklabels=['Visible','Visible\n+ color','Infrared','Infrared\n+ color'],ylabel='Typical mass-prediction error (%)',ylim=(0,13),title='A. Infrared improves stellar weighing')
    ax.text(.02,.97,'26 stars within the training brightness range; whole binaries held out',transform=ax.transAxes,va='top',fontsize=8.5,color='#555555')
    ax=axes[0,1];actual=np.array([float(r['mass']) for r in ext]);pred=np.array([float(r['predicted_mass']) for r in ext]);line=np.linspace(0,.76,200)
    ax.fill_between(line,.9*line,1.1*line,color='#e4eeee',label='Within 10%');ax.plot(line,line,color='#555555',lw=1)
    ax.scatter(actual,pred,s=48,color='#257d79',edgecolors='white',linewidths=.6)
    ax.set(xlim=(.12,.75),ylim=(.12,.75),xlabel='Mass from a companion orbit (solar masses)',ylabel='Mass predicted from infrared light',title='B. Separate binary systems agree closely')
    ax.text(.04,.95,f"22 stars / 11 systems\nMedian error: {a['external_metrics']['median_absolute_fractional_error_percent']:.1f}%\nAll 22 within 10%",transform=ax.transAxes,va='top',fontsize=11)
    ax=axes[1,0];bins=list(reversed(primary['required_mass_by_acceleration']));values=[r['median_required_stellar_multiplier'] for r in bins]
    bars=ax.bar(range(4),values,color=['#257d79','#639b94','#c28f46','#b16b39'],width=.65)
    for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+.18,f'{value:.2f}x',ha='center',fontsize=12)
    ax.axhline(1,color='#555555',ls='--',lw=1);ax.set(ylim=(0,9.1),xticks=range(4),xticklabels=['Strongest','Stronger','Weaker','Weakest'],xlabel='Newtonian pull predicted by the ordinary-matter templates',ylabel='Required stellar-template mass multiplier',title='C. One mass correction cannot fit every regime')
    ax.text(.02,.97,'Median within each galaxy, then across galaxies in each bin',transform=ax.transAxes,va='top',fontsize=8.5,color='#555555')
    ax=axes[1,1];x=np.array([float(r['required_alpha_inner_third']) for r in galaxies]);y=np.array([float(r['required_alpha_outer_third']) for r in galaxies]);lo=min(x.min(),y.min())*.75;hi=max(x.max(),y.max())*1.35
    line=np.geomspace(lo,hi,100);ax.plot(line,line,color='#555555',lw=1,label='Same correction');ax.plot(line,2*line,color='#aaaaaa',ls='--',lw=1,label='Twice the correction')
    ax.scatter(x,y,s=20,color='#b16b39',alpha=.7);ax.set(xscale='log',yscale='log',xlim=(lo,hi),ylim=(lo,hi),xlabel='Required multiplier in inner third',ylabel='Required multiplier in outer third',title='D. The correction usually grows outward')
    ax.text(.04,.95,f"{primary['inner_to_outer_ratio']['above_one']}/86 above equal correction\nTypical outer / inner ratio: {primary['inner_to_outer_ratio']['median']:.2f}x",transform=ax.transAxes,va='top',fontsize=11)
    ax.legend(loc='lower right',frameon=False,fontsize=9)
    for ax in axes.flat:ax.grid(axis='y',color='#dddddd',alpha=.4);ax.set_axisbelow(True)
    fig.text(.085,.035,'Stellar K-band calibration and galaxy 3.6-micron mass-to-light ratios are different quantities.\nGalaxy multipliers rescale fixed stellar force templates; they are not independently measured local mass profiles.',fontsize=10,color='#555555')
    fig.savefig(out/'mass-audit-patterns.png',dpi=170)
    with (out/'mass-audit-patterns.svg').open('w',encoding='utf-8',newline='\n') as svg:
        fig.savefig(svg,format='svg',metadata={'Date':'2026-09-05'})
    svg_path=out/'mass-audit-patterns.svg'
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig)
    lines=[]
    for key,label in [('V','Visible light'),('V_color','Visible light + V-K color'),('K','Infrared K light'),('K_color','Infrared K light + V-K color')]:
        full=s['binary_optical']['metrics'][key];supported=optical['metrics'][key]
        lines.append(f"| {label} | {full['median_absolute_fractional_error_percent']:.1f}% | {supported['median_absolute_fractional_error_percent']:.1f}% |")
    manga_rows=[]
    for kind in ['ridge','trees']:
        cv={r['group']:r for r in m['cross_validation'][kind]['summary']};trans={r['group']:r for r in m['transport'][kind]['summary']}
        for feature in ['color','color_and_spectrum']:
            manga_rows.append(f"| {kind}; {feature.replace('_',' ')} | {cv[feature]['mse_gain_percent']:.1f}% | {trans[feature]['mse_gain_percent']:.1f}% |")
    sensrows=[]
    for name,value in g['sensitivities'].items():
        z=value['inner_to_outer_ratio'];sensrows.append(f"| {name} | {value['galaxies']} | {z['above_one']} | {z['median']:.2f}x |")
    bins_table=[]
    for b,label in zip(reversed(primary['required_mass_by_acceleration']),['Strongest: >= 1e-9 m/s²','Stronger: 1e-10 to 1e-9','Weaker: 1e-11 to 1e-10','Weakest: < 1e-11']):
        bins_table.append(f"| {label} | {b['galaxies']} | {b['median_required_stellar_multiplier']:.2f}x |")
    report=f"""# Stellar mass audit with dark matter fixed to zero

Completed 2026-09-05. The data support useful corrections from light and color, but this analysis does not find an individual-star mass error large enough to explain the galaxy results. The clearest galaxy pattern is that the required correction grows outward and as the predicted ordinary-matter pull becomes weaker.

This is a completed, scoped data audit. It does not establish that every ordinary-matter explanation has been tested, or that a new gravity formula has been found.

![Measured patterns](mass-audit-patterns.png)

## What was actually tested

The physical mass calculations use Newtonian gravity and a dark-matter contribution of exactly zero. No halo or empirical extra-force interpolation was fitted. Binary-star masses come from their mutual orbits. Galaxy rotation curves test the gravitational field of the whole galaxy; they do not weigh individual orbiting stars.

The audit uses 28 nearby stars with both visible and infrared measurements; 62 separate binary systems for a broader infrared calibration; 22 stars in 11 external eclipsing binaries for validation; 86 SPARC galaxies at 1,684 radii; and 585 MaNGA galaxies with transport to 243 other galaxies. These are published observations and existing project development sets. They are not a new, untouched observational confirmation campaign.

## Light and color do matter for stellar mass estimates

We fitted simple light-to-mass relations to the HST binary sample, always withholding both stars of the test binary. Degree and regularization were selected using training binaries only. The figures below are median absolute fractional mass errors, not uncertainty estimates for all stars.

| Inputs | All 28 held-out stars | 26 stars within training brightness range |
|---|---:|---:|
{chr(10).join(lines)}

Visible light alone is a worse predictor than infrared light in this sample. Adding V-K color helps the visible-light model. Adding that color to infrared light does not improve this small test. That last negative result prevents treating every extra color as automatically useful.

The two out-of-range cases are GJ22A and GJ1245C. The latter drives much of the all-sample squared-error penalty: the visible-only model overpredicts its mass by a factor of 2.83 when extrapolating beyond the training brightness range. This is an extrapolation failure, not evidence that the measured star is hiding that much mass. The 26-star comparison is an explicitly post-primary domain diagnostic; it leaves the original predictions and complete 28-star results intact. Source: [Benedict et al. 2016](https://arxiv.org/abs/1608.04775).

## Independent stellar mass checks show modest discrepancies

For the broader calibration we predicted the **sum** of the two component masses and compared it with the measured total orbital mass of each of 62 binary systems. We did not create component mass labels using a light-to-mass formula. A fixed fifth-degree relation in absolute Ks magnitude was fitted with monotonicity constraints and evaluated in five whole-system folds.

The held-system median mass error is {a['calibration_metrics']['median_absolute_fractional_error_percent']:.2f}%; its RMS fractional error is {a['calibration_metrics']['rms_fractional_error_percent']:.2f}%. Restricting the calibration to the 28 systems with orbital mass errors at most 5% gives a {a['quality_sensitivity']['baseline']['median_absolute_fractional_error_percent']:.2f}% median error and {a['quality_sensitivity']['baseline']['rms_fractional_error_percent']:.2f}% RMS error.

We then froze the fit and predicted 22 stars in 11 eclipsing binaries absent from the calibration systems. These stars span 0.174 to 0.690 solar masses. Their median mass error is **{a['external_metrics']['median_absolute_fractional_error_percent']:.2f}%**, RMS error **{a['external_metrics']['rms_fractional_error_percent']:.2f}%**, and all 22 predictions are within 10% of the orbital masses. The mean prediction bias is {a['external_metrics']['mean_predicted_minus_actual_percent']:.2f}%, too small to justify a multiple-fold increase in individual stellar masses in this sample.

That small bias is not a precision measurement of a universal correction: a coherent one-sigma parallax stress on the calibration systems moves the external predictions by about 4.4%. Absolute magnitude and orbital mass were moved together under that stress because both depend on distance. Resampling confidence intervals are conditional on the predictions and omit the full calibration posterior and systematic uncertainty.

Metallicity does not improve the tested infrared relation. The source flags two L-dwarf metallicities as extrapolated; removing those flags and comparing the same 60 systems gives a {v['valid_metallicity']['comparison']['mse_gain_percent']:.1f}% change in squared prediction error, with a paired interval spanning zero gain. This tests one simple metallicity term over the available nearby populations, not every chemical or activity effect.

External eclipsing masses are independent of the fitted relation, but their separated infrared brightnesses use spectral-template conversion of measured contrasts. This is not completely free of atmosphere/photometry assumptions. The validation stars were also checked in the original publication; our results are a reproducible reanalysis. Source: [Mann et al. 2019 and its data](https://arxiv.org/abs/1811.06938).

## Increasing every star's mass by one factor does not fit the galaxies

For SPARC, the fixed baseline is disk mass-to-light ratio 0.5 and bulge ratio 0.7 at 3.6 microns. With a common stellar multiplier alpha, the tested equation is:

`V_Newton² = gas_scale × Vgas × |Vgas| + alpha × (0.5 × Vdisk² + 0.7 × Vbulge²)`

The signed gas term is retained, including outward contributions from the gas geometry. The fitted alpha range is 0.05 to 40; no galaxy's best common multiplier reaches its upper limit. The ordinary gas template and the stellar geometry stay fixed in the primary run.

A global multiplier learned from other galaxies is **{min(primary['global_alpha_by_fold']):.2f} to {max(primary['global_alpha_by_fold']):.2f}x** across the validation folds. It still leaves a median sampled speed error of **{primary['global_held_galaxy']['median_absolute_fractional_error_percent']:.1f}%**. This is a required multiplier estimated from dynamics, not evidence that the stars truly weigh three times more.

Even assigning every galaxy its own best multiplier, using all of its observed radii, leaves a {primary['all_radii_per_galaxy_fit_diagnostic']['median_absolute_fractional_error_percent']:.1f}% median sampled speed error. Only {primary['per_galaxy_90percent_radii_within10percent_speed']} of 86 galaxies have at least 90% of sampled radii within 10% in speed. Allowing separate disk and bulge multipliers lowers the in-sample median error to {primary['all_radii_two_component_fit_diagnostic']['median_absolute_fractional_error_percent']:.1f}%. These fits are diagnostics, not independent validation.

The spatial pattern is more informative: **{primary['inner_to_outer_ratio']['above_one']} of 86 galaxies require a larger multiplier in their outer third than in their inner third**. The median outer-to-inner ratio is **{primary['inner_to_outer_ratio']['median']:.2f}x**. Calibrating a galaxy's multiplier only on its inner half leaves predictions in its outer half, on average across galaxies, **{abs(primary['outer_after_inner_calibration']['mean_predicted_minus_actual_percent']):.1f}% too slow**.

| Nominal ordinary-matter acceleration | Galaxies contributing | Required stellar-template multiplier |
|---|---:|---:|
{chr(10).join(bins_table)}

Each bin is summarized within a galaxy before taking a median over galaxies. A galaxy can contribute to several bins. These multipliers answer how much the **entire fixed stellar template** would have to be rescaled to match a particular radius. They are not measurements of local stellar mass density: a disk's gravity depends on matter at many radii.

The outward pattern survives the targeted checks:

| Check | Galaxies | Larger correction outside | Typical outer/inner ratio |
|---|---:|---:|---:|
{chr(10).join(sensrows)}

Doubling the observed atomic-gas template is a sensitivity test, not a measurement of missing molecular gas. Source: [SPARC observations and mass templates](https://arxiv.org/abs/1606.09251).

## Galaxy colors still predict motion when estimated mass is removed

We removed catalog stellar mass, mass surface density, inferred ages, specific star-formation rate, and the mass-size crossing proxy from the MaNGA model inputs. The baseline uses angular size, projected shape, light concentration, surface brightness, redshift, and signal-to-noise. We then add g-r color, or g-r plus measured spectral summaries.

| Model and extra inputs | Squared-error reduction: 585-galaxy folds | Reduction: separate 243-galaxy sample |
|---|---:|---:|
{chr(10).join(manga_rows)}

Color alone improves the separate-sample result by about 22–25%; replacing redshift with its logarithm to better represent distance scaling still gives about 21%. These percentages are reductions in prediction error, not percentages of missing mass. The response is the spread in stellar velocities, not ordered circular rotation.

This strengthens the case for studying stellar populations. It does not show that mass is misestimated by the amount needed for the SPARC curves. Both MaNGA samples were previously examined; original admission required a valid mass estimate; and spectra and velocity dispersion share a measurement pipeline. This uses available catalog color/spectral summaries, not every raw wavelength or a resolved, independently calibrated stellar census. [SDSS pipeline documentation](https://www.sdss4.org/dr17/manga/manga-analysis-pipeline/)

## What an ordinary-matter explanation still has to demonstrate

The individual-star Ks calibration cannot be multiplied into a galaxy's integrated 3.6-micron mass-to-light ratio. Different wavelengths, populations of faint stars, bright giants, remnants, dust, and spatial gradients intervene. We therefore did not invent such a transfer or call dynamics-fitted galaxy masses independent measurements.

The remaining material explanation needs independently supported changes in the **number, mix, or spatial distribution of stars and gas**, large enough to supply the required outer pull while keeping inner predictions correct. Nearby dwarf-star calibrations give no support here for multiplying every individual star's mass by three to eight. They also do not measure every galaxy's faint-star abundance or remnant inventory.

This audit does not refit a full velocity cube, warps, streaming, pressure support, distance/inclination errors, or their covariance. Nor does it inventory every molecular/hot-gas phase. Those limits leave the broader explanation unresolved. The completed result is narrower: light/color contain real predictive information, but a universal stellar-mass rescaling does not explain the observed galaxy profiles in this zero-dark-matter Newtonian model.

## Reproduction and checks

Source URLs, raw-file hashes, the exact protocol, predictions, galaxy-by-galaxy requirements, and verification results are stored beside this report. Raw observations remain in the ignored private cache. Verification reparses the original stellar source tables, checks binary identities and flux addition, replays all 1,684 Newtonian radius predictions and 828 MaNGA response values, tests that changing held-out targets leaves predictions unchanged, and recovers known injected 50% and threefold mass scales. Computation is table-sized and ran on CPU.

The first attempt stopped at a soft-monotonicity assertion in a binary sensitivity check. It is retained as an incomplete attempt in `stellar-mass-audit-001`; the completed run uses hard linear monotonicity constraints in `stellar-mass-audit-002`. A threaded-tree equality check was changed to an absolute 1e-12 tolerance to allow floating-point summation-order differences; measured changes are recorded in `verification.json`.

To reproduce into a fresh directory, run these scripts with a Python environment containing NumPy, SciPy, scikit-learn, Matplotlib, requests, and threadpoolctl:

```text
python scripts/acquire_gravity_mass_audit.py
python scripts/run_gravity_stellar_mass_audit.py --output work/gravity-first-principles/stellar-mass-audit-replay-001
python scripts/verify_gravity_stellar_mass_audit.py --output work/gravity-first-principles/stellar-mass-audit-replay-001
python scripts/report_gravity_stellar_mass_audit.py --output work/gravity-first-principles/stellar-mass-audit-replay-001
```
"""
    (out/'findings.md').write_text(report,encoding='utf-8',newline='\n')
    result=dict(status=s['status'],dark_matter_contribution=0,verification=v['status'],
                external_stellar_mass_median_error_percent=a['external_metrics']['median_absolute_fractional_error_percent'],
                external_stars=22,external_binary_systems=11,galaxies=86,radii=1684,
                galaxies_requiring_larger_outer_correction=primary['inner_to_outer_ratio']['above_one'],
                median_outer_inner_required_multiplier=primary['inner_to_outer_ratio']['median'],
                global_mass_multiplier_range=[min(primary['global_alpha_by_fold']),max(primary['global_alpha_by_fold'])],
                global_held_galaxy_median_sampled_speed_error_percent=primary['global_held_galaxy']['median_absolute_fractional_error_percent'],
                independent_integrated_galaxy_mass_calibration=False,
                conclusion='Individual-star color/mass calibration is useful but no large universal mass underestimate is supported by these binaries. The zero-dark-matter Newtonian galaxy models require larger, spatially varying corrections. Population counts and full kinematics remain unresolved.',
                report='findings.md',figure='mass-audit-patterns.png')
    save(out/'review-summary.json',result)
    files=['configs/gravity_stellar_mass_audit_v1.json','scripts/acquire_gravity_mass_audit.py','scripts/run_gravity_stellar_mass_audit.py','scripts/verify_gravity_stellar_mass_audit.py','scripts/report_gravity_stellar_mass_audit.py']
    save(out/'implementation-manifest.json',dict(files=[dict(path=p,sha256_LF=hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()) for p in files]))
    if export:
        export.mkdir(parents=True,exist_ok=True)
        shutil.copy2(out/'findings.md',export/'Gravity-stellar-mass-audit.md')
        shutil.copy2(out/'mass-audit-patterns.png',export/'mass-audit-patterns.png')
        shutil.copy2(out/'review-summary.json',export/'Gravity-stellar-mass-audit-summary.json')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--export-dir',type=Path)
    args=parser.parse_args();out=args.output if args.output.is_absolute() else ROOT/args.output
    main(out,args.export_dir)
