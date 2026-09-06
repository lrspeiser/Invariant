"""Replay and review NGC3198 packets; no new source or response reconstruction."""
from __future__ import annotations
import csv
import json
import os
import sys
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mond_atlas_common import read_json,write_json,write_csv,digest
from build_mond_atlas_ngc3198_source_checked import verify_bindings

P=Path(__file__).resolve().parent
COMPONENTS=('stellar_luminosity','atomic_helium','co21')


def main():
    out=P/'findings-001'
    if out.exists():raise FileExistsError('immutable report already exists')
    verify_bindings(read_json(P/'freeze.json')['bindings'])
    source=P/'run-001';verify_bindings(read_json(source/'artifact-hashes.json'))
    c=read_json(ROOT/'configs/mond_atlas_ngc3198_source_v1.json');s=read_json(source/'summary.json')
    cases={r['id']:r for r in s['cases']};out.mkdir()
    checks=[];negative=[]
    for case in s['cases']:
        if digest(ROOT/case['packet'])!=case['packet_sha256']:raise ValueError('changed packet')
        h=case['grid']['spacing_kpc'];g=case['grid']
        with np.load(ROOT/case['packet']) as p:
            for component in COMPONENTS:
                r=case['components'][component];axis=p[component+'_axis'];xx,yy=np.meshgrid(axis,axis,indexing='ij');rad=np.hypot(xx,yy)
                taper=np.clip((g['cutoff_kpc']-rad)/(g['cutoff_kpc']-g['taper_start_kpc']),0,1)
                observed=p[component+'_observed'];coverage=p[component+'_coverage']
                calc=dict(untapered_in_field_signed_integral=float(observed.sum()*h*h*1e6),signed_measured_integral=float((observed*taper).sum()*h*h*1e6),negative_projection_added_integral=float((np.maximum(-observed,0)*taper).sum()*h*h*1e6),conditional_zero_integral=float(p[component+'_zero'].sum()*h*h*1e6),conditional_annular_integral=float(p[component+'_annular'].sum()*h*h*1e6))
                errors={k:abs(v-r[k])/max(abs(v),abs(r[k]),1) for k,v in calc.items()}
                errors['full_field_loss_closure']=abs(r['input_signed_integral']-r['outside_field_signed_integral']-calc['untapered_in_field_signed_integral'])/max(abs(r['input_signed_integral']),1)
                errors['signed_plus_negative_projection']=abs(calc['signed_measured_integral']+calc['negative_projection_added_integral']-calc['conditional_zero_integral'])/max(abs(calc['conditional_zero_integral']),1)
                if max(errors.values())>c['benchmarks']['conservative_sum_relative_max']:raise ValueError('packet integral replay failed')
                checks.append(dict(case=case['id'],component=component,relative_errors=errors,passed=True))
                negative.append(dict(case=case['id'],component=component,negative_observed_cells_inside_cutoff=int(np.sum((observed<0)&(rad<g['cutoff_kpc']))),coverage_above_one_cells_inside_cutoff=int(np.sum((coverage>1)&(rad<g['cutoff_kpc']))),maximum_area_coverage=float(coverage.max())))
            for conv in c['mass_conversion_cases']:
                for fill in ('zero','annular'):
                    actual=sum(float(p[name+'_'+fill].sum()*h*h*1e6)*f for name,f in zip(COMPONENTS,[conv['stellar_ml'],1.,conv['alpha_co10']/conv['r21']]))
                    row=next(r for r in s['mass_cases'] if r['case']==case['id'] and r['conversion']==conv['id'] and r['fill']==fill)
                    if abs(actual/row['total_msun']-1)>1e-12:raise ValueError('mass replay failed')
    distance=[]
    for name in ('distance_low','distance_high'):
        ratio=(cases[name]['geometry']['distance_mpc']/cases['nominal']['geometry']['distance_mpc'])**2
        for r in [x for x in s['mass_cases'] if x['case']==name]:
            ref=next(x for x in s['mass_cases'] if x['case']=='nominal' and x['conversion']==r['conversion'] and x['fill']==r['fill'])
            err=abs(r['total_msun']/ref['total_msun']/ratio-1)
            distance.append(dict(case=name,conversion=r['conversion'],fill=r['fill'],expected_ratio=ratio,relative_error=err))
            if err>c['benchmarks']['distance_scaling_relative_max']:raise ValueError('distance scaling failed')
    with (source/'source-annuli.csv').open() as f:annuli=list(csv.DictReader(f))
    convergence=[]
    for name in COMPONENTS:
        select=lambda case:[r for r in annuli if r['case']==case and r['component']==name and float(r['radius_kpc'])<c['source_grid']['cutoff_kpc']]
        reference=select('nominal')
        for case in ('pixels_one','pixels_two'):
            a=[];b=[];weights=[]
            for r,t in zip(reference,select(case)):
                if r['signed_mean'] and t['signed_mean'] and r['accepted']=='True' and t['accepted']=='True':
                    a.append(float(r['signed_mean']));b.append(float(t['signed_mean']));weights.append(float(r['radius_kpc']))
            a,b,w=map(np.array,(a,b,weights));err=float(np.sum(w*abs(a-b))/max(np.sum(w*abs(a)),1e-30))
            convergence.append(dict(component=name,comparison=case+'_vs_four',matched_annuli=len(a),annular_relative_l1=err,over_frozen_followup_flag=err>c['benchmarks']['actual_annular_convergence_relative_l1_flag'],map_relative_l1=cases[case]['comparison_to_nominal_on_same_angular_index'][name]['coordinate_matched_signed_relative_l1']))
    nom=next(r for r in s['mass_cases'] if r['case']=='nominal' and r['conversion']=='nominal' and r['fill']=='annular')
    sensitivity=[dict(case=r['case'],total_msun=r['total_msun'],relative_to_nominal=r['total_msun']/nom['total_msun'],inclination_deg=cases[r['case']]['geometry']['inclination_deg']) for r in s['mass_cases'] if r['conversion']=='nominal' and r['fill']=='annular']
    truncation=[]
    for name,r in cases['nominal']['components'].items():
        inp=r['input_signed_integral'];truncation.append(dict(component=name,full_input_signed_integral=inp,outside_grid_signed_integral=r['outside_field_signed_integral'],outside_grid_signed_fraction=r['outside_field_signed_integral']/inp,in_grid_taper_removed_signed_integral=r['untapered_in_field_signed_integral']-r['signed_measured_integral'],full_input_to_tapered_signed_loss_fraction=1-r['signed_measured_integral']/inp,coverage_fraction=r['observed_area_fraction_inside_cutoff'],negative_projection_added_integral=r['negative_projection_added_integral'],annular_minus_zero_integral=r['conditional_annular_integral']-r['conditional_zero_integral']))
    write_json(out/'packet-replay.json',dict(passed=True,component_cases=checks,mass_rows_replayed=72,distance_scaling=distance,source_response_arrays_opened=0,raw_source_arrays_reopened=False))
    write_csv(out/'pixel-convergence.csv',convergence);write_csv(out/'geometry-mass-sensitivity.csv',sensitivity);write_csv(out/'aperture-losses.csv',truncation);write_csv(out/'signed-and-coverage-diagnostics.csv',negative)
    factors=[.6,1.,4.35/.65];titles=['Stellar light × assumed M/L','Atomic gas + helium','CO(2–1) × assumed conversion']
    with np.load(ROOT/cases['nominal']['packet']) as packet:
        fig,axes=plt.subplots(2,3,figsize=(13,8.4),constrained_layout=True)
        for j,(name,factor) in enumerate(zip(COMPONENTS,factors)):
            axis=packet[name+'_axis'];d=axis[1]-axis[0];extent=[axis[0]-d/2,axis[-1]+d/2]*2
            coverage=packet[name+'_coverage'];mass=packet[name+'_mean']*factor
            data=np.ma.masked_where((coverage<.5)|~np.isfinite(mass)|(mass<=0),mass)
            cmap=plt.get_cmap('magma').copy();cmap.set_bad('#cccccc')
            im=axes[0,j].imshow(data.T,origin='lower',extent=extent,cmap=cmap,norm=LogNorm(vmin=.1,vmax=300))
            axes[0,j].set_title(titles[j],fontsize=10)
            cv=axes[1,j].imshow(coverage.T,origin='lower',extent=extent,cmap='viridis',vmin=0,vmax=1.5)
            axes[1,j].set_title('Covered area inside cutoff: '+format(cases['nominal']['components'][name]['observed_area_fraction_inside_cutoff'],'.1%'),fontsize=10)
            for ax in axes[:,j]:
                ax.add_patch(plt.Circle((0,0),28,fill=False,color='#50a9dc',linestyle='--',linewidth=.8))
                ax.set(xlim=(-30,30),ylim=(-30,30),xlabel='Major axis (kpc)',ylabel='Deprojected minor axis (kpc)')
        fig.colorbar(im,ax=axes[0].tolist(),label='Conditional density (solar masses/pc²)',shrink=.8)
        fig.colorbar(cv,ax=axes[1].tolist(),label='Quadrature-estimated coverage (may exceed 1)',shrink=.8)
        fig.suptitle('NGC3198: conditional flat-disk source grids\nNative beams differ; gray = insufficient coverage or nonpositive value; dashed circle = cutoff',fontsize=12)
        fig.savefig(out/'tracer-maps.png',dpi=145);plt.close(fig)
    choices=[('distance_low','nominal','Distance 12.355 Mpc'),('nominal','nominal','Nominal: 13.987 Mpc'),('distance_high','nominal','Distance 15.619 Mpc'),('nominal','low_conversion','Lower conversions'),('nominal','high_conversion','Higher conversions')]
    selected=[(label,next(r for r in s['mass_cases'] if r['case']==case and r['conversion']==conv and r['fill']=='annular')) for case,conv,label in choices]
    fig,ax=plt.subplots(figsize=(10,4.8),constrained_layout=True);left=np.zeros(len(selected))
    for key,label,color in [('stellar_msun','Stars','#bf693d'),('atomic_helium_msun','Atomic gas + He','#386fa4'),('molecular_helium_msun','Molecular gas + He','#60915e')]:
        values=np.array([r[key]/1e9 for _,r in selected]);ax.barh(np.arange(len(selected)),values,left=left,label=label,color=color);left+=values
    ax.set_yticks(np.arange(len(selected)),[label for label,_ in selected]);ax.invert_yaxis();ax.set_xlim(0,left.max()*1.12)
    for i,total in enumerate(left):ax.text(total+left.max()*.01,i,f'{total:.2f}',va='center',fontsize=9)
    ax.set_xlabel('Conditional tapered-aperture mass (billion solar masses)');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3,fontsize=9)
    ax.set_title('NGC3198: sensitivity to distance and conversion assumptions\nAnnular fill; same angular aperture; these alternatives are not confidence bounds')
    fig.savefig(out/'mass-sensitivity.png',dpi=145);plt.close(fig)
    write_json(out/'summary.json',dict(status='EXECUTED_AND_PACKET_REPLAYED',disposition='SOURCE_BLOCKED',object_id='NGC3198',geometry_registration_cases=12,component_cases=36,mass_conversion_fill_rows=72,nominal_mass=nom,convergence=convergence,aperture_losses=truncation,sensitivity=sensitivity,new_gravity_scores=0,observed_response_arrays_opened=0,gpu_used=False,numerical_threads=1,bindings={p.relative_to(ROOT).as_posix():digest(p) for p in [Path(__file__),source/'summary.json',P/'freeze.json']}))
    lines=['# NGC3198 executed source-only pilot','','Disposition: **SOURCE_BLOCKED**. Twelve registered geometry/quadrature cases, 36 component packets and 72 conversion/fill mass rows executed using the unchanged generic builder. One CPU numerical thread; no GPU, new downloads, motion/lensing response arrays or gravity scores. These are repeated conditional representations of the same observations, not independent measurements.','',
    'Nine independent tests pass before source construction and again inside the unchanged runner. Thirty actual-header checks pass before construction; four checks on actual supported image pixels also pass before rebinning. All 12 saved packets and 72 mass rows replay; D² scaling passes with the angular aperture held fixed.','',
    f"Nominal annular-fill conditional mass: **{nom['total_msun']/1e9:.3f} billion solar masses**: stars {nom['stellar_msun']/1e9:.3f}, atomic gas including helium {nom['atomic_helium_msun']/1e9:.3f}, molecular gas including helium {nom['molecular_helium_msun']/1e9:.3f}. Fixed assumptions are M/L=.6, alpha_CO10=4.35 and R21=.65. These are aperture integrals, not whole-galaxy masses.",'',
    '| Source | Covered area inside 28 kpc | Signed loss outside grid | Full-input loss after taper |','|---|---:|---:|---:|']
    for r in truncation:lines.append(f"| {r['component']} | {r['coverage_fraction']:.2%} | {r['outside_grid_signed_fraction']:.2%} | {r['full_input_to_tapered_signed_loss_fraction']:.2%} |")
    lines+=['','Loss fractions use signed integrals. The photometric P4 outer isophote (375 arcsec, 25.429 kpc) sets the 28 kpc cutoff, 24–28 kpc taper and +/-32 kpc axis-center field. No motion extent selected the aperture. The gas loss is material and cannot be treated as empty exterior space.','', '| Component | One vs four subdivisions | Two vs four subdivisions |','|---|---:|---:|']
    for name in COMPONENTS:
        r=[x for x in convergence if x['component']==name];lines.append(f"| {name} | {r[0]['annular_relative_l1']:.3%}{' FLAG' if r[0]['over_frozen_followup_flag'] else ''} | {r[1]['annular_relative_l1']:.3%}{' FLAG' if r[1]['over_frozen_followup_flag'] else ''} |")
    lines+=['','These compare matched accepted annular signed means, radius weighted, against four subdivisions. The unchanged 3% threshold is a follow-up flag. Cell-scale L1 changes and coverage above one remain in CSV diagnostics; passing an annular comparison does not certify cell-scale convergence or observation resolution. No extra reconstruction is performed to remove flags.','',
    'Signed negative stellar and CO measurements are preserved in observed grids. Conditional nonnegative projection adds '+f"{cases['nominal']['components']['co21']['negative_projection_added_integral']*4.35/.65/1e9:.4f} billion solar masses to the CO-derived zero-fill mass"+' under nominal conversion. Annular versus zero fill is an alternative coverage treatment, not an uncertainty interval. Native CO EMOM0 is an area-weighted diagnostic, not propagated covariance.','',
    'Both P5-to-P1 checkerboard transfer receipts pass; only their shifts are applied. Original P1 Gaia all-catalog failure is retained alongside the later finite-footprint strict pass (9 validation stars; median .257 arcsec, p90 .317 arcsec). Core TAN explicitly omits inherited SIP. Registration scale/background do not calibrate stellar mass.','',
    'P5 STELLAR_MASS_MAP is cleaned flux in MJy/sr; the ICA mask has inherited but semantically incorrect intensity units. HI uses the original CLEAN beam history (11.43108 × 9.36252 arcsec); CO header beam is 13.396779 arcsec. Native beams differ and are neither matched nor deconvolved. Missing/blanked support is not measured zero. HERACLES masking has prior HI-velocity dependence even though this pilot opens no velocities.','',
    'Flat-disk inclination is 71.923 degrees at nominal q0=.13. Warp, bulge, depth, source covariance, calibration, missing baryonic phases and exterior mass remain unresolved. No unique 3D source or observational response admission follows. The generic runner’s prior_initial_control_failure field refers to the preserved historical NGC2976 uniform-box test repair, not a failed NGC3198 case.','',
    'Reproduce from the repository root with Python313:','', '```powershell', '$env:PYTHONDONTWRITEBYTECODE="1"', 'python -B scripts/build_mond_atlas_ngc3198_source_checked.py verify-freeze', '# Frozen settings; select a never-used run-NNN pair for a replay:', 'python -B scripts/build_mond_atlas_ngc3198_source_checked.py run --run-id run-002', '```','',
    'The saved run-001 and findings-001 are immutable. The checked runner verifies all new bindings plus the unchanged legacy generic-source-001 freeze. Initial preflight is already executed and may not overwrite its receipts. report_source.py reproduces findings in a fresh checkout with no findings-001. No further run is launched by this report.','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(nominal=nom,convergence=convergence,truncation=truncation),indent=2))


if __name__=='__main__':main()
