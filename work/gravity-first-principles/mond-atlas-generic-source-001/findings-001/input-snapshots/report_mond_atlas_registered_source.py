"""Review the frozen NGC2976 tracer pilot without reading motion responses."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_registered_source import source_coordinates,transfer_matrix,inclination
from build_mond_atlas_registered_source import load_sources,reference_wcs


def run(source,output):
    if output.exists():raise FileExistsError('immutable report exists')
    output.mkdir(parents=True)
    config=read_json(ROOT/'configs/mond_atlas_ngc2976_source_v1.json')
    summary=read_json(source/'summary.json')
    for path,expected in read_json(source/'artifact-hashes.json').items():
        if digest(ROOT/path)!=expected:raise ValueError('changed source result: '+path)
    for case in summary['cases']:
        if digest(ROOT/case['packet'])!=case['packet_sha256']:raise ValueError('changed source packet')
    inputs,p1,transfers,bindings,metadata=load_sources(config)
    sys.path.insert(0,str(ROOT/'tests'))
    from test_mond_atlas_registered_source import independent_mapping
    geometry=dict(config['geometry'],inclination_deg=inclination(config['geometry']))
    checks=[]
    for name,image,header,good,_,_ in inputs:
        iy,ix=np.nonzero(good);selection=np.linspace(0,len(ix)-1,101).astype(int)
        xy=np.column_stack((ix[selection],iy[selection]));step=1.
        for tr in (transfers if name=='stellar_luminosity' else [None]):
            transform=transfer_matrix(p1,tr['fit']['shift']) if tr else None
            actual=source_coordinates(header,xy,geometry,transform)
            # This independent helper uses Astropy positions plus trigonometric
            # gnomonic coordinates. Pass explicit two-axis integer header.
            hh=dict(reference_wcs(header).to_header());hh.update(NAXIS=2,NAXIS1=image.shape[1],NAXIS2=image.shape[0])
            h1=dict(reference_wcs(p1).to_header());h1.update(NAXIS=2,NAXIS1=int(p1['NAXIS1']),NAXIS2=int(p1['NAXIS2']))
            delta=[]
            for direction in ([step,0],[0,step]):
                a=independent_mapping(hh,xy+direction,geometry,h1 if tr else None,tr['fit']['shift'] if tr else None)
                b=independent_mapping(hh,xy-direction,geometry,h1 if tr else None,tr['fit']['shift'] if tr else None)
                delta.append(np.column_stack(((a[0]-b[0])/(2*step),(a[1]-b[1])/(2*step))))
            reference_area=abs(delta[0][:,0]*delta[1][:,1]-delta[0][:,1]*delta[1][:,0])*np.cos(np.deg2rad(geometry['inclination_deg']))
            area_error=float(np.max(abs(actual[2]/reference_area-1)))
            pixel_error=None
            if tr:
                v=actual[3];world=np.column_stack((np.rad2deg(np.arctan2(v[:,1],v[:,0]))%360,np.rad2deg(np.arcsin(v[:,2]))))
                w1=reference_wcs(p1);w5=reference_wcs(header)
                reference=w1.wcs_world2pix(w5.wcs_pix2world(xy,0),0)+tr['fit']['shift']
                pixel_error=float(np.max(abs(w1.wcs_world2pix(world,0)-reference)))
            passed=area_error<config['benchmarks']['projected_area_finite_difference_relative_max']
            passed=passed and (pixel_error is None or pixel_error<config['benchmarks']['astropy_composed_pixel_error_max'])
            checks.append(dict(component=name,translation=tr['fit']['shift'] if tr else None,
                               finite_difference_area_relative_max=area_error,composed_pixel_error_max=pixel_error,passed=passed))
    write_json(output/'actual-area-and-translation-checks.json',checks)
    if not all(row['passed'] for row in checks):raise ValueError('actual-source independent area/translation check failed')
    cases={row['id']:row for row in summary['cases']}
    with (source/'source-annuli.csv').open() as stream:annuli=list(csv.DictReader(stream))
    converged=[]
    for name in ('stellar_luminosity','atomic_helium','co21'):
        select=lambda case:[r for r in annuli if r['case']==case and r['component']==name and float(r['radius_kpc'])<config['source_grid']['cutoff_kpc']]
        ref=select('nominal')
        for case in ('pixels_one','pixels_two'):
            trial=select(case);a=[];b=[];weights=[]
            for r,t in zip(ref,trial):
                if r['signed_mean'] and t['signed_mean'] and r['accepted']=='True' and t['accepted']=='True':
                    a.append(float(r['signed_mean']));b.append(float(t['signed_mean']));weights.append(float(r['radius_kpc']))
            a=np.array(a);b=np.array(b);weights=np.array(weights)
            err=float(np.sum(weights*abs(a-b))/max(np.sum(weights*abs(a)),1e-30))
            converged.append(dict(component=name,comparison=case+'_vs_four',matched_annuli=len(a),annular_relative_l1=err,
                                  over_frozen_followup_flag=err>config['benchmarks']['actual_annular_convergence_relative_l1_flag'],
                                  map_relative_l1=cases[case]['comparison_to_nominal_on_same_angular_index'][name]['coordinate_matched_signed_relative_l1']))
    write_csv(output/'pixel-convergence.csv',converged)
    reference_mass=next(row['total_msun'] for row in summary['mass_cases'] if row['case']=='nominal' and row['conversion']=='nominal' and row['fill']=='annular')
    sensitivity=[]
    for row in summary['mass_cases']:
        if row['conversion']=='nominal' and row['fill']=='annular':
            sensitivity.append(dict(case=row['case'],total_msun=row['total_msun'],relative_to_nominal=row['total_msun']/reference_mass,
                                    inclination_deg=cases[row['case']]['geometry']['inclination_deg']))
    write_csv(output/'geometry-mass-sensitivity.csv',sensitivity)
    factors={'stellar_luminosity':.6,'atomic_helium':1.,'co21':4.35/.65}
    titles={'stellar_luminosity':'Cleaned stellar light × assumed M/L','atomic_helium':'Atomic hydrogen + helium','co21':'CO(2–1) × assumed conversion'}
    with np.load(ROOT/cases['nominal']['packet']) as packet:
        fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
        for col,name in enumerate(factors):
            axis=packet[name+'_axis'];d=axis[1]-axis[0];extent=[axis[0]-d/2,axis[-1]+d/2]*2
            coverage=packet[name+'_coverage'];mass=packet[name+'_mean']*factors[name]
            data=np.ma.masked_where((coverage<.5)|~np.isfinite(mass)|(mass<=0),mass)
            cmap=plt.get_cmap('magma').copy();cmap.set_bad('#c8c8c8')
            im=axes[0,col].imshow(data.T,origin='lower',extent=extent,cmap=cmap,norm=LogNorm(vmin=.1,vmax=300))
            axes[0,col].set_title(titles[name],fontsize=10)
            cm=axes[1,col].imshow(np.clip(coverage,0,1).T,origin='lower',extent=extent,cmap='viridis',vmin=0,vmax=1)
            axes[1,col].set_title('Usable coverage: '+format(cases['nominal']['components'][name]['observed_area_fraction_inside_cutoff'],'.1%'))
        for ax in axes.flat:ax.set(xlim=(-6,6),ylim=(-6,6),xlabel='Major axis (kpc)',ylabel='Deprojected minor axis (kpc)')
        fig.colorbar(im,ax=axes[0].tolist(),label='Conditional surface density (solar masses/pc²)',shrink=.85)
        fig.colorbar(cm,ax=axes[1].tolist(),label='Covered area fraction',shrink=.85)
        fig.suptitle('NGC2976: registered tracer grids under a fixed flat-disk geometry\nNative beams differ; gray indicates insufficient coverage or nonpositive values',fontsize=13)
        fig.savefig(output/'tracer-maps.png',dpi=145);plt.close(fig)
    selected=[]
    for case,conv,label in [('distance_low','nominal','Distance 2.904 Mpc'),('nominal','nominal','Nominal: 3.611 Mpc'),('distance_high','nominal','Distance 4.318 Mpc'),('nominal','low_conversion','Lower conversion assumptions'),('nominal','high_conversion','Higher conversion assumptions')]:
        row=next(r for r in summary['mass_cases'] if r['case']==case and r['conversion']==conv and r['fill']=='annular')
        selected.append((label,row))
    fig,ax=plt.subplots(figsize=(10,4.7),constrained_layout=True);left=np.zeros(len(selected))
    for key,label,color in [('stellar_msun','Stars','#bf693d'),('atomic_helium_msun','Atomic gas + He','#386fa4'),('molecular_helium_msun','Molecular gas + He','#60915e')]:
        values=np.array([r[key]/1e9 for _,r in selected]);ax.barh(np.arange(len(selected)),values,left=left,label=label,color=color);left+=values
    ax.set_yticks(np.arange(len(selected)),[label for label,_ in selected]);ax.invert_yaxis()
    ax.set_xlabel('Conditional mass in the same angular aperture (billion solar masses)');ax.legend(loc='lower right',fontsize=9)
    ax.set_title('NGC2976: source assumptions change the inferred mass\nAnnular fill and outer taper; illustrative alternatives, not confidence bounds')
    for i,total in enumerate(left):ax.text(total+.035,i,format(total,'.2f'),va='center',fontsize=9)
    ax.set_xlim(0,4.);fig.savefig(output/'mass-sensitivity.png',dpi=145);plt.close(fig)
    write_json(output/'summary.json',dict(status='SOURCE_SENSITIVITY_REVIEWED',object_id='NGC2976',actual_geometry_checks=checks,
               convergence=converged,mass_sensitivity=sensitivity,nominal_total_msun=reference_mass,
               low_conversion_total_msun=selected[3][1]['total_msun'],high_conversion_total_msun=selected[4][1]['total_msun'],
               bindings={str(p.relative_to(ROOT)):digest(p) for p in [source/'summary.json',Path(__file__),ROOT/'tests/test_mond_atlas_registered_source.py']},
               independent_motion_or_gravity_score=False,source_noise_likelihood_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.source.resolve(),args.output.resolve())
