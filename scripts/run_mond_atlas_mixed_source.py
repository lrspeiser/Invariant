"""Source-only mixed vertical populations, using the validated projection operator."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_source_projection import projection_matrix,project,fit_nonnegative,weighted_relative_rms


def run(config_path,output,private):
    config=read_json(config_path);protocol=read_json(ROOT/config['prior_projection_protocol'])
    prior=read_json(ROOT/config['prior_projection_run']);assert prior['all_optimizers_converged']
    assert digest(ROOT/config['prior_projection_protocol'])==prior['protocol_sha256']
    for path,expected in prior['code_hashes'].items():assert digest(ROOT/path)==expected
    audit=read_json(ROOT/protocol['source_audit']);packet_path=ROOT/audit['source_packet'];assert digest(packet_path)==audit['source_packet_sha256']
    if output.exists() or private.exists():raise FileExistsError('immutable output')
    output.mkdir(parents=True);private.mkdir(parents=True)
    with np.load(packet_path) as f:packet={k:f[k] for k in f.files}
    component=config['source_component'];axis=packet[component+'_axis'];d=axis[1]-axis[0];x,y=np.meshgrid(axis,axis,indexing='ij');r=np.hypot(x,y)
    mean=packet[component+'_mean'];coverage=packet[component+'_coverage'];cfg=protocol['source_fit']
    valid=np.isfinite(mean)&(coverage>=cfg['minimum_cell_coverage']);target=np.where(valid,mean,0)
    weight=np.where(valid&(r<cfg['fitted_radius_kpc']),np.clip(coverage,0,1),0)
    evaluation=np.where(valid&(r<cfg['reported_radius_kpc']),np.clip(coverage,0,1),0);support=r<cfg['source_support_radius_kpc']
    heights=config['vertical_components_kpc'];inc=audit['protocol']['geometry']['inclination_deg']
    operators=[projection_matrix(len(axis),d,h,inc) for h in heights];rows=[];products=[]
    for fraction in config['thin_light_fraction_cases']:
        label='thin-fraction-'+str(fraction).replace('.','p');matrix=fraction*operators[0]+(1-fraction)*operators[1]
        recovered,fit=fit_nonnegative(target,weight,matrix,support,cfg['regularization'],cfg['max_iterations'],cfg['projected_gradient_relative_rms_tolerance'])
        predicted=project(recovered,matrix)
        # Verify superposed 3D light projects by the independent component route.
        separately=fraction*project(recovered,operators[0])+(1-fraction)*project(recovered,operators[1])
        assert np.allclose(predicted,separately,rtol=1e-12,atol=1e-10)
        rms=weighted_relative_rms(predicted,target,evaluation);path=private/(label+'.npz')
        np.savez_compressed(path,axis=axis,intrinsic_effective_surface=recovered,projected_surface=predicted,source_mean=mean,
            evaluation_weight=evaluation,thin_light_fraction=fraction,heights_kpc=heights)
        result=dict(thin_light_fraction=fraction,thin_height_kpc=heights[0],thick_height_kpc=heights[1],
            source_image_relative_rms=rms,optimizer_converged=fit['converged'],iterations=fit['iterations'],
            recovered_light_integral_lsun=float(recovered.sum()*d*d*1e6),
            finite_field_projection_flux_fraction=float(predicted.sum()/recovered.sum()),
            observed_height_inferred=False,noise_calibrated_likelihood=False)
        rows.append(result);products.append(dict(path=str(path.relative_to(ROOT)),sha256=digest(path)))
        write_json(output/(label+'-optimizer.json'),fit);print(result,flush=True)
    write_csv(output/'mixed-height-source-closure.csv',rows)
    write_json(output/'summary.json',dict(status='SOURCE_ONLY_MIXED_HEIGHT_DIAGNOSTIC',admission_disposition='SOURCE_BLOCKED',
        config=config,config_sha256=digest(config_path),source_cases_executed=len(rows),
        all_optimizers_converged=all(r['optimizer_converged'] for r in rows),products=products,
        source_bindings={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/config['prior_projection_protocol'],ROOT/config['prior_projection_run'],ROOT/protocol['source_audit'])},
        code_hashes={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),ROOT/'scripts/mond_atlas_source_projection.py')},
        response_files_opened=[],kinematic_response_scores_computed=0,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_mixed_source_v1.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--private',type=Path,required=True)
    a=p.parse_args();run(a.config.resolve(),a.output.resolve(),a.private.resolve())
