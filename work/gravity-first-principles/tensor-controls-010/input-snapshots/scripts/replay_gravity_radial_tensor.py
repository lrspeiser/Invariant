"""Canonical source and all quadrature cases on the refined radial mesh."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from audit_gravity_matched_tensor import (
    derivative_check,
    maxima,
    norm,
    scales,
    serial,
    source_errors,
)

from invariant_gravity_extensions.exterior_moments import ExteriorMomentField
from invariant_gravity_extensions.length_galaxy_development import regular_disks
from invariant_gravity_extensions.matched_tensor import MatchedTensorPotential
from invariant_gravity_extensions.mixed_source import hankel_mixed_jet, leading_tail_mixed_jet
from invariant_gravity_extensions.vertical_green import Sech2VerticalGreen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    base = ROOT/'work/gravity-first-principles'
    qpath = ROOT/'configs/gravity_tensor_quadrature_v1.json'
    registered = json.loads(qpath.read_bytes())
    folders = {'radial_coarse':'tensor-quadrature-radial-001','wavenumber_coarse':'tensor-quadrature-wavenumber-001',
        'cutoff_200':'tensor-quadrature-cutoff-001','vertical_coarse':'tensor-quadrature-vertical-001','tail_extent':'tensor-quadrature-extent-001'}
    canonical = {'id':'canonical','transform':'work/gravity-first-principles/potential-join-001/transform_r128_k64.json',
        'cutoff':400,'vertical_intervals':2400,'vertical_extent':24}
    cases = [canonical,*registered['cases']]
    variants = ['primary','height_half']
    units_path,profile_path = base/'map-source-003/result.json',base/'map-source-003/source_profiles.json'
    previous_path = base/'matched-tensor-002/result.json'
    registry_path = ROOT/'configs/gravity_sigma_directions_v17.json'
    registry = json.loads(registry_path.read_bytes())
    direction = next(d for d in registry['directions'] if d['id']=='invariant_length_screening')
    expected_hashes = {**direction['tensor_quadrature_checkpoint']['result_hashes'],
        **direction['cutoff_refinement_checkpoint']['result_hashes'],**direction['matched_tensor_checkpoint']['result_hashes']}
    paths = {Path(__file__),ROOT/'scripts/audit_gravity_matched_tensor.py',qpath,units_path,profile_path,previous_path,registry_path,
        *[ROOT/c['transform'] for c in cases],*list((ROOT/'src/invariant_gravity_extensions').glob('*.py'))}
    for folder in [*folders.values(),'tensor-cutoff-refinement-001']:
        p = base/folder/'result.json'
        if sha256(p.read_bytes()).hexdigest()!=expected_hashes[folder]:
            raise ValueError(f'Retained result changed: {folder}')
        paths.add(p)
    for variant in variants:
        pilot = 'tensor-source-003' if variant=='primary' else 'tensor-source-004'
        paths.add(base/pilot/'mixed_table.json')
        paths.add(base/f'exterior-moment-002/moments_{variant}_reference.json')
        paths.add(base/f'tensor-cutoff-refinement-001/mixed_{variant}.json')
        paths.add(base/f'tensor-cutoff-refinement-001/fields_{variant}_radial_only.json')
        paths.update(base/folder/f'mixed_{variant}.json' for folder in folders.values())
    hashes = {p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    for p in sorted(paths):
        target = args.output/'input-snapshots'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())
    config = {'registration':'After radial-only cutoff repair, before full replay. Generate only added radial nodes; preserve all original mixed samples bit for bit. All six cases and both thicknesses are required.',
        'cases':cases,'targets':registered['targets'],'probe_entries_per_thickness':2842,
        'derivative_steps':[.001,.0005],'fine_derivative_target':1e-4,
        'derivative_rule':'Retain old derivative probes and add every new radial boundary at the old heights; central diagnostics retained, both one-sided radial stencils qualify interfaces',
        'cutoff_200_new_nodes':'Reuse retained radial-refinement samples, not a new integration',
        'scope':'Sampled numerical admission for a development field solve, not uniform error bounds or an observational/theory validation',
        'new_observational_scores':0}

    def write(name,value):
        with (args.output/name).open('x',encoding='utf8',newline='\n') as handle:
            json.dump(serial(value),handle,indent=2,allow_nan=False)
            handle.write('\n')

    write('started.json',{'config':config,'input_hashes':hashes,'started_utc':datetime.now(UTC).isoformat()})
    try:
        profile = json.loads(profile_path.read_bytes())['profiles'][-1]
        G = json.loads(units_path.read_bytes())['config']['units']['G_kpc_kms2_msun']
        previous = json.loads(previous_path.read_bytes())['config']
        providers,source_data = {},{}
        for variant in variants:
            _,disks = regular_disks(profile,{'id':variant,**({'height_factor':.5} if variant=='height_half' else {})})
            moments = json.loads((base/f'exterior-moment-002/moments_{variant}_reference.json').read_bytes())
            exterior = ExteriorMomentField(moments,G,minimum_radius=60.)
            probes = json.loads((base/f'tensor-cutoff-refinement-001/fields_{variant}_radial_only.json').read_bytes())
            source_data[variant] = (disks,moments,exterior,np.array(probes['R_kpc']),np.array(probes['z_kpc']))
        cache,rows = {},[]
        for case in cases:
            raw = json.loads((ROOT/case['transform']).read_bytes())
            k,w,S = [np.array(raw[key]) for key in ['k','wavenumber_weights','surface_hankel']]
            keep = k < case['cutoff']
            k,w,S = k[keep],w[keep],S[:,keep]
            key = (sha256(k.tobytes()).hexdigest(),case['vertical_intervals'],case['vertical_extent'])
            if key not in cache:
                cache = {key:{}}
            green = Sech2VerticalGreen(intervals=case['vertical_intervals'],extent=case['vertical_extent'])
            for variant in variants:
                disks,moments,exterior,R,Z = source_data[variant]
                old_path = base/folders.get(case['id'],'tensor-quadrature-cutoff-001')/f'mixed_{variant}.json'
                old_raw = json.loads(old_path.read_bytes())
                r,z,old = [np.array(old_raw[k]) for k in ['radius_kpc','height_kpc','mixed']]
                if case['id']=='canonical':
                    pilot = 'tensor-source-003' if variant=='primary' else 'tensor-source-004'
                    ref = json.loads((base/pilot/'mixed_table.json').read_bytes())
                    ri,zi = np.searchsorted(ref['radius_kpc'],r),np.searchsorted(ref['height_kpc'],z)
                    old = np.array(ref['mixed'])[:,:,ri][:,:,:,zi]
                mid = (r[:-1]+r[1:])/2
                added = mid[(r[:-1]>=40)&(r[1:]<=80)]
                rr = np.sort(np.r_[r,added])
                old_i,new_i = np.searchsorted(rr,r),np.searchsorted(rr,added)
                mixed = np.empty((4,4,len(rr),len(z)))
                mixed[:,:,old_i,:] = old
                if case['id']=='cutoff_200':
                    refined = json.loads((base/f'tensor-cutoff-refinement-001/mixed_{variant}.json').read_bytes())
                    ir,iz = np.searchsorted(refined['radius_kpc'],added),np.searchsorted(refined['height_kpc'],z)
                    new = np.array(refined['mixed'])[:,:,ir][:,:,:,iz]
                else:
                    vertical = []
                    for name in raw['components']:
                        h = disks[name].height
                        if h not in cache[key]:
                            print(f"{case['id']} {variant}: vertical h={h:g}",flush=True)
                            cache[key][h] = green.jet(k*h,z/h)/h**np.arange(4)[:,None,None]
                        vertical.append(cache[key][h])
                    new = hankel_mixed_jet(k,w,S,np.array(vertical),added,z,G)
                    tail,_ = leading_tail_mixed_jet(disks,raw['components'],k,w,S,added,z,G,case['cutoff'],precision={'digits':50,'low_k_limit':8.})
                    new += tail
                mixed[:,:,new_i,:] = new
                assert mixed[:,:,old_i,:].tobytes()==old.tobytes()
                write(f"mixed_{case['id']}_{variant}.json",{'radius_kpc':rr,'height_kpc':z,'mixed':mixed,'old_nodes_bit_identical':True})
                provider = MatchedTensorPotential(rr,z,mixed,exterior)
                old_provider = MatchedTensorPotential(r,z,old,exterior)
                if case['id']=='canonical':
                    providers[variant] = provider
                    ref_fields = old_provider.fields(R,Z)
                else:
                    ref_fields = providers[variant].fields(R,Z)
                fields = provider.fields(R,Z)
                h,half = min(d.height for d in disks.values()),profile['stellar_half_mass_radius_kpc']
                acceleration = G*moments['compact_source_mass']/half**2
                scale = scales(ref_fields,R,Z,h,acceleration,half)
                errors = {'force':np.linalg.norm(fields['gradient_R_z']-ref_fields['gradient_R_z'],axis=0)/scale['force'],
                    'hessian':norm(fields['hessian_RR_Rz_zz_pp']-ref_fields['hessian_RR_Rz_zz_pp'],[1,2,1,1])/scale['hessian'],
                    'third':norm(fields['third_RRR_RRz_Rzz_zzz_Rpp_zpp']-ref_fields['third_RRR_RRz_Rzz_zzz_Rpp_zpp'],[1,3,3,1,3,3])/scale['third']}
                identities = source_errors(fields,disks,R,Z,G,h)
                row = {'case':case['id'],'variant':variant,'probe_entries':R.size,'shape':[len(rr),len(z)],
                    'change_reference':'old canonical mesh' if case['id']=='canonical' else 'refined canonical mesh',
                    'field_change':maxima(errors,R,Z),'source_identities':maxima(identities,R,Z)}
                row['within_field_and_source_targets'] = all(v['value']<config['targets'][k]
                    for group in [row['field_change'],row['source_identities']] for k,v in group.items())
                if case['id']=='canonical':
                    DR,DZ = np.meshgrid(np.unique(np.r_[previous['derivative_radii_kpc'],added]),previous['derivative_heights_kpc'],indexing='ij')
                    df = provider.fields(DR,DZ)
                    ds = scales(df,DR,DZ,h,acceleration,half)
                    checks = []
                    for step in config['derivative_steps']:
                        for stencil in ['central','left','right']:
                            de = derivative_check(provider,DR,DZ,step,ds,radial_stencil=stencil)
                            checks.append({'step':step,'stencil':stencil,'errors':maxima(de,DR,DZ)})
                    extra = maxima(source_errors(df,disks,DR,DZ,G,h),DR,DZ)
                    row.update({'derivative_probe_entries':DR.size,'independent_derivatives':checks,'additional_source_identities':extra,
                        'fine_derivatives_pass':all(v['value']<1e-4 for c in checks if c['step']==.0005 and c['stencil']!='central' for v in c['errors'].values()),
                        'additional_source_checks_pass':all(v['value']<config['targets'][k] for k,v in extra.items())})
                rows.append(row)
                write(f"fields_{case['id']}_{variant}.json",{'R_kpc':R,'z_kpc':Z,'fields':fields,'changes':errors,'source_errors':identities})
                print(f"{case['id']} {variant}: source/field targets {row['within_field_and_source_targets']}",flush=True)
                write(f"row_{case['id']}_{variant}.json",row)
        assert all(sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
        passed = all(r['within_field_and_source_targets'] and r.get('fine_derivatives_pass',True)
            and r.get('additional_source_checks_pass',True) for r in rows)
        write('result.json',{'config':config,'rows':rows,'all_registered_cases_complete':True,
            'all_sampled_numerical_checks_pass':passed,'ready_for_development_field_solve':passed,
            'uniform_error_bound':False,'new_observational_scores':0,'completed_utc':datetime.now(UTC).isoformat()})
        print(f'All sampled replay checks pass: {passed}',flush=True)
    except Exception as exc:
        write('failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise


if __name__=='__main__':
    main()
