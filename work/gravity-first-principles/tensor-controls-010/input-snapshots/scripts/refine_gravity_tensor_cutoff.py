"""Separate radial and vertical mesh refinements of the failed cutoff case."""
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
from audit_gravity_matched_tensor import maxima, norm, scales, serial, source_errors

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
    case_path = base/'tensor-quadrature-cutoff-001/result.json'
    if sha256(case_path.read_bytes()).hexdigest()!='4eaa80f4a1562f4ad14e8ddf3760dffcd53f674694da0a1de9d1d33c24f3de4c':
        raise ValueError('Retained cutoff failure changed')
    transform_path = base/'potential-join-001/transform_r128_k64.json'
    profile_path,units_path = base/'map-source-003/source_profiles.json',base/'map-source-003/result.json'
    paths = [Path(__file__),ROOT/'scripts/audit_gravity_matched_tensor.py',case_path,transform_path,profile_path,units_path,
        *sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]
    for variant in ['primary','height_half']:
        paths += [base/f'tensor-quadrature-cutoff-001/{kind}_{variant}.json' for kind in ['mixed','fields']]
        paths.append(base/f'exterior-moment-002/moments_{variant}_reference.json')
    hashes = {p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in paths}
    for p in paths:
        target = args.output/'input-snapshots'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())
    config = {'registration':'After the exposed mixed-input attribution; before refinement execution. No physical source, cutoff, quadrature, tolerance or gravity parameter changes.',
        'radial_rule':'Bisect every old radial interval entirely within 40..80 kpc',
        'vertical_rule':'Bisect every old vertical interval entirely within 0..0.2 kpc',
        'probe_rule':'Retain all 1562 old entries and add quarter/three-quarter positions in every newly refined radial and near-plane vertical interval as a Cartesian product',
        'old_samples':'Overwrite old-node intersections with byte-identical retained values; retain newly evaluated values only at new nodes',
        'variants':['primary','height_half'],'meshes':['baseline','radial_only','vertical_only','both'],
        'targets':{'force':1e-4,'hessian':.002,'third':.01,'density':.002,'density_gradient':.01},
        'full_source_admitted':False,'new_observational_scores':0}

    def write(name,value):
        with (args.output/name).open('x',encoding='utf8',newline='\n') as handle:
            json.dump(serial(value),handle,indent=2,allow_nan=False)
            handle.write('\n')

    write('started.json',{'config':config,'input_hashes':hashes,'started_utc':datetime.now(UTC).isoformat()})
    try:
        raw = json.loads(transform_path.read_bytes())
        k,w,S = [np.array(raw[key]) for key in ['k','wavenumber_weights','surface_hankel']]
        keep = k < 200
        k,w,S = k[keep],w[keep],S[:,keep]
        profile = json.loads(profile_path.read_bytes())['profiles'][-1]
        G = json.loads(units_path.read_bytes())['config']['units']['G_kpc_kms2_msun']
        green = Sech2VerticalGreen(intervals=2400,extent=24.)
        summaries = []
        for variant in config['variants']:
            raw_table = json.loads((base/f'tensor-quadrature-cutoff-001/mixed_{variant}.json').read_bytes())
            r,z,old = [np.array(raw_table[key]) for key in ['radius_kpc','height_kpc','mixed']]
            rm,zm = (r[:-1]+r[1:])/2,(z[:-1]+z[1:])/2
            added_r = rm[(r[:-1]>=40)&(r[1:]<=80)]
            added_z = zm[(z[:-1]>=0)&(z[1:]<=.2)]
            rr,zz = np.sort(np.r_[r,added_r]),np.sort(np.r_[z,added_z])
            ri,zi = np.searchsorted(rr,r),np.searchsorted(zz,z)
            definition = {'id':variant,**({'height_factor':.5} if variant=='height_half' else {})}
            _,disks = regular_disks(profile,definition)
            vertical = []
            for name in raw['components']:
                print(f'Refinement {variant}: vertical {name}',flush=True)
                h = disks[name].height
                vertical.append(green.jet(k*h,zz/h)/h**np.arange(4)[:,None,None])
            print(f'Refinement {variant}: expanded mixed table',flush=True)
            mixed = hankel_mixed_jet(k,w,S,np.array(vertical),rr,zz,G)
            tail,_ = leading_tail_mixed_jet(disks,raw['components'],k,w,S,rr,zz,G,200.,precision={'digits':50,'low_k_limit':8.})
            mixed += tail
            recomputation = np.max(abs(mixed[:,:,ri][:,:,:,zi]-old),axis=(2,3))
            for i,index in enumerate(ri):
                mixed[:,:,index,zi] = old[:,:,i,:]
            assert np.array_equal(mixed[:,:,ri][:,:,:,zi],old)
            write(f'mixed_{variant}.json',{'radius_kpc':rr,'height_kpc':zz,'mixed':mixed,
                'old_node_recomputation_maximum_absolute_changes':recomputation,'retained_old_nodes_exact':True})
            old_fields = json.loads((base/f'tensor-quadrature-cutoff-001/fields_{variant}.json').read_bytes())
            rstart = rr[:-1][(rr[:-1]>=40)&(rr[1:]<=80)]
            rwidth = np.diff(rr)[(rr[:-1]>=40)&(rr[1:]<=80)]
            zstart = zz[:-1][(zz[:-1]>=0)&(zz[1:]<=.2)]
            zwidth = np.diff(zz)[(zz[:-1]>=0)&(zz[1:]<=.2)]
            pr = np.sort(np.r_[rstart+rwidth/4,rstart+3*rwidth/4])
            pz = np.sort(np.r_[zstart+zwidth/4,zstart+3*zwidth/4])
            NR,NZ = np.meshgrid(pr,pz,indexing='ij')
            R,Z = np.r_[old_fields['R_kpc'],NR.ravel()],np.r_[old_fields['z_kpc'],NZ.ravel()]
            moments = json.loads((base/f'exterior-moment-002/moments_{variant}_reference.json').read_bytes())
            exterior = ExteriorMomentField(moments,G,minimum_radius=60.)
            baseline = MatchedTensorPotential(r,z,old,exterior).fields(R,Z)
            h,half = min(d.height for d in disks.values()),profile['stellar_half_mass_radius_kpc']
            scale = scales(baseline,R,Z,h,G*moments['compact_source_mass']/half**2,half)
            rows = []
            for name,gr,gz,data in [('baseline',r,z,old),('radial_only',rr,z,mixed[:,:,:,zi]),
                                  ('vertical_only',r,zz,mixed[:,:,ri]),('both',rr,zz,mixed)]:
                fields = MatchedTensorPotential(gr,gz,data,exterior).fields(R,Z)
                errors = {'force':np.linalg.norm(fields['gradient_R_z']-baseline['gradient_R_z'],axis=0)/scale['force'],
                    'hessian':norm(fields['hessian_RR_Rz_zz_pp']-baseline['hessian_RR_Rz_zz_pp'],[1,2,1,1])/scale['hessian'],
                    'third':norm(fields['third_RRR_RRz_Rzz_zzz_Rpp_zpp']-baseline['third_RRR_RRz_Rzz_zzz_Rpp_zpp'],[1,3,3,1,3,3])/scale['third']}
                identities = source_errors(fields,disks,R,Z,G,h)
                row = {'mesh':name,'shape':[len(gr),len(gz)],'source_identities':maxima(identities,R,Z),
                    'change_from_baseline':maxima(errors,R,Z)}
                row['within_registered_targets'] = all(v['value']<config['targets'][key]
                    for group in [row['source_identities'],row['change_from_baseline']] for key,v in group.items())
                rows.append(row)
                write(f'fields_{variant}_{name}.json',{'R_kpc':R,'z_kpc':Z,'fields':fields,'source_errors':identities,'changes':errors})
            summaries.append({'variant':variant,'retained_probe_entries':len(old_fields['R_kpc']),
                'new_probe_entries':NR.size,'rows':rows})
            print(json.dumps(serial(summaries[-1])),flush=True)
        assert all(sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
        write('result.json',{'config':config,'summaries':summaries,'completed_utc':datetime.now(UTC).isoformat(),
            'all_quadrature_cases_replayed':False,'full_source_admitted':False,'new_observational_scores':0})
    except Exception as exc:
        write('failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise


if __name__=='__main__':
    main()
