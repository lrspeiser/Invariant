"""Response-free fidelity and numerical audit of existing cluster sources."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from invariant_gravity_extensions.cluster_pressure import (
    DEVELOPMENT_CLUSTERS,
    GM_SUN,
    KPC,
    MU_E,
    PROTON_MASS,
    G,
)
from invariant_gravity_extensions.smooth_spherical_source import (
    build_cluster_sources,
    cluster_source_fields,
)


def serial(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,dict):
        return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [serial(v) for v in value]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config_path=ROOT/'configs/gravity_xcop_smooth_source_v1.json'
    config=json.loads(config_path.read_bytes())
    parent=ROOT/config['source_packet']
    paths=[Path(__file__),config_path,parent,ROOT/'tests/test_gravity_smooth_spherical_source.py',
           *sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]
    hashes={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in paths}
    for p in paths:
        target=args.output/'input-snapshots'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())

    def write(name,value):
        with (args.output/name).open('x',encoding='utf8',newline='\n') as f:
            json.dump(serial(value),f,indent=2,sort_keys=True,allow_nan=False)
            f.write('\n')

    provenance={'config':config,'input_hashes':hashes,'started_utc':datetime.now(UTC).isoformat(),
                'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'new_raw_or_reserved_data_accessed':False,'pressure_used':False}
    write('started.json',provenance)
    try:
        packets=json.loads(parent.read_bytes())['packets']
        assert {p['cluster'] for p in packets}==DEVELOPMENT_CLUSTERS
        source_keys=['cluster','density_radius_kpc','ne_cm3','ne_low_error','ne_high_error','stellar']
        packets=[{k:p[k] for k in source_keys} for p in packets]
        write('source_only_packets.json',packets)
        rows=[]
        for packet in packets:
            r=np.geomspace(packet['density_radius_kpc'][0],packet['density_radius_kpc'][-1],config['probe_count'])*KPC
            measured=np.asarray(packet['density_radius_kpc'])*KPC
            for width in [config['primary_log_radius_width'],*config['sensitivity_widths']]:
                print(f"Source {packet['cluster']}, log-width {width}",flush=True)
                kwargs={'width':width,'outer_factor':config['outer_factor'],'outer_slope':config['outer_slope']}
                coarse=build_cluster_sources(packet,nodes=config['coarse_nodes'],**kwargs)
                fine=build_cluster_sources(packet,nodes=config['fine_nodes'],**kwargs)
                a,b=cluster_source_fields(coarse,r,{}),cluster_source_fields(fine,r,{})
                norms=[b['gbar'],b['gbar']/r+4*np.pi*G*b['density'],
                       b['gbar']/r**2+4*np.pi*G*(b['density']/r+abs(b['density_gradient']))]
                differences={key:abs(a[key]-b[key])/norm for key,norm in zip(['gbar','gbar_first','gbar_second'],norms,strict=True)}
                maximum={key:float(np.max(value)) for key,value in differences.items()}
                density=fine['gas'].evaluate(measured)['density']/(1e6*MU_E*PROTON_MASS)
                residual=density-packet['ne_cm3']
                errors=np.where(residual>=0,packet['ne_high_error'],packet['ne_low_error'])
                sigma=residual/errors
                stellar=None
                if fine['stellar'] is not None:
                    old=packet['stellar']
                    sm=np.asarray(old['mass_msun'])
                    sr=np.asarray(old['radius_kpc'])*KPC
                    prediction=fine['stellar'].evaluate(sr)['mass']*G/GM_SUN
                    stellar={'predicted_mass_msun':prediction,'fraction_change_from_inherited_monotone':prediction/np.maximum.accumulate(sm)-1,
                             'fraction_change_from_raw':prediction/sm-1,'metadata':fine['stellar'].metadata}
                mass_errors={key:abs(fine[key].cumulative_mass[-1]/fine[key].expected_total_mass-1)
                             for key in ['gas','stellar'] if fine[key] is not None}
                gates=config['numerical_gates']
                numerical=(maximum['gbar']<gates['maximum_acceleration_change'] and
                           maximum['gbar_first']<gates['maximum_first_derivative_change'] and
                           maximum['gbar_second']<gates['maximum_second_derivative_change'] and
                           max(mass_errors.values())<gates['maximum_total_mass_error'])
                fidelity=(np.max(abs(sigma))<config['source_fidelity']['primary_maximum_gas_density_shift_in_quoted_errors'] and
                          (stellar is None or np.max(abs(stellar['fraction_change_from_inherited_monotone']))<config['source_fidelity']['primary_maximum_stellar_mass_fraction_change_from_inherited_monotone_curve']))
                rows.append({'cluster':packet['cluster'],'width':width,'probes_m':r,'fields':b,
                             'refinement':differences,'maximum_refinement':maximum,'gas_density_cm3':density,
                             'gas_density_source_standardized_shift':sigma,'maximum_gas_density_source_shift':float(np.max(abs(sigma))),
                             'stellar':stellar,'total_mass_relative_errors':mass_errors,'numerical_pass':numerical,
                             'source_fidelity_within_primary_limits':fidelity})
        primary=[r for r in rows if r['width']==config['primary_log_radius_width']]
        if any(sha256((ROOT/p).read_bytes()).hexdigest()!=v for p,v in hashes.items()):
            raise RuntimeError('input changed during source audit')
        result={**provenance,'rows':rows,'all_primary_numerical_pass':all(r['numerical_pass'] for r in primary),
                'all_primary_fidelity_pass':all(r['source_fidelity_within_primary_limits'] for r in primary),
                'gravity_predictions':None,'gravity_rejection':False}
        write('result.json',result)
        write('receipt.json',{'status':'SOURCE_AUDIT_RETAINED','result_sha256':sha256((args.output/'result.json').read_bytes()).hexdigest()})
        print(json.dumps({'primary_numerical':result['all_primary_numerical_pass'],'primary_fidelity':result['all_primary_fidelity_pass'],
                          'maximum_primary_gas_shift':max(r['maximum_gas_density_source_shift'] for r in primary)}))
    except Exception as exc:
        write('failure.json',{'status':'SOURCE_EXECUTION_FAILURE_RETAINED','error':str(exc)})
        raise


if __name__=='__main__':
    main()
