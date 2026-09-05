"""Post hoc decomposition of all retained 1-Mpc source-width cases; no fits."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

root = Path(__file__).parent / 'Invariant'
sys.path.insert(0, str(root / 'src'))
from invariant_gravity_extensions.cluster_pressure import G, KPC
from invariant_gravity_extensions.length_cluster_pressure import array_packet, pressure_context
from invariant_gravity_extensions.length_screening import LengthScreening
from invariant_gravity_extensions.smooth_spherical_source import build_cluster_sources, spherical_length_anomaly

dest = root / 'work/gravity-first-principles/cluster-force-terms-001'
dest.mkdir(exist_ok=False)
config_path = root / 'configs/gravity_length_cluster_pressure_extended_v1.json'
config = json.loads(config_path.read_bytes())
packet_path = root / config['source_packet']
parent_path = root / 'work/gravity-first-principles/length-cluster-pressure-extended-001/result.json'
paths = [config_path, packet_path, parent_path, *sorted((root / 'src/invariant_gravity_extensions').glob('*.py'))]
hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
snap = dest / 'inputs'
for p in paths:
    target = snap / p.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(p.read_bytes())
(dest / 'runner.py').write_bytes(Path(__file__).read_bytes())
(dest / 'started.json').write_text(json.dumps({'input_sha256': hashes, 'scope': __doc__}, indent=2), encoding='utf-8')
parent = json.loads(parent_path.read_bytes())
packets = [array_packet(p) for p in json.loads(packet_path.read_bytes())['packets']]
models = [m for m in parent['models'] if m.get('length_pc') == 1000000]
scenarios = [s for s in parent['scenarios'] if s['id'] in ('nominal', 'source_width_0.005', 'source_width_0.01')]
rows = []
for packet in packets:
    for scenario in scenarios:
        source = build_cluster_sources(packet, nodes=config['source_control']['fine_nodes'], **scenario['source'])
        context = pressure_context(packet, source, scenario['values'], nodes=config['pressure_control']['fine_nodes'])
        f = context['fields']
        r, g, first, second = [f[k] for k in ('radius_m', 'gbar', 'gbar_first', 'gbar_second')]
        gas_gradient = source['gas'].evaluate(r)['density_gradient']
        star_gradient = f['density_gradient'] - gas_gradient
        for model in models:
            ell, a0 = model['length_pc']*KPC/1000, model['a0_m_s2']
            spec = LengthScreening(model['shape'], model['epsilon'])
            x = g*g/a0**2
            h = ell**2*(first**2+2*(g/r)**2)/a0**2
            px, ph, k1, k2, fraction = spec.partials(x, h)
            dx = 2*g*first/a0**2
            dh_without_second = ell**2*4*g/r*(first/r-g/r**2)/a0**2
            dph_without_second = ((k1+fraction*k2)*dx+fraction*k2*dh_without_second)/(x+h)
            coefficient = -ell**2*(ph+first*fraction*k2/(x+h)*ell**2*2*first/a0**2)
            terms = {
                'algebraic': (1+px)*g,
                'reaction_without_second': -ell**2*(dph_without_second*first+2*ph*(first-g/r)/r),
                'second_geometric': coefficient*(-2*first/r+2*g/r**2),
                'gas_density_gradient': coefficient*4*np.pi*G*gas_gradient,
                'stellar_density_gradient': coefficient*4*np.pi*G*star_gradient,
            }
            force = g+spherical_length_anomaly(spec, r, g, first, second, ell, a0)
            reconstructed = sum(terms.values())
            scale = np.maximum(abs(force), sum(abs(t) for t in terms.values()))
            err = float(np.max(abs(force-reconstructed)/scale))
            assert err < 1e-12
            bad = force <= 0
            index = int(np.argmin(force))
            rows.append(dict(cluster=packet['cluster'], scenario=scenario['id'], model=model['id'],
                bad_nodes=int(sum(bad)), decomposition_relative_error=err,
                minimum_force_m_s2=float(force[index]), radius_at_minimum_kpc=float(r[index]/KPC),
                terms_at_minimum={k:float(v[index]) for k,v in terms.items()},
                bad_nodes_positive_without_stellar_gradient=int(sum(bad & ((force-terms['stellar_density_gradient'])>0))),
                diagnostic_only='Removing a term is not an alternative physical law or a source correction.'))
        print(packet['cluster']+' '+scenario['id']+' complete', flush=True)
        (dest / 'rows_partial.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
assert len(rows) == 216
assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(root))] for p in paths)
(dest / 'result.json').write_text(json.dumps({'input_sha256':hashes, 'rows':rows, 'new_observational_scores':0,
    'family_exclusions':0, 'all_inputs_unchanged':True}, indent=2), encoding='utf-8')
print('Completed 216 force decompositions', flush=True)
