"""Describe retained source-width sensitivity without selecting a replacement source."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).parent / 'Invariant'
base = root / 'work/gravity-first-principles/length-cluster-pressure-extended-001'
paths = [base / name for name in ('result.json', 'source_preflight.json')]
result, preflight = [json.loads(p.read_bytes()) for p in paths]
models = {m['id']: m for m in result['models']}
selected = {key for key, m in models.items() if m.get('length_pc') == 1000000}
assert len(selected) == 9
rows = []
for scenario in result['scenarios']:
    if scenario['id'] not in ('nominal', 'source_width_0.005', 'source_width_0.01'):
        continue
    source = scenario['source']
    key = [source['width'], source['outer_factor'], source['outer_slope'], 0]
    sources = {x['cluster']: x for x in preflight['rows'] if x['source_key'] == key}
    controls = [x for x in result['numerical_controls'] if x['model'] in selected and x['scenario'] == scenario['id']]
    assert len(controls) == 72 and len(sources) == 8
    for c in controls:
        s = sources[c['cluster']]
        failures = c['failures']
        forces = [v for f in failures.values() for v in f.get('bad_force_m_s2', [])]
        assert all(v is not None and v <= 0 for v in forces)
        rows.append(dict(model=c['model'], cluster=c['cluster'], scenario=scenario['id'],
                         width=source['width'], numerical_pass=c['numerical_pass'],
                         failed_grids=list(failures), minimum_bad_force_m_s2=min(forces) if forces else None,
                         source_within_primary_limits=s['within_primary_source_limits'],
                         stellar_profile_present=s['stellar'] is not None,
                         maximum_stellar_mass_shift=(max(map(abs, s['stellar']['fraction_from_monotone'])) if s['stellar'] else None),
                         maximum_gas_shift_quoted_errors=s['maximum_gas_smoothing_shift']))
assert len(rows) == 216
summary = []
for width in (.0025, .005, .01):
    group = [r for r in rows if r['width'] == width]
    summary.append(dict(width=width, cases=len(group), failed_cases=sum(not r['numerical_pass'] for r in group),
                        outside_source_limits=sorted({r['cluster'] for r in group if not r['source_within_primary_limits']}),
                        failed_clusters=sorted({r['cluster'] for r in group if not r['numerical_pass']})))
out = dict(scope='Post hoc descriptive audit of existing development results; no source selection, rescoring, or family exclusion.',
           input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
           summary=summary, rows=rows,
           conclusion='Recorded force failures depend on source smoothing. The widest source removes failures but violates the inherited source-fidelity limit for A85. Neither selecting that source nor declaring physical exclusion is justified by this audit.')
dest = root / 'work/gravity-first-principles/cluster-width-failure-audit-001'
dest.mkdir(exist_ok=True)
target = dest / 'result.json'
assert not target.exists()
target.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
(dest / 'audit.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(summary, indent=2))
