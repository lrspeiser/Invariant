"""Read-only coverage/count audit: no observational values or scores computed."""
import collections,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'work/gravity-first-principles'
paths=['mond-atlas-relay-001/README.md','mond-atlas-clock-relay-001/README.md',
 'mond-atlas-clock-relay-001/run001/candidate-formulas.json',
 'mond-atlas-clock-relay-001/physics/scale-repair/PREFLIGHT.md',
 'mond-atlas-clock-relay-001/physics/scale-repair/README.md',
 'mond-atlas-clock-relay-001/physics/scale-repair/run001/candidates.json',
 'mond-atlas-clock-relay-001/source-audit/core-repair/PREFLIGHT.md',
 'mond-atlas-clock-relay-001/source-audit/core-repair/README.md',
 'mond-atlas-clock-relay-001/source-audit/core-repair/run001/pre-access-bindings.json',
 'structure-environment-motion-001/report.md','broad-pattern-findings-001/findings.md',
 'broad-pattern-findings-001/hypothesis-inventory.csv',
 'mond-atlas-motion-controls-001/README.md','mond-atlas-motion-covariance-001/README.md']
files=[BASE/p for p in paths]+[ROOT/'docs/GRAVITY_PATTERN_SYSTEM_TASKS.md',ROOT/'configs/mond_atlas_clock_relay_v1.json']
groups=[]
for relative in [paths[2],paths[5],paths[8]]:
    value=json.loads((BASE/relative).read_text(encoding='utf-8'))
    candidates=value['candidate_grid'] if isinstance(value,dict) else value
    groups.append(dict(path=relative,settings=len(candidates),family_counts=dict(collections.Counter(c['family'] for c in candidates))))
out=dict(scope='READ_ONLY_COVERAGE_NO_NEW_RESPONSE_SCORING',groups=groups,total_evaluated_settings=sum(g['settings'] for g in groups),
    counting_convention='Initial eleven labels reduce to nine algebraic forms after merging fixed/adjusted baseline labels; central repair adds one force shape. Source-scaling variants, overlapping mixture endpoints and zero-strength duplicates are explicitly not independent physical mechanisms.',
    bindings={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(HERE/'inventory.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(groups,indent=2))
