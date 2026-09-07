"""Scoped, byte-verified publication of the executed research increment."""
import hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[3];P=Path(__file__).resolve().parent
prior=R/'work/gravity-first-principles/mond-atlas-execution-028/publication-manifest.json'
assert not (P/'publication-manifest.json').exists()
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=R).strip()
entries=json.loads(prior.read_text())['files']
for e in entries:assert hashlib.sha256((R/e['path']).read_bytes()).hexdigest()==e['sha256'],e['path']
for path,snapshot in [('docs/MOND_OBSERVATION_ATLAS_GOAL.md','prior-goal-handoff.md'),('docs/GRAVITY_PATTERN_SYSTEM_TASKS.md','prior-task-plan.md')]:
    target=R/path;assert target.read_bytes()==(P/snapshot).read_bytes()
    with target.open('a',encoding='utf-8',newline='') as f:f.write('\n'+(P/'handoff-addendum.md').read_text(encoding='utf-8'))
paths={e['path'] for e in entries};paths.add(prior.relative_to(R).as_posix())
for pattern in ['mond_atlas_local_calibration*.py','mond_atlas_local_gas_calibration*.py','mond_atlas_noise_reference.py','mond_atlas_noise_transfer*.py','mond_atlas_static_susceptibility.py','mond_atlas_log_susceptibility.py']:
    paths.update(p.relative_to(R).as_posix() for p in (R/'scripts').glob(pattern))
for name in ['mond-atlas-local-calibration-001','mond-atlas-local-gas-calibration-001','mond-atlas-noise-reference-001','mond-atlas-noise-transfer-001','mond-atlas-static-susceptibility-001','mond-atlas-log-susceptibility-001','mond-atlas-execution-029']:
    for p in (R/'work/gravity-first-principles'/name).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='publication-manifest.json':paths.add(p.relative_to(R).as_posix())
bound=[]
for path in sorted(paths):
    assert '/private/' not in path and Path(path).suffix.lower() not in ['.fits','.npz','.npy','.zip','.gz','.tar','.pdf']
    data=(R/path).read_bytes();bound.append(dict(path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
manifest=P/'publication-manifest.json';manifest.write_text(json.dumps(dict(status='VALIDATED_FOR_ORDINARY_GIT_PUBLICATION',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),goal_complete=False,raw_observation_files_included=0,files=bound),indent=2)+'\n',encoding='utf-8')
paths.add(manifest.relative_to(R).as_posix());private=R/'work/private/mond-atlas-execution-029';private.mkdir(exist_ok=False)
(private/'paths.nul').write_bytes(b'\0'.join(p.encode() for p in sorted(paths))+b'\0')
print(json.dumps(dict(manifest_entries=len(bound),stage_paths=len(paths))))
