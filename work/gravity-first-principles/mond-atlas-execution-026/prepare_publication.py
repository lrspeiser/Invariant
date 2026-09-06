"""Exact scoped publication for the executed uncertainty/environment increment."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=Path(__file__).resolve().parent
prior=ROOT/'work/gravity-first-principles/mond-atlas-execution-025/publication-manifest.json'
assert not (P/'publication-manifest.json').exists()
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT).strip()
entries=json.loads(prior.read_text(encoding='utf-8'))['files']
for entry in entries:assert hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
paragraph=(P/'handoff-addendum.md').read_text(encoding='utf-8')
for path,snapshot in [('docs/MOND_OBSERVATION_ATLAS_GOAL.md','prior-goal-handoff.md'),('docs/GRAVITY_PATTERN_SYSTEM_TASKS.md','prior-task-plan.md')]:
    target=ROOT/path;assert target.read_bytes()==(P/snapshot).read_bytes()
    with target.open('a',encoding='utf-8',newline='') as f:f.write('\n'+paragraph)
paths={e['path'] for e in entries};paths.add(prior.relative_to(ROOT).as_posix())
for pattern in ['mond_atlas_external_program.py','mond_atlas_noise_stationary*.py','mond_atlas_noise_scale_channel*.py','mond_atlas_selection_transfer*.py']:
    paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'scripts').glob(pattern))
for name in ['mond-atlas-external-program-001','mond-atlas-noise-stationary-001','mond-atlas-noise-scale-channel-001','mond-atlas-selection-transfer-001','mond-atlas-execution-026']:
    for p in (ROOT/'work/gravity-first-principles'/name).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='publication-manifest.json':paths.add(p.relative_to(ROOT).as_posix())
bound=[]
for path in sorted(paths):
    assert '/private/' not in path and Path(path).suffix.lower() not in ['.fits','.npz','.npy','.zip','.gz','.tar','.pdf']
    data=(ROOT/path).read_bytes();bound.append(dict(path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
manifest=P/'publication-manifest.json';manifest.write_text(json.dumps(dict(status='VALIDATED_FOR_ORDINARY_GIT_PUBLICATION',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),goal_complete=False,raw_observation_files_included=0,files=bound),indent=2)+'\n',encoding='utf-8')
paths.add(manifest.relative_to(ROOT).as_posix());private=ROOT/'work/private/mond-atlas-execution-026';private.mkdir(exist_ok=False)
(private/'paths.nul').write_bytes(b'\0'.join(p.encode() for p in sorted(paths))+b'\0')
print(json.dumps(dict(manifest_entries=len(bound),stage_paths=len(paths))))
