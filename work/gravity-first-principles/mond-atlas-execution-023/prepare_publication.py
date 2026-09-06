"""One-use scoped publication preparation after explicit prior-byte archival."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
prior=root/'work/gravity-first-principles/mond-atlas-execution-022/publication-manifest.json'
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root).strip()
paragraph='''
## Execution-023: real radial clock/relay formula comparisons

Ran the frozen 713-candidate comparison on the RTX 5090 using 102 eligible
galaxies and 2212 radii from the 139 historically exposed identities. Reserved
archive members were not parsed by the fit runners. Independent replay checks
all 180 selections and 79,632 initial held-family predictions. Global parameters
are selected with training velocities; only predictor inputs are source-only.
Adjusted algebraic MOND wins every initial training selection; absorption selects
zero opacity. Original clock potential has excessive inner attraction. Separate,
explicitly post-hoc mass-scale and central-core repairs are recorded with frozen
grids and no claims of fresh confirmation. See the complete results and limits:
`work/gravity-first-principles/mond-atlas-clock-relay-001/README.md`.

These are source-backed radial empirical tests, not an observed 3D operator or
evidence that time supplies energy. Energy exchange, source histories, actual
distributed 3D source fields, cluster/Solar System transfer and lensing remain
unresolved. No parameter is inferred independently of training responses merely
because its formula uses photometry. Overall research goal remains unfinished.
'''
for path,old in [('docs/MOND_OBSERVATION_ATLAS_GOAL.md','prior-goal-handoff.md'),('docs/GRAVITY_PATTERN_SYSTEM_TASKS.md','prior-task-plan.md')]:
    p=root/path;assert p.read_bytes()==(out/old).read_bytes()
    with p.open('a',encoding='utf-8',newline='') as f:f.write(paragraph)
paths={v['path'] for v in json.loads(prior.read_text(encoding='utf-8'))['files']};paths.add(prior.relative_to(root).as_posix())
paths.update(['configs/mond_atlas_clock_relay_v1.json','scripts/mond_atlas_clock_relay.py','scripts/run_mond_atlas_clock_relay.py','scripts/summarize_mond_atlas_clock_relay.py','scripts/run_mond_atlas_clock_scale_repair.py','scripts/run_mond_atlas_clock_core_repair.py','tests/test_mond_atlas_clock_relay.py','tests/test_mond_atlas_clock_scale_repair.py','tests/test_mond_atlas_clock_core_repair.py'])
for name in ['mond-atlas-clock-relay-001','mond-atlas-execution-023']:
    for p in (root/'work/gravity-first-principles'/name).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='publication-manifest.json':paths.add(p.relative_to(root).as_posix())
entries=[]
for path in sorted(paths):
    assert '/private/' not in path and Path(path).suffix.lower() not in ['.fits','.npy','.npz','.tar','.gz','.zip','.pdf']
    data=(root/path).read_bytes();entries.append(dict(path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
manifest=out/'publication-manifest.json';manifest.write_text(json.dumps(dict(status='VALIDATED_FOR_ORDINARY_GIT_PUBLICATION',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),goal_complete=False,raw_observation_files_included=0,files=entries),indent=2)+'\n',encoding='utf-8')
paths.add(manifest.relative_to(root).as_posix());private=root/'work/private/mond-atlas-execution-023';private.mkdir(exist_ok=True)
(private/'paths.nul').write_bytes(b'\0'.join(p.encode() for p in sorted(paths))+b'\0');print(json.dumps(dict(manifest_entries=len(entries),stage_paths=len(paths))))
