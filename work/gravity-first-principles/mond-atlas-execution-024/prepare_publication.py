"""Prepare explicitly scoped audit and nonclock development publication."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
prior=root/'work/gravity-first-principles/mond-atlas-execution-023/publication-manifest.json'
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root).strip()
paragraph='''
## Execution-024: coverage correction and broader nonclock comparisons

Audited 25 proposed mechanisms and confirmed that earlier numerical checks did
not establish comprehensive mechanism coverage. Parameter settings were not
independent ideas. Actual 3D distribution, directional transport, nested geometry,
motion couplings and memory remain incomplete. The audit is preserved at
`work/gravity-first-principles/mond-atlas-coverage-audit-001/README.md`.

Expanded surface coherence, passive/active re-emission and finite return proxies
with continuous global parameters, three starts, central/outer/mass-scale changes,
same 102-galaxy cohort, and gas-rich/stellar-rich transfer in both directions.
Independent replay and retained failures accompany all completed results at
`work/gravity-first-principles/mond-atlas-nonclock-robust-001/README.md`.
This improves testing of specific radial proxies; it does not complete the 25
mechanism program or supply 3D/source/noise/history/lensing admission. Goal unfinished.
'''
for path,old in [('docs/MOND_OBSERVATION_ATLAS_GOAL.md','prior-goal-handoff.md'),('docs/GRAVITY_PATTERN_SYSTEM_TASKS.md','prior-task-plan.md')]:
    p=root/path;assert p.read_bytes()==(out/old).read_bytes()
    with p.open('a',encoding='utf-8',newline='') as f:f.write(paragraph)
paths={v['path'] for v in json.loads(prior.read_text(encoding='utf-8'))['files']};paths.add(prior.relative_to(root).as_posix())
paths.update(['scripts/run_mond_atlas_coherence_robust.py','scripts/run_mond_atlas_return_robust.py','tests/test_mond_atlas_coherence_robust.py','tests/test_mond_atlas_return_robust.py'])
for name in ['mond-atlas-coverage-audit-001','mond-atlas-nonclock-robust-001','mond-atlas-execution-024']:
    for p in (root/'work/gravity-first-principles'/name).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='publication-manifest.json':paths.add(p.relative_to(root).as_posix())
entries=[]
for path in sorted(paths):
    assert '/private/' not in path and Path(path).suffix.lower() not in ['.fits','.npy','.npz','.tar','.gz','.zip','.pdf']
    data=(root/path).read_bytes();entries.append(dict(path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
manifest=out/'publication-manifest.json';manifest.write_text(json.dumps(dict(status='VALIDATED_FOR_ORDINARY_GIT_PUBLICATION',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),goal_complete=False,raw_observation_files_included=0,files=entries),indent=2)+'\n',encoding='utf-8')
paths.add(manifest.relative_to(root).as_posix());private=root/'work/private/mond-atlas-execution-024';private.mkdir(exist_ok=True)
(private/'paths.nul').write_bytes(b'\0'.join(p.encode() for p in sorted(paths))+b'\0');print(json.dumps(dict(manifest_entries=len(entries),stage_paths=len(paths))))
