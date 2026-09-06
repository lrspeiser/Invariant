"""Prepare scoped cumulative manifest after prior integrity verification."""
import hashlib,json,subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
prior=root/'work/gravity-first-principles/mond-atlas-execution-021/publication-manifest.json'
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root).strip(),'Index must start empty'
paragraph='''
## Execution-022: parallel relay and published halo tests

Completed three parallel agent investigations plus coordinator halo and geometry
tests: absorption/redirection, distributed secondary sources, finite memory and
feedback, and cumulative return shapes. Twenty tests pass. All 525 selected fit
rows from 175 SPARC galaxies match source tables; 504 pilot vectors independently
match galpy within 1.69e-12. This is fitted-halo calibration, not observational
validation. A manufactured distributed disk becomes nearly spherical far out,
but its inner strength differs from the point-calibrated halo. Absorption alone
weakens the tested attraction; strong feedback lengthens memory and can become
unstable. Scalar disk boosts fail off-plane vector geometry. Two finite-return
shapes interpolate NFW/Burkert well but degrade in outer extrapolation. All
failures and scope limits are retained. See
`work/gravity-first-principles/mond-atlas-relay-001/README.md`.

Next priority: a conservative distributed response with ordinary-matter rules
for strength, scale and finite extent, then real-source and held-galaxy tests.
No new observed full-field, cluster, Solar System or lensing score is admitted.
The existing data/noise and field-convergence blockers remain. Goal unfinished.
'''
for path,archive in [('docs/MOND_OBSERVATION_ATLAS_GOAL.md','prior-goal-handoff.md'),('docs/GRAVITY_PATTERN_SYSTEM_TASKS.md','prior-task-plan.md')]:
    p=root/path
    assert p.read_bytes()==(out/archive).read_bytes(),'Handoff changed since archival'
    with p.open('a',encoding='utf-8',newline='') as f:f.write(paragraph)
paths={row['path'] for row in json.loads(prior.read_text(encoding='utf-8'))['files']}
paths.add(prior.relative_to(root).as_posix())
files=['configs/mond_atlas_halo_return_v1.json',
    'scripts/mond_atlas_halo_return.py','scripts/run_mond_atlas_halo_return.py',
    'scripts/mond_atlas_secondary_experiment.py','scripts/mond_atlas_absorption_experiment.py',
    'scripts/mond_atlas_delay_experiment.py','scripts/run_mond_atlas_relay_geometry.py',
    'scripts/verify_mond_atlas_relay.py',
    'tests/test_mond_atlas_halo_return.py','tests/test_mond_atlas_secondary_experiment.py',
    'tests/test_mond_atlas_absorption_experiment.py','tests/test_mond_atlas_delay_experiment.py']
paths.update(files)
for name in ['mond-atlas-halo-return-001','mond-atlas-relay-001','mond-atlas-execution-022']:
    for p in (root/'work/gravity-first-principles'/name).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='publication-manifest.json':paths.add(p.relative_to(root).as_posix())
entries=[]
for path in sorted(paths):
    assert '/private/' not in path and Path(path).suffix.lower() not in ['.fits','.npy','.npz','.tar','.gz','.zip','.pdf']
    data=(root/path).read_bytes();entries.append(dict(path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
manifest=out/'publication-manifest.json'
manifest.write_text(json.dumps(dict(status='VALIDATED_FOR_ORDINARY_GIT_PUBLICATION',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),goal_complete=False,raw_observation_files_included=0,files=entries),indent=2)+'\n',encoding='utf-8')
paths.add(manifest.relative_to(root).as_posix())
private=root/'work/private/mond-atlas-execution-022';private.mkdir(exist_ok=True)
(private/'paths.nul').write_bytes(b'\0'.join(p.encode('utf-8') for p in sorted(paths))+b'\0')
print(json.dumps(dict(manifest_files=len(entries),stage_paths=len(paths))))
