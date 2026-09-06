"""Publish explicitly scoped larger-program evidence; never include private data."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
prior=root/'work/gravity-first-principles/mond-atlas-execution-024/publication-manifest.json'
assert not (out/'publication-manifest.json').exists()
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root).strip()
old=json.loads(prior.read_text(encoding='utf-8'))
for item in old['files']:
    assert hashlib.sha256((root/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
paragraph=(out/'handoff-addendum.md').read_text(encoding='utf-8')
for path,snapshot in [('docs/MOND_OBSERVATION_ATLAS_GOAL.md','prior-goal-handoff.md'),('docs/GRAVITY_PATTERN_SYSTEM_TASKS.md','prior-task-plan.md')]:
    p=root/path;assert p.read_bytes()==(out/snapshot).read_bytes()
    with p.open('a',encoding='utf-8',newline='') as f:f.write('\n'+paragraph)
paths={v['path'] for v in old['files']};paths.add(prior.relative_to(root).as_posix())
paths.update(['configs/mond_atlas_ngc3198_source_v1.json','scripts/build_mond_atlas_ngc3198_source_checked.py'])
patterns=['*mond_atlas_ngc3198_source_v2.py','mond_atlas_dynamic_program*.py','mond_atlas_dynamic_partition*.py','mond_atlas_noise_extension.py','mond_atlas_noise_joint_program*.py','mond_atlas_refraction_program*.py','mond_atlas_spatial_program.py']
for pattern in patterns:
    paths.update(p.relative_to(root).as_posix() for p in (root/'scripts').glob(pattern))
paths.add('tests/test_mond_atlas_ngc3198_source_v2.py')
for name in ['mond-atlas-generic-source-002','mond-atlas-ngc3198-recovery-001','mond-atlas-dynamic-program-001','mond-atlas-noise-extension-001','mond-atlas-noise-joint-program-001','mond-atlas-refraction-program-001','mond-atlas-spatial-program-001','mond-atlas-execution-025']:
    folder=root/'work/gravity-first-principles'/name;assert folder.exists(),name
    for p in folder.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='publication-manifest.json':paths.add(p.relative_to(root).as_posix())
entries=[]
for relative in sorted(paths):
    assert '/private/' not in relative
    assert Path(relative).suffix.lower() not in ['.fits','.npy','.npz','.tar','.gz','.zip','.pdf']
    data=(root/relative).read_bytes()
    assert len(data)<50*1024**2,relative
    entries.append(dict(path=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
manifest=out/'publication-manifest.json'
manifest.write_text(json.dumps(dict(status='VALIDATED_FOR_ORDINARY_GIT_PUBLICATION',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),goal_complete=False,raw_observation_files_included=0,files=entries),indent=2)+'\n',encoding='utf-8')
paths.add(manifest.relative_to(root).as_posix())
private=root/'work/private/mond-atlas-execution-025';private.mkdir(exist_ok=False)
(private/'paths.nul').write_bytes(b'\0'.join(p.encode() for p in sorted(paths))+b'\0')
print(json.dumps(dict(manifest_entries=len(entries),stage_paths=len(paths))))
