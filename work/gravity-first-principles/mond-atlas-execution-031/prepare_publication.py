"""Publish only the verified increment, preserving prior research bytes."""
import ast,csv,gzip,hashlib,io,json,math,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[3];P=Path(__file__).resolve().parent
prior=R/'work/gravity-first-principles/mond-atlas-execution-030/publication-manifest.json'
assert not (P/'publication-manifest.json').exists()
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=R).strip()
entries=json.loads(prior.read_text())['files']
for e in entries:assert hashlib.sha256((R/e['path']).read_bytes()).hexdigest()==e['sha256'],e['path']
patterns=['mond_atlas_aperture_covariance*.py','mond_atlas_aperture_scorer*.py',
          'mond_atlas_column_balance*.py','mond_atlas_hi_column*.py',
          'mond_atlas_native_emission*.py','mond_atlas_column_force*.py']
scripts=sorted({p for pattern in patterns for p in (R/'scripts').glob(pattern)})
tracked=set(subprocess.check_output(['git','ls-files'],cwd=R,text=True).splitlines())
new={p.relative_to(R).as_posix() for p in scripts}
for p in scripts:
    text=p.read_text(encoding='utf-8');compile(text,str(p),'exec')
    for node in ast.walk(ast.parse(text)):
        imports=([node.module] if isinstance(node,ast.ImportFrom) else
                 [alias.name for alias in node.names] if isinstance(node,ast.Import) else [])
        for module in imports:
            if not module:continue
            target=R/'scripts'/(module.split('.')[0]+'.py')
            if target.is_file():assert target.relative_to(R).as_posix() in tracked|new,str(target)
receipt_count=0
for relative in ['work/gravity-first-principles/mond-atlas-hi-column-001/completion-receipt.json',
                 'work/gravity-first-principles/mond-atlas-native-emission-001/completion.json']:
    receipt=json.loads((R/relative).read_text())
    for path,expected in receipt['files'].items():
        assert hashlib.sha256((R/path).read_bytes()).hexdigest()==expected,path
        receipt_count+=1
compressed='work/gravity-first-principles/mond-atlas-column-force-001/run004/column-force-table.csv.gz'
artifact=json.loads((R/Path(compressed).parent/'artifact-receipt.json').read_text())
packed=(R/compressed).read_bytes();data=gzip.decompress(packed)
assert hashlib.sha256(packed).hexdigest()==artifact['compressed_sha256']
assert hashlib.sha256(data).hexdigest()==artifact['csv_sha256']
rows=list(csv.DictReader(io.StringIO(data.decode('utf-8'))))
assert len(rows)==artifact['rows'] and list(rows[0])==artifact['columns']
assert all(math.isfinite(float(row[key])) for row in rows for key in artifact['numeric_columns'])
(P/'publication-checks.json').write_text(json.dumps(dict(prior_files_verified=len(entries),
    new_scripts_compiled=len(scripts),direct_local_imports_available=True,
    completion_receipt_hashes_verified=receipt_count,derived_compressed_rows_verified=len(rows),
    raw_observations_in_publication=0,observed_gravity_scores=0,goal_complete=False),indent=2)+'\n')
for path,snapshot in [('docs/MOND_OBSERVATION_ATLAS_GOAL.md','prior-goal-handoff.md'),
                      ('docs/GRAVITY_PATTERN_SYSTEM_TASKS.md','prior-task-plan.md')]:
    target=R/path;assert target.read_bytes()==(P/snapshot).read_bytes()
    with target.open('a',encoding='utf-8',newline='') as f:f.write('\n'+(P/'handoff-addendum.md').read_text(encoding='utf-8'))
paths={e['path'] for e in entries};paths.add(prior.relative_to(R).as_posix());paths.update(new)
for name in ['mond-atlas-aperture-covariance-001','mond-atlas-aperture-scorer-001',
             'mond-atlas-column-balance-001','mond-atlas-hi-column-001',
             'mond-atlas-native-emission-001','mond-atlas-column-force-001','mond-atlas-execution-031']:
    for p in (R/'work/gravity-first-principles'/name).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='publication-manifest.json':paths.add(p.relative_to(R).as_posix())
bound=[]
for path in sorted(paths):
    assert '/private/' not in path
    if path!=compressed:assert Path(path).suffix.lower() not in ['.fits','.npz','.npy','.zip','.gz','.tar','.pdf'],path
    blob=(R/path).read_bytes();bound.append(dict(path=path,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
manifest=P/'publication-manifest.json'
manifest.write_text(json.dumps(dict(status='VERIFIED_FOR_ORDINARY_PUBLICATION',
    base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
    goal_complete=False,raw_observation_files_included=0,explicit_derived_gzip=compressed,files=bound),indent=2)+'\n')
paths.add(manifest.relative_to(R).as_posix())
private=R/'work/private/mond-atlas-execution-031';private.mkdir(exist_ok=False)
(private/'paths.nul').write_bytes(b'\0'.join(p.encode() for p in sorted(paths))+b'\0')
print(json.dumps(dict(manifest_entries=len(bound),stage_paths=len(paths),compiled_scripts=len(scripts))))
