"""Scoped execution033 preparation; never stages, commits or pushes."""
import argparse,ast,csv,gzip,hashlib,io,json,math,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[3];P=Path(__file__).resolve().parent
PRIOR=R/'work/gravity-first-principles/mond-atlas-execution-032/publication-manifest.json'
PATTERNS=['mond_atlas_alternative_hi*.py','mond_atlas_alternative_spectra*.py','mond_atlas_centroid_diagnostic*.py','mond_atlas_covariance_transfer*.py','mond_atlas_pressure_estimator*.py']
PACKAGES=['mond-atlas-alternative-hi-001','mond-atlas-alternative-spectra-001','mond-atlas-centroid-diagnostic-001','mond-atlas-covariance-transfer-001','mond-atlas-pressure-estimator-001','mond-atlas-execution-033']
DOCS={'docs/MOND_OBSERVATION_ATLAS_GOAL.md':'prior-goal-handoff.md','docs/GRAVITY_PATTERN_SYSTEM_TASKS.md':'prior-task-plan.md'}
OLD_GZIP='work/gravity-first-principles/mond-atlas-column-force-001/run004/column-force-table.csv.gz'
NEW_GZIP=['work/gravity-first-principles/mond-atlas-source-sensitivity-001/fields001/radial-fields.csv.gz','work/gravity-first-principles/mond-atlas-source-sensitivity-001/hi001/pressure-profiles.csv.gz']
ALLOWED_GZIP={OLD_GZIP,*NEW_GZIP}
RECEIPTS=['mond-atlas-alternative-hi-001/completion-receipt.json','mond-atlas-alternative-spectra-001/completion.json','mond-atlas-centroid-diagnostic-001/completion-receipt.json','mond-atlas-covariance-transfer-001/run001/bindings.json','mond-atlas-pressure-estimator-001/run001/bindings.json','mond-atlas-covariance-transfer-001/independent-review/receipt.json','mond-atlas-execution-033/pressure-review/receipt.json']

def sha(blob):return hashlib.sha256(blob).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=R)
def relative(path):return path.relative_to(R).as_posix()

def verify_index():
    """Read-only index check, including LFS pointers and local content objects."""
    manifest=P/'publication-manifest.json';m=read(manifest)
    expected={e['path']:e for e in m['files']}
    blob=manifest.read_bytes();expected[relative(manifest)]={'path':relative(manifest),'bytes':len(blob),'sha256':sha(blob)}
    # Both changed handoff files and every research path are in the manifest.
    staged=set(git('diff','--cached','--name-only').decode().splitlines())
    assert staged and staged<=set(expected),('Unexpected staged files',sorted(staged-set(expected)))
    keys=sorted(expected)
    output=subprocess.check_output(['git','cat-file','--batch'],cwd=R,input=b''.join((':'+k+'\n').encode() for k in keys))
    offset=0;lfs=[];ordinary=0
    objectroot=Path(git('rev-parse','--git-common-dir').decode().strip())/'lfs/objects'
    if not objectroot.is_absolute():objectroot=R/objectroot
    for name in keys:
        end=output.index(b'\n',offset);header=output[offset:end].split();assert len(header)==3 and header[1]==b'blob',(name,header)
        length=int(header[2]);data=output[end+1:end+1+length];offset=end+length+2;e=expected[name]
        if data.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
            assert name in ALLOWED_GZIP,('Unexpected LFS path',name)
            fields=dict(line.split(' ',1) for line in data.decode().splitlines())
            assert fields['oid']=='sha256:'+e['sha256'] and int(fields['size'])==e['bytes'],name
            oid=e['sha256'];obj=objectroot/oid[:2]/oid[2:4]/oid
            assert obj.stat().st_size==e['bytes'] and sha(obj.read_bytes())==oid,('Missing/mismatched local LFS object',name)
            lfs.append(dict(path=name,content_sha256=oid,content_bytes=e['bytes'],pointer_sha256=sha(data),local_lfs_object_verified=True))
        else:
            assert len(data)==e['bytes'] and sha(data)==e['sha256'],('Index content differs',name)
            ordinary+=1
    assert offset==len(output)
    print(json.dumps(dict(status='INDEX_CONTENT_AND_LFS_VERIFIED',manifest_entries=len(m['files']),ordinary_blobs_verified=ordinary,lfs=lfs,staged_paths=len(staged),no_index_mutation=True),indent=2))

def prepare():
    assert not (P/'publication-manifest.json').exists(),'Already frozen; never overwrite'
    for generated in ['publication-checks.json','lfs-publication-plan.json']:assert not (P/generated).exists(),generated
    assert not (R/'work/private/mond-atlas-execution-033').exists(),'Pathspec directory already exists'
    assert not git('diff','--cached','--name-only').strip(),'Index must be empty before prepare'
    entries=read(PRIOR)['files'];assert len(entries)==2233,len(entries)
    for e in entries:assert sha((R/e['path']).read_bytes())==e['sha256'],e['path']
    for doc,snapshot in DOCS.items():assert (R/doc).read_bytes()==(P/snapshot).read_bytes(),doc
    assert (P/'README.md').is_file() and (P/'handoff-addendum.md').is_file()
    scripts=sorted({p for pattern in PATTERNS for p in (R/'scripts').glob(pattern)})
    nested_python=[p for name in PACKAGES for p in (R/'work/gravity-first-principles'/name).rglob('*.py') if '__pycache__' not in p.parts]
    for path in nested_python:compile(path.read_text(encoding='utf-8'),str(path),'exec')
    new={relative(p) for p in scripts};tracked=set(git('ls-files').decode().splitlines())
    for path in scripts:
        text=path.read_text(encoding='utf-8');compile(text,str(path),'exec')
        for node in ast.walk(ast.parse(text)):
            imports=([node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else [])
            for module in imports:
                if module:
                    target=R/'scripts'/(module.split('.')[0]+'.py')
                    if target.is_file():assert relative(target) in tracked|new,('Unpublished dependency',str(target))
    receipt_count=0
    receipt_paths={R/'work/gravity-first-principles'/name for name in RECEIPTS}
    for name in PACKAGES:
        for path in (R/'work/gravity-first-principles'/name).rglob('*.json'):
            if 'receipt' in path.name or 'completion' in path.name:receipt_paths.add(path)
    for path in sorted(receipt_paths):
        receipt=read(path);bindings=receipt.get('files',receipt.get('bindings',receipt.get('public_files')))
        if bindings is None and all(isinstance(v,str) and len(v)==64 for v in receipt.values()):bindings=receipt
        if bindings is None:
            assert path not in {R/'work/gravity-first-principles'/n for n in RECEIPTS},('Required receipt has no hash map',str(path))
            continue
        assert isinstance(bindings,dict) and bindings,str(path)
        for name,entry in bindings.items():
            expected=entry['sha256'] if isinstance(entry,dict) else entry
            actual=Path(name)
            if not actual.is_absolute():actual=R/actual
            assert sha(actual.read_bytes())==expected,('Completion binding mismatch',str(path),name)
            if isinstance(entry,dict) and 'bytes' in entry:assert actual.stat().st_size==entry['bytes'],name
            receipt_count+=1
    cov=read(R/'work/gravity-first-principles/mond-atlas-covariance-transfer-001/run001/summary.json')
    assert cov['status']=='WESTERN_COVARIANCE_TRANSFER_EXECUTED' and cov['fits']==1392 and cov['variants']==29
    assert cov['source_region_reads']==0 and len(cov['summaries'])==2
    assert (R/'work/gravity-first-principles/mond-atlas-pressure-estimator-001/run001/summary.json').is_file()
    alt=read(R/'work/gravity-first-principles/mond-atlas-alternative-spectra-001/completion.json')
    assert alt['observed_source_spectra_read']==0
    derived=[];source=R/'work/gravity-first-principles/mond-atlas-source-sensitivity-001'
    for a in read(source/'derived-artifacts.json'):
        path=source/a['path'];assert relative(path) in NEW_GZIP
        packed=path.read_bytes();raw=gzip.decompress(packed)
        assert len(packed)==a['compressed_bytes'] and sha(packed)==a['sha256']
        assert len(raw)==a['csv_bytes'] and sha(raw)==a['csv_sha256']
        rows=list(csv.DictReader(io.StringIO(raw.decode())))
        assert len(rows)==a['rows'] and list(rows[0])==a['columns']
        finite=0;blanks=0
        for row in rows:
            for value in row.values():
                if value=='':blanks+=1;continue
                try:number=float(value)
                except ValueError:continue
                assert math.isfinite(number),('Nonfinite derived value',relative(path))
                finite+=1
        # Receipts count scientific numeric columns, excluding azimuth count etc.;
        # every parseable numeric value is additionally checked here.
        assert finite>=a['finite_numeric_values'] and blanks==a['explicit_undefined_blanks']
        derived.append(dict(path=relative(path),rows=len(rows),numeric_values_checked=finite,sha256=a['sha256']))
    assert {e['path'] for e in derived}==set(NEW_GZIP)
    lfs=[]
    for name in sorted(ALLOWED_GZIP):
        attribute=git('check-attr','filter','--',name).decode().strip();assert attribute.endswith(': lfs'),attribute
        blob=(R/name).read_bytes();lfs.append(dict(path=name,content_sha256=sha(blob),content_bytes=len(blob),expected_lfs_oid='sha256:'+sha(blob),index_check='run --verify-index after explicit scoped git add'))
    paths={e['path'] for e in entries}|{relative(PRIOR)}|new
    for name in PACKAGES:
        for path in (R/'work/gravity-first-principles'/name).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.name!='publication-manifest.json':paths.add(relative(path))
    for path in paths:
        assert '/private/' not in path
        if path not in ALLOWED_GZIP:assert Path(path).suffix.lower() not in ['.fits','.npz','.npy','.zip','.gz','.tar','.pdf'],path
    # All checks above are read-only. Parent-requested document append happens now.
    addendum=(P/'handoff-addendum.md').read_text(encoding='utf-8')
    for doc in DOCS:
        with (R/doc).open('a',encoding='utf-8',newline='') as f:f.write('\n'+addendum)
    checks=dict(prior_files_verified=len(entries),prior_entries_preserved=True,new_scripts_compiled=len(scripts),nested_python_compiled=len(nested_python),direct_local_imports_available=True,completion_receipt_hashes_verified=receipt_count,derived_compressed_tables=derived,covariance_transfer_fits=cov['fits'],covariance_variants=cov['variants'],nested_receipts_considered=len(receipt_paths),raw_observations_in_publication=0,observed_gravity_scores=0,goal_complete=False)
    write(P/'publication-checks.json',checks);write(P/'lfs-publication-plan.json',dict(status='CONTENT_AND_LFS_ATTRIBUTES_VERIFIED_INDEX_CHECK_PENDING',files=lfs,attributes_modified=False))
    paths.update({relative(P/'publication-checks.json'),relative(P/'lfs-publication-plan.json')})
    bound=[]
    for path in sorted(paths):
        blob=(R/path).read_bytes();bound.append(dict(path=path,bytes=len(blob),sha256=sha(blob)))
    manifest=P/'publication-manifest.json'
    write(manifest,dict(status='VERIFIED_FOR_ORDINARY_PUBLICATION',base_commit=git('rev-parse','HEAD').decode().strip(),goal_complete=False,raw_observation_files_included=0,explicit_derived_gzip=sorted(ALLOWED_GZIP),prior_manifest_entries=2233,files=bound))
    paths.add(relative(manifest));private=R/'work/private/mond-atlas-execution-033';private.mkdir(exist_ok=False)
    (private/'paths.nul').write_bytes(b'\0'.join(p.encode() for p in sorted(paths))+b'\0')
    print(json.dumps(dict(manifest_entries=len(bound),stage_paths=len(paths),compiled_scripts=len(scripts),pathspec=str(private/'paths.nul'),no_stage_commit_or_push_performed=True)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-index',action='store_true');args=parser.parse_args()
    verify_index() if args.verify_index else prepare()
