"""Verify completed packages and prepare the strictly scoped publication helper."""
import hashlib,json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3]
P=Path(__file__).resolve().parent
B=R/'work/gravity-first-principles'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
names=['scripts/build_mond_atlas_ngc2903_source.py','scripts/mond_atlas_common.py']
save(B/'mond-atlas-co-photometry-001/supplemental-dependency-receipt.json',dict(status='POST_ACCESS_DEPENDENCY_RECEIPT',prospectively_frozen=False,reason='Transitive omissions identified by independent review; original bindings unchanged.',files={n:sha(R/n) for n in names}))
count=0
for pkg,file in [('mond-atlas-co-photometry-001','run001/bindings.json'),('mond-atlas-selection-ablation-001','completion-receipt.json')]:
    obj=json.loads((B/pkg/file).read_text());bindings=obj.get('files',obj)
    for name,value in bindings.items():assert sha(R/name)==value,name;count+=1
prior=json.loads((B/'mond-atlas-execution-026/publication-manifest.json').read_text())
for e in prior['files']:assert sha(R/e['path'])==e['sha256'],e['path']
checks=[]
for path in ['work/gravity-first-principles/mond-atlas-noise-mode-power-001/test_mode_power.py','work/gravity-first-principles/mond-atlas-selection-ablation-001/independent-review/replay.py','scripts/mond_atlas_external_boundary_review.py']:
    proc=subprocess.run([sys.executable,path],cwd=R,capture_output=True,text=True)
    checks.append(dict(path=path,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr));assert proc.returncode==0,proc.stderr
save(P/'coordinator-verification.json',dict(prior_files_verified=len(prior['files']),bound_inputs_verified=count,checks=checks,goal_complete=False))
src=(B/'mond-atlas-execution-026/prepare_publication.py').read_text()
src=src.replace('execution-025','execution-026').replace("'mond-atlas-execution-026'","'mond-atlas-execution-027'").replace('private/mond-atlas-execution-026','private/mond-atlas-execution-027')
a=src.index('for pattern in ');b=src.index('    for p in ',a)
src=src[:a]+"for pattern in ['mond_atlas_external_boundary*.py','mond_atlas_noise_mode_power.py','mond_atlas_selection_ablation*.py','mond_atlas_co_photometry.py']:\n    paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'scripts').glob(pattern))\nfor name in ['mond-atlas-external-boundary-001','mond-atlas-noise-mode-power-001','mond-atlas-selection-ablation-001','mond-atlas-co-photometry-001','mond-atlas-execution-027']:\n"+src[b:]
(P/'prepare_publication.py').write_text(src,encoding='utf-8')
print(json.dumps(dict(prior_files=len(prior['files']),bindings=count,checks=len(checks))))
