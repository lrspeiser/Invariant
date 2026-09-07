"""Scoped byte-preserving publication; no private arrays or unrelated files."""
import hashlib
import json
import subprocess
from pathlib import Path
R = Path(__file__).resolve().parents[3]
P = Path(__file__).resolve().parent
prior = R/'work/gravity-first-principles/mond-atlas-execution-029/publication-manifest.json'
assert not (P/'publication-manifest.json').exists()
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=R).strip()
entries = json.loads(prior.read_text())['files']
for e in entries:
    assert hashlib.sha256((R/e['path']).read_bytes()).hexdigest() == e['sha256'], e['path']
patterns = ['mond_atlas_log_source*.py', 'mond_atlas_newton_spectral*.py',
            'mond_atlas_observation_admission*.py', 'mond_atlas_selection_second_galaxy*.py',
            'mond_atlas_force_emission*.py']
scripts = sorted({p for pattern in patterns for p in (R/'scripts').glob(pattern)})
for p in scripts:
    compile(p.read_text(encoding='utf-8'), str(p), 'exec')
receipt = R/'work/gravity-first-principles/mond-atlas-observation-admission-001/completion-receipt.json'
checks = json.loads(receipt.read_text())['files']
for path, expected in checks.items():
    assert hashlib.sha256((R/path).read_bytes()).hexdigest() == expected, path
(P/'publication-checks.json').write_text(json.dumps(dict(
    prior_files_verified=len(entries), new_python_scripts_compiled=len(scripts),
    admission_receipt_files_verified=len(checks), observed_gravity_scores=0,
    failures_retained=True, goal_complete=False), indent=2)+'\n', encoding='utf-8')
for path, snapshot in [('docs/MOND_OBSERVATION_ATLAS_GOAL.md', 'prior-goal-handoff.md'),
                       ('docs/GRAVITY_PATTERN_SYSTEM_TASKS.md', 'prior-task-plan.md')]:
    target = R/path
    assert target.read_bytes() == (P/snapshot).read_bytes()
    with target.open('a', encoding='utf-8', newline='') as f:
        f.write('\n'+(P/'handoff-addendum.md').read_text(encoding='utf-8'))
paths = {e['path'] for e in entries}
paths.add(prior.relative_to(R).as_posix())
paths.update(p.relative_to(R).as_posix() for p in scripts)
for name in ['mond-atlas-log-source-001', 'mond-atlas-newton-spectral-001',
             'mond-atlas-observation-admission-001', 'mond-atlas-selection-second-galaxy-001',
             'mond-atlas-force-emission-001', 'mond-atlas-execution-030']:
    for p in (R/'work/gravity-first-principles'/name).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name != 'publication-manifest.json':
            paths.add(p.relative_to(R).as_posix())
bound = []
for path in sorted(paths):
    assert '/private/' not in path and Path(path).suffix.lower() not in ['.fits', '.npz', '.npy', '.zip', '.gz', '.tar', '.pdf']
    data = (R/path).read_bytes()
    bound.append(dict(path=path, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
manifest = P/'publication-manifest.json'
manifest.write_text(json.dumps(dict(status='VALIDATED_FOR_ORDINARY_GIT_PUBLICATION',
    base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=R, text=True).strip(),
    goal_complete=False, raw_observation_files_included=0, files=bound), indent=2)+'\n', encoding='utf-8')
paths.add(manifest.relative_to(R).as_posix())
private = R/'work/private/mond-atlas-execution-030'
private.mkdir(exist_ok=False)
(private/'paths.nul').write_bytes(b'\0'.join(p.encode() for p in sorted(paths))+b'\0')
print(json.dumps(dict(manifest_entries=len(bound), stage_paths=len(paths), scripts=len(scripts))))
