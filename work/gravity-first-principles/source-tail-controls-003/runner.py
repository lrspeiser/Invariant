"""Run the exact current focused CI lint/test commands with input hashes."""
import json
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
workflow = ROOT/'.github/workflows/gravity-extensions.yml'
commands = [shlex.split(line.strip()[5:]) for line in workflow.read_text().splitlines()
            if line.strip().startswith(('run: python -m ruff check ', 'run: python -m pytest '))]
assert len(commands) == 2
paths = {workflow, Path(__file__), *list((ROOT/'src/invariant_gravity_extensions').glob('*.py'))}
for command in commands:
    for value in command:
        p = ROOT/value
        if p.is_file():
            paths.add(p)
paths.update((ROOT/'configs').glob('gravity_*_v10.json'))
paths.update((ROOT/'configs').glob('gravity_*source*audit_v1.json'))
paths.update([ROOT/'docs/GRAVITY_SOURCE_TAIL_DERIVATION.md', ROOT/'docs/GRAVITY_SOURCE_TAIL_RESULTS.md'])
hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def write(name, value):
    with (HERE/name).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


write('started.json', {'started_utc': datetime.now(UTC).isoformat(), 'input_hashes': hashes,
    'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
    'longdouble_mantissa_bits': np.finfo(np.longdouble).nmant})
rows = []
for command in commands:
    command[0] = sys.executable
    run = subprocess.run(command, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT/'src'),
        'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}, capture_output=True, text=True, check=False)
    rows.append({'command': command, 'exit_code': run.returncode, 'stdout': run.stdout, 'stderr': run.stderr})
    print(run.stdout, run.stderr, flush=True)
unchanged = all(sha256((ROOT/p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
write('result.json', {'rows': rows, 'inputs_unchanged': unchanged,
    'all_commands_pass': unchanged and all(row['exit_code'] == 0 for row in rows)})
write('receipt.json', {'result_sha256': sha256((HERE/'result.json').read_bytes()).hexdigest()})
if not unchanged or any(row['exit_code'] for row in rows):
    raise SystemExit(1)
