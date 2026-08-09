param(
    [Parameter(Mandatory = $true)]
    [string]$Repo,
    [string]$Python = "python",
    [string]$Generator = "generator-v2\target\release\sigma-generator-v2.exe",
    [int]$Threads = 8
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

& $Python -m sigma_theory_compiler.cli knowledge-build --repo $Repo --ontology configs\gate_ontology.json --database runs\knowledge-base\evidence.sqlite --summary runs\knowledge-base\summary.json
& $Python -m sigma_theory_compiler.cli formula-prioritize --database runs\knowledge-base\evidence.sqlite --output runs\knowledge-base\formula-priority.json
& $Generator basis --config configs\generator_v2_billion.json --output runs\knowledge-base\basis-v2.json
& $Generator run --config configs\generator_v2_billion.json --output runs\generator-v2\billion-survivor-export.json --threads $Threads --block-size 4194304 --survivor-dir runs\generator-v2\survivors
& $Python -m sigma_theory_compiler.cli survivor-audit --manifest runs\generator-v2\billion-survivor-export.json --survivor-dir runs\generator-v2\survivors --output runs\generator-v2\billion-survivor-audit.json
& $Python -m sigma_theory_compiler.cli dense-static-gpu --manifest runs\generator-v2\billion-survivor-export.json --survivor-dir runs\generator-v2\survivors --basis runs\knowledge-base\basis-v2.json --config configs\generator_v2_billion.json --status-dir runs\generator-v2\dense-status --output runs\generator-v2\billion-dense-static-gpu.json
& $Python -m sigma_theory_compiler.cli dense-static-crosscheck --dense-report runs\generator-v2\billion-dense-static-gpu.json --basis runs\knowledge-base\basis-v2.json --survivor-dir runs\generator-v2\survivors --status-dir runs\generator-v2\dense-status --output runs\generator-v2\billion-dense-static-crosscheck.json
& $Python -m sigma_theory_compiler.cli generated-prioritize --manifest runs\generator-v2\billion-survivor-export.json --survivor-dir runs\generator-v2\survivors --basis runs\knowledge-base\basis-v2.json --database runs\knowledge-base\evidence.sqlite --dense-report runs\generator-v2\billion-dense-static-gpu.json --dense-status-dir runs\generator-v2\dense-status --max-fronts 8 --output runs\knowledge-base\generated-priority-dense.json

Write-Host "Priority pipeline complete: runs\knowledge-base\generated-priority-dense.json"
