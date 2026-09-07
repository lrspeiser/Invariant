"""Verify CUDA against actual-source CPU at frozen nuisance corners."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
import cupy as cp
from mond_atlas_actual_spectra import ActualSpectrumModel
from mond_atlas_aperture_injection_gpu import GPUCallback

R=Path(__file__).resolve().parents[1]
P=R/'work/gravity-first-principles/mond-atlas-aperture-injection-001'

def run():
    out=P/'gpu001';out.mkdir(exist_ok=False)
    cache=R/'work/private/mond-atlas-actual-spectra-001/cache001/p0p03125_z48.npz'
    paths=[Path(__file__),cache,P/'GPU_ADDENDUM.md',R/'scripts/mond_atlas_actual_spectra.py',
           R/'scripts/mond_atlas_aperture_injection_gpu.py',R/'scripts/mond_atlas_native_emission_sampled.py',
           R/'scripts/mond_atlas_native_emission.py',R/'scripts/mond_atlas_native_selection.py']
    (out/'bindings.json').write_text(json.dumps({p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    rows=[]
    for branch in ('boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated'):
        cpu=ActualSpectrumModel(cache,branch=branch);gpu=GPUCallback(cpu)
        for parameters in [[0.,10.,1.]]+list(itertools.product((-30.,30.),(3.,20.),(.5,2.))):
            start=time.monotonic();reference=cpu.predict(parameters);cpu_seconds=time.monotonic()-start
            start=time.monotonic();actual=gpu.predict(parameters);gpu_seconds=time.monotonic()-start
            absolute=float(np.max(abs(reference-actual)));relative=float(np.linalg.norm(reference-actual)/np.linalg.norm(reference))
            rows.append(dict(branch=branch,parameters=list(parameters),max_absolute_error=absolute,
                             relative_error=relative,cpu_seconds=cpu_seconds,gpu_seconds=gpu_seconds,
                             passed=absolute<1e-10 and relative<1e-10))
    props=cp.cuda.runtime.getDeviceProperties(0)
    result=dict(status='ACTUAL_SOURCE_CUDA_ARITHMETIC_VERIFIED',all_pass=all(row['passed'] for row in rows),
                rows=rows,device=str(props['name']),cupy_version=cp.__version__,source_region_reads=0,
                max_absolute_error=max(row['max_absolute_error'] for row in rows),
                max_relative_error=max(row['relative_error'] for row in rows),
                median_cpu_seconds=float(np.median([row['cpu_seconds'] for row in rows])),
                median_gpu_seconds=float(np.median([row['gpu_seconds'] for row in rows])))
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    assert result['all_pass']

if __name__=='__main__':run()
