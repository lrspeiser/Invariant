"""Fixed central-aperture background and conditional Gaussian trial packets."""
from pathlib import Path
import hashlib,json
import numpy as np

R=Path(__file__).resolve().parents[1]
P=R/'work/gravity-first-principles/mond-atlas-aperture-injection-001'


def run():
    out=P/'noise001';out.mkdir(exist_ok=False)
    private=R/'work/private/mond-atlas-aperture-injection-001/noise001';private.mkdir(parents=True,exist_ok=False)
    config=R/'configs/mond_atlas_aperture_noise_v1.json'
    cfg=json.loads(config.read_text());source=R/cfg['input']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==cfg['input_sha256']
    working=R/'work/gravity-first-principles/mond-atlas-aperture-covariance-001/run001/working-covariance.json'
    covariance=json.loads(working.read_text());C=np.array(covariance['covariance']);L=np.array(covariance['cholesky']);mean=np.array(covariance['mean'])
    assert np.max(abs(L@L.T-C))<1e-12
    paths=[Path(__file__),P/'PREFLIGHT.md',config,source,working]
    (out/'bindings.json').write_text(json.dumps({p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    with np.load(source) as z:east=z['validation']
    assert east.shape==(27,24,24,42) and np.isfinite(east).all()
    native=east[:,6:18,6:18].mean(axis=(1,2))
    direct=np.zeros_like(native)
    for y in range(6,18):
        for x in range(6,18):direct+=east[:,y,x]/144
    error=float(np.max(abs(native-direct)));assert error<1e-12
    real=np.array([native[(np.arange(15)+trial)%27] for trial in range(8)])
    seeds=list(range(9063001,9063009))
    gaussian=np.array([mean+np.random.default_rng(seed).standard_normal((15,42))@L.T for seed in seeds])
    dest=private/'noise.npz'
    np.savez_compressed(dest,real_background=real,gaussian_background=gaussian,
                        covariance=C,western_mean=mean,seeds=seeds)
    receipt=dict(status='EXACT_APERTURE_NOISE_PACKET_READY_FOR_ADMITTED_TEMPLATE',
        background_units='mJy/native_beam',trials_per_family=8,apertures=15,channels=42,
        scalar_mean_max_error=error,source_region_reads=0,gravity_scores=0,
        eastern_exposure='Previously exposed development backgrounds; cyclic assignments share cores',
        path=dest.relative_to(R).as_posix(),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        bytes=dest.stat().st_size,seeds=seeds,template_trials_executed=False)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':run()
