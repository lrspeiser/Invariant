"""Independent global mass/first moment and serialized packet replay."""
from pathlib import Path
import hashlib,json
import numpy as np
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-alternative-hi-001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    summary=json.loads((P/'run001/summary.json').read_text());bindings=json.loads((P/'run001/bindings.json').read_text())
    assert all(sha(R/k)==v for k,v in bindings.items());checks=[]
    for asset,row in zip(summary['assets'],summary['cases']):
        assert asset['case']==row['case'] and asset['step_kpc']==row['step_kpc'];assert sha(R/asset['path'])==asset['sha256']
        with np.load(R/row['source_path']) as z:axis=z['latent_axis'];surface=z['intrinsic_effective_surface']/1.36
        # All source boundary values are zero; exact bilinear hat integral is dx² per node.
        assert np.max(abs(surface[[0,-1],:]))==0 and np.max(abs(surface[:,[0,-1]]))==0
        mass=surface*(axis[1]-axis[0])**2*1e6;expected=np.array([mass.sum(),(mass*axis[:,None]).sum(),(mass*axis[None,:]).sum()])
        with np.load(R/asset['path']) as z:
            m=z['mass_HI_msun'];x=z['x_major_kpc'];y=z['y_minor_kpc'];actual=np.array([m.sum(),m@x,m@y]);err=np.abs(actual-expected)/expected[0]
            assert max(err)<1e-10
            assert np.allclose(z['radius_kpc'],np.hypot(x,y),rtol=0,atol=1e-14)
            assert np.allclose(z['phi_rad'],np.arctan2(y,x),rtol=0,atol=1e-14)
            assert np.allclose(z['flux_jy_km_s']*235631.09406987214*3.611**2,m,rtol=5e-16,atol=1e-12)
            for n in [24,48]:
                zz=z[f'z{n}_kpc'];w=z[f'z{n}_weight'];assert abs(w.sum()-1)<1e-14 and abs(w@zz)<1e-14 and abs(w@abs(zz)-.2)<1e-14 and abs(w@(zz*zz)-.08)<1e-14
        checks.append(dict(case=asset['case'],step_kpc=asset['step_kpc'],mass_and_first_moment_errors_per_total_mass=err.tolist(),pass_all=True))
    result=dict(status='SIX_PACKET_INDEPENDENT_GLOBAL_MOMENT_REPLAY_PASS',checks=checks,observed_source_spectra_read=0,bindings={str(p.relative_to(R)):sha(p) for p in [Path(__file__),P/'run001/summary.json',P/'run001/bindings.json']})
    with (P/'independent-review.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(result['status'])
if __name__=='__main__':main()
