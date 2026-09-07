"""Independent direct Fourier transform check of the actual bilinear maps."""
import hashlib,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from mond_atlas_newton_spectral import coefficients
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-newton-spectral-001/run001'
def main():
    config=json.loads((R/'work/gravity-first-principles/mond-atlas-spatial-program-001/source-bindings.json').read_text());seen=set();results=[]
    for case in config['source_cases']:
        for c in case['components']:
            if c['path'] in seen:continue
            seen.add(c['path'])
            with np.load(R/c['path']) as packet:
                coeff,mass=coefficients(packet,c['conversion_to_msun_pc2'],32.,.0625);nodes=packet['latent_axis'];surf=packet['intrinsic_effective_surface'];h=nodes[1]-nodes[0]
                for i,j in [(0,0),(1,2),(-3,7),(100,-77),(255,0),(-256,18)]:
                    kx,ky=2*np.pi*np.array([i,j])/32
                    # Direct integral of each linear tent via its analytic transform.
                    tx=h*np.sinc(kx*h/(2*np.pi))**2*np.exp(-1j*kx*nodes);ty=h*np.sinc(ky*h/(2*np.pi))**2*np.exp(-1j*ky*nodes)
                    exact=(tx@surf@ty)*c['conversion_to_msun_pc2']*1e6/32**2
                    error=float(abs(coeff[i%512,j%512]-exact)/(mass/32**2));assert error<1e-11
                    results.append(dict(path=c['path'],mode=[i,j],error_relative_to_mass_mode=error))
    out=dict(status='PASS',unique_packets=len(seen),modes_checked=len(results),max_error=max(r['error_relative_to_mass_mode'] for r in results),results=results,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (P/'coefficient-review.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(out['max_error'])
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
