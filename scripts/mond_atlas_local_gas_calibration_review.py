"""Independent quadrature, transcription and unit audit of the gas-only profile."""
import hashlib,json,re
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-local-gas-calibration-001'
def run():
    out=P/'independent-review';out.mkdir(exist_ok=True)
    b=json.loads((P/'run001/bindings.json').read_text());s=json.loads((P/'run001/summary.json').read_text());raw=(R/'work/private/mond-atlas-local-gas-calibration-001/paper.html').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==b['paper_sha256'] and len(raw)==b['paper_bytes']
    for path,h in b['files'].items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==h
    html=raw.decode();table=html[html.index('<table id="S5.T2.4"'):];table=table[:table.index('</table>')]
    # Independently transcribed Table 2 hydrogen nuclei profiles, not H2 molecules.
    profiles=[('H2',.15,105,2),('CNM',.80,127,2),('WNM1',.13,318,2),('WNM2',.077,403,1),('HII',.0154,1590,1)]
    assert 'Includes He and heavier elements with 40%' in table
    assert 'Gum Nebula that is not included' in table
    transcribed={(float(n),int(h)) for n,h in re.findall(r'alttext="([0-9.]+)\\exp-\(z/([0-9]+)~',table)}
    for name,n,h,power in profiles:assert (n,h) in transcribed,(name,n,h)
    # SI conversion: nuclei cm^-3 -> kg m^-3 -> solar masses pc^-3.
    from astropy.constants import m_p,pc,M_sun
    conversion=(1.4*m_p.to('kg').value)*(pc.to('cm').value**3)/M_sun.to('kg').value
    assert abs(conversion/s['rho_conversion_factor']-1)<1e-14
    x,w=leggauss(512);rows=[];errors=[];totals=[]
    for ell in [250,500]:
        values=[]
        for name,n,h,power in profiles:
            z=6*ell*(x+1);density=n*np.exp(-(z/h)**power);kernel=np.exp(-.5*(z/ell)**2)/(np.sqrt(2*np.pi)*ell)
            integral=float(12*ell*np.dot(w,density*kernel));values.append(integral*conversion)
            original=next(r for r in s['rows'] if r['component']==name and r['ell_pc']==ell);error=abs(integral/original['smoothed_nH_cm3']-1);assert error<1e-10;errors.append(error)
            rows.append(dict(component=name,ell_pc=ell,independent_nuclei_cm3=integral,relative_error=error))
        rho=sum(values);eps=(rho+.002)/(rho+.01);original=next(t for t in s['totals'] if t['ell_pc']==ell);assert abs(eps-original['epsilon_gas_only'])<1e-12
        totals.append(dict(ell_pc=ell,rho_msun_pc3=rho,epsilon_gas_only=eps))
    result=dict(status='PASS_RESTRICTED_GAS_PROFILE_CALCULATION',rows=rows,totals=totals,max_relative_error=max(errors),quadrature='512-point Gauss-Legendre on z=0..12ell, doubled; omitted Gaussian probability <4e-33',paper_sha256=b['paper_sha256'],hydrogen_nuclei_not_molecules=True,helium_factor=1.4,actual_lab_epsilon_established=False,source_3d_environment_established=False,review_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(totals));print(max(errors))
if __name__=='__main__':run()
