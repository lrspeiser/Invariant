"""Independent matching/force/pressure/support and unknown-flux decomposition."""
from pathlib import Path
import sys,json,hashlib,csv,gzip
import numpy as np
from scipy.interpolate import PchipInterpolator
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R/'scripts'))
from mond_atlas_actual_spectra import ActualSpectrumModel
P=Path(__file__).resolve().parent.parent;O=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def csvrows(p):
    with gzip.open(p,'rt',newline='') as f:return list(csv.DictReader(f))
def main():
    S=R/'work/gravity-first-principles/mond-atlas-source-sensitivity-001';summary=read(P/'cache001/summary.json');bindings=read(P/'cache001/bindings.json');assert all(sha(R/k)==v for k,v in bindings.items())
    hi=read(R/'work/gravity-first-principles/mond-atlas-alternative-hi-001/run001/summary.json');packets={a['path']:a for a in hi['assets']}
    fr=csvrows(S/'fields001/radial-fields.csv.gz');pr=csvrows(S/'hi001/pressure-profiles.csv.gz');records=[];bound=[];maxerr=0.
    for a in summary['assets']:
        assert sha(R/a['cache'])==a['sha256'];assert sha(R/a['source_packet'])==packets[a['source_packet'].replace('\\','/')]['sha256']
        with np.load(R/a['cache']) as z:d={k:z[k].copy() for k in z.files}
        with np.load(R/a['source_packet']) as z:
            assert np.array_equal(z['radius_kpc'],d['radius_kpc']) and np.array_equal(z['phi_rad'],d['phi_rad']) and np.array_equal(z['flux_jy_km_s'],d['planar_flux'])
            assert np.allclose(z['mass_HI_msun'],z['flux_jy_km_s']*235631.09406987214*3.611**2,rtol=5e-16,atol=1e-12)
        r=d['radius_kpc'];branch='common30_gaussian_approximation' if a['case'].startswith('common30') else a['case'];p=sorted([x for x in pr if x['branch']==branch and float(x['radial_spacing_kpc'])==.00625],key=lambda x:float(x['r_kpc']));rr=np.array([float(x['r_kpc']) for x in p]);den=PchipInterpolator(rr,[float(x['raw_sigma_hi']) for x in p],extrapolate=False)(r);grad=PchipInterpolator(rr,[float(x['smoothed_gradient']) for x in p],extrapolate=False)(r);inside=(r>=rr.min())&(r<=rr.max())&np.isfinite(den)&(den>0)&np.isfinite(grad)
        assert np.array_equal(den,d['den_h0p1'],equal_nan=True) and np.array_equal(grad,d['pressure_gradient'],equal_nan=True)
        f={}
        for grid in ['nf','lf']:
            comp=[]
            for c in ['stellar_luminosity','atomic_helium','co21']:
                rows=sorted([x for x in fr if x['case']==a['case'] and x['grid']==grid and x['component']==c and int(x['azimuth_nodes'])==2048],key=lambda x:float(x['r_kpc']));rad=np.array([float(x['r_kpc']) for x in rows]);inside&=(r>=rad.min())&(r<=rad.max());comp.append(PchipInterpolator(rad,[float(x['gbar_inward']) for x in rows],extrapolate=False)(r))
            f[grid]=np.array(comp)
        assert np.array_equal(f['nf'],d['force_h0p1_newton'],equal_nan=True) and np.array_equal(f['nf']+f['lf'],d['force_h0p1_newton_plus_log'],equal_nan=True) and np.array_equal(inside,d['force_supported'])
        grouped=d['grouped_flux'];assert np.isfinite(grouped).all() and (grouped>=0).all()
        masks={'unsupported_inner':(~inside)&(r<.05),'unsupported_outer':(~inside)&(r>6.025),'unsupported_other':(~inside)&(r>=.05)&(r<=6.025)}
        rec=dict(case=a['case'],label=a['label'],categories={k:dict(planar_groups=int(m.sum()),total_planar_flux_jy_km_s=float(d['planar_flux'][m].sum()),max_aperture_weighted_flux_jy_km_s=float(grouped[:,m].sum(1).max())) for k,m in masks.items()});records.append(rec)
        for gravity in ['newton','newton_plus_log']:
            for pressure in [5,10,15]:
                field=f['nf'].sum(0)+(f['lf'].sum(0) if gravity=='newton_plus_log' else 0);v2=r*(field+pressure**2*grad/den);valid=inside&np.isfinite(v2)&(v2>=0);negative=inside&np.isfinite(v2)&(v2<0)
                invalidflux=grouped[:,~valid].sum(1)
                for branch in ['boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated']:
                    model=ActualSpectrumModel(R/a['cache'],model=gravity,pressure_reference=pressure,branch=branch);assert np.array_equal(model.valid,valid)
                    predicted=2*invalidflux[:,None]*model.bound_coefficients[None,:];err=float(np.max(abs(predicted-model.unknown_envelope([0,10,2]))));maxerr=max(maxerr,err);assert err<1e-10
                    bound.append(dict(case=a['case'],label=a['label'],gravity=gravity,pressure=pressure,branch=branch,negative_supported_groups=int(negative.sum()),negative_supported_max_aperture_flux=float(grouped[:,negative].sum(1).max()),unknown_envelope_max_mjy_beam=float(predicted.max())))
    result=dict(status='MATCHED_ALTERNATIVE_CACHE_SUPPORT_REPLAY_PASS',caches=len(records),envelope_cases=len(bound),max_envelope_error=maxerr,unsupported_categories=records,envelopes=bound,source_region_spectra_read=0,limitations='Interpolation arithmetic independently replayed, same declared PCHIP rule. Native beam integration inherited, not independently reconstructed here. Envelope coefficient inherited already audited signed spectral operator; invalid flux independently recomputed.',bindings={str(p.relative_to(R)):sha(p) for p in [Path(__file__),P/'cache001/summary.json',P/'cache001/bindings.json']})
    with (O/'cache-receipt.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ['bindings','unsupported_categories','envelopes']}))
if __name__=='__main__':main()
