"""Build target-free grouped actual-emitter caches from streamed3D source."""
from mond_atlas_actual_spectra import *
import json,csv,gzip,hashlib,time
from scipy.interpolate import PchipInterpolator
from mond_atlas_hi_refinement import load_planar,iter_native_batches
from mond_atlas_hi_column import smooth
from mond_atlas_native_emission_sampled import grouped_aperture_flux

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def run():
    started=time.monotonic();out=PACKAGE/'cache001';out.mkdir(exist_ok=False);priv=PRIVATE/'cache001';priv.mkdir(parents=True,exist_ok=True)
    source_summary=ROOT/'work/gravity-first-principles/mond-atlas-hi-refinement-001/run001/summary.json';source=json.loads(source_summary.read_text(encoding='utf-8'));table=ROOT/'work/gravity-first-principles/mond-atlas-column-force-001/run004/column-force-table.csv.gz';column=ROOT/'work/private/mond-atlas-hi-column-001/run001/f4-column.npz'
    deps=[Path(__file__),ROOT/'scripts/mond_atlas_actual_spectra.py',PACKAGE/'PREFLIGHT.md',source_summary,table,column,ROOT/'scripts/mond_atlas_hi_refinement.py',ROOT/'scripts/mond_atlas_hi_column.py',ROOT/'scripts/mond_atlas_native_emission_sampled.py',ROOT/'scripts/mond_atlas_native_emission.py',ROOT/'scripts/mond_atlas_native_selection.py']+[ROOT/a['path'] for a in source['assets']]
    save(out/'bindings.json',{str(p.relative_to(ROOT)):digest(p) for p in deps})
    with gzip.open(table,'rt',newline='') as f:rows=list(csv.DictReader(f))
    with np.load(column) as z:rawr=z['raw_radius_kpc'];rawsig=z['raw_sigma_HI_msun_pc2']
    inst=load_instrument();assets=[]
    for asset in source['assets']:
        path=ROOT/asset['path'];assert digest(path)==asset['sha256'];planar=load_planar(path);r=planar['radius_kpc'];phi=planar['phi_rad'];gradient=np.empty_like(r)
        for i in range(0,len(r),512):_,gradient[i:i+512]=smooth(rawr,rawsig,r[i:i+512])
        arrays=dict(radius_kpc=r,phi_rad=phi,pressure_gradient=gradient,sin_inclination=np.sin(np.deg2rad(source['geometry']['inclination_deg'])),planar_flux=planar['flux_jy_km_s']);inside=None
        for height in ['h0p1','h0p4']:
            def select(component):return sorted([row for row in rows if row['case']==f'f4-stars-{height}' and row['component']==component],key=lambda row:float(row['r_kpc']))
            total=select('total');rr=np.array([float(row['r_kpc']) for row in total]);supported=(r>=rr.min())&(r<=rr.max());inside=supported if inside is None else inside&supported
            arrays[f'den_{height}']=PchipInterpolator(rr,[float(row['sigma_hi_msun_pc2']) for row in total],extrapolate=False)(r)
            for model,key in [('newton','newton_gbar'),('newton_plus_log','newton_plus_log_gbar')]:arrays[f'force_{height}_{model}']=np.array([PchipInterpolator(rr,[float(row[key]) for row in select(comp)],extrapolate=False)(r) for comp in ['stellar_luminosity','atomic_helium','co21']])
        arrays['force_supported']=inside
        for order in [24,48]:
            grouped=np.zeros((15,len(r)));totalflux=0.;nodes=0
            for batch in iter_native_batches(planar,vertical_order=order,max_nodes=65536):
                grouped+=grouped_aperture_flux(batch['xy_pixel'],batch['flux_jy_km_s'],batch['planar_index'],inst,group_count=len(r));totalflux+=batch['flux_jy_km_s'].sum();nodes+=len(batch['planar_index'])
            closure=abs(totalflux/planar['flux_jy_km_s'].sum()-1);assert closure<1e-8
            label=f"p{str(asset['step_kpc']).replace('.','p')}_z{order}";target=priv/f'{label}.npz';np.savez_compressed(target,**arrays,grouped_flux=grouped)
            assets.append(dict(label=label,step_kpc=asset['step_kpc'],vertical_order=order,cache=str(target.relative_to(ROOT)),sha256=digest(target),planar_nodes=len(r),projected_nodes=nodes,flux_closure=closure,force_unsupported_groups=int((~inside).sum()),total_input_flux_jy_km_s=float(planar['flux_jy_km_s'].sum())));print(label,'complete',flush=True)
    save(out/'summary.json',dict(status='ACTUAL_GROUPED_SOURCE_CACHES_BUILT',assets=assets,observed_source_spectra_read=0,seconds=time.monotonic()-started));assert all(digest(ROOT/p)==h for p,h in json.loads((out/'bindings.json').read_text()).items())
if __name__=='__main__':run()
