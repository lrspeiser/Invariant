"""Actual full-emitter feasibility; no invented velocities at invalid nodes."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import csv,gzip,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from mond_atlas_hi_column import smooth
from mond_atlas_column_balance import FACTORS
from mond_atlas_pressure_support import SurfaceColumn,surface_balance
from mond_atlas_native_emission_sampled import load_instrument,grouped_aperture_flux,BRANCHES
from mond_atlas_native_selection import spectral_matrix

R=Path(__file__).resolve().parents[1]
P=R/'work/gravity-first-principles/mond-atlas-column-balance-001'


def run():
    start=time.monotonic();out=P/'emitter002';out.mkdir(exist_ok=False)
    source=R/'work/private/mond-atlas-hi-column-001/centroid-projection001/f4-emission-0.0625.npz'
    table=R/'work/gravity-first-principles/mond-atlas-column-force-001/run004/column-force-table.csv.gz'
    packet=R/'work/private/mond-atlas-hi-column-001/run001/f4-column.npz'
    paths=[Path(__file__),P/'EMITTER_PREFLIGHT.md',source,table,packet,
        R/'scripts/mond_atlas_hi_column.py',R/'scripts/mond_atlas_column_balance.py',
        R/'scripts/mond_atlas_pressure_support.py',R/'scripts/mond_atlas_native_emission_sampled.py',
        R/'scripts/mond_atlas_native_emission.py',R/'scripts/mond_atlas_native_selection.py']
    (out/'bindings.json').write_text(json.dumps({p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    with np.load(source) as z:
        planar,groups=np.unique(np.column_stack([z['radius_kpc'],z['phi_rad']]),axis=0,return_inverse=True)
        inst=load_instrument()
        grouped=grouped_aperture_flux(z['xy_pixel'],z['flux_jy_km_s'],groups,inst,group_count=len(planar))
        intrinsic=np.bincount(groups,weights=z['flux_jy_km_s'],minlength=len(planar))
    rr=planar[:,0]
    unique_r,radial_index=np.unique(rr,return_inverse=True)
    with np.load(packet) as z:smoothed,gradient=smooth(z['raw_radius_kpc'],z['raw_sigma_HI_msun_pc2'],unique_r)
    with gzip.open(table,'rt',newline='') as f:rows=list(csv.DictReader(f))
    bounds={}
    for branch in BRANCHES:
        H,grid,width=spectral_matrix(inst['parent_channels'],branch)
        operator=inst['continuum']@H
        bounds[branch]=float(np.max(np.sum(abs(operator),axis=1))/(abs(inst['velocity_increment_km_s'])*width))*1000
    results=[]
    for case in sorted(set(row['case'] for row in rows)):
        def select(name):return sorted([r for r in rows if r['case']==case and r['component']==name],key=lambda r:float(r['r_kpc']))
        total=select('total');r=np.array([float(row['r_kpc']) for row in total]);sigma=np.array([float(row['sigma_hi_msun_pc2']) for row in total])
        assert unique_r.min()>=r.min() and unique_r.max()<=r.max()
        den=PchipInterpolator(r,sigma)(unique_r);assert np.all(den>0)
        component=[select(name) for name in ('stellar_luminosity','atomic_helium','co21')]
        for model,key in [('newton','newton_gbar'),('newton_plus_log','newton_plus_log_gbar')]:
            base=np.array([PchipInterpolator(r,[float(row[key]) for row in c])(unique_r) for c in component])
            for material,factors in FACTORS.items():
                force=np.array(factors)@base;hi=factors[1]
                for reference in (5.,10.,15.):
                    balance=surface_balance(SurfaceColumn(unique_r,den*hi,smoothed*hi*reference**2,gradient*hi*reference**2),force)
                    invalid=(~balance.feasible)[radial_index]
                    weighted=grouped[:,invalid].sum(axis=1)*hi
                    radii=rr[invalid]
                    results.append(dict(case=case,model=model,material=material,pressure_reference_km_s=reference,
                        status=balance.status,invalid_planar_groups=int(invalid.sum()),
                        minimum_rotation_squared=float(balance.rotation_squared.min()),
                        invalid_radius_range_kpc=None if not invalid.any() else [float(radii.min()),float(radii.max())],
                        invalid_intrinsic_flux_fraction=float(intrinsic[invalid].sum()/intrinsic.sum()),
                        invalid_aperture_weighted_flux_jy_km_s_beam=weighted.tolist(),
                        max_invalid_aperture_flux_fraction=float(np.max(weighted/(grouped.sum(axis=1)*hi))),
                        max_unknown_spectral_contribution_mjy_beam={b:float(weighted.max()*bound) for b,bound in bounds.items()}))
    summary=dict(status='FULL_EMITTER_STEADY_BALANCE_TEST',planar_groups=len(planar),
        unique_radii=len(unique_r),profiles=len(results),fully_feasible_profiles=sum(r['status']=='STEADY_CIRCULAR_SOLUTION' for r in results),
        results=results,arbitrary_line_operator_bounds_mjy_per_Jykms=bounds,
        observed_source_spectra_read=0,observed_gravity_scores=0,seconds=time.monotonic()-start,
        scope='Declared centroid source and interpolated force table; not a full fluid solution or source-quadrature admission')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}))
    print(json.dumps([r for r in results if r['material']=='baseline']))


if __name__=='__main__':run()
