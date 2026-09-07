"""Independent header, source-input and manufactured design checks; no response values."""
from pathlib import Path
import json,re,csv,hashlib
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SpectralCoord,SkyCoord
import astropy.units as u
from scipy.special import ndtr
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'work/gravity-first-principles/mond-atlas-observation-admission-001'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    out=P/'metadata-and-design-checks.json';assert not out.exists();rows=[]
    for name in ['NGC2976','NGC3198']:
        source=read(ROOT/f'work/gravity-first-principles/mond-atlas-native-spectral-001/{name}.json');h=fits.getheader(ROOT/source['source_path']);w=WCS(h).spectral
        native=w.pixel_to_world_values(np.arange(h['NAXIS3']));unit=u.Unit(w.world_axis_units[0]);rest=h['RESTFREQ']*u.Hz
        v=SpectralCoord(native*unit,doppler_rest=rest,doppler_convention='radio').to_value(u.km/u.s)
        manual=299792.458*(1-native/h['RESTFREQ']) if unit.is_equivalent(u.Hz) else native/1000
        beams=[tuple(map(float,m.groups())) for line in h['HISTORY'] if (m:=re.search(r'AIPS\s+CLEAN BMAJ=\s*([\d.E+-]+) BMIN=\s*([\d.E+-]+) BPA=\s*([\d.E+-]+)',line))]
        assert len(set(beams))==1
        rows.append(dict(galaxy=name,native_ctype=w.wcs.ctype[0],native_units=str(unit),radio_velocity_km_s_first_last=[float(v[0]),float(v[-1])],radio_strictly_decreasing=bool(np.all(np.diff(v)<0)),independent_conversion_error_km_s=float(np.max(abs(v-manual))),restfreq_hz=float(rest.value),beam_arcsec=[beams[0][0]*3600,beams[0][1]*3600],beam_PA_deg=beams[0][2],native_cube_values_read=False))
    integrity=read(P/'run004/integrity.json');packets=[]
    for x in integrity['source_arrays']:
        with np.load(ROOT/x['path']) as z:
            surf=z['intrinsic_effective_surface'];axis=z['latent_axis'];boundary=np.r_[surf[0],surf[-1],surf[:,0],surf[:,-1]]
            packets.append(dict(path=x['path'],source_shape=list(surf.shape),axis_shape=list(axis.shape),all_finite=bool(np.isfinite(surf).all()),nonnegative=bool((surf>=0).all()),zero_boundaries=bool((boundary==0).all()),axis_strictly_increasing=bool((np.diff(axis)>0).all()),target_nonfinite=int((~np.isfinite(z['target'])).sum()),source_mean_masked_nonfinite=int((~np.isfinite(z['source_mean'])).sum()),negative_source_means_are_not_mass=True))
    # This tests only geometry's ability to support a declared three-parameter spectrum model.
    with (P/'run004/frozen-geometric-apertures.csv').open(newline='') as f:ap=list(csv.DictReader(f))
    g=read(ROOT/'work/gravity-first-principles/mond-atlas-generic-source-001/run-002/summary.json')['cases'][0]['geometry'];h=fits.getheader(ROOT/'work/private/things-observable-12gal-003/NGC_2976_NA_CUBE_THINGS.FITS');cw=WCS(h).celestial;origin=SkyCoord(g['ra_deg']*u.deg,g['dec_deg']*u.deg);pa=np.deg2rad(g['pa_deg']);inc=np.deg2rad(g['inclination_deg']);directions=[]
    for r in ap:
        if r['subset']!='training':continue
        sky=cw.pixel_to_world(float(r['x_center']),float(r['y_center'])).transform_to(origin.skyoffset_frame());e,n=sky.lon.rad,sky.lat.rad
        major=e*np.sin(pa)+n*np.cos(pa);minor=(e*np.cos(pa)-n*np.sin(pa))/np.cos(inc);directions.append(major/np.hypot(major,minor))
    edges=np.sort(WCS(h).spectral.pixel_to_world_values(np.arange(43)-.5)/1000)
    def model(p):
        mean=p[0]+50*np.sin(inc)*np.asarray(directions)
        return (p[2]*np.diff(ndtr((edges[:,None]-mean[None,:])/p[1]),axis=0)).ravel()
    p=np.array([0.,10.,1.]);steps=np.array([.001,.001,.0001]);J=np.column_stack([(model(p+np.eye(3)[i]*steps[i])-model(p-np.eye(3)[i]*steps[i]))/(2*steps[i]) for i in range(3)])*np.array([30,17,1.5]);sv=np.linalg.svd(J,compute_uv=False)
    values=dict(status='METADATA_AND_MANUFACTURED_CHECKS_PASS',header_rows=rows,source_packets=packets,synthetic_training_design=dict(parameters=3,training_apertures=len(directions),velocity_for_manufactured_design_km_s=50,scaled_singular_values=sv.tolist(),ratio=float(sv[-1]/sv[0]),rank=int(np.linalg.matrix_rank(J)),not_actual_signal_identifiability=True),correction='NGC3198 increasing FREQ does NOT mean increasing radio velocity; both radio velocity grids decrease. Earlier failed script notes describe coordinate orientation only.',new_observed_scores=0)
    assert all(x['all_finite'] and x['nonnegative'] and x['zero_boundaries'] for x in packets);assert all(x['independent_conversion_error_km_s']<1e-8 for x in rows);assert sv[-1]/sv[0]>1e-6
    out.write_text(json.dumps(values,indent=2),encoding='utf8');print(json.dumps(values['synthetic_training_design']))
if __name__=='__main__':main()
