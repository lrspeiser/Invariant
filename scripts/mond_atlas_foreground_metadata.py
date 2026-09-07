"""Header-only spectral-coordinate audit; no science-array access."""
from pathlib import Path
import json,hashlib,warnings,urllib.request,re
import numpy as np
import astropy
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.constants import c

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'work/gravity-first-principles/mond-atlas-foreground-metadata-001'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

out=P/'run001';out.mkdir(exist_ok=False)
cfgpath=ROOT/'configs/mond_atlas_native_selection_v1.json'
config=json.loads(cfgpath.read_text());cube=ROOT/config['cube_path']
assert digest(cube)==config['cube_sha256']
bindings={p.relative_to(ROOT).as_posix():digest(p) for p in [Path(__file__),P/'SCOPE.md',cfgpath,cube]}
save(out/'bindings.json',bindings)
h=fits.getheader(cube)
keys=['NAXIS','NAXIS3','NAXIS4','CTYPE3','CUNIT3','CRPIX3','CRVAL3','CDELT3','RESTFREQ','RESTFRQ','VELREF','SPECSYS','SSYSOBS','VELOSYS','DATE-OBS','OBJECT','TELESCOP']
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always');wcs=WCS(h);spectral=wcs.spectral
messages=[str(w.message) for w in caught]
assert h['CTYPE3']=='VELO-HEL' and h['VELREF']==258
assert spectral.wcs.ctype[0]=='VRAD' and spectral.wcs.specsys=='BARYCENT'
indices=np.array([10,20,21,30],float)
world=spectral.all_pix2world(indices[:,None],0)[:,0]
manual=h['CRVAL3']+(indices+1-h['CRPIX3'])*h['CDELT3']
error=float(np.max(abs(world-manual)));assert error<1e-9
roundtrip=float(np.max(abs(spectral.all_world2pix(world[:,None],0)[:,0]-indices)));assert roundtrip<1e-10
radio=(world*u.Unit(str(spectral.wcs.cunit[0]))).to(u.km/u.s)
rest=h['RESTFREQ']*u.Hz
frequency=radio.to(u.Hz,equivalencies=u.doppler_radio(rest))
optical=frequency.to(u.km/u.s,equivalencies=u.doppler_optical(rest))
analytic=radio/(1-radio/c.to(u.km/u.s))
operror=float(np.max(abs(optical.value-analytic.value)));assert operror<1e-7
rows=[dict(stored_index_zero_based=int(i),fits_channel_one_based=int(i+1),native_radio_barycentric_kms=float(v),
           same_frame_optical_parameter_kms=float(o),rest_frame_frequency_hz=float(f),
           optical_minus_radio_kms=float(o-v)) for i,v,o,f in zip(indices,radio.value,optical.value,frequency.value)]
save(out/'header-and-coordinates.json',dict(cube=config['cube_path'],sha256=config['cube_sha256'],
    original_header={k:h.get(k) for k in keys},astropy_version=astropy.__version__,wcslib_ctype=str(spectral.wcs.ctype[0]),
    normalized_unit=str(spectral.wcs.cunit[0]),normalized_specsys=spectral.wcs.specsys,warnings=messages,
    coordinates=rows,independent_linear_error_m_per_s=error,pixel_roundtrip_error=roundtrip,
    optical_reparameterization_error_kms=operror,no_time_dependent_reference_frame_correction=True,
    science_array_values_accessed=False))
source_rows=[]
for url in ['https://arxiv.org/html/0810.2125','https://arxiv.org/html/1903.03767',
            'https://arxiv.org/html/1903.03767v1/pvslice.png',
            'https://arxiv.org/html/1903.03767v1/profile_DRAO_N2976.svg']:
    with urllib.request.urlopen(url,timeout=30) as response:
        raw=response.read(2000001);mime=response.headers.get('content-type')
    assert len(raw)<=2000000
    row=dict(url=url,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),content_type=mime,raw_saved=False)
    if 'html' in mime:
        text=raw.decode('utf8')
        row['frame_term_occurrences']={term:len(re.findall(term,text,re.I)) for term in ['heliocentric','barycentric','LSRK','LSRD','LSR']}
    source_rows.append(row)
save(out/'paper-source-evidence.json',dict(sources=source_rows,
    publisher_url='https://academic.oup.com/mnras/article/486/1/504/5374532',
    publisher_fetch='Web tool failed; author arXiv preprint used.',
    figure3_visual_axis='Velocity (km/s); no rest-frame qualifier seen',
    figure5_svg_note='No textual frame metadata found; glyph-only labels are not a positive frame identification.',
    drao_frame='UNRESOLVED from inspected paper text/figure3',drao_doppler_convention='UNRESOLVED',
    numerical_crossmatch_admitted=False))
save(out/'literature-metadata.json',dict(
    things=dict(source='https://arxiv.org/html/0810.2125',table2_observation_center_kms=3.0,
        table2_definition='Barycentric (heliocentric), optical observing setup',table5_and_figure30_systemic_kms=2.6,
        native_cube_differs_in_doppler_parameterization=True),
    drao=dict(source='https://arxiv.org/html/1903.03767',section='2.3',published_velocity_intervals_kms=[[-76.3,-40.0],[-23.5,19.3]],
        published_peaks_kms=[-56.5,-2.1],reference_frame_unestablished=True,
        values_not_transferred_to_things_channels=True,different_observation_and_resolution=True)))
assert all(digest(ROOT/k)==v for k,v in bindings.items())
save(out/'summary.json',dict(status='METADATA_AUDIT_COMPLETED_FRAME_CROSSMATCH_BLOCKED',
    actual_spectral_type='VRAD from legacy VELO-HEL/VELREF258; not FELO-HEL',science_arrays_opened=False,
    foreground_diagnosed=False,channels_masked=0,gravity_scores=0,all_bindings_reverified=True))
print(json.dumps(rows,indent=2))
