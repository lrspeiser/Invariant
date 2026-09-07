"""Post-run descriptive accounting only; no fitted parameters or new trials."""
from mond_atlas_selection_second_galaxy import *
def run():
    out=P/'descriptive-accounting';out.mkdir(exist_ok=False);det=read_json(P/'run001/detector-before-east.json');history=read_json(ROOT/'work/gravity-first-principles/mond-atlas-native-spectral-001/NGC3198.json');prov=history['provenance'];keep=prov['parent_channel_indices_zero_based'];op=continuum_operator(108,prov['continuum_fit_parent_indices_zero_based'],keep,1);h=fits.getheader(ROOT/history['source_path']);beam=beam_from_history(h);nc=beam_covariance(beam['major_arcsec'],beam['minor_arcsec'],beam['pa_deg'],h['CDELT2']*3600,h['CDELT1']*3600);nk=gaussian_kernel(nc);ek=gaussian_kernel(np.eye(2)*(30/FWHM_SIGMA/abs(h['CDELT1']*3600))**2-nc);rows=[]
    for branch in BRANCHES:
        H,grid,width=spectral_matrix(108,branch)
        for center in [10,36,61]:
            norm=None
            for kind in KINDS:
                for shift in [0,-4,4]:
                    raw=source(kind,grid,width,keep[center],shift=shift);pre=convolve_spatial((H@raw.reshape(len(grid),-1)).reshape(108,121,121),nk)
                    if norm is None:norm=pre[keep].max()
                    post=(op@pre.reshape(108,-1)).reshape(72,121,121)/norm;signal=convolve_spatial(post,ek)[:,20:101,20:101]
                    for amplitude in [5,10]:rows.append(dict(branch=branch,center=center,kind=kind,shift=shift,amplitude=amplitude,actual_maximum_detector_signal_over_sigma=float(np.max(signal/np.array(det['sigma'])[:,None,None])*amplitude*det['full_scale'])))
    write_csv(out/'actual-detector-SNR.csv',rows)
    with (P/'run001/case-summary.csv').open(newline='',encoding='utf-8') as f:cases=list(csv.DictReader(f))
    paired=[]
    for r in cases:
        if r['kind']=='rotation':continue
        ref=next(v for v in cases if v['kind']=='rotation' and all(v[k]==r[k] for k in ['detector','group','branch','center','shift','amplitude']))
        paired.append({**{k:r[k] for k in ['detector','group','branch','center','shift','amplitude','kind']},'mean_true_full_retention_difference':float(r['true_full_retained'])-float(ref['true_full_retained'])})
    write_csv(out/'paired-morphology.csv',paired)
    summary=dict(scope='Post-run amplitude interpretation and paired arithmetic, no retuning',detector_SNR_ranges={str(a):[min(r['actual_maximum_detector_signal_over_sigma'] for r in rows if r['amplitude']==a),max(r['actual_maximum_detector_signal_over_sigma'] for r in rows if r['amplitude']==a)] for a in [5,10]},full_empirical_max_abs_morphology_retention_difference=max(abs(r['mean_true_full_retention_difference']) for r in paired if r['detector']=='full_30arcsec' and r['group']=='empirical'))
    write_json(out/'summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':run()
