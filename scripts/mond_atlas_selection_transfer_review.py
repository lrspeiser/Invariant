"""Separate algebra and table replay; no observational array access."""
import csv,json,math
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from mond_atlas_selection_transfer import intrinsic,ROOT,PKG
from run_mond_atlas_native_selection import write_json,sha

out=PKG/'run001'
maximum=0.
for label in ['rotation','warp','streaming']:
    # Separately written CDF cell integral, polar velocity expression.
    a=np.arange(81)-40
    horizontal=np.broadcast_to(a[None,:]*1.5,(81,81))
    vertical=np.broadcast_to(a[:,None]*3.,(81,81))
    radius=np.sqrt(horizontal**2+vertical**2)
    angle=np.arctan2(vertical,horizontal)
    twist=(math.pi/6)*radius/(radius+15) if label=='warp' else 0.
    velocity=4*np.tanh(radius/15)*np.cos(angle-twist)
    if label=='streaming': velocity+=2*np.tanh(radius/15)*np.sin(angle)
    brightness=np.exp(-math.log(2)*radius**2/225)
    sd=1/math.sqrt(2*math.log(2))
    for width in [1.,.5]:
        grid=np.arange(64)*width
        center=31.
        hi=(grid[:,None,None]+width/2-center-velocity)/sd
        lo=(grid[:,None,None]-width/2-center-velocity)/sd
        reference=(ndtr(hi)-ndtr(lo))*sd*math.sqrt(2*math.pi)/width*brightness
        calculated,_=intrinsic(label,grid,width,center)
        maximum=max(maximum,float(np.max(abs(reference-calculated))))
assert maximum<1e-12
trials=list(csv.DictReader((out/'trials.csv').open()))
cases=list(csv.DictReader((out/'case-summary.csv').open()))
pairs=list(csv.DictReader((out/'paired-morphology.csv').open()))
max_error=0.
for case in cases:
    selected=[r for r in trials if all(r[k]==case[k] for k in ['group','branch','center','amplitude','kind'])]
    for metric in ['true_flux_fraction_retained','paired_selected_flux_difference_over_reference','selected_noisy_flux_over_reference']:
        mean=sum(float(r[metric]) for r in selected)/len(selected)
        max_error=max(max_error,abs(mean-float(case[metric+'_mean'])))
    truth=float(case['true_flux_fraction_retained_mean'])
    paired=float(case['paired_selected_flux_difference_over_reference_mean'])
    assert (truth>=.9 and abs(paired-1)<=.1)==(case['adequate_recovery']=='True')
for pair in pairs:
    selected=[r for r in trials if all(r[k]==pair[k] for k in ['group','branch','center','amplitude'])]
    base={r['draw']:float(r['true_flux_fraction_retained']) for r in selected if r['kind']=='rotation'}
    other={r['draw']:float(r['true_flux_fraction_retained']) for r in selected if r['kind']==pair['kind']}
    delta=sum(other[k]-base[k] for k in base)/len(base)
    max_error=max(max_error,abs(delta-float(pair['mean_retention_difference'])))
    assert (abs(delta)<=.05)==(pair['transfer_gate_pass']=='True')
assert max_error<1e-12
bindings=json.loads((out/'pre-access-bindings.json').read_text())['bindings']
assert all(sha(ROOT/k)==v for k,v in bindings.items())
write_json(PKG/'algebra-table-review.json',dict(
    separate_template_cdf_and_polar_algebra_max_abs=maximum,
    table_replay_max_abs=max_error,case_gates_replayed=len(cases),
    paired_gates_replayed=len(pairs),bound_files_reverified=len(bindings),
    same_author_separate_algebra=True,independent_external_review_claimed=False,
    observational_arrays_opened=False,
    limitation='Does not independently replay the full injection operator; inherited direct convolution controls apply.'))
