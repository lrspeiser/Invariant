"""Separate spectral/continuum bookkeeping and table replay; no cube pixels."""
import csv,json
import numpy as np
from mond_atlas_selection_ablation import ROOT,PKG
from mond_atlas_selection_transfer import intrinsic
from run_mond_atlas_native_selection import write_json,sha

out=PKG/'run001'
ledger=list(csv.DictReader((out/'stage-ledger.csv').open()))
profiles=list(csv.DictReader((out/'channel-ledger.csv').open()))
mat=json.loads((out/'linear-operators.json').read_text())
prov=json.loads((ROOT/'work/gravity-first-principles/mond-atlas-native-spectral-001/NGC2976.json').read_text())['provenance']
n=prov['parent_channel_count']; indices=prov['parent_channel_indices_zero_based']; fit=prov['continuum_fit_parent_indices_zero_based']
design=np.column_stack([np.ones(n),np.arange(n,dtype=float)])
max_ledger=0.; max_profile=0.
for row in ledger:
    branch=row['branch']; center=int(row['center']); kind=row['kind']
    b=mat['branches'][branch]; grid=np.array(b['precell_centers']); width=b['precell_width']
    raw,_=intrinsic(kind,grid,width,indices[center])
    if branch=='boxcar_independent': parent=raw
    else:
        parent=raw[:-2]*.25+raw[1:-1]*.5+raw[2:]*.25
        if branch=='boxcar_hanning_decimated': parent=parent[::2]
    stored=parent[indices]
    post=stored.copy()
    if row['continuum']=='actual':
        coef=np.linalg.lstsq(design[fit],parent[fit].reshape(len(fit),-1),rcond=None)[0]
        post-=(design[indices]@coef).reshape(stored.shape)
    vals={'intrinsic_pre_H_weighted':raw.sum()*width,'parent_post_H':parent.sum(),
          'stored_pre_continuum':stored.sum(),'stored_post_continuum':post.sum()}
    for key,val in vals.items():max_ledger=max(max_ledger,abs(float(row[key])/val-1))
    rr=[r for r in profiles if all(r[k]==row[k] for k in ['branch','center','kind','continuum'])]
    for key,total in [('post_continuum','stored_post_continuum'),('native','native_post_continuum'),('detector','detector_post_continuum')]:
        max_profile=max(max_profile,abs(sum(float(r[key]) for r in rr)/float(row[total])-1))
assert max_ledger<1e-10 and max_profile<1e-10
trials=list(csv.DictReader((out/'trials.csv').open())); effects=list(csv.DictReader((out/'paired-effects.csv').open()))
maximum=0.
for e in effects:
    rr=[r for r in trials if all(r[k]==e[k] for k in ['group','branch','center','kind','amplitude'])]
    cells={(r['draw'],r['continuum'],r['threshold']):float(r['true_flux_fraction_retained']) for r in rr}
    values=[]
    for draw in sorted({r['draw'] for r in rr}):
        a=cells[(draw,'actual','per_channel')];b=cells[(draw,'source_disabled','per_channel')]
        c=cells[(draw,'actual','fixed_global')];d=cells[(draw,'source_disabled','fixed_global')]
        values.append({'remove_source_continuum':b-a,'fixed_threshold':c-a,'interaction':d-c-b+a}[e['effect']])
    mean=sum(values)/len(values);maximum=max(maximum,abs(mean-float(e['mean'])))
    assert (abs(mean)>.05)==(e['material']=='True')
assert maximum<1e-12
bindings=json.loads((out/'pre-access-bindings.json').read_text())['bindings']
assert all(sha(ROOT/k)==v for k,v in bindings.items())
write_json(PKG/'separate-bookkeeping-review.json',dict(
    same_author_separate_implementation=True,shared_intrinsic_template=True,
    direct_spectral_filter_and_separate_polynomial_lstsq=True,
    ledger_relative_error=max_ledger,profile_sum_relative_error=max_profile,
    paired_effect_absolute_error=maximum,paired_effects_replayed=len(effects),
    saved_source_arrays_opened=False,independent_external_review_claimed=False,
    all_bound_files_reverified=True))
