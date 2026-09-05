"""Recalculate saved metrics, authenticate sources, and test target isolation."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import run_gravity_broad_patterns as broad
import run_gravity_population_patterns as pop

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'work/gravity-first-principles/broad-patterns-001'

def load(p):return json.loads(p.read_text())

def check_sparc():
    result=load(D/'primary.json');rows=list(csv.DictReader((D/'predictions.csv').open()));assert len(rows)==1684
    for score in result['galaxy_scores']:
        rs=[r for r in rows if r['name']==score['name'] and r['supported']=='True']
        assert len(rs)==score['positions']
        for output,col in [('base_mse','baseline'),('model_mse',score['model'])]:
            mse=np.mean([(float(r['y'])-float(r[col]))**2 for r in rs]);assert abs(mse-score[output])<1e-14
    # Changing one outer fold's target must not alter its inner-selected model.
    d=broad.make_data();fold=broad.split(d['name'],'broad-A',5);changed={**d,'y':d['y'].copy()};changed['y'][fold==0]+=3
    replay,pred,_=broad.evaluate(changed,only_selector=True)
    original=next(r for r in result['choices'] if r['model']=='selected_physical' and r['outer_fold']==0)
    assert replay['choices'][0]==original
    saved=np.array([float(r['selected_physical']) for r in rows]);assert np.allclose(pred['selected_physical'][fold==0],saved[fold==0],rtol=0,atol=1e-12)
    # Feature injection is a computational sensitivity check, not physical proof.
    feature=d['F'][:,d['keys'].index('local_atomic_force_share')]
    unconditioned={**d,'y':broad.prior.rar(d['x'],np.log10(1.2e-10))-.15*feature}
    unrestricted=broad.evaluate(unconditioned,kind='rar',only_selector=True)[0]['summary'][0]['mse_gain_percent']
    B=broad.prior.basis(d['x'])
    feature=feature-B@broad.prior.smooth_fit(B,feature,broad.prior.weights(d['name']))
    synthetic={**d,'y':broad.prior.rar(d['x'],np.log10(1.2e-10))-.15*feature}
    injected,_,_=broad.evaluate(synthetic,kind='rar',only_selector=True)
    assert injected['summary'][0]['mse_gain_percent']>50
    return dict(saved_metrics_recalculated=True,outer_target_does_not_choose_its_model=True,
        injected_effect_gain_percent=injected['summary'][0]['mse_gain_percent'],
        unconditioned_injection_gain_percent=unrestricted,
        injection_limit='The residualized feature search is sensitive to an added conditional effect; it does not fully recover every acceleration-dependent mean shift. The unconditioned injection is retained as a counterexample to broad detection-power claims.')

def check_manga():
    result=load(pop.OUT/'result.json');a,_=pop.rows(12);b,_=pop.rows(13)
    for item,rs in [(12,a),(13,b)]:
        stem='item-12-manga-dynamical-age' if item==12 else 'item-13-manga-relaxation-mergers'
        directory=ROOT/('runs/gravity/roadmap/'+stem+'-v1-source')
        receipt=load(ROOT/('runs/gravity/roadmap/'+stem+'-v1.json'))
        for filename,key in [('response-source.json','response_source_sha256'),('sample-manifest.json','sample_manifest_sha256'),('predictor-source.json','predictor_source_sha256')]:
            digest=hashlib.sha256((directory/filename).read_bytes().replace(b'\r\n',b'\n')).hexdigest();assert digest==receipt['inputs'][key]
        source={r['plateifu']:r for r in load(directory/'response-source.json')['records']}
        for r in rs:assert abs(float(r['stellar_sigma_1re_km_s'])-float(source[r['plateifu']]['stellar_sigma_1re']))<1e-8
    assert not set(r['mangaid'] for r in a)&set(r['mangaid'] for r in b)
    for kind in ['ridge','trees']:
        for prefix,group in [('predictions','runs'),('transport','transport'),('disturbance','disturbance')]:
            rows=list(csv.DictReader((pop.OUT/(prefix+'_'+kind+'.csv')).open()))
            baseline=np.array([(float(r['y'])-float(r['baseline']))**2 for r in rows])
            for s in result[group][kind]['summary']:
                e=np.array([(float(r['y'])-float(r[s['group']]))**2 for r in rows])
                assert abs(100*(1-e.mean()/baseline.mean())-s['mse_gain_percent'])<1e-10
    assert result['crossing_alias_max_residual']<1e-9
    return dict(source_receipt_hashes_pass=True,raw_dispersion_replay=True,saved_metrics_recalculated=True,
                transport_samples_disjoint=True,crossing_clock_not_independent=True)

def main():
    a=check_sparc();b=check_manga()
    sources=['configs/gravity_broad_pattern_search_v1.json','configs/sparc_rotation_curves_full_v1.json','configs/sparc_surface_brightness_exploration_v1.json','work/gravity-first-principles/map-response-metadata-001/SPARC_Lelli2016c.mrt','scripts/run_gravity_broad_patterns.py','scripts/run_gravity_population_patterns.py','scripts/check_gravity_composition_mass_calibration.py','scripts/run_gravity_matched_concentration.py']
    receipt=dict(status='PASS_COMPUTATIONAL_AND_SOURCE_CHECKS_NOT_PHYSICS_ADMISSION',sparc=a,manga=b,
        source_hashes=[dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in sources])
    broad.save(D/'verification.json',receipt);print(json.dumps(receipt,indent=2)[:1800])

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
