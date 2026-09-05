"""Produce the bounded hypothesis inventory and readable broad-search report."""
import hashlib
import json
import re
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_gravity_broad_patterns import ROOT,save,csvsave

D=ROOT/'work/gravity-first-principles/broad-pattern-findings-001'
OUT=ROOT.parents[1]/'outputs'

def read(p):return json.loads((ROOT/p).read_text())

def main():
    D.mkdir(parents=True,exist_ok=True);assert not any(D.iterdir())
    s=read('work/gravity-first-principles/broad-patterns-001/result.json')
    p=read('work/gravity-first-principles/population-patterns-001/result.json')
    mass=read('work/gravity-first-principles/composition-mass-control-001/result.json')['real']
    verify=read('work/gravity-first-principles/broad-patterns-001/verification.json')
    assert verify['status'].startswith('PASS')
    # Index earlier receipts without pretending to rerun or accept their conclusions.
    historical=[]
    for path in sorted((ROOT/'runs/gravity/roadmap').glob('item-*.json')):
        match=re.match(r'item-(\d+)-',path.name)
        if not match or int(match[1])>45:continue
        try:obj=json.loads(path.read_text())
        except (ValueError,UnicodeDecodeError):continue
        if 'decision' in obj:historical.append(dict(item=int(match[1]),path=str(path.relative_to(ROOT)),decision=obj['decision'],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),status='historical receipt indexed, not newly validated'))
    roadmap=(ROOT/'docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md').read_text(encoding='utf-8')
    mechanisms=[]
    fresh={2:'Partial new shape/morphology tests; exact 3D anisotropy unavailable',3:'New surface proxies tested; true volume density unmeasured',4:'New compactness/potential proxies tested',7:'New composition lead, substantially reduced by stellar-mass calibration',8:'New radial gradient/curvature tests',9:'New inner/outer radial light/force tests; no full 3D shells',10:'Partial profile-boundary tests; coverage diagnostic',12:'New spectral-population proxy prediction and sample transport; orbit age unmeasured',13:'New imaging disturbance tests give model-dependent results',14:'No direct resonance phase measurement; radial profile proxies only',15:'Source-only period/crossing proxies tested and algebraically identified',44:'Partial size and radial-scale ratios tested',45:'New limited nonlinear transforms and combined ridge'}
    for line in roadmap.splitlines():
        m=re.match(r'(\d+)\. \*\*(.*?)\*\* — (.*)',line)
        if m and int(m[1])<=45:
            num=int(m[1]);mechanisms.append(dict(item=num,name=m[2],proposal=m[3],current_test=fresh.get(num,'No new direct test in this round; requires a derived observable prediction and appropriate source data'),earlier_receipts=[r['path'] for r in historical if r['item']==num]))
    assert len(mechanisms)==45
    save(D/'hypothesis-inventory.json',dict(scope='45 finite mechanism families from the existing roadmap, not all possible theories',empty_space_explanation='set aside per user; historical records preserved',mechanisms=mechanisms,historical_receipts=historical))
    csvsave(D/'hypothesis-inventory.csv',[dict(item=r['item'],name=r['name'],current_test=r['current_test'],historical_receipts=len(r['earlier_receipts'])) for r in mechanisms])
    ranked=sorted(s['runs']['primary'],key=lambda r:-r['mse_gain_percent'])
    csvsave(D/'sparc-family-scores.csv',[dict(model=r['model'],log_acceleration_mse_gain_percent=r['mse_gain_percent'],fractional_speed_mse_gain_percent=r['fractional_gain_percent'],kms_mse_gain_percent=r['kms_gain_percent'],galaxies_improving=r['galaxies_improving'],diagnostic=r['diagnostic'],algebraic_alias=r['algebraic_alias']) for r in ranked])
    # Distinct panels keep unlike targets from looking like one physical contest.
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.4))
    names=['local_atomic_force_share','global_atomic_fraction','stellar_surface_density','radius','truncated_potential','force_slope','outer_stellar_contrast']
    labels=['Gas vs stars: force share','Gas vs stars: global mixture','Local stellar concentration','Radius / predicted orbital period','Potential-depth proxy','Radial force gradient','Outer stellar profile']
    values=[next(r['mse_gain_percent'] for r in ranked if r['model']==k) for k in names]
    axes[0].barh(labels[::-1],values[::-1],color=['#24838a' if v>0 else '#c67862' for v in values[::-1]])
    axes[0].axvline(0,color='black',lw=.7);axes[0].set(xlabel='Change in prediction MSE: positive is better (%)',title='SPARC: acceleration beyond ordinary matter\n39 descriptors screened; 86 galaxies')
    groups=['break','balmer','star_formation','all_population','crossing_proxy'];labels=['4000 Å stellar-population break','Balmer absorption','Star-formation indicators','All population indicators','Crossing-time proxy']
    for offset,kind,color in [(-.18,'ridge','#487ead'),(.18,'trees','#d99839')]:
        vals=[next(r['mse_gain_percent'] for r in p['runs'][kind]['summary'] if r['group']==k) for k in groups]
        axes[1].barh(np.arange(5)+offset,vals,height=.32,label='Linear control' if kind=='ridge' else 'Nonlinear control',color=color)
    axes[1].set_yticks(np.arange(5),labels);axes[1].invert_yaxis();axes[1].axvline(0,color='black',lw=.7)
    axes[1].set(xlabel='Change in prediction MSE: positive is better (%)',title='MaNGA: stellar velocity dispersion\n585 galaxies; a different target from SPARC');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(D/'broad-patterns.png',dpi=160);plt.close(fig)
    summary=dict(status='BROAD_DEVELOPMENT_SCREEN_COMPLETE_NO_CAUSAL_GRAVITY_LAW',mechanism_inventory=45,sparc_descriptors=39,sparc_galaxies=86,sparc_scored_positions=1654,manga_primary=585,manga_transport=243,
        best_sparc_descriptor='local_atomic_force_share',nominal_sparc_gain_percent=7.443273804764683,
        calibrated_mass_additional_composition_gain_percent=mass['incremental_composition_gain_percent'],
        population_primary_gain_percent=[next(r['mse_gain_percent'] for r in p['runs'][k]['summary'] if r['group']=='all_population') for k in ['ridge','trees']],
        population_transport_gain_percent=[next(r['mse_gain_percent'] for r in p['transport'][k]['summary'] if r['group']=='all_population') for k in ['ridge','trees']],
        all_possible_theories_exhausted=False,actual_orbit_age_measured=False,total_3d_density_measured=False,new_gravity_laws=0)
    save(D/'summary.json',summary)
    registry=json.loads((OUT/'Sigma-gravity-directions-v36.json').read_text());registry['predecessor']='Sigma-gravity-directions-v36.json'
    registry['current_user_exclusions']=['Empty-space explanation set aside; preserve historical evidence.']
    registry['broad_observable_pattern_search']=dict(summary=summary,evidence='work/gravity-first-principles/broad-pattern-findings-001',next='Separate stellar-population/mass-calibration effects from an acceleration residual before proposing an age-dependent force law.')
    save(D/'Sigma-gravity-directions-v37.json',registry)
    for a,b in [('summary.json','Gravity-broad-patterns-summary.json'),('hypothesis-inventory.csv','Gravity-hypothesis-inventory.csv'),('broad-patterns.png','broad-patterns.png'),('Sigma-gravity-directions-v37.json','Sigma-gravity-directions-v37.json')]:shutil.copy2(D/a,OUT/b)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
