"""Independent factorial effects and stage-ledger replay, no cube arrays."""
import csv,hashlib,itertools,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];OWN=Path(__file__).resolve().parent;RUN=OWN.parent/'run001'


def read(path):return list(csv.DictReader(path.open(encoding='utf-8')))
def run():
    trials=read(RUN/'trials.csv');aggregates=read(RUN/'case-summary.csv');effects=read(RUN/'paired-effects.csv');gaps=read(RUN/'placement-gaps.csv');ledger=read(RUN/'stage-ledger.csv');channels=read(RUN/'channel-ledger.csv');groups={}
    keys=['group','branch','kind','amplitude','center','continuum','threshold']
    for row in trials:groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    for key in groups:groups[key]=sorted(groups[key],key=lambda r:int(r['draw']))
    maximum=0.
    def values(key):return np.array([float(r['true_flux_fraction_retained']) for r in groups[key]])
    def stats(v):return dict(mean=float(v.mean()),sd=float(v.std(ddof=1)) if len(v)>1 else 0.,minimum=float(v.min()),maximum=float(v.max()))
    for row in aggregates:
        key=tuple(row[k] for k in keys);v=values(key);s=stats(v)
        for field,computed in [('mean_retention',s['mean']),('sd',s['sd']),('minimum',s['minimum']),('maximum',s['maximum']),('mean_paired_flux',np.mean([float(r['paired_selected_flux_difference_over_reference']) for r in groups[key]]))]:maximum=max(maximum,abs(computed-float(row[field])))
    materials=0
    for row in effects:
        base=tuple(row[k] for k in keys[:5]);a=values(base+('actual','per_channel'));b=values(base+('source_disabled','per_channel'));c=values(base+('actual','fixed_global'));d=values(base+('source_disabled','fixed_global'))
        v={'remove_source_continuum':b-a,'fixed_threshold':c-a,'interaction':d-c-b+a}[row['effect']]
        for k,val in stats(v).items():maximum=max(maximum,abs(val-float(row[k])))
        material=abs(v.mean())>.05;assert material==(row['material']=='True');materials+=int(material)
    for row in gaps:
        base=tuple(row[k] for k in keys[:4]);suffix=(row['continuum'],row['threshold']);v=values(base+('20',)+suffix)-.5*(values(base+('10',)+suffix)+values(base+('30',)+suffix));s=stats(v)
        for key,val in [('center20_minus_outermean',s['mean']),('sd',s['sd']),('minimum',s['minimum']),('maximum',s['maximum'])]:maximum=max(maximum,abs(val-float(row[key])))
    assert maximum<1e-12
    previous=read(ROOT/'work/gravity-first-principles/mond-atlas-selection-transfer-001/run001/trials.csv');lookup={tuple(row[k] for k in ['group','branch','center','kind','draw','amplitude']):row for row in trials if row['continuum']=='actual' and row['threshold']=='per_channel'};prior_error=0.;count=0
    metrics=['true_flux_fraction_retained','paired_selected_flux_difference_over_reference','selected_noisy_flux_over_reference','reference_flux_jy_kms','post_continuum_flux_over_reference','selected_voxel_fraction']
    for row in previous:
        if row['group']=='gaussian':continue
        now=lookup[tuple(row[k] for k in ['group','branch','center','kind','draw','amplitude'])];prior_error=max(prior_error,max(abs(float(now[k])-float(row[k])) for k in metrics));assert now['peak_selected']==row['peak_selected'];count+=1
    assert count==702 and prior_error<1e-10
    ledger_error=0.
    for row in ledger:
        rr=[r for r in channels if all(r[k]==row[k] for k in ['branch','center','kind','continuum'])];assert len(rr)==42
        for key,channelkey in [('stored_pre_continuum','pre_continuum'),('stored_post_continuum','post_continuum'),('native_post_continuum','native'),('detector_post_continuum','detector')]:ledger_error=max(ledger_error,abs(sum(float(r[channelkey]) for r in rr)-float(row[key]))/max(abs(float(row[key])),1e-30))
    sigma={int(r['stored_channel']):float(r['western_mad']) for r in channels};median=float(np.median(list(sigma.values())));assert all(float(r['global_mad'])==median for r in channels)
    continuum_max=max(abs(float(r['continuum_signed_fraction'])-1) for r in ledger if r['continuum']=='actual');removed=max(abs(float(r['mean'])) for r in effects if r['effect']=='remove_source_continuum')
    mean_gap=lambda group,threshold:float(np.mean([float(r['center20_minus_outermean']) for r in gaps if r['group']==group and r['continuum']=='actual' and r['threshold']==threshold]))
    before=mean_gap('empirical','per_channel');after=mean_gap('empirical','fixed_global');reduction=100*(1-after/before)
    bindings=json.loads((RUN/'pre-access-bindings.json').read_text(encoding='utf-8'))['bindings']
    for p,h in bindings.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    result=dict(status='PASS_WITH_SCOPE_LIMITS',empirical_trials=sum(r['group']=='empirical' for r in trials),noiseless_trials=sum(r['group']=='noiseless' for r in trials),aggregate_rows=len(aggregates),paired_effect_rows=len(effects),gap_rows=len(gaps),all_gate_and_table_max_error=maximum,material_effects=materials,prior_replayed=count,prior_max_error=prior_error,ledger_rows=len(ledger),channel_ledger_rows=len(channels),ledger_relative_error=ledger_error,global_threshold_is_fixed_western_median=True,western_global_mad=median,maximum_source_continuum_fraction_loss=continuum_max,maximum_retention_change_from_source_continuum_removal=removed,empirical_center20_gap_per_channel=before,empirical_center20_gap_global=after,empirical_gap_relative_reduction_percent=reduction,noiseless_gap_per_channel=mean_gap('noiseless','per_channel'),noiseless_gap_global=mean_gap('noiseless','fixed_global'),all_input_hashes_verified=len(bindings),raw_background_or_observed_velocities_opened=False)
    (OWN/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':run()
