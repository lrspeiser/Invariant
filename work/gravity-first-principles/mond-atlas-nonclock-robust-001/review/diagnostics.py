"""Post-hoc paired bootstrap and influence diagnostics of frozen predictions."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
rng=np.random.default_rng(9062491)
outputs=[];influence=[];bindings={}
for branch,families,filename,column in [('return',['finite_mix','truncated_point_kernel','finite_flat_bridge'],'per-galaxy.csv','mse_logspeed'),('coherence',['coherence_free'],'galaxy-scores.csv','mse')]:
    path=HERE.parent/branch/'run001'/filename
    bindings[str(path.relative_to(HERE.parent))]=hashlib.sha256(path.read_bytes()).hexdigest()
    rows=list(csv.DictReader(path.open(encoding='utf-8',newline='')))
    names=sorted(set(r['galaxy'] for r in rows));assert len(names)==102
    def avg(f):
        result=[]
        for n in names:
            values=[float(r[column]) for r in rows if r['galaxy']==n and r['family']==f]
            assert len(values)==3
            result.append(np.mean(values))
        return np.array(result)
    base=avg('mond_adjusted')
    for family in families:
        candidate=avg(family);d=base-candidate
        samples=rng.choice(d,(4000,len(d))).mean(axis=1)
        loo=100*(d.sum()-d)/(base.sum()-base)
        biggest=int(np.argmax(abs(d)));positive=int(np.argmax(d))
        outputs.append(dict(branch=branch,family=family,galaxies=len(names),mean_mse_difference=float(d.mean()),mse_gain_percent=float(100*d.sum()/base.sum()),bootstrap95_mean_difference=np.quantile(samples,[.025,.975]).tolist(),descriptive_bootstrap_positive_fraction=float((samples>0).mean()),largest_absolute_contributor=names[biggest],largest_absolute_share_of_absolute_total=float(abs(d[biggest])/abs(d).sum()),largest_positive_contributor=names[positive],largest_positive_share_of_net_gain=float(d[positive]/d.sum()),leave_one_out_gain_min_percent=float(loo.min()),leave_one_out_gain_max_percent=float(loo.max()),galaxies_improved_above_1e_minus12=int((d>1e-12).sum())))
        influence.extend(dict(branch=branch,family=family,galaxy=n,mean_mse_mond=float(base[i]),mean_mse_candidate=float(candidate[i]),difference=float(d[i]),leave_this_galaxy_out_gain_percent=float(loo[i])) for i,n in enumerate(names))
bindings['DIAGNOSTIC_ADDENDUM.md']=hashlib.sha256((HERE/'DIAGNOSTIC_ADDENDUM.md').read_bytes()).hexdigest()
bindings['diagnostics.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(HERE/'diagnostics.json').write_text(json.dumps(dict(status='POST_HOC_DESCRIPTIVE_NO_REFITTING_NOT_SIGNIFICANCE',seed=9062491,replicates=4000,results=outputs,bindings=bindings),indent=2)+'\n',encoding='utf-8')
with (HERE/'galaxy-influence.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(influence[0]));writer.writeheader();writer.writerows(influence)
print(json.dumps(outputs,indent=2))
