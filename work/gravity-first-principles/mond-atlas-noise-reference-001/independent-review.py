"""Independent gamma quadrature and finite binomial sum; no source arrays."""
from pathlib import Path
import csv,json,math,hashlib
from scipy.integrate import quad

P=Path(__file__).resolve().parent;ROOT=P.parents[2]
source=ROOT/'work/gravity-first-principles/mond-atlas-noise-dc-regularization-001/run001/channel-scores.csv'
previous=json.loads((P/'run001/summary.json').read_text())
bound=json.loads((P/'run001/bindings.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(ROOT/k)==v for k,v in bound.items())
rows=list(csv.DictReader(source.open()));n=27;m=len(rows);assert m==42
def pdf_q(q):
    return math.exp((n/2-1)*math.log(q)-n*q/2+(n/2)*math.log(n/2)-math.lgamma(n/2)) if q>0 else 0.
prob=quad(pdf_q,.8,1.2,epsabs=1e-13,epsrel=1e-13)[0]
count=sum(.8<=float(r['q'])<=1.2 for r in rows)
tail=sum(math.comb(m,k)*prob**k*(1-prob)**(m-k) for k in range(count+1))
replayed=[];maxerror=0.
for row,prior in zip(rows,previous['rows']):
    assert row['channel']==prior['channel']
    q=float(row['q']);lo=quad(pdf_q,0,q,epsabs=1e-12,epsrel=1e-12)[0]
    hi=quad(pdf_q,q,math.inf,epsabs=1e-12,epsrel=1e-12)[0]
    assert abs(lo+hi-1)<1e-10
    p=min(1.,2*min(lo,hi));adjusted=min(1,m*p)
    maxerror=max(maxerror,abs(p-prior['conditional_two_sided_p']),abs(adjusted-prior['bonferroni_p']))
    replayed.append(dict(channel=int(row['channel']),q=q,p=p,bonferroni_p=adjusted))
assert count==previous['descriptive_pass_count']
assert abs(prob-previous['reference_pass_probability'])<1e-10
assert abs(tail-previous['independent_channel_probability_pass_count_le_observed'])<1e-10
assert maxerror<1e-10
result=dict(status='INDEPENDENT_CONDITIONAL_ARITHMETIC_PASSED',
    method='Direct normalized gamma density of q integrated by quadrature; exact finite binomial sum, no scipy.stats.chi2/binom calls',
    bound_inputs_reverified=True,cores=n,channels=m,descriptive_pass_count=count,
    conditional_q_standard_deviation=math.sqrt(2/n),probability_in_descriptive_band=prob,
    expected_count=m*prob,independent_channel_count_sd=math.sqrt(m*prob*(1-prob)),
    independent_channel_count_lower_tail=tail,raw_p_range=[min(r['p'] for r in replayed),max(r['p'] for r in replayed)],
    adjusted_p_range=[min(r['bonferroni_p'] for r in replayed),max(r['bonferroni_p'] for r in replayed)],
    family_rejections=sum(r['bonferroni_p']<.05 for r in replayed),maximum_probability_replay_error=maxerror,
    known_correct_mean_and_variance_assumed=True,independent_gaussian_cores_assumed=True,
    binomial_requires_channel_independence=True,expected_count_does_not_require_channel_independence=True,
    bonferroni_requires_valid_marginals_not_channel_independence=True,
    fitted_pipeline_calibrated=False,noise_model_validated=False,source_arrays_opened=False,
    script_sha256=sha(Path(__file__)),rows=replayed)
with (P/'independent-review.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
print(result['raw_p_range'],result['adjusted_p_range'],prob,tail,maxerror)
