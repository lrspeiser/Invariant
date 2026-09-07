"""Conditional finite-sample reference; not fitted-pipeline calibration."""
import csv,hashlib,json,math
from pathlib import Path
from scipy.stats import chi2,binom
from scipy.integrate import quad
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-noise-reference-001'
def run():
    out=P/'run001';out.mkdir(exist_ok=False)
    src=R/'work/gravity-first-principles/mond-atlas-noise-dc-regularization-001/run001/channel-scores.csv'
    bound={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),P/'SCOPE.md',src]}
    (out/'bindings.json').write_text(json.dumps(bound,indent=2)+'\n')
    rows=list(csv.DictReader(src.open()));n=27;m=len(rows);assert m==42
    probability=float(chi2.cdf(1.2*n,n)-chi2.cdf(.8*n,n));count=sum(.8<=float(r['q'])<=1.2 for r in rows)
    density=lambda x:math.exp((n/2-1)*math.log(x)-x/2-n/2*math.log(2)-math.lgamma(n/2)) if x>0 else 0
    norm,err=quad(density,0,math.inf,epsabs=1e-12,epsrel=1e-12);assert abs(norm-1)<1e-10
    independent,_=quad(density,.8*n,1.2*n,epsabs=1e-12,epsrel=1e-12);assert abs(independent-probability)<1e-10
    for r in rows:
        q=float(r['q']);pv=min(1.,2*min(float(chi2.cdf(n*q,n)),float(chi2.sf(n*q,n))))
        r.update(q=q,conditional_two_sided_p=pv,bonferroni_p=min(1.,m*pv))
    result=dict(status='RETROSPECTIVE_CONDITIONAL_NULL_REFERENCE',cores=n,channels=m,descriptive_pass_count=count,reference_pass_probability=probability,expected_pass_count=m*probability,independent_channel_count_sd=math.sqrt(m*probability*(1-probability)),independent_channel_probability_pass_count_le_observed=float(binom.cdf(count,m,probability)),family_alpha=.05,bonferroni_rejections=sum(r['bonferroni_p']<.05 for r in rows),minimum_bonferroni_p=min(r['bonferroni_p'] for r in rows),pdf_integral=norm,probability_integral_error=abs(independent-probability),rows=rows,actual_core_independence_validated=False,parameter_estimation_uncertainty_included=False,likelihood_admitted=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':run()
