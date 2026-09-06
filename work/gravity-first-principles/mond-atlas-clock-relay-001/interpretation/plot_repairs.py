"""Plot saved results only; no parameter selection."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
package=Path(__file__).resolve().parents[1]
original=json.loads((package/'run001/summary.json').read_text(encoding='utf-8'))
mass=json.loads((package/'physics/scale-repair/run001/summary.json').read_text(encoding='utf-8'))
core=json.loads((package/'source-audit/core-repair/run001/summary.json').read_text(encoding='utf-8'))
old={v['family']:v['rmse_dex'] for v in original['metrics']}
scale={v['family']:v['rmse_dex'] for v in mass['metrics']}
cores={v['family']:v['rmse_dex'] for v in core['summary']}
fig,axes=plt.subplots(1,2,figsize=(11.5,4.8))
labels=['Original clock potential','Mass-based scale','Central taper only','Mass scale + central taper','Fixed MOND','Adjusted MOND']
vals=[old['clock_potential'],scale['clock_potential'],cores['clock_core_original_scale'],cores['clock_core_mass_scale'],old['mond_fixed'],old['mond_adjusted']]
axes[0].barh(labels,vals,color=['#bd7b20','#d99930','#b4af38','#679438','#4477aa','#228877']);axes[0].invert_yaxis()
axes[0].set_xlabel('Held-galaxy log-speed RMSE (dex), lower is better');axes[0].set_title('Clock-like force: tested development repairs')
original_regions=list(csv.DictReader((package/'run001/strata.csv').open(encoding='utf-8')))
core_regions=list(csv.DictReader((package/'source-audit/core-repair/run001/region-signed-bias.csv').open(encoding='utf-8')))
groups=['inner_r_over_Rd_lt1','middle_1to3','outer_ge3']
for f,label,color in [('clock_potential','Original clock','#bd7b20'),('mond_fixed','Fixed MOND','#4477aa'),('mond_adjusted','Adjusted MOND','#228877')]:
    v=[np.sqrt(np.mean([float(r['logspeed_rmse'])**2 for r in original_regions if r['family']==f and r['group']==g])) for g in groups]
    axes[1].plot(range(3),v,'o-',label=label,color=color)
v=[np.sqrt(np.mean([float(r['rmse_dex'])**2 for r in core_regions if r['family']=='clock_core_mass_scale' and r['region']==g])) for g in ['inner_lt1','middle_1to3','outer_ge3']]
axes[1].plot(range(3),v,'o-',label='Repaired clock',color='#679438')
axes[1].set_xticks(range(3),['Inner < Rd','Middle 1–3 Rd','Outer >=3 Rd']);axes[1].set_ylabel('Log-speed RMSE (dex)');axes[1].legend();axes[1].set_title('Central mismatch shrinks; outer gap remains')
fig.suptitle('102 galaxies, 2,212 radii | repairs are post-hoc development, not independent confirmation',fontsize=10)
fig.tight_layout();fig.savefig(package/'interpretation/clock-repair-comparison.png',dpi=160)
