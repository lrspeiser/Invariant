import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path(__file__).parent;root=base/'Invariant';outputs=base.parent/'outputs'
dest=root/'work/gravity-first-principles/sparc-pattern-findings-001'
result=json.loads((dest/'result.json').read_text());pred=json.loads((dest/'predictions.json').read_text())
strata=[]
for salt in sorted(set(r['salt'] for r in pred)):
    rows=[r for r in pred if r['salt']==salt]
    cut=float(np.median([np.median(r['rar']) for r in rows]))
    for label,subset in [('lower_speed',[r for r in rows if np.median(r['rar'])<=cut]),('higher_speed',[r for r in rows if np.median(r['rar'])>cut])]:
        b=np.mean([np.mean((np.array(r['vobs'])-r['rar'])**2) for r in subset])
        a=np.mean([np.mean((np.array(r['vobs'])-r['predicted'])**2) for r in subset])
        strata.append(dict(salt=salt,group=label,n=len(subset),threshold_RAR_kms=cut,kms_mse_gain=float(1-a/b)))
(dest/'speed_strata.json').write_text(json.dumps(dict(rows=strata,scope='Post-result descriptive median split using baryon-predicted speed; not a fitted activation threshold.'),indent=2),encoding='utf-8')

fig,axes=plt.subplots(2,2,figsize=(11,7.4),layout='constrained')
for ax,name in zip(axes.flat,['NGC0055','UGC07524','NGC3521','NGC3741']):
    rows=[r for r in pred if r['name']==name]; r=np.array(rows[0]['r']);v=np.array(rows[0]['vobs'])
    curves=np.array([row['predicted'] for row in rows]);rar=np.array(rows[0]['rar'])
    ax.errorbar(r,v,yerr=rows[0]['error'],fmt='.',color='#333333',alpha=.8,label='SPARC measurements')
    ax.plot(r,rar,color='#b35806',label='RAR benchmark')
    ax.plot(r,curves.mean(axis=0),color='#2166ac',label='Gas-profile correction')
    ax.fill_between(r,curves.min(axis=0),curves.max(axis=0),color='#2166ac',alpha=.18,label='Two splits; not a confidence band')
    gains=[1-np.mean(((v-c)/rar)**2)/np.mean(((v-rar)/rar)**2) for c in curves]
    text='improves' if min(gains)>0 else 'worsens'
    ax.set(title=f'{name}: {text} in both splits',xlabel='Radius (kpc)',ylabel='Rotation speed (km/s)')
    ax.grid(alpha=.18)
axes[0,0].legend(fontsize=8)
fig.suptitle('Real SPARC examples: a promising feature, an incomplete model\nSelected successes and failures from exposed development data',fontsize=12)
fig.savefig(outputs/'SPARC-pattern-examples.png',dpi=170);plt.close(fig)

report='''# SPARC deep analysis: patterns found, and limits

The most useful new lead is a relationship to the radial gas-contribution profile, particularly the difference between the local gas contribution and its broader exterior average. This is a statistical source-profile signal on existing development data, not proof that outside gas creates an anomalous inward force. Simple compactness/brightness labels do not provide a robust explanation of the residuals.

We analyzed all 139 exposed SPARC development galaxies and 2,720 radii. The nominal RAR score was independently reproduced (chi-square 130714.6893155). Only the historical exploration whitelist was analyzed. The model predicts fractional velocity residuals from baryonic components, surface brightness, and radius, with no observed velocity as an input. A poison test confirms that changing observed velocities and errors leaves source features unchanged.

## Pattern 1: most error concerns whole-galaxy normalization

Under equal-galaxy weighting of squared fractional velocity residuals, 71.25% of the RAR error lies in the mean offset of each galaxy; 28.75% remains as variation along its curve. This is a mathematical decomposition, not evidence that calibration explains 71% of the discrepancy. Distance, inclination, stellar mass, and genuine physics could all contribute to an offset.

Changing only the common stellar mass-to-light assumptions moves the median galaxy's fractional velocity residual from +3.78% (lighter stars) through -4.70% (nominal) to -9.65% (heavier stars). Positive means observed speed exceeds the benchmark. Thus a naive inference that all galaxies need stronger or weaker gravity is not robust to these source assumptions. The chosen scenarios are sensitivity brackets, not measured uncertainty intervals.

## Pattern 2: surrounding gas-profile information contributes beyond local inputs

Local inputs included acceleration and its square, radius relative to the stellar-disk speed peak, gas and bulge component fractions, surface brightness, and baryonic slope. Nonlocal inputs add interior/exterior averages at logarithmic radial reaches 0.25 and 1, minus the local value. Reach 1 has an exponential weight that declines by a factor e over a radius ratio e; it is not a physical length in kpc.

The initial complete nonlocal model reduced equal-galaxy squared fractional residuals by 7.09% and 3.88% in two independently assigned five-fold galaxy partitions. Local structure alone reduced them by 4.65% and 2.47%. Ridge strength was selected using three inner galaxy folds within every training set. These are whole-galaxy development predictions, not independent historical confirmation.

As a diagnostic, we removed each galaxy's observed mean offset and predicted the remaining shape. Complete nonlocal features improved shape error by 5.12% and 5.51%, whereas acceleration-only features slightly worsened it. This oracle centering uses the test galaxy's observed mean solely to isolate shape; those numbers must not be advertised as complete speed predictions.

Feature ablations localize most of the additional signal to broad gas-contribution structure. Brightness averaging alone contributed little beyond local inputs. Exterior-only features carried more of the gain than interior-only features in this tested regression. In all 30 original training fits (two partitions, five folds, three stellar-mass assumptions), the coefficient of exterior-minus-local gas contribution was positive. Correlated coefficients do not uniquely identify a physical cause.

In plain language: where the model says the gas contribution becomes more important farther out than it is locally, the statistical fit tends to raise the local speed relative to its other predictions. That is the concrete relation worth investigating. It is not simply 'gas-rich galaxies rotate faster,' nor a measured count of nested spheres or spin coherence.

Twelve within-galaxy shuffles of nonlocal feature alignment in each partition weakened the original improvement in every trial. That small, post-result diagnostic supports useful radial alignment but is not a calibrated discovery significance. Bootstrap intervals for the original nonlocal gain included zero in one partition and omitted repeated model selection, so uncertainty remains substantial.

## Pattern 3: a bad source descriptor exaggerated some results

SPARC's signed gas contribution can be negative near a gas deficit. It must stay signed in the gravitational calculation. But dividing that component by a nearly cancelled total created extreme 'fraction' features: DDO064 reached an absolute ratio 18.87 under the lighter-stellar assumption. A coverage-control fit then failed badly in one partition; this failure remains recorded.

The successor retains the exact original baryonic force and RAR predictions but defines gas and bulge descriptors relative to the sum of component magnitudes. They lie between zero and one without clipping or deleting galaxies. They are component-magnitude descriptors, not true measured mass fractions. This repair was motivated after inspecting the failure, so all later results are explicitly post-selection development.

With that bounded representation, adding the broad gas profile improves over the local model in both galaxy partitions under all three stellar-mass scenarios, both for the full fractional residual and the oracle-centered shape diagnostic. The absolute size of the gain varies. This survival is more useful than a single optimized score.

## Pattern 4: the candidate helps lower-speed systems but damages higher-speed ones

The bounded local-plus-broad-gas model's nominal whole-galaxy outcomes are:

| Metric | Partition A | Partition B |
|---|---:|---:|
| Equal-galaxy fractional error reduction | 8.77% | 6.28% |
| Galaxies improved on that metric | 73/139 | 76/139 |
| Median galaxy improvement | 2.04% | 6.98% |
| Published-error-weighted chi-square change | 2.06% worse | 7.02% worse |
| Equal-galaxy squared km/s error change | 2.40% worse | 10.69% worse |

The different metrics expose a tradeoff, not contradictory calculations. In a descriptive split at the median baryon-predicted speed (80.52 km/s), the 70 lower-speed galaxies improve their squared km/s error by 12.85% and 9.66%; the 69 higher-speed galaxies worsen by 7.44% and 17.40%. This threshold was not fitted as a physical transition, and it should not become one merely because the table looks suggestive.

NGC0055 and UGC07524 improve in both partitions; NGC3521 and NGC3741 worsen in both. The accompanying figure shows these selected successes and failures. They are illustrations, not an independent sample or a complete account of influence. NGC3741 also warns against a blanket statement that all low-speed galaxies benefit.

## What this suggests for formulas

A concrete development template is

    V_trial = V_RAR [1 + local_correction + A * (broad_exterior_gas_descriptor - local_gas_descriptor) + ...].

The learned models used shared training coefficients, with no galaxy-specific gravitational constants. This template is a statistical diagnostic, not a derived acceleration law: it does not yet guarantee conservation, a local limit, mass scaling, lensing, or stable dynamics. A new force model should reproduce the source-profile dependence, including suppression or sign changes where needed, rather than adopt these regression coefficients as fundamental constants.

The next physical question is whether the low/high-speed tradeoff comes from source reconstruction, a missing geometric term, or a real activation scale. Speed inferred from the target cannot be used as a hidden switch. The source-derived acceleration/size/structure must predict when any extra response turns on. A full disk calculation is needed before interpreting the exterior descriptor as a shell-theorem violation.

## Checks and unresolved alternatives

Observational coverage controls added location within the measured radial span and total span. They did not absorb the nominal gas-profile benefit, but their lighter-stellar failure exposed the unstable ratio described above. These coverage variables are not physical gravity inputs. The coverage audit predates the bounded repair and is not a completed combined validation of every successor.

No complete marginalization over distance, inclination, thickness, molecular gas, pressure support, or correlated errors was done. Brightness and gas dependence can reflect those effects rather than modified gravity. These data do not independently test source spin, random motions, external companions, clusters, or light bending. Historical exposure and the sequence of refinements prohibit an independent-confirmation claim.

The strongest conclusion is therefore: there is a reproducible development lead tied to the broader gas profile, but the present formula is not an across-metric improvement over RAR. The overall normalization problem is larger than the shape gain. Both findings should guide the next physical model.
'''
report+='\nEvidence packages:\n'+ '\n'.join(f'- {k}: {v}' for k,v in result['input_hashes'].items())+'\n'
(dest/'report.md').write_text(report,encoding='utf-8')
(outputs/'SPARC-deeper-pattern-findings.md').write_text(report,encoding='utf-8')
(dest/'report_runner.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(strata,indent=2))
