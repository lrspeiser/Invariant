import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
groups=json.loads((P/'independent-review/arrangement-summary.json').read_text(encoding='utf-8'))['groups']
fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
colors={'f1-stars-h0p1':'#2166ac','f4-stars-h0p1':'#67a9cf','f1-stars-h0p4':'#b2182b','f4-stars-h0p4':'#ef8a62'}
for case,color in colors.items():
    points=sorted([v for v in groups if v['case']==case and v['z']==0],key=lambda v:v['r'])
    label=case.replace('stars-','').replace('h0p','h=0.')+' kpc'
    for ax,key in zip(axes,['azimuth_rms_fraction','tangential_rms_fraction']):
        ax.plot([p['r'] for p in points],[100*p[key] for p in points],'-o',label=label,color=color,linewidth=2)
for ax,title in zip(axes,['Variation around the same radius','Tangential pull relative to vector strength']):
    ax.set_title(title,fontsize=12);ax.set_xlabel('Radius in galaxy plane (kpc)');ax.set_ylabel('RMS fraction (%)');ax.set_ylim(bottom=0);ax.grid(alpha=.2)
axes[0].legend(fontsize=8,frameon=False)
fig.suptitle('NGC2976: conditional distributed secondary force',fontsize=16)
fig.text(.5,-.025,'Predictions from tracer maps plus assumed depth; no observed gravity or motion fit.',ha='center',fontsize=10)
fig.savefig(P/'geometry-patterns.png',dpi=160,bbox_inches='tight')
