"""Correct longitude labels and use matching colors for antipodal axis markers."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
os.environ['MPLCONFIGDIR']=str(ROOT/'work/private/mond-atlas-sky-alignment-mpl')
import csv,json,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).parent
with (p/'run-001/sample.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
cfg=json.loads((ROOT/'configs/mond_atlas_sky_alignment_v1.json').read_text(encoding='utf-8'))
l=np.array([float(r['l_deg']) for r in rows]);b=np.array([float(r['b_deg']) for r in rows]);y=np.array([float(r['target']) for r in rows])
fig=plt.figure(figsize=(12,6),layout='constrained');ax=fig.add_subplot(111,projection='mollweide')
v=ax.scatter(-np.deg2rad((l+180)%360-180),np.deg2rad(b),c=y,cmap='coolwarm',vmin=-.2,vmax=.2,s=35,edgecolors='#555',linewidths=.3)
for color,(name,(lon,lat)) in zip(['#238b45','#e08000'],cfg['axes'].items()):
    for k,(ll,bb) in enumerate([(lon,lat),((lon+180)%360,-lat)]):
        ax.scatter(-np.deg2rad((ll+180)%360-180),np.deg2rad(bb),c=color,marker='*',s=200,edgecolors='black',linewidths=.6,label=name if k==0 else None)
ticks=np.arange(-150,151,30);ax.set_xticks(np.deg2rad(ticks));ax.set_xticklabels([str(int(-t)%360)+' deg' for t in ticks],fontsize=8)
ax.grid(alpha=.3);ax.legend(loc='lower left',fontsize=9)
ax.set_title('Galaxy sky positions versus fixed CMB axes\nGalactic longitude increases leftward; latitude is measured from the Milky Way plane')
fig.colorbar(v,ax=ax,shrink=.8,label='log10(observed / algebraic MOND speed); colors clipped at +/-0.2')
fig.suptitle('86 development galaxies; stars mark both ends of each axis',fontsize=11)
fig.savefig(p/'sky-map-reviewed.png',dpi=160);plt.close(fig)
