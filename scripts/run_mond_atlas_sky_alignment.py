"""Frozen fixed-axis observational-direction diagnostic and predictive test."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,csv,sys,subprocess,json
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest,canonical_name
from mond_atlas_sky_alignment import sky_features,associations,residualizer
from mond_atlas_pattern_learning import galaxy_folds,nested_predictions


def run(out):
    out.mkdir(parents=True,exist_ok=False);cfgpath=ROOT/'configs/mond_atlas_sky_alignment_v1.json';cfg=read_json(cfgpath)
    bound=[cfgpath,ROOT/cfg['sample'],ROOT/cfg['coordinates'],Path(__file__),ROOT/'scripts/mond_atlas_sky_alignment.py',
           ROOT/'scripts/mond_atlas_pattern_learning.py',ROOT/'tests/test_mond_atlas_sky_alignment.py',out.parent/'PREFLIGHT.md']
    write_json(out/'bindings.json',dict(files={p.relative_to(ROOT).as_posix():digest(p) for p in bound},new_associations_computed=False))
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_mond_atlas_sky_alignment.py'],cwd=ROOT,capture_output=True,text=True)
    (out/'tests.log').write_text(test.stdout+test.stderr,encoding='utf-8')
    if test.returncode:raise RuntimeError('Pre-access tests failed')
    with (ROOT/cfg['coordinates']).open(encoding='utf-8') as f:catalog=list(csv.DictReader(l for l in f if not l.startswith('#')))
    index={}
    for number,row in enumerate(catalog):index.setdefault(canonical_name(row['name']),[]).append((number,row))
    with (ROOT/cfg['sample']).open(encoding='utf-8') as f:sample=list(csv.DictReader(f))
    accepted=[];excluded=[]
    for row in sample:
        match=index.get(canonical_name(row['galaxy']),[])
        if len(match)!=1:
            excluded.append(dict(galaxy=row['galaxy'],reason='missing_or_ambiguous_direct_PROBES_match'));continue
        number,m=match[0]
        try:
            ra,dec,d,ext=map(float,[m['RA'],m['DEC'],m['distance'],m['ext_r']])
            if not np.isfinite([ra,dec,d,ext]).all() or d<=0 or not (0<=ra<360 and -90<=dec<=90):raise ValueError()
        except ValueError:
            excluded.append(dict(galaxy=row['galaxy'],reason='invalid_coordinate_distance_extinction'));continue
        accepted.append(dict(**row,ra_deg=ra,dec_deg=dec,log10_distance=np.log10(d),ext_r=ext,
                             coordinate_row=number,coordinate_name=m['name'],coordinate_survey=m['RC_survey']))
    names=[r['galaxy'] for r in accepted];assert len(names)==len(set(names))
    features=cfg['features']+['log10_distance','ext_r']
    x=np.array([[float(r[k]) for k in features] for r in accepted]);y=np.array([float(r['target']) for r in accepted])
    sky,labels,meta=sky_features([r['ra_deg'] for r in accepted],[r['dec_deg'] for r in accepted],cfg['axes'])
    for i,r in enumerate(accepted):
        r.update({k:float(v[i]) for k,v in meta.items()});r.update({k:float(sky[i,j]) for j,k in enumerate(labels)})
    write_csv(out/'sample.csv',accepted);write_csv(out/'excluded.csv',excluded)
    rows,null,yr,sr,rank=associations(x,y,sky,labels,cfg['permutations'],cfg['seed'])
    write_csv(out/'associations.csv',rows);write_json(out/'maxstat-shuffles.json',null)
    strictx=np.column_stack([x,sky[:,6:]]);m,strict_rank=residualizer(strictx);strict=[]
    for i in range(6):
        z=m@sky[:,i];a=m@y
        strict.append(dict(feature=labels[i],partial_r=float(a@z/(np.linalg.norm(a)*np.linalg.norm(z))),
                           remaining_sky_variance_fraction=float(z@z/np.sum((sky[:,i]-sky[:,i].mean())**2))))
    write_csv(out/'strict-axis-controls.csv',strict)
    # Retain confounding with every input, rather than only the strongest one.
    write_csv(out/'input-sky-correlations.csv',[dict(input=f,sky=l,r=float(np.corrcoef(x[:,i],sky[:,j])[0,1])) for i,f in enumerate(features) for j,l in enumerate(labels)])
    bundles={'baseline':[], 'quadrupole_axis':[0,1], 'octopole_axis':[2,3], 'galactic_latitude':[6,7], 'ecliptic_axis':[8], 'all_sky':list(range(10))}
    partitions=[(str(seed),galaxy_folds(names,seed,cfg['outer_folds'])) for seed in cfg['fold_seeds']]
    partitions.append(('galactic_octants',meta['octant']))
    metrics=[];predrows=[];selection=[]
    for split,fold in partitions:
        cache={}
        for bundle,cols in bundles.items():
            design=np.column_stack([x,sky[:,cols]])
            pred,chosen=nested_predictions(design,y,fold,'linear_ridge',cfg)
            cache[bundle]=pred;selection.extend(dict(split=split,bundle=bundle,**r) for r in chosen)
            predrows.extend(dict(galaxy=n,split=split,bundle=bundle,fold=int(fold[i]),target=float(y[i]),prediction=float(pred[i])) for i,n in enumerate(names))
        b=np.mean((cache['baseline']-y)**2)
        for bundle,pred in cache.items():
            mse=float(np.mean((pred-y)**2));metrics.append(dict(split=split,bundle=bundle,rmse_dex=np.sqrt(mse),mse_gain_percent=100*(b-mse)/b))
    write_csv(out/'prediction-metrics.csv',metrics);write_csv(out/'predictions.csv',predrows);write_csv(out/'selected-penalties.csv',selection)
    summary=dict(status='EXPLORATORY_FIXED_SKY_AXES_TESTED',galaxies=len(y),excluded=len(excluded),tests_passed=4,
        axes=cfg['axes'],axis_separation_deg=float(np.rad2deg(np.arccos(np.clip(np.dot(__import__('mond_atlas_sky_alignment').vector(*cfg['axes']['quadrupole']),__import__('mond_atlas_sky_alignment').vector(*cfg['axes']['octopole'])),-1,1)))),
        nuisance_rank=rank,strict_rank=strict_rank,octant_counts={str(k):int(np.sum(meta['octant']==k)) for k in np.unique(meta['octant'])},
        associations=rows,strict_controls=strict,prediction_metrics=metrics,limitations=cfg['limitations'],
        reference_fractions_are_confirmatory_p_values=False,spin_axis_test=False,new_gravity_law=False,goal_complete=False)
    write_json(out/'summary.json',summary)
    os.environ['MPLCONFIGDIR']=str(ROOT/'work/private/mond-atlas-sky-alignment-mpl')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(12,5),layout='constrained');ax=fig.add_subplot(121,projection='mollweide')
    lon=np.deg2rad((meta['l_deg']+180)%360-180)
    points=ax.scatter(-lon,np.deg2rad(meta['b_deg']),c=y,cmap='coolwarm',vmin=-.2,vmax=.2,s=24)
    from mond_atlas_sky_alignment import vector
    for label,(l,b) in cfg['axes'].items():
        for ll,bb in [(l,b),((l+180)%360,-b)]:ax.scatter(-np.deg2rad((ll+180)%360-180),np.deg2rad(bb),marker='*',s=110,label=label if bb==b else None)
    ax.grid(alpha=.3);ax.legend(fontsize=7,loc='lower left');ax.set_title('86 directly matched galaxies: Galactic sky')
    fig.colorbar(points,ax=ax,shrink=.7,label='log10 observed / MOND speed (color clipped at +/-0.2)')
    ax=fig.add_subplot(122);ax.scatter(sr[:,1],yr,s=24,alpha=.75)
    slope=rows[1]['partial_slope_dex_per_feature'];xx=np.array([sr[:,1].min(),sr[:,1].max()]);ax.plot(xx,slope*xx,color='black')
    ax.set(xlabel='Quadrupole axial alignment after controls',ylabel='Speed residual after controls (dex)',title='Descriptive fit; not an independent confirmation')
    fig.savefig(out/'sky-alignment.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(galaxies=len(y),octants=summary['octant_counts'],associations=rows,prediction_metrics=metrics),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();run(args.output.resolve())
