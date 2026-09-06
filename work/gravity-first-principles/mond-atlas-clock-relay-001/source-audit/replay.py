"""Independent NumPy replay; imports no production formula or loader helpers."""
import csv, hashlib, json, math
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
RUN=HERE.parent/'run001'
OUT=HERE/'replay'
OUT.mkdir(exist_ok=False)
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
csvread=lambda p:list(csv.DictReader(p.open(encoding='utf-8',newline='')))
config=read(ROOT/'configs/mond_atlas_clock_relay_v1.json')
candidates=read(RUN/'candidate-formulas.json')
registered=read(ROOT/config['registered_development_names'])['names']
nameset=set(registered)
metadata={}
for line in (ROOT/config['source_metadata']).read_text(encoding='utf-8').splitlines():
    t=line.split()
    if t and t[0] in nameset:
        metadata[t[0]]=(float(t[5]),float(t[7]),float(t[11]),float(t[13]),int(t[17]))
data=[];opened=[];names=[]
with zipfile.ZipFile(ROOT/config['source_archive']) as archive:
    members={Path(p).name:p for p in archive.namelist()}
    for name in sorted(registered):
        raw=archive.read(members[name+'_rotmod.dat'])
        opened.append(dict(galaxy=name,sha256=hashlib.sha256(raw).hexdigest()))
        a=np.loadtxt(raw.decode('utf-8').splitlines())
        inc,lum,rd,hi,quality=metadata[name]
        valid=[]
        for j,row in enumerate(a):
            r,v,e,gas,disk,bulge,sb,sbb=row
            okay=all(math.isfinite(float(z)) for z in row) and min(r,v,e)>0
            okay=okay and all(gas*abs(gas)+factor*(.5*disk*disk+.7*bulge*bulge)>0 for factor in config['mass_factors'])
            if okay:valid.append((j,row))
        if quality>2 or not 30<=inc<=80 or rd<=0 or lum<=0 or hi<0 or len(valid)<5:continue
        names.append(name)
        for j,row in valid:data.append((name,j,*row,lum,rd,hi))
assert names==read(RUN/'cohort.json')['names']
assert len(data)==read(RUN/'cohort.json')['radial_rows']
assert len(opened)==139
original_members={r['galaxy']:r['sha256'] for r in read(RUN/'access-receipt.json')['opened_archive_members']}
assert all(original_members[r['galaxy']]==r['sha256'] for r in opened)
ids=np.array([d[0] for d in data]);radids=np.array([d[1] for d in data])
r,v,e,gas,disk,bulge,sb,sbb,lum,rd,hi=np.array([d[2:] for d in data],float).T
y=np.log10(v)
G=4.30091727003628e-6
a0=1.2e-10*3.085677581491367e19/1e6

def independent(candidate):
    f=candidate['family'];mf=candidate['mf']
    v2=gas*np.abs(gas)+mf*(.5*disk*disk+.7*bulge*bulge)
    if f.startswith('newton'):return np.log10(v2)/2
    if f.startswith('mond'):
        z=candidate.get('a0_factor',1)*a0*r
        return np.log10(v2*(1+np.sqrt(1+4*z/v2))/2)/2
    if f=='absorption_proxy':
        return .5*(np.log10(v2)-candidate['kappa']*.5*mf*np.maximum(sb,0)/100/np.log(10))
    if f=='surface_relay':
        return .5*np.log10(v2*(1+candidate['beta']*candidate['sigma0']/(candidate['sigma0']+.5*mf*np.maximum(sb,0))))
    GM=G*1e9*(.5*mf*lum+1.33*hi)
    if f=='clock_potential':
        scale=candidate['clock_factor']*a0*rd
        extra_v2=candidate['beta']*GM*scale*r/((r+rd)*(scale*(r+rd)+GM))
    else:
        length=candidate['length_factor']*rd
        if f=='kernel_point':
            u=np.minimum(r/length,candidate['cutoff'])
            # Separate formula; small-argument integral series avoids cancellation.
            shape=np.log1p(u)-u/(1+u)
            small=u<.001
            shape[small]=sum((-1.)**k*(k-1)/k*u[small]**k for k in range(2,14))
            extra_v2=candidate['eta']*GM*shape/r
        else:
            p2=r/(r+length)**2
            p3=r*r/(r+length)**3
            q=0 if f=='finite_p2' else 1 if f=='finite_p3' else candidate['q']
            extra_v2=candidate['eta']*GM*((1-q)*p2+q*p3)
    return np.log10(v2+extra_v2)/2

pred=np.vstack([independent(c) for c in candidates])
assert np.isfinite(pred).all()
loss=np.array([np.mean((pred[:,ids==n]-y[ids==n])**2,axis=1) for n in names]).T
folds={}
for seed in config['fold_seeds']:
    order=sorted(names,key=lambda n:hashlib.sha256(f'{seed}|{n}'.encode()).digest())
    fold={n:i%5 for i,n in enumerate(order)}
    assert fold=={x['galaxy']:int(x['fold']) for x in csvread(RUN/f'folds-{seed}.csv')}
    folds[seed]=fold
training_error=0.
for row in csvread(RUN/'all-training-losses.csv'):
    seed=int(row['seed']);held=int(row['fold']);ix=int(row['candidate_index'])
    train=np.array([folds[seed][n]!=held for n in names])
    training_error=max(training_error,abs(float(loss[ix,train].mean())-float(row['training_mse'])))
assert training_error<1e-10
selections=read(RUN/'selections.json');choice={};boundary={};selection_gap=0.
gridkeys={'mf':'mass_factors','a0_factor':'a0_factors','kappa':'opacities_per_100_msun_pc2','beta':'strengths','eta':'strengths','sigma0':'surface_scales_msun_pc2','length_factor':'length_factors','clock_factor':'clock_factors','q':'core_mixture_weights'}
for row in selections:
    seed=row['seed'];family=row['family'];held=row['fold'];ix=row['candidate_index']
    assert row['candidate']==candidates[ix]
    train=np.array([folds[seed][n]!=held for n in names])
    options=[i for i,c in enumerate(candidates) if family=='training_selected' or c['family']==family]
    gap=float(loss[ix,train].mean()-loss[options][:,train].mean(axis=1).min())
    selection_gap=max(selection_gap,gap)
    assert gap<1e-10
    choice[(seed,family,held)]=ix
    if candidates[ix]['family'] not in ('newton_fixed','mond_fixed'):
        for key,gridkey in gridkeys.items():
            if key in candidates[ix]:
                vals=config[gridkey];value=candidates[ix][key]
                label=(family,key)
                entry=boundary.setdefault(label,dict(family=family,parameter=key,selections=0,at_lower=0,at_upper=0,grid_min=min(vals),grid_max=max(vals)))
                entry['selections']+=1;entry['at_lower']+=int(value==min(vals));entry['at_upper']+=int(value==max(vals))
families=config['families']+['training_selected'];heldpred={};allrows=[]
for seed in config['fold_seeds']:
    for family in families:
        ix=np.array([choice[(seed,family,folds[seed][n])] for n in ids])
        p=pred[ix,np.arange(len(ids))];heldpred[(seed,family)]=p
        allrows.extend(dict(seed=seed,family=family,galaxy=str(n),radial_index=int(radids[j]),candidate_index=int(ix[j]),log10_predicted_speed=float(p[j])) for j,n in enumerate(ids))
galerror=speederror=0.
for row in csvread(RUN/'galaxy-held-scores.csv'):
    p=heldpred[(int(row['seed']),row['family'])];mask=ids==row['galaxy']
    galerror=max(galerror,abs(float(np.mean((p[mask]-y[mask])**2))-float(row['mse_logspeed'])))
    speederror=max(speederror,abs(float(np.mean((10**p[mask]-v[mask])**2))-float(row['mse_speed_kms'])))
assert galerror<1e-10 and speederror<1e-7
rowlookup={(str(n),int(j)):i for i,(n,j) in enumerate(zip(ids,radids))};raderror=0.
for row in csvread(RUN/'selected-radial-residuals.csv'):
    i=rowlookup[(row['galaxy'],int(row['radial_index']))]
    p=heldpred[(int(row['seed']),'training_selected')][i]
    raderror=max(raderror,abs(p-float(row['log10_predicted_speed'])),abs(p-y[i]-float(row['log10_predicted_over_observed'])))
assert raderror<1e-10
def writecsv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
writecsv(OUT/'all-family-held-predictions.csv',allrows)
writecsv(OUT/'parameter-boundaries.csv',list(boundary.values()))
boundpaths=[ROOT/'configs/mond_atlas_clock_relay_v1.json',Path(__file__),ROOT/config['source_archive'],ROOT/config['source_metadata'],ROOT/config['registered_development_names'],HERE/'inventory.json',*sorted(RUN.iterdir())]
receipt=dict(status='PASS_INDEPENDENT_NUMPY_REPLAY',uses_production_formula_or_loader=False,galaxies=len(names),radial_rows=len(ids),historical_member_bodies_opened=len(opened),reserved_member_bodies_opened=0,candidates=len(candidates),selections=len(selections),held_prediction_rows=len(allrows),training_loss_max_abs=training_error,selected_candidate_loss_gap_max=selection_gap,galaxy_log_mse_max_abs=galerror,galaxy_speed_mse_max_abs=speederror,selected_radial_max_abs=raderror,
    correction='run001 summary source_only_parameters:true must be read as source-only predictor inputs. Parameters are learned from training Vobs and are not derived from source data alone.',
    boundary_note='Boundary frequencies describe only selected frozen candidates. Zero strengths are included lower boundaries; some other parameters become unidentifiable at zero strength. Do not interpret them as physical estimates.',
    bindings={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in boundpaths if p.is_file()})
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k!='bindings'},indent=2))
