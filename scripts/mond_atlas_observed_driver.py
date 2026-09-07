"""Two-phase source-conditioned observed-spectrum driver; no implicit admission."""
import argparse, ast, csv, hashlib, json
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.linalg import solve_triangular
from mond_atlas_aperture_scorer import training_fit, freeze_predictions, evaluate, FrozenPrediction, arrays
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'work/gravity-first-principles/mond-atlas-observed-driver-001'
APERTURES=ROOT/'work/gravity-first-principles/mond-atlas-observation-admission-001/run004/frozen-geometric-apertures.csv'
COVARIANCE=ROOT/'work/gravity-first-principles/mond-atlas-aperture-covariance-001/run001/working-covariance.json'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,allow_nan=False)


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def header_digest(path): return hashlib.sha256(fits.getheader(path).tostring().encode()).hexdigest()
def resolve(path): return (ROOT/path).resolve()


def verify_bindings(bindings):
    if not bindings: raise ValueError('Empty bindings')
    for name,sha in bindings.items():
        if digest(resolve(name))!=sha: raise ValueError('Changed binding: '+name)


def code_closure():
    todo=[Path(__file__).resolve(), ROOT/'scripts/mond_atlas_actual_spectra.py']; found={}
    while todo:
        path=todo.pop()
        if str(path) in found: continue
        found[str(path)]=digest(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            names=([node.module] if isinstance(node,ast.ImportFrom) else [x.name for x in node.names] if isinstance(node,ast.Import) else [])
            for name in names:
                if name:
                    local=ROOT/'scripts'/(name.split('.')[0]+'.py')
                    if local.exists(): todo.append(local.resolve())
    return found


def apertures(path=APERTURES):
    with Path(path).open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    if len(rows)!=15: raise ValueError('Exactly fifteen fixed apertures required')
    for row in rows:
        for key in ('x_start','x_stop','y_start','y_stop'): row[key]=int(row[key])
        if row['x_stop']-row['x_start']!=12 or row['y_stop']-row['y_start']!=12: raise ValueError('Exactly144 pixels required')
    if sum(r['subset']=='training' for r in rows)!=10 or sum(r['subset']=='evaluation' for r in rows)!=5: raise ValueError('Ten/five split required')
    # Overlap would expose held pixels through training slices.
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if max(a['x_start'],b['x_start'])<min(a['x_stop'],b['x_stop']) and max(a['y_start'],b['y_start'])<min(a['y_stop'],b['y_stop']): raise ValueError('Apertures overlap')
    return rows


def read_aperture_means(cube,rows,subset):
    """Only selected slices materialize. FITS byte hashing is a separate operation."""
    if subset not in ('training','evaluation'): raise ValueError('Invalid phase')
    ids=[i for i,r in enumerate(rows) if r['subset']==subset]
    result=[]
    with fits.open(cube,memmap=True) as hdus:
        h=hdus[0].header
        if str(h.get('BUNIT','')).strip().upper()!='JY/BEAM': raise ValueError('Native Jy/beam required')
        if h.get('BSCALE',1)!=1 or h.get('BZERO',0)!=0: raise ValueError('Scaled FITS unsupported')
        data=hdus[0].data
        if data.ndim==4 and data.shape[0]==1: data=data[0]
        if data.ndim!=3 or data.shape[0]!=42: raise ValueError('42 native channels required')
        for i in ids:
            a=rows[i];x0,x1,y0,y1=[a[k] for k in ('x_start','x_stop','y_start','y_stop')]
            if x1-x0!=12 or y1-y0!=12 or min(x0,y0)<0 or x1>data.shape[2] or y1>data.shape[1]: raise ValueError('Invalid aperture extent')
            patch=np.array(data[:,y0:y1,x0:x1],dtype=float,copy=True)
            if patch.shape!=(42,12,12) or not np.isfinite(patch).all(): raise ValueError('Selected aperture nonfinite')
            result.append(1000*patch.sum(axis=(1,2))/144)
    return ids,np.asarray(result)


def pointer(doc,path):
    for key in path: doc=doc[key]
    return doc


def validate_admission(path):
    """External reviewed evidence, not a self-issued success boolean."""
    a=read_json(path)
    if a.get('schema')!='observed-driver-admission-v1' or a.get('reviewer')!='parent' or a.get('phase')!='fit_and_evaluation': raise ValueError('Parent full-phase admission required')
    verify_bindings(a['bindings'])
    history_path=ROOT/'work/gravity-first-principles/mond-atlas-native-spectral-001/NGC2976.json'
    if a['bindings'].get(str(history_path.resolve()))!=digest(history_path): raise ValueError('Instrument history binding required')
    if resolve(a['cube'])!=resolve(read_json(history_path)['source_path']): raise ValueError('Cube differs from native instrument source')
    for name,sha in code_closure().items():
        if a['bindings'].get(name)!=sha: raise ValueError('Missing current code binding: '+name)
    for name in (str(APERTURES.resolve()),str(COVARIANCE.resolve()),str(resolve(a['cube'])),str(resolve(a['cache']))):
        if name not in a['bindings']: raise ValueError('Missing mandatory asset binding: '+name)
    if header_digest(resolve(a['cube']))!=a['cube_header_sha256']: raise ValueError('Header changed')
    if canonical(a['cases'])!=a['case_sha256']: raise ValueError('Case declaration changed')
    cases=a['cases']; ids=[c['id'] for c in cases]
    if not cases or len(set(ids))!=len(ids): raise ValueError('Unique declared cases required')
    groups={}
    for c in cases:
        if set(c)!={'id','kwargs'}: raise ValueError('Unexpected case keys')
        kw=c['kwargs']
        if set(kw)!={'height','model','pressure_reference','spin','branch','material'}: raise ValueError('Every model argument must be explicit')
        if kw['branch'] not in ('boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated'): raise ValueError('Unknown spectral branch')
        group=canonical({k:v for k,v in kw.items() if k!='spin'})
        groups.setdefault(group,[]).append(kw.get('spin'))
    if any(sorted(s)!=[-1,1] for s in groups.values()): raise ValueError('Both spins exactly once per physical case required')
    if set(a['evidence'])!={'spectral_convergence','source_assumptions','background_injection'}: raise ValueError('All evidence categories required')
    for category,entry in a['evidence'].items():
        ep=resolve(entry['path'])
        if digest(ep)!=entry['sha256'] or a['bindings'].get(str(ep))!=entry['sha256']: raise ValueError('Evidence hash mismatch')
        doc=read_json(ep)
        if sorted(entry['covered_case_ids'])!=sorted(ids): raise ValueError('Incomplete case coverage: '+category)
        checks=entry['assertions']
        if not checks: raise ValueError('Explicit evidence assertions required')
        if not entry.get('required_row_ids') or len(set(entry['required_row_ids']))!=len(entry['required_row_ids']): raise ValueError('Required rows must be named')
        if set(x['row_id'] for x in checks)!=set(entry['required_row_ids']): raise ValueError('Unchecked evidence rows')
        for check in checks:
            value=pointer(doc,check['pointer']);op=check['operator'];expected=check['expected']
            if op=='equals': ok=type(value)==type(expected) and value==expected
            elif op=='less_equal': ok=isinstance(value,(int,float)) and np.isfinite(value) and value<=expected
            elif op=='greater_equal': ok=isinstance(value,(int,float)) and np.isfinite(value) and value>=expected
            else: raise ValueError('Unknown assertion operator')
            if not ok: raise ValueError('Failed evidence '+category+': '+check['row_id'])
    if not a.get('source_bindings') or any(name not in a['bindings'] for name in a['source_bindings']): raise ValueError('Source provenance bindings required')
    return a


def fit_and_freeze(out,rows,training,mean,covariance,cases,model_factory,bindings):
    """No evaluation-response argument exists. Shared manufactured/real core."""
    out=Path(out);out.mkdir(exist_ok=False,parents=True)
    train=[i for i,r in enumerate(rows) if r['subset']=='training'];held=[i for i,r in enumerate(rows) if r['subset']=='evaluation']
    arrays(training,mean,covariance)
    if len(training)!=len(train): raise ValueError('Training rows differ')
    records=[]
    for number,case in enumerate(cases):
        model=model_factory(case); callback=lambda inds,p: model.predict(p)[list(inds)]
        fit=training_fit(callback,train,training,mean,covariance)
        record={'case':case,'fit':fit}
        if fit.get('evaluation_allowed'):
            frozen=freeze_predictions(callback,fit,held)
            envelope=np.asarray(model.unknown_envelope(fit['parameters']))[held]
            if envelope.shape!=(5,42) or not np.isfinite(envelope).all() or np.any(envelope<0): raise ValueError('Invalid unknown-emission envelope')
            name=f'prediction-{number:04d}.npz'
            np.savez(out/name,indices=np.array(held),values=frozen.values,absolute_unknown_envelope_mjy_beam=envelope)
            record.update(prediction=name,prediction_sha256=digest(out/name))
        write_json(out/f'fit-{number:04d}.json',record);records.append(record)
    manifest={'schema':'frozen-observed-predictions-v1','bindings':bindings,'training_indices':train,'evaluation_indices':held,'cases':records,'apertures_sha256':canonical(rows),'mean':np.asarray(mean).tolist(),'covariance':np.asarray(covariance).tolist(),'interpretation':'historically exposed, source-conditioned descriptive prediction; no evaluation selection'}
    manifest['fit_files']={f'fit-{n:04d}.json':digest(out/f'fit-{n:04d}.json') for n in range(len(records))}
    write_json(out/'frozen-manifest.json',manifest)
    write_json(out/'freeze-complete.json',{'manifest_sha256':digest(out/'frozen-manifest.json')})
    return manifest


def verify_frozen(directory,rows):
    directory=Path(directory)
    seal=read_json(directory/'freeze-complete.json')
    if digest(directory/'frozen-manifest.json')!=seal['manifest_sha256']: raise ValueError('Frozen manifest changed')
    m=read_json(directory/'frozen-manifest.json')
    if m['schema']!='frozen-observed-predictions-v1' or m['apertures_sha256']!=canonical(rows): raise ValueError('Frozen metadata mismatch')
    verify_bindings(m['bindings'])
    for name,sha in m['fit_files'].items():
        if digest(directory/name)!=sha: raise ValueError('Fit receipt changed')
    for record in m['cases']:
        if 'prediction' in record and digest(directory/record['prediction'])!=record['prediction_sha256']: raise ValueError('Frozen prediction changed')
    return m


def evaluate_saved(directory,manifest,data):
    results=[];lower=np.linalg.cholesky(manifest['covariance']);absolute_inverse=abs(solve_triangular(lower,np.eye(42),lower=True))
    for record in manifest['cases']:
        if 'prediction' not in record: results.append({'case':record['case'],'status':'NO_IDENTIFIABLE_FROZEN_PREDICTION'});continue
        with np.load(Path(directory)/record['prediction']) as z:
            frozen=FrozenPrediction(tuple(z['indices'].tolist()),z['values'].copy());delta=z['absolute_unknown_envelope_mjy_beam'].copy()
        if list(frozen.indices)!=manifest['evaluation_indices']: raise ValueError('Held index mismatch')
        score=evaluate(frozen,data,manifest['mean'],manifest['covariance'])
        radius=np.linalg.norm(delta@absolute_inverse.T,axis=1);center=np.sqrt(42*np.array(score['per_aperture_q_per_channel']))
        score['unknown_emission_q_lower_bound']=(np.maximum(0,center-radius)**2/42).tolist();score['unknown_emission_q_upper_bound']=((center+radius)**2/42).tolist()
        results.append({'case':record['case'],'score':score,'interpretation':'valid-emitter component prediction plus conservative unknown-emission interval; no law selected'})
    return results


def run(phase,out,admission):
    # No cube-value access occurs before all external evidence and byte bindings pass.
    a=validate_admission(admission);rows=apertures();noise=read_json(COVARIANCE)
    if noise.get('covariance_units')!='(mJy/native_restoring_beam)^2': raise ValueError('Unexpected covariance units')
    arrays(np.zeros((1,42)),noise['mean'],noise['covariance'])
    binding=dict(a['bindings']);binding[str(Path(admission).resolve())]=digest(admission)
    if phase=='fit':
        from mond_atlas_actual_spectra import ActualSpectrumModel
        factory=lambda case:ActualSpectrumModel(resolve(a['cache']),**case['kwargs'])
        if Path(out).exists(): raise ValueError('Fresh fit directory required before response access')
        for case in a['cases']:
            candidate=factory(case)
            if candidate.grouped_flux.shape[0]!=15: raise ValueError('Cache aperture count mismatch')
            del candidate
        _,training=read_aperture_means(resolve(a['cube']),rows,'training')
        return fit_and_freeze(out,rows,training,noise['mean'],noise['covariance'],a['cases'],factory,binding)
    m=verify_frozen(out,rows)
    if m['mean']!=noise['mean'] or m['covariance']!=noise['covariance']: raise ValueError('Frozen noise changed')
    if m['bindings']!=binding or [r['case'] for r in m['cases']]!=a['cases']: raise ValueError('Admission/fit binding mismatch')
    if not any('prediction' in r for r in m['cases']): raise ValueError('No identifiable prediction; no evaluation read')
    evaluation_path=Path(out)/'evaluation.json'
    if evaluation_path.exists(): raise ValueError('Evaluation already exists')
    _,held=read_aperture_means(resolve(a['cube']),rows,'evaluation')
    result={'results':evaluate_saved(out,m,held),'frozen_manifest_sha256':digest(Path(out)/'frozen-manifest.json'),'admission_sha256':digest(admission),'selection':'none; both spins retained'}
    write_json(evaluation_path,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['fit','evaluate']);parser.add_argument('--out',required=True);parser.add_argument('--admission',default=str(PACKAGE/'parent-admission.json'));args=parser.parse_args()
    run(args.phase,args.out,args.admission)
