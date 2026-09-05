"""Independent source and saved-score checks for the concentration pilot."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'work/gravity-first-principles/matched-concentration-001'

def main():
    result=json.loads((D/'result.json').read_text())
    curves={g['name']:g for g in json.loads((ROOT/'configs/sparc_rotation_curves_full_v1.json').read_text())['galaxies']}
    photo=json.loads((ROOT/'configs/sparc_surface_brightness_exploration_v1.json').read_text())
    allowed={g['galaxy'] for g in photo['galaxies']}
    receipt=json.loads((D/'source-retrieval.json').read_text())
    archive=ROOT/receipt['private_path']
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
    checked=0
    with ZipFile(archive) as z:
        for g in photo['galaxies']:
            raw=z.read(Path(g['source_member']).name)
            assert hashlib.sha256(raw).hexdigest()==g['source_member_sha256']
            rows=[l.split() for l in raw.decode().splitlines() if l and not l.startswith('#')]
            assert [r[:6] for r in rows]==curves[g['galaxy']]['rows']
            assert [r[6:] for r in rows]==g['rows']
            checked+=len(rows)
    metadata=ROOT/'work/gravity-first-principles/map-response-metadata-001'
    assert hashlib.sha256((metadata/'SPARC_Lelli2016c.mrt').read_bytes()).hexdigest()==json.loads((metadata/'receipt.json').read_text())['table_sha256']
    records=list(csv.DictReader((D/'positions.csv').open()))
    assert set(r['name'] for r in records)<=allowed
    for r in records:
        radius=float(r['r']); gas=float(r['gas'])
        b2=gas*abs(gas)+.5*float(r['disk'])**2+.7*float(r['bul'])**2
        assert abs(float(r['x'])-np.log10(b2*1e6/(radius*3.085677581491367e19)))<1e-12
        assert abs(float(r['y'])-np.log10(float(r['v'])**2*1e6/(radius*3.085677581491367e19)))<1e-12
    scores=list(csv.DictReader((D/'galaxy-scores.csv').open()))
    for s in scores:
        rows=[r for r in records if r['name']==s['name'] and r['prediction_support']=='True']
        assert len(rows)==int(s['positions'])
        for model in ['rar','rar_density','flexible','flexible_density']:
            mse=np.mean([(float(r['y'])-float(r[model]))**2 for r in rows])
            assert abs(mse-float(s[model]))<1e-14
    pairs=list(csv.DictReader((D/'matched-pairs.csv').open()))
    names=[r[k] for r in pairs for k in ['diffuse','dense']]
    assert len(names)==len(set(names))
    assert all(abs(float(r['x_diffuse'])-float(r['x_dense']))<=.05 and float(r['concentration_ratio'])>=10**.5 for r in pairs)
    # Independently recompute pair medians from exported observed positions.
    for p in pairs:
        residuals=[]
        for label in ['diffuse','dense']:
            x=float(p['x_'+label]); b=int(np.floor(x/.1))
            rs=[r for r in records if r['name']==p[label] and int(np.floor(float(r['x'])/.1))==b]
            residuals.append(np.median([float(r['y'])-(float(r['x'])-np.log10(1-np.exp(-np.sqrt(10**float(r['x'])/1.2e-10)))) for r in rs]))
        assert abs(residuals[0]-residuals[1]-float(p['diffuse_minus_dense_residual_dex']))<1e-12
    spec=importlib.util.spec_from_file_location('matched',ROOT/'scripts/run_gravity_matched_concentration.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data,_=module.prepare(module.read_inputs()[0])
    for name in ['DDO154','NGC3198','NGC2403']:
        test=data['name']==name
        if not np.any(test):continue
        prediction=module.fit_predict(data,~test,test)
        for model in ['rar','rar_density','flexible','flexible_density']:
            saved=np.array([float(r[model]) for r in records if r['name']==name])
            assert np.allclose(saved,prediction[model],rtol=0,atol=1e-12)
    verification=dict(status='PASS',authenticated_source_rows=checked,scored_galaxies=len(scores),
        disjoint_pairs=len(pairs),units_and_signed_gas=True,independent_saved_score_recalculation=True,
        pair_recalculation=True,selected_prediction_replay=True,reserved_photometry_excluded=True,
        result_sha256=hashlib.sha256((D/'result.json').read_bytes()).hexdigest())
    (D/'verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps(verification,indent=2))

if __name__=='__main__':main()
