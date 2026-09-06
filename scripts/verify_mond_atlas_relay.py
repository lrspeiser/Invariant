"""Independent aggregate tests, output bindings and geometry identity checks."""
import csv,hashlib,io,json,sys,unittest
from pathlib import Path
import numpy as np
from mond_atlas_halo_return import field,mn_field,curl

def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'work/gravity-first-principles/mond-atlas-relay-001'
    suite=unittest.TestSuite()
    for pattern in ['test_mond_atlas_halo_return.py','test_mond_atlas_secondary_experiment.py','test_mond_atlas_absorption_experiment.py','test_mond_atlas_delay_experiment.py']:
        suite.addTests(unittest.defaultTestLoader.discover(str(root/'tests'),pattern=pattern))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (out/'combined-test-log.txt').write_text(stream.getvalue(),encoding='utf-8')
    if not result.wasSuccessful():raise RuntimeError('Combined verification failed')
    rows=list(csv.DictReader((out/'geometry/geometry.csv').open(encoding='utf-8')))
    errors=[];curl_errors=[]
    G=4.30091727003628e-6
    def gamma(r):return np.linalg.norm(field([r,0,0],8.53702e6,19.5725,'NFW',G))/np.linalg.norm(mn_field([r,0,0]))
    for row in rows:
        p=np.array([float(row['R_kpc']),0,float(row['z_kpc'])]);r=np.linalg.norm(p);b=mn_field(p)
        e=np.linalg.norm(np.cross(p,b))/(r*np.linalg.norm(b))
        errors.append(abs(e-float(row['best_scalar_relative_vector_error'])))
        derivative=(gamma(r+1e-4)-gamma(r-1e-4))/2e-4
        expected=derivative*np.cross(p/r,b)
        actual=curl(lambda v:gamma(np.linalg.norm(v))*mn_field(v),p,h=5e-5)
        curl_errors.append(float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1.)))
    if max(errors)>1e-12 or max(curl_errors)>1e-6:raise RuntimeError('Geometry identity failed')
    bindings={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        root/'scripts/run_mond_atlas_relay_geometry.py',root/'scripts/verify_mond_atlas_relay.py',out/'geometry/geometry.csv',out/'geometry/results.json']}
    summary=dict(tests_run=result.testsRun,passed=True,projection_identity_max_absolute_error=max(errors),curl_identity_max_scaled_error=max(curl_errors),bindings=bindings)
    (out/'combined-verification.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
