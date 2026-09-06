"""Frozen manufactured-disk direction check from halo-return preflight."""
import csv,json
from pathlib import Path
import numpy as np
from mond_atlas_halo_return import field,mn_field,curl

def main():
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/mond_atlas_halo_return_v1.json').read_text(encoding='utf-8'))
    out=root/'work/gravity-first-principles/mond-atlas-relay-001/geometry'
    out.mkdir(exist_ok=False)
    G=config['G_kpc_kms2_per_msun'];rows=[]
    def target(p):return field(p,8.53702e6,19.5725,'NFW',G)
    def scalar(p):
        r=np.linalg.norm(p)
        return mn_field(p)*np.linalg.norm(target([r,0,0]))/np.linalg.norm(mn_field([r,0,0]))
    for R in config['field_R_kpc']:
        for z in config['field_z_kpc']:
            p=np.array([R,0,z]);h=target(p);b=mn_field(p);r=np.linalg.norm(p)
            projected=b*np.dot(b,h)/np.dot(b,b)
            rows.append(dict(R_kpc=R,z_kpc=z,
                best_scalar_relative_vector_error=float(np.linalg.norm(projected-h)/np.linalg.norm(h)),
                midplane_matched_relative_vector_error=float(np.linalg.norm(scalar(p)-h)/np.linalg.norm(h)),
                normalized_curl=float(r*np.linalg.norm(curl(scalar,p))/np.linalg.norm(scalar(p)))))
    with (out/'geometry.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    off=[v for v in rows if v['z_kpc']!=0]
    results=dict(disposition='THEORY_BENCHMARK_ONLY',points=len(rows),
        meaning='Manufactured Miyamoto-Nagai disk, not measured Milky Way mass. Scalar projection is a pointwise best possible fit, not a predictive model.',
        max_irreducible_vector_error=max(v['best_scalar_relative_vector_error'] for v in off),
        median_irreducible_vector_error=float(np.median([v['best_scalar_relative_vector_error'] for v in off])),
        max_normalized_curl=max(v['normalized_curl'] for v in off),
        examples=[v for v in rows if v['R_kpc']==4 and v['z_kpc'] in [0,2,8]])
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8');print(json.dumps(results,indent=2))

if __name__=='__main__':main()
