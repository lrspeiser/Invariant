"""Independent sparse operator and saved-field comparison audit."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts'))
import mond_atlas_external_program as parent
OWN=Path(__file__).resolve().parent


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def operator_review():
    shape=(7,8,9);spacing=(.3,.4,.2);axes=[np.arange(n)*h for n,h in zip(shape,spacing)];x,y,z=np.meshgrid(*axes,indexing='ij')
    eps=.3+.2*np.exp(-((x-.7)**2+(y-1.1)**2+(z-.6)**2));bc=-x+.3*z
    interior=list(np.ndindex(tuple(n-2 for n in shape)));loc={tuple(k+1 for k in p):i for i,p in enumerate(interior)};A=lil_matrix((len(loc),len(loc)));b=np.zeros(len(loc))
    for idx,row in loc.items():
        for axis,h in enumerate(spacing):
            for step in [-1,1]:
                neighbor=list(idx);neighbor[axis]+=step;neighbor=tuple(neighbor);face=2*eps[idx]*eps[neighbor]/(eps[idx]+eps[neighbor])/h**2
                A[row,row]+=face
                if neighbor in loc:A[row,loc[neighbor]]-=face
                else:b[row]+=face*bc[neighbor]
    reference=spsolve(A.tocsr(),b);phi,record=parent.solve(eps,spacing,bc);actual=np.array([phi[p] for p in loc]);difference=float(np.max(abs(actual-reference)))
    independent_residual=float(np.linalg.norm(A@actual-b)/np.linalg.norm(b));assert difference<1e-9 and abs(independent_residual-record['boundary_forcing_relative_residual'])<1e-12
    binding=json.loads((parent.P/'run001/bindings.json').read_text(encoding='utf-8'));verified=[]
    for name,sha in binding.items():assert digest(ROOT/name)==sha;verified.append(name)
    manifest=json.loads(parent.m.MANIFEST.read_text(encoding='utf-8'));paths={c['path']:c['sha256'] for case in manifest['source_cases'] for c in case['components']}
    for path,sha in paths.items():assert digest(ROOT/path)==sha
    result=dict(status='PASS',shape=shape,spacing=spacing,independent_sparse_max_potential_error=difference,independent_relative_residual=independent_residual,parent_relative_residual=record['boundary_forcing_relative_residual'],bound_file_hashes_verified=verified,all_unique_source_component_hashes_verified=list(paths),source_arrays_or_observed_velocities_opened=False)
    (OWN/'field-review-operator.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');return result


def field_review():
    out=parent.P/'run001';summary=json.loads((out/'summary.json').read_text(encoding='utf-8'));rows=list(csv.DictReader((out/'vectors.csv').open(encoding='utf-8')));sets={}
    for row in rows:
        key=(float(row['ell']),row['grid'],int(row['direction']));sets.setdefault(key,[]).append(row)
    vectors={k:np.array([[float(r[c]) for c in ['gx','gy','gz']] for r in v]) for k,v in sets.items()};z=np.array([float(r['z']) for r in next(iter(sets.values()))][:-1]);checks=[]
    for ref in summary['comparisons']:
        old,new=ref['comparison'].split('_to_');a=vectors[(ref['ell'],old,ref['direction'])].copy();b=vectors[(ref['ell'],new,ref['direction'])].copy()
        if ref['center_relative']:a-=a[-1];b-=b[-1]
        a=a[:-1];b=b[:-1]
        metric=lambda aa,bb:float(np.sqrt(np.sum((aa-bb)**2))/max(np.sqrt(np.sum(bb**2)),1e-8*np.sqrt(len(bb))))
        rms=metric(a,b);groups=[metric(a[z==v],b[z==v]) for v in [0,.2,.5,1]];passed=rms<.05 and max(groups)<.08
        assert abs(rms-ref['rms'])<1e-12 and all(abs(v-r['relative'])<1e-12 for v,r in zip(groups,ref['groups'])) and passed==ref['passed']
        checks.append(dict(ell=ref['ell'],direction=ref['direction'],comparison=ref['comparison'],center_relative=ref['center_relative'],rms=rms,groups=groups,passed=passed))
    effects=[]
    for ell in [.25,.5]:
        for direction in [0,2]:
            g=vectors[(ell,'finer',direction)];center=g[-1];relative=g[:-1]-center;applied=np.eye(3)[direction]
            gates=[v for v in checks if v['ell']==ell and v['direction']==direction];groups=[]
            for height in [0,.2,.5,1]:
                mask=z==height;gg=relative[mask];groups.append(dict(height=height,relative_vector_rms_per_unit_applied_field=float(np.sqrt(np.mean(np.sum(gg*gg,axis=1)))),parallel_signed_mean=float(np.mean(gg[:,direction])),transverse_rms=float(np.sqrt(np.mean(np.sum(gg*gg,axis=1)-gg[:,direction]**2)))))
            effects.append(dict(ell=ell,direction=direction,all_gates_passed=all(v['passed'] for v in gates),center_acceleration_per_unit_field=center.tolist(),center_field_magnitude=float(np.linalg.norm(center)),raw_vector_rms=float(np.sqrt(np.mean(np.sum(g[:-1]**2,axis=1)))),background_subtracted_rms=float(np.sqrt(np.mean(np.sum((g[:-1]-applied)**2,axis=1)))),center_relative_rms=float(np.sqrt(np.mean(np.sum(relative**2,axis=1)))),groups=groups))
    result=dict(status='PASS_REPLAY_NOT_AUTOMATIC_PHYSICS_ADMISSION',vector_rows=len(rows),comparison_count=len(checks),all_gate_replays_exact=True,comparisons=checks,fine_response_per_unit_applied_field=effects,observed_external_field_used=False,files_verified={p.name:digest(p) for p in [out/'summary.json',out/'vectors.csv']})
    (OWN/'field-review-results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');return result


if __name__=='__main__':
    print(json.dumps(operator_review(),indent=2))
    if (parent.P/'run001/summary.json').exists():print(json.dumps(field_review(),indent=2))
