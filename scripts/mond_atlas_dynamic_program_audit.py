"""Independent mechanical residual checks and metadata-only source inventory."""
import json,hashlib,itertools
import numpy as np
from mond_atlas_dynamic_program import ROOT,OUT,mechanics


def run():
    rng=np.random.default_rng(9753);x=rng.normal(size=(4,3));v=.3*rng.normal(size=(4,3));m=np.array([.7,1.,1.4,2.])
    pairs=list(itertools.combinations(range(4),2));q=np.linspace(.01,.15,6);w=np.linspace(-.02,.01,6)
    def kinetic_and_p(xx,vv,kind):
        kinetic=.5*np.sum(m[:,None]*vv*vv);p=m[:,None]*vv
        if kind=='kinetic':
            for i,j in pairs:
                d=xx[i]-xx[j];dv=vv[i]-vv[j];k=.5*m[i]*m[j]/(m[i]+m[j])*np.exp(-d@d/2)
                kinetic+=.5*k*(dv@dv);p[i]+=k*dv;p[j]-=k*dv
        return kinetic,p
    def lagrangian(xx,vv,kind,qq):
        kinetic,_=kinetic_and_p(xx,vv,kind);pot=0.
        for k,(i,j) in enumerate(pairs):
            r2=np.sum((xx[i]-xx[j])**2);pot-=m[i]*m[j]/np.sqrt(r2+.05**2)
            if kind.startswith('memory'):
                omega=.2 if kind=='memory_slow' else 2.;h=.15*np.sqrt(m[i]*m[j])*np.exp(-r2/2);pot+=.5*omega**2*(qq[k]-h)**2
        return kinetic-pot
    errors=[]
    for kind in ['newton','kinetic','memory_slow','memory_fast']:
        a=mechanics(x,v,m,kind,q,w)[0];step=1e-5
        dP=(kinetic_and_p(x+step*v,v+step*a,kind)[1]-kinetic_and_p(x-step*v,v-step*a,kind)[1])/(2*step)
        dx=np.zeros_like(x)
        for i in range(4):
            for j in range(3):
                delta=np.zeros_like(x);delta[i,j]=step
                dx[i,j]=(lagrangian(x+delta,v,kind,q)-lagrangian(x-delta,v,kind,q))/(2*step)
        err=float(np.max(abs(dP-dx)));assert err<1e-6
        errors.append(dict(kind=kind,euler_lagrange_max_absolute_residual=err))
    inventories=[]
    for rel in ['work/gravity-first-principles/things-cube-acquisition-001/receipt.json','work/gravity-first-principles/stellar-co-acquisition-001/receipt.json']:
        path=ROOT/rel;doc=json.loads(path.read_text(encoding='utf-8'))
        selected=[]
        for f in doc['files']:
            if f['name'] in ['NGC2976','NGC3198']:
                p=ROOT/f['file'];selected.append({k:f[k] for k in ['name','role','file','url','sha256'] if k in f}|dict(file_exists=p.exists(),actual_bytes=p.stat().st_size if p.exists() else None,expected_bytes=f.get('bytes')))
        inventories.append(dict(receipt=rel,receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),selected=selected))
    rel='work/wellnet-2026-09/member-dynamics/clean/PRODUCTS.manifest.json';path=ROOT/rel;doc=json.loads(path.read_text(encoding='utf-8'))
    inventories.append(dict(receipt=rel,receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),products=[{k:p[k] for k in ['file','sha256','rows','columns'] if k in p} for p in doc['products'] if 'kin' in p['file']]))
    result=dict(status='PASS',timing='Post-integration independent audit. Pre-integration conservation/covariance/oscillator controls were run, but explicit finite-difference Euler-Lagrange force audit was completed here after integration.',mechanics=errors,inventory=inventories,raw_cube_or_stellar_kinematic_values_opened=False,source_admission='SOURCE_BLOCKED for observational motion/memory fit: no independent 3D source velocities/history plus separate response channel admitted.')
    (OUT/'audit-and-inventory.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(mechanics=errors,inventory_receipts=len(inventories))))


if __name__=='__main__':run()
