"""Reproduce paper-defined halo targets; never score observational velocities."""
import csv, hashlib, json, os, subprocess, sys, tarfile
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from mond_atlas_halo_return import mass_shape, density_shape, field, fit_return, return_shape

ROOT=Path(__file__).resolve().parents[1]

def dump(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def csvwrite(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    os.chdir(ROOT)
    out=Path(sys.argv[1]) if len(sys.argv)>1 else Path('work/gravity-first-principles/mond-atlas-halo-return-001/run001')
    out.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path('configs/mond_atlas_halo_return_v1.json').read_text(encoding='utf-8'))
    receipts=json.loads(Path(config['source_receipts']).read_text(encoding='utf-8'))
    hashes=[]
    for item in receipts['files']:
        p=Path(item['path']); digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest!=item['sha256']:raise RuntimeError('Source hash mismatch '+str(p))
        hashes.append(dict(path=p.as_posix(),sha256=digest))
    check=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_mond_atlas_halo_return.py','-v'],capture_output=True,text=True)
    (out/'benchmark-tests.txt').write_text(check.stdout+check.stderr,encoding='utf-8')
    if check.returncode:raise RuntimeError('Benchmark failed before target calculations')
    dump(out/'source-verification.json',hashes)
    columns='Ydisk e_Ydisk Ybul e_Ybul D e_D inc e_inc V200 e_V200 C200 e_C200 rs e_rs log_rhos e_log_rhos log_M200 e_log_M200 alpha e_alpha Chi'.split()
    catalog=[]; members=[]
    with tarfile.open('work/private/mond-atlas-halo-return-001/sources/Fits.tar.gz') as tar:
        tables=sorted([m for m in tar if m.isfile() and m.name.startswith('Fits/ByGalaxy/Table/') and m.name.endswith('.mrt')],key=lambda m:m.name)
        if len(tables)!=175:raise RuntimeError('Unexpected source table inventory')
        for member in tables:
            raw=tar.extractfile(member).read();members.append(dict(member=member.name,sha256=hashlib.sha256(raw).hexdigest()))
            found=[]
            for line in raw.decode('utf-8').splitlines():
                tokens=line.split()
                if not tokens or tokens[0] not in config['catalog_models']:continue
                if len(tokens)!=22:raise RuntimeError('Unexpected parameter row '+member.name)
                row=dict(galaxy=Path(member.name).stem,model=tokens[0],**dict(zip(columns,map(float,tokens[1:]))))
                row['rho_s_msun_kpc3']=10**row['log_rhos']*1e9
                if row['rs']<=0 or not np.isfinite([row[k] for k in columns if k!='Chi']+[row['rho_s_msun_kpc3']]).all():raise RuntimeError('Invalid halo parameters')
                row['nonfinite_published_chi']=not np.isfinite(row['Chi'])
                catalog.append(row);found.append(tokens[0])
            if sorted(found)!=sorted(config['catalog_models']):raise RuntimeError('Missing/duplicate selected model')
    csvwrite(out/'published-halo-parameters.csv',catalog);dump(out/'table-members.json',members)
    mw_path=Path('work/private/mond-atlas-halo-return-001/sources/PJM16_best.Tpot')
    halo=list(map(float,mw_path.read_text(encoding='utf-8').splitlines()[-1].split()))
    if halo[1:4]!=[1.,1.,3.] or halo[5]!=0.:raise RuntimeError('Unexpected author halo law')
    mw=dict(galaxy='MilkyWay',model='NFW-author',rho_s_msun_kpc3=halo[0],rs=halo[4])
    dump(out/'milky-way-author-parameters.json',mw)
    fits=[];scores=[]; radial=[]
    for profile in ['NFW','Burkert']:
        for kind in ['inverse_radius','bounded_p2','bounded_p3']:
            fit=fit_return(profile,kind,config['training_x']);fits.append(dict(profile=profile,**fit))
    dump(out/'training-fits.json',fits) # Freeze training selection before held radii.
    for fit in fits:
        for split,key in [('train','training_x'),('interpolation','interpolation_x'),('extrapolation','extrapolation_x')]:
            x=np.array(config[key]);target=mass_shape(x,fit['profile'])/x**2
            pred=return_shape(x,fit['kind'],fit['a'],fit['L']);rel=pred/target-1
            scores.append(dict(profile=fit['profile'],kind=fit['kind'],split=split,rms_relative=float(np.sqrt(np.mean(rel**2))),max_abs_relative=float(np.max(abs(rel)))))
    csvwrite(out/'shape-scores.csv',scores)
    G=config['G_kpc_kms2_per_msun']; errors=[]
    for item in catalog+[mw]:
        profile='Burkert' if item['model'].startswith('Burkert') else 'NFW'
        for x in config['training_x']+config['interpolation_x']+config['extrapolation_x']:
            r=x*item['rs']; g=4*np.pi*G*item['rho_s_msun_kpc3']*item['rs']*float(mass_shape(x,profile))/x**2
            radial.append(dict(galaxy=item['galaxy'],model=item['model'],x=x,r_kpc=r,g_extra_kms2_per_kpc=g,equivalent_halo_v_kms=np.sqrt(r*g)))
    csvwrite(out/'scaled-halo-targets.csv',radial)
    vectors=[]
    pilots=[mw]+[row for row in catalog if row['galaxy'] in ['NGC2976','NGC3198']]
    for item in pilots:
        profile='Burkert' if item['model'].startswith('Burkert') else 'NFW'
        for R in config['field_R_kpc']:
            for z in config['field_z_kpc']:
                p=np.array([R,0,z]);r=np.linalg.norm(p);x=r/item['rs']
                # Independent density integral supplies the reconstructed cumulative response.
                m=quad(lambda t:float(density_shape(x*t,profile))*t*t,0,1,epsabs=1e-12,epsrel=1e-12)[0]*x**3
                H=4*np.pi*G*item['rho_s_msun_kpc3']*item['rs']**3*m
                reconstructed=-H*p/r**3; target=field(p,item['rho_s_msun_kpc3'],item['rs'],profile,G)
                error=float(np.linalg.norm(reconstructed-target)/np.linalg.norm(target));errors.append(error)
                vectors.append(dict(galaxy=item['galaxy'],model=item['model'],R_kpc=R,z_kpc=z,gR=float(target[0]),gz=float(target[2]),return_gR=float(reconstructed[0]),return_gz=float(reconstructed[2]),relative_error=error))
    csvwrite(out/'pilot-vector-reproduction.csv',vectors)
    summary=dict(disposition='THEORY_BENCHMARK_ONLY',galaxies=175,catalog_rows=len(catalog),scaled_target_rows=len(radial),pilot_vectors=len(vectors),exact_reconstruction_max_relative_error=max(errors),tests_passed=5,
        interpretation='Inverse reconstruction of fitted halo profiles, not a prediction from baryons or a validation against new observations.',source_hashes_verified=len(hashes),nonfinite_published_chi_rows=sum(row['nonfinite_published_chi'] for row in catalog))
    dump(out/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    xx=np.geomspace(.02,15,400)
    for ax,profile in zip(axes,['NFW','Burkert']):
        ax.loglog(xx,mass_shape(xx,profile)/xx**2,color='black',label='Published profile shape',linewidth=2)
        for fit in fits:
            if fit['profile']==profile:ax.loglog(xx,return_shape(xx,fit['kind'],fit['a'],fit['L']),label=fit['kind'])
        ax.axvspan(.03,2,color='grey',alpha=.12,label='Training radius range');ax.set_title(profile);ax.set_xlabel('Radius / halo scale radius');ax.set_ylabel('Extra acceleration / (4 pi G rho_s r_s)');ax.legend(fontsize=7)
    fig.suptitle('Return formulas versus fitted halo shapes | mathematical benchmark, not new observations',fontsize=10)
    fig.tight_layout();fig.savefig(out/'halo-shape-comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
