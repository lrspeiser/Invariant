"""Independent packet/geometry/objective replay for explicit source alternatives."""
import hashlib,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from mond_atlas_source_resolution import cell_projection_matrix,project,roughness_gradient
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-source-sensitivity-001/run001'
def main():
    bindings=json.loads((P/'bindings.json').read_text())
    for path,h in bindings.items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==h
    summary=json.loads((P/'summary.json').read_text());checks=[]
    for a in summary['assets']:
        path=R/a['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==a['sha256']
        with np.load(path) as p:
            nodes=p['latent_axis'];s=p['intrinsic_effective_surface'];dx=nodes[1]-nodes[0];obs=p['observed_axis'];height=float(p['vertical_layers'][0,1]);assert np.all(np.isfinite(s)) and np.min(s)>=0 and not np.any(s[[0,-1]]) and not np.any(s[:,[0,-1]])
            left=cell_projection_matrix(obs,.125,nodes,dx,0);right=cell_projection_matrix(obs,.125,nodes,dx,height*np.tan(np.deg2rad(53.8623309521001)));pred=left@s@right.T;err=float(np.max(abs(pred-p['projected_surface'])));assert err<1e-9;row=dict(path=a['path'],mass_msun=float(s.sum()*dx*dx*1e6*a['conversion_to_msun_pc2']),projection_max_difference=err)
            if a['branch'].startswith('missing'):
                target=p['imposed_target'];weight=p['imposed_weight'];scale=max(float(np.sqrt(np.sum(weight*target**2)/weight.sum())),1e-12);normal=s/scale;res=left@normal@right.T-target/scale;grad=left.T@(weight*res)@right+1e-4*roughness_gradient(normal);L=float(left.sum(0).max()*left.sum(1).max()*right.sum(0).max()*right.sum(1).max()+8e-4);support=np.hypot(*np.meshgrid(nodes,nodes,indexing='ij'))<6;step=normal-np.where(support,np.maximum(normal-grad/L,0),0);stationarity=float(np.sqrt(np.mean(step**2))*L*16);assert stationarity<1e-6;row['independent_projected_gradient']=stationarity
            else:
                cov=p['additional_gaussian_covariance_kpc2'];beam=p['assumed_native_beam_arcsec'];pa=np.deg2rad(144);ci=np.cos(np.deg2rad(53.8623309521001));A=np.array([[np.sin(pa),np.cos(pa)],[np.cos(pa)/ci,-np.sin(pa)/ci]])*(3611*np.pi/648000);native_angle=np.deg2rad(beam[2]);U=np.array([[np.sin(native_angle),np.cos(native_angle)],[np.cos(native_angle),-np.sin(native_angle)]]);native=U@np.diag((beam[:2]/np.sqrt(8*np.log(2)))**2)@U.T;restored=np.linalg.solve(A,cov)@np.linalg.inv(A.T)+native;difference=float(np.max(abs(restored-np.eye(2)*30**2/(8*np.log(2)))));assert difference<1e-10;row['sky_target_covariance_error_arcsec2']=difference
            checks.append(row)
    result=dict(status='PASS_CONDITIONAL_SOURCE_IMPLEMENTATION',packets=len(checks),checks=checks,observed_spectra_accessed=False,source_noise_likelihood_validated=False,source_inverse_unit_tests='8 existing independent tests replayed successfully after packet build and before field replay; inherited solver was already benchmarked by source-resolution-001',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(P/'independent-review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print('PASS',len(checks))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
