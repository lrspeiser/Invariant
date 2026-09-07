"""Replay old full-node versus streamed coordinates without observeddata."""
from pathlib import Path
import json,hashlib
import numpy as np
from mond_atlas_hi_refinement import iter_native_batches,ROOT,P

def main():
 out=P/'projection-parity.json'
 if out.exists():raise FileExistsError('Preserve existing parity receipt; use fresh output forreplay')
 new=ROOT/'work/private/mond-atlas-hi-refinement-001/run001/HI-planar-0.0625.npz';old=ROOT/'work/private/mond-atlas-hi-column-001/centroid-projection001/f4-emission-0.0625.npz';batch=next(iter_native_batches(new,24,max_nodes=65536))
 with np.load(old) as z:xy=float(np.max(abs(batch['xy_pixel'][:128]-z['xy_pixel'][:128])));flux=float(np.max(abs(batch['flux_jy_km_s'][:128]-z['flux_jy_km_s'][:128])))
 assert xy<1e-6 and flux<1e-12
 out.write_text(json.dumps(dict(status='PASS_OLD_FULL3D_VS_NEW_STREAMED_API',samples=128,native_xy_max_error_pixel=xy,node_flux_max_error_jy_km_s=flux,old_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),new_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),observed_response_access=False),indent=2),encoding='utf8')
if __name__=='__main__':main()
