from pathlib import Path
import numpy as np
from sigma_theory_compiler import open_gravity_dynamic_source_memory_gw150914_response_v5 as m

def test_bounded_fractional_grids():
    root=Path(__file__).resolve().parents[1]; c=m.load_config(root); m.validate(root,c); base=m.v3.load_config(root)
    t=m.grid(base["analysis"]["time_grid_seconds"]); d=m.grid(base["analysis"]["delta_LH_grid_seconds"])
    assert len(t)==410 and len(d)==82
    assert t[0]==-.05 and t[-1]<=.05 and d[-1]<=.01

def test_fractional_origin_correlation():
    n=32768; sr=4096; f=np.fft.rfftfreq(n,1/sr); mask=(f>=20)&(f<=512); bins=np.flatnonzero(mask); q=np.exp(1j*f[mask])
    s=m.corr(q,q,bins,n,f[mask],-.05,410)
    assert np.all(np.isfinite(s)) and len(s)==410
