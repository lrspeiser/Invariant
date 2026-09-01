from pathlib import Path
from sigma_theory_compiler import open_gravity_dynamic_source_memory_gw150914_response_v4 as m


def test_contract_and_correlation_grid():
    root=Path(__file__).resolve().parents[1]
    c=m.load_config(root); m.validate(root,c)
    base=m.v3.load_config(root)
    assert len(m.v3.declared_grid(base["analysis"]["time_grid_seconds"]))==411
    assert len(m.v3.declared_grid(base["analysis"]["delta_LH_grid_seconds"]))==83
    assert c["response_access_at_freeze"]["strain_values_read"]==0


def test_profile_recovers_zero_shift():
    import numpy as np
    n=32768; band=np.arange(160,4097); q=np.exp(1j*band*.01); d=q.copy()
    s=m._correlation_scores(d,q,band,n,np.array([-1,0,1]))
    assert int(np.argmax(s))==1
