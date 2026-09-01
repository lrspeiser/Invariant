from pathlib import Path

from sigma_theory_compiler import open_gravity_dynamic_source_memory_gw150914_response_v3 as m


def test_frozen_contract_and_kernel_equivalences():
    root = Path(__file__).resolve().parents[1]
    config = m.load_config(root)
    m.validate_frozen_contract(root, config)
    templates = m.build_templates(root, config)
    assert set(templates) == set(m.KERNEL_IDS) | {"C04_SOURCE_RINGDOWN"}
    assert m.normalized_overlap(templates["K00_INSTANTANEOUS"], templates["K01_RETARDED"]) < 0.999
    assert m.normalized_overlap(templates["K02_EXPONENTIAL"], templates["K06_STOCHASTIC_OU"]) > 1 - 1e-12


def test_time_grids_are_exact():
    root = Path(__file__).resolve().parents[1]
    config = m.load_config(root)
    assert len(m.declared_grid(config["analysis"]["time_grid_seconds"])) == 411
    assert len(m.declared_grid(config["analysis"]["delta_LH_grid_seconds"])) == 83
