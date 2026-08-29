from __future__ import annotations

from pathlib import Path

from sigma_theory_compiler.gravity_item49_pseudorandom_exploration import (
    load_config as load_item49_config,
)
from sigma_theory_compiler.gravity_item51_gpu_screening import (
    _canonical_symbolic_keys,
)
from sigma_theory_compiler.gravity_item54_equivalence_detection import (
    _control_programs,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_item54_freezes_layered_nondestructive_equivalence() -> None:
    config = load_config(ROOT, require_bound=False)
    assert config["candidate_pool"]["expected_unique_ordinals"] == 878
    assert len(config["equivalence_layers"]["behavioral"]["environments"]) == 5
    assert config["equivalence_layers"]["behavioral"]["response_fields_used"] == 0
    assert config["equivalence_layers"]["behavioral"][
        "global_algebraic_identity_claimed"
    ] is False
    assert config["preservation_policy"]["original_ordinals_deleted"] == 0
    assert config["preservation_policy"]["lineage_records_deleted"] == 0
    assert config["preservation_policy"]["protected_archive_references_deleted"] == 0


def test_symbolic_control_pairs_canonicalize_as_frozen() -> None:
    programs = _control_programs()
    keys = _canonical_symbolic_keys(programs, load_item49_config(ROOT))
    assert keys[0] == keys[1]  # exact duplicate
    assert keys[2] == keys[3]  # commutative product rewrite
    assert keys[4] == keys[5]  # two zero-producing structures
    assert keys[6] == keys[7]  # max/min equal-operand unary collapse
    assert keys[8] != keys[9]  # adjacent outer cells remain separate
