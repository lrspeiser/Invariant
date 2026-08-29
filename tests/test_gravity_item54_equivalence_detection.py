from __future__ import annotations

from pathlib import Path
import json

from sigma_theory_compiler.gravity_item49_pseudorandom_exploration import (
    load_config as load_item49_config,
)
from sigma_theory_compiler.gravity_item51_gpu_screening import (
    _canonical_symbolic_keys,
)
from sigma_theory_compiler.gravity_item54_equivalence_detection import (
    _control_programs,
    build_aggregate_result,
    build_control_test,
    build_equivalence_manifest,
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


def test_recorded_item54_equivalence_and_controls_are_exactly_replayable() -> None:
    config = load_config(ROOT)
    source = ROOT / config["paths"]["source_dir"]
    equivalence = json.loads(
        (source / config["paths"]["equivalence_manifest"]).read_text(encoding="utf-8")
    )
    controls = json.loads(
        (source / config["paths"]["control_test"]).read_text(encoding="utf-8")
    )
    aggregate = json.loads(
        (ROOT / config["paths"]["aggregate_result"]).read_text(encoding="utf-8")
    )
    assert equivalence == build_equivalence_manifest(ROOT)
    assert controls == build_control_test(ROOT)
    assert aggregate == build_aggregate_result(ROOT)
    assert equivalence["counts"]["symbolic_equivalence_classes"] == 878
    assert equivalence["counts"]["multi_environment_behavioral_equivalence_classes"] == 877
    aliases = [
        row for row in equivalence["behavioral_classes"] if row["member_count"] > 1
    ]
    assert len(aliases) == 1
    assert aliases[0]["member_ordinals"] == [341_577_670_407, 341_715_123_975]
    assert aggregate["claims"]["roadmap_item_54_complete"] is True
    assert aggregate["claims"]["formula_family_pruned"] is False
