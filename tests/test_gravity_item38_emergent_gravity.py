import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.gravity_item38_emergent_gravity import (
    GravityItem38Error,
    _contract_digest,
    admissible_candidates,
    decode_candidate,
    fixed_control_multiplier,
    generate_raw_candidates,
    load_config,
    predict_multiplier,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_item38_config_binds_v2_counterexample_policy_and_zero_response_access() -> None:
    config = load_config(ROOT)
    assert config["item"] == 38
    assert config["scope"]["exploration_response_opened"] is False
    assert config["scope"]["confirmation_opening_authorized"] is False
    assert config["scope"]["paid_api_calls_authorized"] is False
    assert config["discovery_policy"][
        "single_empirical_counterexample_is_not_a_formula_or_family_veto"
    ]
    assert config["discovery_policy"]["counterexample_count_alone_is_never_decisive"]
    assert not config["discovery_policy"]["finite_empirical_sample_may_prune_family"]


def test_item38_contract_digest_ignores_only_future_bindings() -> None:
    config = load_config(ROOT)
    changed = deepcopy(config)
    changed["scientific_freeze_commit"] = "a" * 40
    changed["source_metadata_freeze_commit"] = "b" * 40
    assert _contract_digest(changed) == _contract_digest(config)
    changed["candidate_generator"]["raw_candidate_cells"] += 1
    assert _contract_digest(changed) != _contract_digest(config)


def test_item38_raw_grammar_has_exact_equal_capacity() -> None:
    config = load_config(ROOT)
    raw = generate_raw_candidates(config)
    assert len(raw["candidate_id"]) == 262144
    assert np.array_equal(raw["candidate_id"], np.arange(262144))
    assert {int(lane): int(np.sum(raw["lane"] == lane)) for lane in range(4)} == {
        0: 65536,
        1: 65536,
        2: 65536,
        3: 65536,
    }


def test_item38_candidate_equations_are_finite_positive_and_target_blind() -> None:
    config = load_config(ROOT)
    raw = generate_raw_candidates(config)
    sample_ids = np.linspace(0, len(raw["candidate_id"]) - 1, 512, dtype=np.int64)
    sample = {key: value[sample_ids] for key, value in raw.items()}
    u = np.logspace(-8, 8, 65)
    values = predict_multiplier(sample, u, config)
    assert values.shape == (512, 65)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 1.0)
    forbidden = json.dumps(config["data_source"]["forbidden_inputs"]).casefold()
    assert "observed esd" in forbidden
    assert config["candidate_generator"]["post_response_cells"] == 0


def test_item38_admission_is_reproducible_and_keeps_all_niches() -> None:
    config = load_config(ROOT)
    admitted, audit = admissible_candidates(config, batch_size=16384)
    assert 0 < audit["admitted_candidates"] <= 262144
    assert sum(audit["admitted_by_lane"].values()) == audit["admitted_candidates"]
    assert all(audit["admitted_by_lane"][str(lane)] > 0 for lane in range(4))
    assert 0 < audit["behavioral_equivalence_classes"] <= audit["admitted_candidates"]
    replay, replay_audit = admissible_candidates(config, batch_size=32768)
    assert np.array_equal(admitted["candidate_id"], replay["candidate_id"])
    assert audit["behavioral_equivalence_classes"] == replay_audit[
        "behavioral_equivalence_classes"
    ]


def test_item38_fixed_known_controls_have_expected_limits() -> None:
    u = np.asarray([1e-8, 1.0, 1e8])
    baryonic = fixed_control_multiplier("baryonic_newton", u)
    verlinde = fixed_control_multiplier("verlinde_point_mass", u)
    mond = fixed_control_multiplier("mond_RAR", u)
    assert np.array_equal(baryonic, np.ones(3))
    assert verlinde[0] > verlinde[1] > verlinde[2] > 1.0
    assert mond[0] > mond[1] > mond[2] >= 1.0
    assert verlinde[0] == pytest.approx(1.0 + np.sqrt(1.0 / (6e-8)))


def test_item38_decoder_discloses_known_and_potentially_new_labels() -> None:
    config = load_config(ROOT)
    known = decode_candidate(0, config)
    new = decode_candidate(3 * 65536, config)
    assert known["lane"] == "verlinde_elastic_strain"
    assert known["creativity_label"] == "known_formula_plus_declared_transition_extension"
    assert new["lane"] == "collective_information_compression"
    assert new["creativity_label"].startswith("potentially_new_synthesis")


def test_item38_rejects_tampered_counterexample_and_response_boundaries(
) -> None:
    config_path = ROOT / "configs" / "gravity_item38_emergent_gravity_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["discovery_policy"]["finite_empirical_sample_may_prune_family"] = True
    config["data_source"]["archive_payload_may_be_read_before_binding"] = True
    with pytest.raises(GravityItem38Error):
        validate_config(ROOT, config)
