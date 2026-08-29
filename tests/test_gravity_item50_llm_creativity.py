from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest

from sigma_theory_compiler.gravity_item49_pseudorandom_exploration import (
    _admissible_parameter_table,
    load_config as load_item49_config,
)
from sigma_theory_compiler.gravity_item50_llm_creativity import (
    GravityItem50Error,
    _canonical_structure,
    _control_structures,
    _expand_structures,
    _generation_prompt,
    _normalize_proposal,
    _normalize_generation_output,
    _proposal_schema,
    _unique_structures,
    build_preflight_manifest,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _raw_proposal() -> dict[str, object]:
    return {
        "proposal_id": "provider-display-id",
        "title": "Cross-mechanism control",
        "origin_self_assessment": "known_family_combination",
        "known_analogues": ["screened scalar response"],
        "source_domains": ["effective field theory", "control theory"],
        "mechanism": "Combine a dimensionless scale ratio with a bounded action coordinate.",
        "left_primitive_id": 106,
        "left_transform": "tanh_2x",
        "right_primitive_id": 361,
        "right_transform": "sin_pi_x",
        "binary_operator": "weighted_difference",
        "mixing": 4.0,
        "suggested_amplitude": 6.0,
        "suggested_acceleration_exponent": 0.2,
        "suggested_transition_u": 50.0,
        "why_not_merely_a_rewrite": "It combines two independently generated mechanisms.",
        "expected_observational_signature": "A low-acceleration crossover.",
        "cheapest_falsifier": "A held-out cross-scale object test.",
        "likely_failure_mode": "The cluster response may be too strong.",
    }


def test_item50_freeze_budget_ensemble_and_prompt_boundary() -> None:
    config = load_config(ROOT)
    assert config["provider"]["maximum_successful_calls"] == 9
    assert config["provider"]["maximum_provider_attempts"] == 11
    assert config["provider"]["maximum_total_proposals"] == 48
    assert config["provider"]["conservative_maximum_campaign_usd"] == "25.000000"
    assert config["provider"]["user_authorized_maximum_usd"] == "1000.000000"
    assert {
        row["model"]
        for row in config["ensemble"]["generation_calls"]
        + config["ensemble"]["critic_calls"]
    } == {"claude-opus-5", "claude-sonnet-5", "claude-opus-4-8"}
    prompt = _generation_prompt(ROOT, config, config["ensemble"]["generation_calls"][0])
    assert len(prompt.encode()) < config["provider"]["maximum_prompt_bytes_per_call"]
    assert '"id":439' in prompt
    assert "log10_observed_quantity" not in prompt
    assert "object_losses" not in prompt
    assert "selected_ordinal" not in prompt
    preflight = build_preflight_manifest(ROOT, live=False)
    assert preflight["response_fields_in_provider_prompts"] == []
    assert preflight["paid_inference_calls"] == 0


def test_proposal_schema_and_local_normalization_preserve_lineage() -> None:
    config = load_config(ROOT)
    config49 = load_item49_config(ROOT)
    schema = _proposal_schema(config)
    slots = schema["properties"]["proposals"]["required"]
    assert len(slots) == 8
    normalized = _normalize_proposal(
        _raw_proposal(),
        call=config["ensemble"]["generation_calls"][0],
        slot=1,
        config=config,
        config49=config49,
    )
    assert normalized["proposal_id"] == "item50-opus5-mechanism-01"
    assert normalized["provider_proposal_id"] == "provider-display-id"
    assert normalized["suggested_outer_cell_physically_admitted"] is True
    assert normalized["historical_novelty_claimed"] is False
    assert normalized["retained_regardless_of_origin_or_critic_label"] is True

    off_grid = _raw_proposal()
    off_grid["suggested_transition_u"] = 0.6
    retained = _normalize_proposal(
        off_grid,
        call=config["ensemble"]["generation_calls"][0],
        slot=2,
        config=config,
        config49=config49,
    )
    assert retained["suggested_outer_cell_physically_admitted"] is False
    assert retained["structure_executable_for_frozen_outer_expansion"] is True
    assert retained["local_compilation_issues"] == [
        "suggested_outer_value_outside_frozen_grid"
    ]

    bad = _raw_proposal()
    bad["origin_self_assessment"] = "genuinely_novel"
    with pytest.raises(GravityItem50Error, match="origin"):
        _normalize_proposal(
            bad,
            call=config["ensemble"]["generation_calls"][0],
            slot=1,
            config=config,
            config49=config49,
        )


def test_matched_control_and_outer_expansion_are_deterministic() -> None:
    config = load_config(ROOT)
    config49 = load_item49_config(ROOT)
    first = _control_structures(config)
    replay = _control_structures(config)
    assert first == replay
    assert len(first) == 48
    assert len({_canonical_structure(row, config) for row in first}) == 48
    expanded = _expand_structures(first, config49)
    assert len(expanded["candidate_id"]) == 48 * 336
    assert len(np.unique(expanded["ordinal"])) == 48 * 336
    assert int(np.sum(_admissible_parameter_table(config49))) == 336


def test_symbolic_structure_dedup_does_not_delete_lineage() -> None:
    config = load_config(ROOT)
    row = _control_structures(config)[0]
    unique, mapping, audit = _unique_structures([row, dict(row)], config)
    assert len(unique) == 1
    assert mapping == [0, 0]
    assert audit["raw_structures"] == 2
    assert audit["symbolic_structure_duplicates"] == 1


def test_malformed_provider_slot_is_retained_without_becoming_a_formula() -> None:
    config = load_config(ROOT)
    config49 = load_item49_config(ROOT)
    slots = {
        f"idea_{index:02d}": json.dumps(_raw_proposal())
        for index in range(1, 9)
    }
    slots["idea_03"] = "idea_03_placeholder"
    proposals = _normalize_generation_output(
        {"proposals": slots},
        config["ensemble"]["generation_calls"][0],
        config,
        config49,
    )
    assert len(proposals) == 8
    assert proposals[2]["retained_regardless_of_origin_or_critic_label"] is True
    assert proposals[2]["structure_executable_for_frozen_outer_expansion"] is False
    assert proposals[2]["local_compilation_issues"] == ["provider_slot_not_json"]
