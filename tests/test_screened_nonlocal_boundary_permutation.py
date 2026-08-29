from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.screened_nonlocal_boundary_permutation import (
    CONFIG_PATH,
    ScreenedNonlocalPermutationError,
    _analytic_arrays,
    _base_factor,
    _behavior_indices,
    _feature_banks,
    _raw_stream_digest,
    behavior_representatives,
    decode_ordinal,
    encode_ordinal,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def _item59_config() -> dict[str, object]:
    path = ROOT / "configs/gravity_item59_xcop_forward_observable_gate_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_unbound_contract_is_exactly_the_frozen_four_to_the_tenth_grammar() -> None:
    config = _config()
    validate_config(ROOT, config, require_bound=False)
    grammar = config["grammar"]
    assert len(grammar["factor_order"]) == 10
    assert all(len(grammar[name]) == 4 for name in grammar["factor_order"])
    assert grammar["raw_ordinal_count"] == 4**10 == 1_048_576
    assert grammar["all_factors_exhausted"] is True
    assert grammar["sampling_allowed"] is False


def test_sampling_sealed_access_and_single_counterexample_veto_are_rejected() -> None:
    sampled = deepcopy(_config())
    sampled["grammar"]["sampling_allowed"] = True
    with pytest.raises(ScreenedNonlocalPermutationError, match="exhaustive"):
        validate_config(ROOT, sampled, require_bound=False)

    opened = deepcopy(_config())
    opened["data_contract"]["direct_lensing_target_rows_allowed"] = 1
    with pytest.raises(ScreenedNonlocalPermutationError, match="sealed or new response"):
        validate_config(ROOT, opened, require_bound=False)

    terminal = deepcopy(_config())
    terminal["claim_policy"]["single_counterexample_terminal"] = True
    with pytest.raises(ScreenedNonlocalPermutationError, match="counterexample policy"):
        validate_config(ROOT, terminal, require_bound=False)


@pytest.mark.parametrize(
    "ordinal",
    [0, 1, 3, 4, 255, 65_535, 524_287, 1_048_574, 1_048_575],
)
def test_raw_ordinal_encoding_is_a_bijection(ordinal: int) -> None:
    config = _config()
    indices = decode_ordinal(ordinal, config)
    assert encode_ordinal(indices, config) == ordinal
    assert set(indices) == set(config["grammar"]["factor_order"])


def test_raw_ordinal_stream_and_behavioral_equivalence_are_fully_accounted() -> None:
    config = _config()
    representatives = behavior_representatives(config)
    controls = representatives["alpha"] == 0
    dynamics = ~controls

    assert _raw_stream_digest(config) == (
        "1f7a6345e9b0e88fbda1b3deadf54bb6f18ccbf548a244bf2de33179c243c0ff"
    )
    assert len(representatives["representative_ordinal"]) == 196_612
    assert np.count_nonzero(controls) == 4
    assert np.count_nonzero(dynamics) == 196_608
    assert np.all(representatives["raw_multiplicity"][controls] == 4**8)
    assert np.all(representatives["raw_multiplicity"][dynamics] == 4)
    assert np.sum(representatives["raw_multiplicity"]) == 4**10
    assert np.all(representatives["lensing_branch"] == 0)


def test_analytic_gates_retain_controls_and_filter_nonzero_candidates() -> None:
    config = _config()
    representatives = behavior_representatives(config)
    analytic = _analytic_arrays(representatives, config)
    controls = representatives["alpha"] == 0

    assert np.all(analytic["admitted"][controls])
    assert not np.any(analytic["qualifying"][controls])
    assert np.any(analytic["qualifying"])
    assert np.any(~analytic["admitted"])
    assert np.all(analytic["positive_pass"])
    assert np.all(analytic["epsilon_min"] > 0.0)
    assert np.all(analytic["local_transition"] >= 0.0)
    assert np.all(analytic["local_transition"] <= 1.0)


def test_feature_banks_cover_every_dynamics_subfactor_and_stay_finite() -> None:
    config = _config()
    radius = np.geomspace(0.3, 2_000.0, 31)
    gbar = np.geomspace(5.0e-9, 5.0e-13, len(radius))
    banks = _feature_banks(radius, gbar, config, _item59_config())

    assert banks["base"].shape == (4, len(radius))
    assert banks["x"].shape == (4**3, len(radius))
    assert banks["transition"].shape == (4**4, len(radius))
    assert np.all(np.isfinite(banks["base"]))
    assert np.all(np.isfinite(banks["x"]))
    assert np.all(np.isfinite(banks["transition"]))
    assert np.all((banks["transition"] >= 0.0) & (banks["transition"] <= 1.0))

    representatives = behavior_representatives(config)
    x_index, transition_index = _behavior_indices(representatives)
    assert np.min(x_index) == 0 and np.max(x_index) == 63
    assert np.min(transition_index) == 0 and np.max(transition_index) == 255


def test_base_laws_recover_newtonian_high_acceleration_limit() -> None:
    y = np.asarray([5.0e7])
    factors = _base_factor(y, np.arange(4)).reshape(-1)
    assert factors[0] == 1.0
    assert np.allclose(factors, np.ones(4), rtol=0.0, atol=1.0e-5)
