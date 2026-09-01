from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_gp01_entropy_holdout_confirmation_v1 as holdout

ROOT = Path(__file__).resolve().parents[1]


def test_config_is_exact_and_fixed_before_holdout() -> None:
    config = holdout.load_config(ROOT)
    assert config["scope"]["clusters"] == ["A2029", "A3158", "A644", "RXC1825"]
    assert config["fixed_formulas"]["equilibrium"]["id"] == "GP01L-n1"
    assert config["fixed_formulas"]["elliptic"]["id"] == "GP01E-n1-A8-rho10-T10-q2-p2-L0"
    assert config["scope"]["parameter_tuning"] == 0
    assert config["adjudication"][
        "dynamic_history_descendants_retained_regardless_of_static_result"
    ]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("scope", "parameter_tuning"), 1),
        (("scope", "clusters"), ["A2029"]),
        (("fixed_formulas", "elliptic", "A_max"), 16.0),
        (("fixed_formulas", "equilibrium", "n"), 2),
        (("adjudication", "global_theory_pruning_allowed"), True),
    ],
)
def test_config_mutations_fail_closed(path: tuple[str, ...], value: object) -> None:
    config = copy.deepcopy(holdout.load_config(ROOT))
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(holdout.GP01EntropyHoldoutError, match="semantics changed"):
        holdout.validate_config(ROOT, config)


def test_preflight_is_target_free_and_deterministic() -> None:
    first = holdout.build_preflight(ROOT)
    second = holdout.build_preflight(ROOT)
    assert first == second
    assert first["status"] == "READY_FROZEN_ZERO_HOLDOUT_RESPONSE_DECODE"
    assert first["raw_paths_verified_by_metadata_only"] == 14
    assert first["raw_files_opened"] == 0
    assert first["response_rows_decoded"] == 0
    assert first["scores_computed"] == 0
    assert first["preflight_content_sha256"] == holdout._self_hash(
        first, "preflight_content_sha256"
    )


def test_exact_rank_test_has_known_small_sample_resolution() -> None:
    x = [1.0, 2.0, 3.0, 4.0]
    y = [4.0, 3.0, 2.0, 1.0]
    rho = holdout._spearman(x, y)
    assert rho == pytest.approx(-1.0)
    assert holdout._exact_permutation_p(x, y, rho) == pytest.approx(2.0 / 24.0)


def test_atomic_writer_is_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    value = {"a": 1}
    assert holdout._atomic_no_clobber(path, value) == "CREATED"
    assert holdout._atomic_no_clobber(path, value) == "EXISTING_IDENTICAL"
    with pytest.raises(holdout.GP01EntropyHoldoutError, match="refusing to replace"):
        holdout._atomic_no_clobber(path, {"a": 2})


def test_preflight_paths_do_not_embed_scientific_rows() -> None:
    config = holdout.load_config(ROOT)
    encoded = json.dumps(config, sort_keys=True)
    for forbidden in ("observed", "predicted", "pressure_kev_cm3", "temperature_kev"):
        assert f'"{forbidden}"' not in encoded


def test_result_claim_ceiling_is_restrictive_by_contract() -> None:
    config = holdout.load_config(ROOT)
    assert config["adjudication"][
        "standalone_p_less_than_0_05_is_mathematically_unavailable_for_two_sided_exact_spearman"
    ]
    assert config["adjudication"]["global_theory_pruning_allowed"] is False
    assert config["adjudication"]["publication_signal"].startswith(
        "The four-cluster result may motivate"
    )
