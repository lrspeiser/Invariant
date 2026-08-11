from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.einstein_aether_coupling_boundary_kkt_gate import (
    FIRST_BLOCKER,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/einstein_aether_coupling_boundary_kkt_gate.json"
ARTIFACT = ROOT / "runs/engine/einstein-aether-coupling-boundary-kkt-gate.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, object]:
    return build_gate(CONFIG)


def test_exact_rebuild_partition_and_counts(rebuilt: dict[str, object]) -> None:
    artifact = _load(ARTIFACT)
    assert rebuilt == artifact
    _validate_result(artifact)
    assert artifact["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 1}
    assert artifact["gate_counts"] == {
        "generic_symbolic_determinant_identities_pass": 5,
        "D_only_ambient_singular_constrained_full_rank_witnesses": 2,
        "true_constrained_rank_boundary_witnesses": 3,
        "five_mode_linear_positivity_chart_bindings": 1,
        "global_nonlinear_stability_pass": 0,
        "candidate_or_theory_reject": 0,
        "observational_pass": 0,
    }
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_generic_determinants_and_normality_are_exact(rebuilt: dict[str, object]) -> None:
    factors = rebuilt["symbolic_factorization"]
    assert all(factors["identity_checks"].values())
    assert factors["ambient_10x10_determinant"] == (
        "-(c1 + c4)**3*(-M2 + c1 + c3)**5*(2*M2*c1 + 2*M2*c2 + 2*M2*c3 + "
        "2*M2*c4 + c1**2 + 4*c1*c2 + 2*c1*c3 + c1*c4 + 4*c2*c3 + "
        "3*c2*c4 + c3**2 + c3*c4)/512"
    )
    assert factors["unit_normality_rational_factor"] == (
        "-4*(2*M2 + c1 + 3*c2 + c3)/(2*M2*c1 + 2*M2*c2 + 2*M2*c3 + "
        "2*M2*c4 + c1**2 + 4*c1*c2 + 2*c1*c3 + c1*c4 + 4*c2*c3 + "
        "3*c2*c4 + c3**2 + c3*c4)"
    )
    assert factors["constrained_KKT_11x11_determinant"] == (
        "-(c1 + c4)**3*(-M2 + c1 + c3)**5*(2*M2 + c1 + 3*c2 + c3)/128"
    )


def test_D_only_singularity_is_removed_by_unit_constraint(rebuilt: dict[str, object]) -> None:
    witness = rebuilt["exact_witnesses"]["D_only_inside_five_mode_positivity_chart"]
    assert witness["factor_values"] == {
        "D": "0",
        "c14": "1/2",
        "M2_minus_c13": "1",
        "trace": "23/10",
        "c123": "1/10",
        "vector_gradient": "27/23",
    }
    assert witness["ambient_rank"] == 9
    assert witness["tangent_rank"] == 9
    assert witness["KKT_rank"] == 11
    assert witness["ambient_determinant"] == "0"
    assert witness["tangent_determinant"] == "-23/40960"
    assert witness["KKT_determinant"] == "23/10240"
    assert witness["ambient_nullspace"] == [
        ["2/23", "2/23", "2/23", "0", "0", "0", "1", "0", "0", "0"]
    ]
    assert all(witness["five_mode_chart_checks"].values())


def test_independent_D_only_replay_matches_exact_ranks(rebuilt: dict[str, object]) -> None:
    witness = rebuilt["exact_witnesses"]["D_only_independent_replay"]
    assert witness["factor_values"]["D"] == "0"
    assert (witness["ambient_rank"], witness["tangent_rank"], witness["KKT_rank"]) == (
        9,
        9,
        11,
    )
    assert witness["couplings"] == {
        "M2": "1",
        "c1": "1/10",
        "c2": "1/20",
        "c3": "0",
        "c4": "-11/75",
    }


def test_true_boundaries_have_exact_constrained_null_witnesses(rebuilt: dict[str, object]) -> None:
    boundaries = rebuilt["exact_witnesses"]["true_constrained_boundaries"]
    c14 = boundaries["c14_equals_zero"]
    assert (c14["ambient_rank"], c14["tangent_rank"], c14["KKT_rank"]) == (7, 6, 8)
    assert len(c14["KKT_nullspace"]) == 3
    tensor = boundaries["M2_minus_c13_equals_zero"]
    assert (tensor["ambient_rank"], tensor["tangent_rank"], tensor["KKT_rank"]) == (5, 4, 6)
    assert len(tensor["KKT_nullspace"]) == 5
    trace = boundaries["two_M2_plus_c13_plus_3c2_equals_zero"]
    assert (trace["ambient_rank"], trace["tangent_rank"], trace["KKT_rank"]) == (10, 8, 10)
    assert trace["KKT_nullspace"] == [
        ["-16/9", "-16/9", "-16/9", "0", "0", "0", "0", "0", "0", "0", "1"]
    ]


def test_five_mode_chart_connection_is_linear_and_fail_closed(rebuilt: dict[str, object]) -> None:
    chart = rebuilt["reduced_five_mode_chart_binding"]
    assert chart["formal_control_status"] == "pass"
    assert chart["mode_count"] == 5
    assert chart["D_is_not_a_reduced_five_mode_boundary"] is True
    assert chart["shared_constrained_rank_boundaries"] == [
        "c14=0",
        "M2-c13=0 (M2=1 gives 1-c13=0)",
        "2*M2+c13+3*c2=0 (M2=1 gives 2+c13+3*c2=0)",
    ]
    assert "not a generic nonlinear" in chart["scope"]
    assert all(value is False for value in rebuilt["claim_seals"].values())
    assert all(value is False for value in rebuilt["data_seals"].values())


def test_lineage_config_and_result_tampering_fail_closed(
    tmp_path: Path, rebuilt: dict[str, object]
) -> None:
    config = _load(CONFIG)
    tampered = copy.deepcopy(config)
    tampered["source_bindings"]["adm_aether_source"]["file_sha256"] = "0" * 64
    path = tmp_path / "configs" / CONFIG.name
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch"):
        build_gate(path)

    opened = copy.deepcopy(config)
    opened["seals"]["global_nonlinear_stability_claimed"] = True
    path.write_text(json.dumps(opened), encoding="utf-8")
    with pytest.raises(ValueError, match="seal opened"):
        build_gate(path)

    overclaim = copy.deepcopy(rebuilt)
    overclaim["claim_seals"]["global_nonlinear_Hamiltonian_stability_proven"] = True
    overclaim.pop("content_sha256")
    with pytest.raises(ValueError, match="seal opened"):
        _validate_result(overclaim)

    broken = copy.deepcopy(rebuilt)
    broken["exact_witnesses"]["D_only_inside_five_mode_positivity_chart"]["KKT_rank"] = 10
    broken.pop("content_sha256")
    with pytest.raises(ValueError, match="D-only witness lost"):
        _validate_result(broken)


def test_source_bindings_are_exact_and_portable(rebuilt: dict[str, object]) -> None:
    bindings = rebuilt["source_bindings"]
    for label in ("adm_aether_source", "formal_controls", "config", "implementation", "test"):
        binding = bindings[label]
        assert _file_sha(ROOT / binding["path"]) == binding["file_sha256"]
        assert not Path(binding["path"]).is_absolute()
    assert all("C:/" not in str(value) and "C:\\" not in str(value) for value in bindings.values())
