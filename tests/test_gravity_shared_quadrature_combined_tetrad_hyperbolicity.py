from __future__ import annotations

import copy
import hashlib
import json
import threading
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_shared_quadrature_combined_tetrad_hyperbolicity as gate,
)

ROOT = Path(__file__).resolve().parents[1]


def test_symbolic_and_numeric_derivation_passes_exact_inventory() -> None:
    config = gate.load_config(ROOT)
    symbolic, expressions = gate.symbolic_checks()
    numeric = gate.numeric_cases(config)
    assert [row["check_id"] for row in symbolic] == config["machine_check_contract"][
        "required_symbolic_checks"
    ]
    assert len(symbolic) == 37
    assert all(row["passed"] for row in symbolic)
    assert len(numeric) == 4
    assert all(row["passed"] for row in numeric)
    assert all(row["aether_primary_symmetric_theorem_entry"] for row in numeric)
    assert all(row["common_local_Cauchy_time"] for row in numeric)
    assert all(row["aether_modes_not_subluminal"] for row in numeric)
    assert all(row["scalar_transverse_mode_subluminal"] for row in numeric)
    assert not any(row["all_mode_cherenkov_safety"] for row in numeric)
    assert expressions["spin_2_speed_squared"] == "1"
    assert expressions["spin_1_speed_squared"] == "(Delta - 1)**(-2)"
    assert expressions["spin_0_speed_squared"] == "1"
    assert expressions["alpha1"] == "-4*Delta"
    assert expressions["alpha2"] == "0"
    assert expressions["combined_all_mode_cherenkov_safe"] is False
    assert expressions["scalar_longitudinal_speed_squared"] == "1"


def test_receipt_is_exact_rebuild_with_honest_ceiling() -> None:
    stored = gate.check_receipt(ROOT)
    rebuilt = gate.build_receipt(ROOT)
    assert stored == rebuilt
    assert stored["decision"] == gate.DECISION
    assert stored["counts"] == {
        "predecessor_bindings": 2,
        "predecessor_artifacts": 6,
        "symbolic_checks": 37,
        "symbolic_checks_passed": 37,
        "numeric_cases": 4,
        "numeric_cases_passed": 4,
        "aether_physical_modes": 5,
        "combined_physical_modes": 6,
        "observational_files_opened": 0,
        "observational_rows_opened": 0,
        "network_calls_by_builder": 0,
        "model_or_paid_calls": 0,
        "gpu_calls": 0,
    }
    assert stored["adjudication"]["restricted_combined_local_symmetric_hyperbolicity"]
    assert stored["adjudication"]["primary_source_aether_tetrad_symmetric_hyperbolicity"]
    assert stored["adjudication"]["scalar_symmetric_hyperbolicity"]
    assert stored["adjudication"]["combined_direct_sum_principal_structure"]
    assert stored["adjudication"]["common_local_Cauchy_time"]
    assert stored["adjudication"]["aether_modes_not_subluminal"]
    assert stored["adjudication"]["aether_primary_source_PPN_and_speed_necessary_bounds"]
    assert not stored["adjudication"]["all_mode_cherenkov_safety"]
    assert not stored["adjudication"]["exact_preferred_frame_free_limit_regular"]
    assert not stored["adjudication"]["nonzero_W_or_global_background_hyperbolicity"]
    assert not stored["adjudication"]["CP11_4_complete"]
    assert not stored["claim_boundary"]["full_covariant_health_established"]
    assert not stored["claim_boundary"]["global_hyperbolicity_established"]
    assert not stored["claim_boundary"]["all_mode_cherenkov_safety_established"]
    assert stored["claim_boundary"]["aether_literature_necessary_bounds_satisfied_on_locus"]
    assert not stored["claim_boundary"]["observational_support"]
    assert set(stored["zero_access_and_compute"].values()) == {0}


def test_predecessor_commits_and_runtime_control_are_exact() -> None:
    config = gate.load_config(ROOT)
    rows = gate.validate_predecessors(ROOT, config["predecessor_bindings"])
    assert rows == [
        {
            "binding_id": "quadrature_reduced_principal_factorization",
            "git_commit": "80217be5bfed836cb3e06f0910fa4c379b304aea",
            "artifact_count": 4,
            "valid": True,
            "receipt_content_sha256": "15fd7544da7959633f2f9718694d2c32bcd1d4ec4a841568020e1348298c5419",
        },
        {
            "binding_id": "einstein_aether_covariant_strong_hyperbolicity_control",
            "git_commit": "0d11dae4f5b1a6b4e5d9a8c333b650eb60e611f8",
            "artifact_count": 2,
            "valid": True,
            "runtime_control_passed": True,
            "primary_source": "https://arxiv.org/abs/1902.05130",
        },
    ]


@pytest.mark.parametrize(
    "section",
    [
        "predecessor_bindings",
        "frozen_branch_contract",
        "symmetric_hyperbolic_locus_contract",
        "scalar_first_order_contract",
        "combined_principal_contract",
        "ppn_and_physical_gate_boundary",
        "obstruction_contract",
        "machine_check_contract",
        "adjudication",
        "claim_boundary",
        "zero_access_and_compute",
    ],
)
def test_every_frozen_section_rejects_coherent_mutation(section: str) -> None:
    config = copy.deepcopy(gate.load_config(ROOT))
    value = config[section]
    if isinstance(value, list):
        value[0]["binding_id"] += "-changed"
    elif section == "machine_check_contract":
        value["numeric_tolerance"] = 1e-9
    elif section == "adjudication":
        value["CP11_4_complete"] = True
    elif section == "claim_boundary":
        value["full_covariant_health_established"] = True
    elif section == "zero_access_and_compute":
        value["observational_rows_opened"] = 1
    else:
        first = next(iter(value))
        value[first] = f"{value[first]} changed"
    with pytest.raises(gate.QuadratureCombinedHyperbolicityError, match=f"config {section}"):
        gate.validate_config(config)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["adjudication"].__setitem__("CP11_4_complete", True),
        lambda value: value["claim_boundary"].__setitem__(
            "full_covariant_health_established", True
        ),
        lambda value: value["counts"].__setitem__("observational_rows_opened", 1),
        lambda value: value["machine_results"]["numeric_cases"][0].__setitem__("passed", False),
    ],
)
def test_rehashed_receipt_overclaims_and_evidence_mutations_fail(mutation: object) -> None:
    receipt = copy.deepcopy(gate.build_receipt(ROOT))
    mutation(receipt)  # type: ignore[operator]
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    receipt["content_sha256"] = gate._sha(body)
    with pytest.raises(
        gate.QuadratureCombinedHyperbolicityError,
        match="stored receipt differs from exact rebuild",
    ):
        gate.validate_receipt(receipt, ROOT)


def test_atomic_no_clobber_preserves_existing_and_race_bytes(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    gate._atomic_no_clobber(path, b"first\n")
    gate._atomic_no_clobber(path, b"first\n")
    with pytest.raises(gate.QuadratureCombinedHyperbolicityError, match="refusing"):
        gate._atomic_no_clobber(path, b"second\n")
    assert path.read_bytes() == b"first\n"

    raced = tmp_path / "raced.json"
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def publish(payload: bytes) -> None:
        barrier.wait()
        try:
            gate._atomic_no_clobber(raced, payload)
            outcomes.append("pass")
        except gate.QuadratureCombinedHyperbolicityError:
            outcomes.append("reject")

    threads = [
        threading.Thread(target=publish, args=(b"alpha\n",)),
        threading.Thread(target=publish, args=(b"beta\n",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["pass", "reject"]
    assert raced.read_bytes() in {b"alpha\n", b"beta\n"}


def test_config_and_test_bytes_match_frozen_hashes() -> None:
    assert hashlib.sha256((ROOT / gate.CONFIG_PATH).read_bytes()).hexdigest() == (
        gate.EXPECTED_CONFIG_FILE_SHA256
    )
    assert hashlib.sha256((ROOT / gate.TEST_PATH).read_bytes()).hexdigest() == (
        gate.EXPECTED_TEST_FILE_SHA256
    )


def test_cli_status_fields_remain_partial_and_zero_access() -> None:
    receipt = gate.check_receipt(ROOT)
    status = gate._status(receipt)
    assert status["valid"]
    assert status["restricted_combined_symmetric_hyperbolicity"]
    assert not status["full_covariant_health"]
    assert status["observational_rows_opened"] == 0
    assert status["content_sha256"] == receipt["content_sha256"]


def test_receipt_json_is_canonical_and_content_bound() -> None:
    path = ROOT / gate.OUTPUT_PATH
    payload = path.read_bytes()
    assert payload.endswith(b"\n")
    parsed = json.loads(payload)
    assert (
        hashlib.sha256(
            gate._canonical_bytes(
                {key: value for key, value in parsed.items() if key != "content_sha256"}
            ).rstrip(b"\n")
        ).hexdigest()
        == parsed["content_sha256"]
    )
