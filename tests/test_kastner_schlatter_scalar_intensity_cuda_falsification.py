from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.kastner_schlatter_scalar_intensity_cuda_falsification import (
    _load_predecessor,
    deterministic_inputs,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_scalar_intensity_cuda_falsification.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-scalar-intensity-cuda-falsification.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rehash(document: dict) -> None:
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    document["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def test_artifact_validates_and_exact_counters_close() -> None:
    artifact = _load(ARTIFACT)
    validate_campaign(artifact, CONFIG)
    assert artifact["counts"] == {
        "compiler_action_hypotheses": 2,
        "parameter_cases": 2048,
        "spectral_and_radial_samples_per_parameter": 1024,
        "unique_parameter_sample_pairs": 2_097_152,
        "unique_scalar_consequence_values": 4_194_304,
        "exact_sentinel_groups": 4,
        "negative_parameter_controls": 5,
        "gpu_warmup_repetitions": 4,
        "gpu_measured_repetitions": 32768,
        "gpu_kernel_dispatches": 32772,
        "gpu_measured_parameter_sample_pairs": 68_719_476_736,
        "gpu_measured_scalar_consequence_evaluations": 137_438_953_472,
        "observational_records_accessed": 0,
        "paper_qed_or_theory_passes": 0,
    }


def test_inputs_are_deterministic_positive_and_dimensionally_ordered() -> None:
    config = _load(CONFIG)
    first = deterministic_inputs(config)
    second = deterministic_inputs(config)
    assert first.keys() == second.keys()
    for key in first:
        assert first[key].tobytes() == second[key].tobytes()
        assert np.all(first[key] > 0.0)
    assert np.allclose(
        first["A"] / first["B"], first["mu_squared"], rtol=2.3e-16, atol=0.0
    )
    assert np.all(np.diff(first["mu_squared"]) > 0.0)
    assert np.all(np.diff(first["wave_number"]) > 0.0)
    assert np.all(np.diff(first["radius"]) > 0.0)


def test_both_branches_bind_identical_scalar_dynamics_without_paper_claim() -> None:
    artifact = _load(ARTIFACT)
    branches = artifact["branch_records"]
    assert {row["beta"] for row in branches} == {"1/2", "1/4"}
    assert branches[0]["scalar_linearized_outputs_sha256"] == branches[1][
        "scalar_linearized_outputs_sha256"
    ]
    assert all(row["beta_enters_linearized_intensity_operator"] is False for row in branches)
    assert all(row["paper_authorship_or_derivation"] is False for row in branches)
    assert artifact["branch_degeneracy_control"]["maximum_branch_output_difference"] == 0.0


def test_dispersion_green_and_exact_sentinels_close() -> None:
    artifact = _load(ARTIFACT)
    assert artifact["dispersion_control"]["all_registered_domain_omega_squared_positive"] is True
    assert artifact["dispersion_control"]["maximum_gpu_relative_equation_residual"] <= 1e-12
    assert artifact["dispersion_control"]["physical_propagation_claim"] is False
    assert artifact["green_yukawa_control"]["all_registered_domain_denominators_positive"] is True
    assert artifact["green_yukawa_control"]["physical_source_response_claim"] is False
    sentinels = artifact["green_yukawa_control"][
        "source_normalization_and_radial_operator_sentinels"
    ]
    assert sentinels["dispersion_A4_B1_k3"]["absolute_residual"] <= 1e-14
    assert sentinels["gap_A9_B4_k0"]["absolute_residual"] == 0.0
    assert sentinels["green_A1_B1_r1"]["high_precision_absolute_error"] <= 1e-17
    assert sentinels["radial_green_operator_r_positive"]["exact_residual"] == "0"
    assert sentinels["radial_green_operator_r_positive"]["source_flux_limit"].endswith("=1")


def test_parameter_domain_negative_controls_are_explicit() -> None:
    controls = _load(ARTIFACT)["parameter_domain_controls"]
    assert controls["registered_domain"] == "q0>0,A_q>0,B_q>0"
    assert controls["valid_positive_parameter_cases"] == 2048
    assert controls["all_negative_controls_rejected"] is True
    cases = {row["case"] for row in controls["negative_controls"]}
    assert cases == {"A_q=0,B_q>0", "A_q<0,B_q>0", "B_q=0,A_q>0", "B_q<0,A_q>0", "q0<=0"}


def test_gpu_cpu_bounds_and_device_counter_scope() -> None:
    artifact = _load(ARTIFACT)
    assert artifact["gpu_cpu_crosscheck"]["maximum_absolute_error"] <= 1e-8
    assert artifact["gpu_cpu_crosscheck"]["maximum_relative_error"] <= 1e-12
    utilization = artifact["runtime_measurement"]["utilization"]
    assert utilization["available"] is True
    assert utilization["sample_count"] > 0
    assert "device-wide NVML" in utilization["counter_scope"]
    assert "not a sustained or lane-exclusive" in artifact["runtime_measurement"]["scope"]


def test_claim_data_and_host_path_seals() -> None:
    artifact = _load(ARTIFACT)
    assert artifact["scope_boundary"] == {
        "actions_are_compiler_hypotheses": True,
        "actions_present_in_or_derived_from_paper": False,
        "qed_actualization_dynamics_tested": False,
        "paper_transaction_ontology_tested": False,
        "external_green_source_is_mathematical_control_only": True,
    }
    assert artifact["synthetic_only"] is True
    for key in (
        "observations_opened",
        "paper_or_qed_pass",
        "theory_pass",
        "ontology_pass",
        "dark_matter_or_halo_inputs",
        "redshift_or_cosmology_inputs",
        "paid_llm_calls",
    ):
        assert artifact[key] is False
    text = ARTIFACT.read_text(encoding="utf-8").lower()
    assert "c:" + "\\users\\" not in text
    assert "/" + "home/" not in text
    for marker in ("api" + "_key", "bear" + "er ", "s" + "k-"):
        assert marker not in text


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("paper_or_qed_pass", True), "claim or data seal"),
        (
            lambda value: value["scope_boundary"].__setitem__(
                "actions_present_in_or_derived_from_paper", True
            ),
            "compiler-hypothesis scope changed",
        ),
        (
            lambda value: value["branch_records"][0].__setitem__(
                "paper_authorship_or_derivation", True
            ),
            "scalar branch binding changed",
        ),
        (
            lambda value: value["parameter_domain_controls"].__setitem__(
                "all_negative_controls_rejected", False
            ),
            "negative parameter controls changed",
        ),
        (
            lambda value: value["deterministic_manifest"]["array_sha256"].__setitem__(
                "A", "0" * 64
            ),
            "scalar-intensity deterministic manifest changed",
        ),
    ],
)
def test_rehashed_tampering_fails_closed(mutation, message: str) -> None:
    artifact = copy.deepcopy(_load(ARTIFACT))
    mutation(artifact)
    _rehash(artifact)
    with pytest.raises(ValueError, match=message):
        validate_campaign(artifact, CONFIG)


def test_predecessor_hash_tamper_fails_before_cuda() -> None:
    config = _load(CONFIG)
    binding = config["predecessors"]["candidate_action_completion"]
    original = binding["file_sha256"]
    binding["file_sha256"] = ("0" if original[0] != "0" else "1") + original[1:]
    with pytest.raises(ValueError, match="candidate_action_completion file hash mismatch"):
        _load_predecessor(ROOT, "candidate_action_completion", binding)
