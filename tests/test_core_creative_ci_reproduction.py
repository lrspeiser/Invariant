from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sigma_theory_compiler import core_creative_ci_reproduction as R
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _ci_environment() -> dict[str, str]:
    return {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": R.CI_REPOSITORY,
        "GITHUB_WORKFLOW": R.CI_WORKFLOW,
        "INVARIANT_EVIDENCE_JOB": R.CI_JOB,
        "GITHUB_RUN_ID": "123456789",
        "GITHUB_RUN_ATTEMPT": "1",
        "INVARIANT_EVIDENCE_HEAD_SHA": "a" * 40,
        "GITHUB_EVENT_NAME": "pull_request",
        "RUNNER_OS": "Linux",
        "RUNNER_ARCH": "X64",
        "RUNNER_NAME": "GitHub Actions 1000000001",
        "INVARIANT_EVIDENCE_ARTIFACT_NAME": "external-creativity-ubuntu-latest-3.12",
        "INVARIANT_EVIDENCE_OPERATING_SYSTEM": "ubuntu-latest",
        "INVARIANT_EVIDENCE_PYTHON_VERSION": "3.12",
    }


def test_ci_probe_replays_both_live_llm_lanes_without_a_credential() -> None:
    receipt = R.build_receipt(ROOT, _ci_environment())
    R.validate_receipt(receipt, ROOT, require_ci_provenance=True)
    projection = receipt["llm_evidence_projection"]
    accelerator = receipt["accelerator_evidence_projection"]
    assert receipt["verification"] == {
        "core_runtime_validated": True,
        "creative_modular_gpu_receipt_validated": True,
        "descendant_runtime_validated": True,
        "new_provider_calls": 0,
        "provider_credential_available_on_reproduction_host": False,
        "status": "PASS_CORE_LLM_EVIDENCE_REPRODUCTION",
    }
    assert projection["core_lane"]["completed_calls"] == 8
    assert projection["descendant_lane"]["completed_calls"] == 6
    assert projection["core_lane"]["model"] == "claude-opus-4-6"
    assert projection["descendant_lane"]["model"] == "claude-opus-4-6"
    assert projection["credential_boundary"]["credential_persisted"] is False
    assert projection["credential_boundary"]["credential_value_recorded"] is False
    assert accelerator["candidate_count"] == 33**5
    assert accelerator["gpu_modular_survivors"] == 1
    assert accelerator["exact_survivors"] == 1
    assert accelerator["sample_crosscheck_agrees"] is True
    assert accelerator["gpu_survival_establishes_proof"] is False
    assert receipt["claim_boundary"]["reproduction_host_reran_cuda"] is False
    assert receipt["claim_boundary"]["literature_novelty_established"] is False


def test_projection_is_stable_across_deterministic_receipt_rebinds() -> None:
    core = json.loads((ROOT / R.CORE_PATH).read_text(encoding="utf-8"))
    descendant = json.loads((ROOT / R.DESCENDANT_PATH).read_text(encoding="utf-8"))
    baseline = R.llm_evidence_projection(core, descendant)
    rebound = deepcopy(core)
    rebound["content_sha256"] = "0" * 64
    rebound["verification"] = {"changed_deterministic_evidence": True}
    rebound["source_bindings"]["multi_host_reproduction_receipt"] = {
        "content_sha256": "1" * 64,
        "path": "replacement.json",
    }
    assert R.llm_evidence_projection(rebound, descendant) == baseline


def test_resealed_live_evidence_projection_substitution_is_rejected() -> None:
    receipt = R.build_receipt(ROOT, _ci_environment())
    changed = deepcopy(receipt)
    changed["llm_evidence_projection"]["core_lane"]["evidence_content_sha256"] = "0" * 64
    changed["llm_evidence_projection_sha256"] = canonical_sha256(
        changed["llm_evidence_projection"]
    )
    body = {key: item for key, item in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(R.CoreCreativeCIReproductionError, match="source binding"):
        R.validate_receipt(changed, ROOT, require_ci_provenance=True)


def test_resealed_accelerator_projection_substitution_is_rejected() -> None:
    receipt = R.build_receipt(ROOT, _ci_environment())
    changed = deepcopy(receipt)
    changed["accelerator_evidence_projection"]["exact_survivors"] = 0
    changed["accelerator_evidence_projection_sha256"] = canonical_sha256(
        changed["accelerator_evidence_projection"]
    )
    changed["content_sha256"] = canonical_sha256(
        {key: item for key, item in changed.items() if key != "content_sha256"}
    )
    with pytest.raises(R.CoreCreativeCIReproductionError, match="accelerator"):
        R.validate_receipt(changed, ROOT, require_ci_provenance=True)


def test_ci_provenance_is_required_for_a_passing_machine_receipt() -> None:
    receipt = R.build_receipt(ROOT, {})
    R.validate_receipt(receipt, ROOT)
    assert receipt["verification"]["status"] == "BLOCKED_INCOMPLETE_CI_PROVENANCE"
    with pytest.raises(R.CoreCreativeCIReproductionError, match="provenance is incomplete"):
        R.validate_receipt(receipt, ROOT, require_ci_provenance=True)
