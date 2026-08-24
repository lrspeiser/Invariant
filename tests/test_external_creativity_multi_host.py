from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sigma_theory_compiler import external_creativity_multi_host as M
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / M.OUTPUT_PATH


def test_downloaded_multi_host_receipt_is_sealed_and_cross_platform() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    M.validate_receipt(value)
    assert value["acquisition"]["artifact_bytes_downloaded_and_hashed"]
    assert value["reproduction"]["received_machines"] == 4
    assert value["reproduction"]["distinct_runner_ids"] == 4
    assert value["reproduction"]["distinct_operating_systems"] == [
        "ubuntu-latest",
        "windows-latest",
    ]
    assert value["acquisition"]["workflow_run_id"] == 32698594929
    assert value["head_sha"] == "56c4b86621ab243f306c51d6c406c695f5c672c4"
    assert value["lean"]["artifact_id"] == 9509721324
    assert value["lean"]["kernel_checked"]
    assert value["reproduction"]["status"] == (
        "PASS_MULTI_HOST_CORE_LLM_EVIDENCE_REPRODUCTION"
    )
    assert value["reproduction"]["core_llm_evidence_reproductions"] == 4
    assert value["reproduction"]["core_new_provider_calls"] == 0
    assert value["reproduction"]["core_llm_evidence_projection_sha256"] == (
        "5f2a010578c99a895b143a1b83ef11f51159e7a7810dad78c83792186ee12433"
    )
    assert all(
        host["core_reproduction"]["status"]
        == "PASS_CORE_LLM_EVIDENCE_REPRODUCTION"
        and host["core_reproduction"]["new_provider_calls"] == 0
        and host["core_reproduction"][
            "provider_credential_available_on_reproduction_host"
        ]
        is False
        for host in value["hosts"]
    )
    assert not value["claim_boundary"]["physical_bare_metal_identity_claimed"]


def test_multi_host_receipt_rejects_collapsed_runner_identity() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    changed = deepcopy(value)
    for host in changed["hosts"]:
        host["runner_id"] = 1
    body = {key: item for key, item in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(M.MultiHostReproductionError, match="policy"):
        M.validate_receipt(changed)


def test_multi_host_receipt_rejects_core_projection_disagreement() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    changed = deepcopy(value)
    changed["hosts"][0]["core_reproduction"]["llm_evidence_projection_sha256"] = (
        "0" * 64
    )
    body = {key: item for key, item in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(M.MultiHostReproductionError, match="policy"):
        M.validate_receipt(changed)
