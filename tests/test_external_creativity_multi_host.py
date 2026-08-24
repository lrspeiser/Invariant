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
    M.validate_receipt(value, ROOT)
    assert value["acquisition"]["artifact_bytes_downloaded_and_hashed"]
    assert value["reproduction"]["received_machines"] == 5
    assert value["reproduction"]["core_reproduction_machines"] == 4
    assert value["reproduction"]["lean_kernel_machines"] == 1
    assert value["reproduction"]["distinct_runner_ids"] == 5
    assert value["reproduction"]["distinct_operating_systems"] == [
        "ubuntu-latest",
        "windows-latest",
    ]
    assert value["acquisition"]["workflow_run_id"] == 32736204652
    assert value["head_sha"] == "3740bc0a955fceea3e305419684dfe7f6422a615"
    assert value["lean"]["artifact_id"] == 9523594616
    assert value["lean"]["kernel_checked"]
    assert value["reproduction"]["status"] == ("PASS_MULTI_HOST_CORE_LLM_EVIDENCE_REPRODUCTION")
    assert value["reproduction"]["core_llm_evidence_reproductions"] == 4
    assert value["reproduction"]["core_new_provider_calls"] == 0
    assert value["reproduction"]["core_llm_evidence_projection_sha256"] == (
        "73af5e6628f6cf6f035add2c499748b591d57e548935962768b5889d0ccb0c57"
    )
    assert value["reproduction"]["core_live_evidence_content_sha256"] == (
        "b13a9da8fd9b8213f6c2e94d91872d3403342f1cddb4be80e7e55e3d3f03bf7e"
    )
    assert all(
        host["core_reproduction"]["status"] == "PASS_CORE_LLM_EVIDENCE_REPRODUCTION"
        and host["core_reproduction"]["new_provider_calls"] == 0
        and host["core_reproduction"]["provider_credential_available_on_reproduction_host"] is False
        and host["core_reproduction"]["live_evidence_content_sha256"]
        == value["reproduction"]["core_live_evidence_content_sha256"]
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
    changed["hosts"][0]["core_reproduction"]["llm_evidence_projection_sha256"] = "0" * 64
    body = {key: item for key, item in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(M.MultiHostReproductionError, match="policy"):
        M.validate_receipt(changed)


def test_multi_host_receipt_rejects_core_live_evidence_disagreement() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    changed = deepcopy(value)
    changed["hosts"][0]["core_reproduction"]["live_evidence_content_sha256"] = "0" * 64
    body = {key: item for key, item in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(M.MultiHostReproductionError, match="policy"):
        M.validate_receipt(changed)


def test_multi_host_receipt_rejects_lean_runner_or_archive_digest_tamper() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    for mutate in (
        lambda changed: changed["lean"].__setitem__("runner_id", changed["hosts"][0]["runner_id"]),
        lambda changed: changed["acquisition"]["archive_digests_bound"].__setitem__(
            0, "sha256:" + "0" * 64
        ),
    ):
        changed = deepcopy(value)
        mutate(changed)
        body = {key: item for key, item in changed.items() if key != "content_sha256"}
        changed["content_sha256"] = canonical_sha256(body)
        with pytest.raises(M.MultiHostReproductionError):
            M.validate_receipt(changed, ROOT)


def test_source_manifest_rejects_collapsed_github_identity(tmp_path: Path) -> None:
    source = json.loads((ROOT / M.CONFIG_PATH).read_text(encoding="utf-8"))
    source["artifacts"][1]["runner_id"] = source["artifacts"][0]["runner_id"]
    path = tmp_path / M.CONFIG_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(M.MultiHostReproductionError, match="identity collapsed"):
        M._load_source(tmp_path)


def test_source_manifest_rejects_wrong_artifact_topology(tmp_path: Path) -> None:
    source = json.loads((ROOT / M.CONFIG_PATH).read_text(encoding="utf-8"))
    source["artifacts"][-1]["artifact_name"] = "external-creativity-unexpected"
    path = tmp_path / M.CONFIG_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(M.MultiHostReproductionError, match="topology"):
        M._load_source(tmp_path)
