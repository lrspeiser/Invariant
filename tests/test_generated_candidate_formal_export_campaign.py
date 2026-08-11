from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.generated_candidate_formal_export_campaign import (
    MARKER,
    SANDBOX_SCHEMA,
    _sha,
    build_generated_candidate_formal_export,
    execute_cadabra_batch_sandbox,
    validate_generated_candidate_formal_export,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "generated_candidate_formal_export_campaign.json"
ARTIFACT = ROOT / "runs" / "engine" / "generated-candidate-formal-export.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fake_executor(script: str, _root: Path, _config: dict) -> dict:
    stdout = MARKER.encode()
    stderr = b""
    body = {
        "schema_version": SANDBOX_SCHEMA,
        "status": "pass",
        "backend_mode": "wsl-local",
        "backend_version": "test",
        "network_namespace_created": True,
        "user_namespace_created": True,
        "shell_invoked": False,
        "return_code": 0,
        "marker": MARKER,
        "marker_count": 1,
        "batch_script_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "batch_script_bytes": len(script.encode()),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stdout_bytes": len(stdout),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stderr_bytes": len(stderr),
        "timeout_seconds": 120,
    }
    return {**body, "content_sha256": _sha(body)}


@pytest.fixture(scope="module")
def export() -> dict:
    return build_generated_candidate_formal_export(
        _load(CONFIG), ROOT, sandbox_executor=_fake_executor
    )


def test_all_163_exact_actions_render_with_separate_metric_routes(export: dict) -> None:
    assert export["candidate_count"] == 163
    assert export["family_counts"] == {
        "AETHER_K1234_PARAMETER_CELL": 128,
        "CONFORMAL_G4_PHI_SCALAR_TENSOR": 1,
        "CUBIC_HORNDESKI_G3_WEAK_CELL": 32,
        "KESSENCE_G2_CONVEX": 2,
    }
    assert export["action_export_counts"] == {
        "exact_rendered": 163,
        "sandbox_parsed_and_canonicalised": 163,
        "rejected": 0,
    }
    assert export["metric_variation_counts"] == {
        "reviewed_adapter_routes_bound": 163,
        "executed_by_this_campaign": 0,
        "formal_passes_inferred": 0,
    }
    assert len({record["cadabra_symbol"] for record in export["candidate_records"]}) == 163
    assert all(record["formal_pass_inferred"] is False for record in export["candidate_records"])


def test_checked_artifact_is_exact_real_sandbox_result(export: dict) -> None:
    checked = _load(ARTIFACT)
    validate_generated_candidate_formal_export(checked)
    assert checked["sandbox_receipt"]["backend_version"] == "2.4.5.4"
    assert checked["sandbox_receipt"]["network_namespace_created"] is True
    assert checked["sandbox_receipt"]["shell_invoked"] is False
    assert (
        checked["candidate_record_registry_root_sha256"]
        == export["candidate_record_registry_root_sha256"]
    )
    assert [record["cadabra_action_expression"] for record in checked["candidate_records"]] == [
        record["cadabra_action_expression"] for record in export["candidate_records"]
    ]


def test_real_network_isolated_cadabra_batch_replays() -> None:
    config = _load(CONFIG)
    pure = build_generated_candidate_formal_export(config, ROOT, sandbox_executor=_fake_executor)
    try:
        receipt = execute_cadabra_batch_sandbox(pure["cadabra_batch_script"], ROOT, config)
    except RuntimeError as error:
        if "requires wsl-local Cadabra" in str(error):
            pytest.skip(str(error))
        raise
    assert receipt["status"] == "pass"
    assert receipt["marker_count"] == 1


def test_tampered_formula_sandbox_and_metric_promotion_reject(export: dict) -> None:
    tampered = copy.deepcopy(export)
    tampered["candidate_records"][0]["cadabra_action_expression"] += " + forbidden"
    tampered["candidate_records"][0]["cadabra_action_expression_sha256"] = hashlib.sha256(
        tampered["candidate_records"][0]["cadabra_action_expression"].encode()
    ).hexdigest()
    tampered["candidate_records"][0]["content_sha256"] = _sha(
        {
            key: value
            for key, value in tampered["candidate_records"][0].items()
            if key != "content_sha256"
        }
    )
    tampered["candidate_record_registry_root_sha256"] = _sha(
        [record["content_sha256"] for record in tampered["candidate_records"]]
    )
    tampered["content_sha256"] = _sha(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError):
        validate_generated_candidate_formal_export(tampered)

    for field, value in (
        ("network_namespace_created", False),
        ("shell_invoked", True),
        ("marker_count", 2),
    ):
        tampered = copy.deepcopy(export)
        tampered["sandbox_receipt"][field] = value
        receipt_body = {
            key: item
            for key, item in tampered["sandbox_receipt"].items()
            if key != "content_sha256"
        }
        tampered["sandbox_receipt"]["content_sha256"] = _sha(receipt_body)
        tampered["content_sha256"] = _sha(
            {key: item for key, item in tampered.items() if key != "content_sha256"}
        )
        with pytest.raises(ValueError):
            validate_generated_candidate_formal_export(tampered)

    tampered = copy.deepcopy(export)
    tampered["candidate_records"][0]["metric_variation_route"][
        "metric_variation_executed_by_this_campaign"
    ] = True
    tampered["candidate_records"][0]["content_sha256"] = _sha(
        {
            key: value
            for key, value in tampered["candidate_records"][0].items()
            if key != "content_sha256"
        }
    )
    tampered["candidate_record_registry_root_sha256"] = _sha(
        [record["content_sha256"] for record in tampered["candidate_records"]]
    )
    tampered["content_sha256"] = _sha(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError):
        validate_generated_candidate_formal_export(tampered)


def test_source_config_and_data_seal_tamper_fail_closed() -> None:
    config = _load(CONFIG)
    for mutation in ("source", "budget", "data"):
        tampered = copy.deepcopy(config)
        if mutation == "source":
            tampered["source_export"]["file_sha256"] = "0" * 64
        elif mutation == "budget":
            tampered["budget"]["maximum_candidates"] = 164
        else:
            tampered["data_eligibility"]["observational_data_opened"] = True
        with pytest.raises(ValueError):
            build_generated_candidate_formal_export(tampered, ROOT, sandbox_executor=_fake_executor)
