from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from sigma_theory_compiler.gravity_lead_parent_registry import (
    CONFIG_PATH,
    LEAD_IDS,
    OUTPUT_PATH,
    GravityLeadParentRegistryError,
    build_receipt,
    load_config,
    validate_config,
    verify_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def test_registry_binds_all_five_leads_without_target_or_compute_authority() -> None:
    config = load_config(ROOT)
    assert tuple(row["lead_id"] for row in config["lead_programs"]) == LEAD_IDS
    assert [row["rank"] for row in config["lead_programs"]] == [1, 2, 3, 4, 5]
    assert config["safety_contract"]["sealed_target_rows_opened"] == 0
    assert not config["safety_contract"]["network_calls_allowed"]
    assert not config["safety_contract"]["gpu_production_allowed"]
    assert not config["safety_contract"]["paid_model_calls_allowed"]


def test_registry_verifies_every_registered_evidence_file() -> None:
    receipt = build_receipt(ROOT)
    assert receipt["decision"] == "PASS_ALL_FIVE_PARENTS_REGISTERED_EVIDENCE_INTACT"
    assert receipt["lead_count"] == 5
    assert receipt["registered_evidence_files"] > 40
    assert receipt["safety"] == {
        "metadata_only": True,
        "raw_payloads_opened": 0,
        "sealed_target_rows_opened": 0,
        "network_calls": 0,
        "gpu_production_runs": 0,
        "paid_model_calls": 0,
    }
    assert all(
        row["registry_status"] == "REGISTERED_EVIDENCE_INTACT"
        for row in receipt["lead_programs"]
    )


def test_every_lead_binds_executable_implementation_source() -> None:
    config = load_config(ROOT)
    for lead in config["lead_programs"]:
        implementations = [
            item for item in lead["evidence"] if item["kind"] == "implementation"
        ]
        assert implementations, lead["lead_id"]
        assert all(item["path"].endswith(".py") for item in implementations)
        assert all(item["path"].startswith("src/sigma_theory_compiler/") for item in implementations)


def test_registry_fails_closed_on_missing_or_tampered_evidence(tmp_path: Path) -> None:
    missing = {"kind": "lead_summary", "path": "missing.json", "sha256": "0" * 64}
    with pytest.raises(GravityLeadParentRegistryError, match="missing"):
        verify_evidence(tmp_path, missing)

    path = tmp_path / "evidence.json"
    path.write_text("{}\n", encoding="utf-8")
    tampered = {"kind": "source_receipt", "path": "evidence.json", "sha256": "0" * 64}
    with pytest.raises(GravityLeadParentRegistryError, match="changed"):
        verify_evidence(tmp_path, tampered)


def test_registry_fails_closed_on_implementation_source_tampering(tmp_path: Path) -> None:
    relative = "src/sigma_theory_compiler/parent.py"
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_text("def replay():\n    return True\n", encoding="utf-8")
    item = {
        "kind": "implementation",
        "path": relative,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    assert verify_evidence(tmp_path, item)["kind"] == "implementation"
    source.write_text("def replay():\n    return False\n", encoding="utf-8")
    with pytest.raises(GravityLeadParentRegistryError, match="changed"):
        verify_evidence(tmp_path, item)


def test_registry_rejects_unsafe_command_and_raw_payload_registration() -> None:
    config = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    unsafe = deepcopy(config)
    unsafe["lead_programs"][0]["safe_commands"] = [
        "python -m sigma_theory_compiler.screened_nonlocal_boundary_permutation evaluate"
    ]
    with pytest.raises(GravityLeadParentRegistryError, match="production"):
        validate_config(unsafe, ROOT)

    raw = deepcopy(config)
    raw["lead_programs"][0]["evidence"][0]["path"] = "sealed-target.fits"
    with pytest.raises(GravityLeadParentRegistryError, match="raw payload"):
        validate_config(raw, ROOT)


def test_emergent_lead_discloses_missing_exact_replay() -> None:
    config = load_config(ROOT)
    emergent = config["lead_programs"][-1]
    assert emergent["lead_id"] == "emergent_gravity_transition"
    assert emergent["bounded_local_rerun"]["possible"] is False
    assert any("compute-manifest" in text for text in emergent["known_limitations"])


def test_stored_receipt_is_exact_replay_of_registry() -> None:
    stored = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    assert stored == build_receipt(ROOT)
