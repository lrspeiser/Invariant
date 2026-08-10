from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.grammar_v3_evidence_knowledge_bridge import (
    GrammarV3EvidenceKnowledgeBridge,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "grammar_v3_evidence_knowledge_bridge.json"
PORTABLE = ROOT / "runs" / "engine" / "grammar-v3-evidence-pareto-report.json"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bridge(database: Path, config: dict | None = None, root: Path = ROOT):
    return GrammarV3EvidenceKnowledgeBridge(database, config or _load(CONFIG), root)


def test_ingest_replay_priority_and_exact_blocker_taxonomy(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path / "knowledge.sqlite")
    assert bridge.ingest() == {
        "accepted_candidates": 6,
        "duplicate_candidates": 0,
        "accepted_packets": 23,
        "duplicate_packets": 3,
        "accepted_source_links": 26,
        "duplicate_source_links": 0,
    }
    assert bridge.ingest() == {
        "accepted_candidates": 0,
        "duplicate_candidates": 6,
        "accepted_packets": 0,
        "duplicate_packets": 26,
        "accepted_source_links": 0,
        "duplicate_source_links": 26,
    }
    report = bridge.priority_report()
    assert report["candidate_count"] == 6
    assert report["candidate_packet_count"] == 11
    assert report["calibration_packet_count"] == 12
    assert report["source_packet_link_count"] == 26
    assert report["candidate_decision_counts"] == {"blocked": 6}
    assert report["evidence_packet_outcome_counts"] == {
        "blocked": 11,
        "pass": 11,
        "reject": 1,
    }
    assert report["calibration_outcome_counts"] == {"pass": 11, "reject": 1}
    assert report["observational_data_opened"] is False
    assert report["paid_llm_spend_usd"] == 0.0
    assert report["data_eligibility"] == {**ELIGIBILITY, "passed": True}
    queue = {item["formula_id"]: item for item in report["pareto_follow_up_queue"]}
    assert len(queue) == 6
    assert report["terminally_excluded"] == []
    blocker_ids = {
        seed_id: {blocker["gate_id"] for blocker in item["blocker_taxonomy"]}
        for seed_id, item in queue.items()
    }
    assert "hypersurface_orthogonal_aether" in blocker_ids[
        "G3-0b8cb2d5591bf50d2465978d"
    ]
    assert "complete_distributed_dirac_boundary_contract" in blocker_ids[
        "G3-a82c572555e5d79686bc4a4a"
    ]
    assert "uniform_weak_cell_principal_common_cone" in blocker_ids[
        "G3-1ee308440d778dfbee8094d2"
    ]
    assert report["priority_axes"] == [
        "formal_pass_count",
        "candidate_evidence_packet_count",
        "source_lineage_depth",
        "blocker_reduction_margin",
    ]
    assert all("truth_score" not in item for item in queue.values())


def test_calibration_packets_never_become_candidate_evidence(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path / "calibration.sqlite")
    bridge.ingest()
    with bridge._connect() as connection:
        calibration = connection.execute(
            "SELECT candidate_id,outcome_class,packet_json FROM evidence_packets "
            "WHERE calibration_only=1 ORDER BY packet_id"
        ).fetchall()
    assert len(calibration) == 12
    assert all(row["candidate_id"] is None for row in calibration)
    packets = [json.loads(row["packet_json"]) for row in calibration]
    assert all(packet["eligible_for_candidate_priority"] is False for packet in packets)
    negative = [packet for packet in packets if packet["outcome_class"] == "reject"]
    assert [packet["control_id"] for packet in negative] == [
        "twisting_unit_aether_hypersurface_orthogonality_negative_control"
    ]


def test_source_tamper_lineage_disk_and_live_database_fail_closed(tmp_path: Path) -> None:
    config = _load(CONFIG)
    config["sources"][0]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source file mismatch"):
        _bridge(tmp_path / "tamper.sqlite", config)

    with pytest.raises(ValueError, match="live campaign watchdog"):
        _bridge(tmp_path / "campaign-v1-live.sqlite")

    copied_root = tmp_path / "copied"
    copied_config = _load(CONFIG)
    for descriptor in copied_config["sources"]:
        source = ROOT / descriptor["path"]
        destination = copied_root / descriptor["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    g3_descriptor = copied_config["sources"][3]
    g3_path = copied_root / g3_descriptor["path"]
    g3 = _load(g3_path)
    g3["candidate_records"][0]["provenance"]["predecessor_provenance_sha256"] = "0" * 64
    body = {key: value for key, value in g3.items() if key != "content_sha256"}
    g3["content_sha256"] = _sha(body)
    g3_path.write_text(json.dumps(g3, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    g3_descriptor["content_sha256"] = g3["content_sha256"]
    g3_descriptor["file_sha256"] = hashlib.sha256(g3_path.read_bytes()).hexdigest()
    changed = _bridge(tmp_path / "lineage.sqlite", copied_config, copied_root)
    with pytest.raises(ValueError, match="prerequisite packet lineage"):
        changed.ingest()

    disk_config = _load(CONFIG)
    disk_config["budget"]["maximum_database_bytes"] = 4096
    disk_limited = _bridge(tmp_path / "disk.sqlite", disk_config)
    with pytest.raises(RuntimeError, match="disk budget"):
        disk_limited.ingest()


def test_portable_pareto_report_is_exact(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path / "portable.sqlite")
    bridge.ingest()
    assert bridge.priority_report() == _load(PORTABLE)
