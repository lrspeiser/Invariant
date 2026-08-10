from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.high_throughput import decode_ordinal
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY, PromotionOrchestrator
from sigma_theory_compiler.rust_promotion_bridge import (
    SERVICE_ELIGIBILITY,
    RustPromotionBridge,
)
from sigma_theory_compiler.rust_streaming_search import (
    PROMOTION_HEADER,
    PROMOTION_MAGIC,
    PROMOTION_RECORD,
)
from sigma_theory_compiler.rust_streaming_service import EXPORT_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "configs" / "generator_v2_billion.json"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _pipeline() -> dict:
    return {
        "schema_version": "sigma-promotion-pipeline-1.0",
        "external_paid_llm_calls": False,
        "maximum_evaluator_attempts": 2,
        "data_eligibility": dict(ELIGIBILITY),
        "stages": [
            {
                "name": "sampled_static",
                "category": "cheap",
                "evaluator_id": None,
                "required_evaluator_binding_sha256": None,
            },
            {
                "name": "symbolic_health",
                "category": "symbolic",
                "evaluator_id": None,
                "required_evaluator_binding_sha256": None,
            },
        ],
    }


def _write_block(
    directory: Path,
    index: int,
    start: int,
    end: int,
    identities: list[tuple[int, int]],
) -> dict:
    generator = json.loads(GENERATOR.read_text(encoding="utf-8"))
    path = directory / f"promotion-{index:08}-{start}-{end}.bin"
    with path.open("wb") as handle:
        handle.write(
            PROMOTION_HEADER.pack(
                PROMOTION_MAGIC, 1, PROMOTION_RECORD.size, start, end, len(identities)
            )
        )
        for ordinal, status in identities:
            decoded = decode_ordinal(
                generator["basis_count"], generator["max_action_terms"], ordinal
            )
            mask = sum(
                1 << position
                for position, sign in enumerate(decoded["signs"])
                if sign > 0
            )
            terms = list(decoded["term_ids"])
            handle.write(
                PROMOTION_RECORD.pack(
                    status,
                    ordinal,
                    len(terms),
                    mask,
                    0,
                    *(terms + [0xFFFF] * (6 - len(terms))),
                )
            )
    raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": "sigma-promotion-survivor-block-1.0",
        "sha256": raw_sha,
        "record_format": "SGPROM1/1 fixed-25-byte status-plus-SGSURV2-identity",
        "record_count": len(identities),
        "pass_count": sum(status == 1 for _, status in identities),
        "ambiguous_count": sum(status == 2 for _, status in identities),
        "source_block_sha256": hashlib.sha256(f"source-{index}".encode()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(f"manifest-{index}".encode()).hexdigest(),
        "generator_config_sha256": hashlib.sha256(GENERATOR.read_bytes()).hexdigest(),
        "manifest_verification_normalization": None,
        "file": path.name,
        "start_ordinal": start,
        "end_ordinal_exclusive": end,
        "source_id": "bounded-fixture-source",
        "result_status_root_sha256": hashlib.sha256(f"result-{index}".encode()).hexdigest(),
    }


def _write_export(path: Path, blocks: list[dict], *, opened: bool = False) -> dict:
    report = {
        "schema_version": EXPORT_SCHEMA,
        "service_id": "SGRS-bounded-fixture",
        "source": {
            "source_id": "bounded-fixture-source",
            "identity_sha256": "a" * 64,
            "next_ordinal": blocks[-1]["end_ordinal_exclusive"] if blocks else 0,
            "stop_ordinal": 20,
            "sequence": len(blocks),
            "deadline_utc": "2099-01-01T00:00:00+00:00",
            "owner_id": None,
            "owner_lease_expires_utc": None,
            "exhausted": False,
        },
        "blocks": blocks,
        "block_count": len(blocks),
        "survivor_identity_count": sum(block["record_count"] for block in blocks),
        "pass_count": sum(block["pass_count"] for block in blocks),
        "ambiguous_count": sum(block["ambiguous_count"] for block in blocks),
        "artifact_bytes": sum((path.parent / block["file"]).stat().st_size for block in blocks),
        "maximum_export_bytes": 1024 * 1024,
        "blocks_root_sha256": _sha(blocks),
        "data_eligibility": {
            **SERVICE_ELIGIBILITY,
            "observational_data_opened": opened,
        },
        "promotion_contract": "sampled-static identities only",
    }
    report["content_sha256"] = _sha(report)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def test_incremental_restart_growth_and_ambiguous_records_are_fail_closed(
    tmp_path: Path,
) -> None:
    first = _write_block(tmp_path, 0, 0, 4, [(0, 1), (1, 2)])
    second = _write_block(tmp_path, 1, 4, 8, [(4, 1), (6, 1)])
    export_path = tmp_path / "portable-export.json"
    _write_export(export_path, [first])
    orchestrator = PromotionOrchestrator(tmp_path / "promotion.sqlite", _pipeline())
    database = tmp_path / "bridge.sqlite"

    batch = RustPromotionBridge(database).import_incremental(
        export_path, GENERATOR, orchestrator, maximum_records=1
    )
    assert batch["registered_candidates"] == 1
    assert batch["cursor"] == {
        "next_block_index": 0,
        "next_record_index": 1,
        "next_interval_start": 0,
    }

    resumed = RustPromotionBridge(database)
    batch = resumed.import_incremental(
        export_path, GENERATOR, orchestrator, maximum_records=1
    )
    assert batch["ambiguous_not_promoted"] == 1
    assert batch["snapshot_exhausted"]
    assert orchestrator.status()["candidate_count"] == 1

    # A later portable snapshot may append a contiguous immutable block.
    _write_export(export_path, [first, second])
    batch = RustPromotionBridge(database).import_incremental(
        export_path, GENERATOR, orchestrator, maximum_records=10
    )
    assert batch["registered_candidates"] == 2
    assert batch["cursor"]["next_block_index"] == 2
    assert orchestrator.status()["candidate_count"] == 3
    assert RustPromotionBridge(database).status()["consumed"] == {
        "ambiguous": 1,
        "pass": 3,
    }

    replay = RustPromotionBridge(database).import_incremental(
        export_path, GENERATOR, orchestrator, maximum_records=10
    )
    assert replay["consumed_records"] == 0
    assert replay["registered_candidates"] == 0


def test_hash_identity_gap_replay_and_policy_tampering_are_rejected(tmp_path: Path) -> None:
    first = _write_block(tmp_path, 0, 0, 4, [(0, 1)])
    second = _write_block(tmp_path, 1, 4, 8, [(4, 1)])
    export_path = tmp_path / "portable-export.json"
    _write_export(export_path, [first, second])
    orchestrator = PromotionOrchestrator(tmp_path / "promotion.sqlite", _pipeline())
    database = tmp_path / "bridge.sqlite"
    RustPromotionBridge(database).import_incremental(
        export_path, GENERATOR, orchestrator, maximum_records=1
    )

    changed = json.loads(json.dumps(first))
    changed["source_block_sha256"] = "f" * 64
    _write_export(export_path, [changed, second])
    with pytest.raises(ValueError, match="changed or replayed"):
        RustPromotionBridge(database).import_incremental(
            export_path, GENERATOR, orchestrator, maximum_records=1
        )

    gap = {**second, "start_ordinal": 5}
    _write_export(export_path, [first, gap])
    with pytest.raises(ValueError, match="gap or replay"):
        RustPromotionBridge(tmp_path / "gap.sqlite").import_incremental(
            export_path, GENERATOR, orchestrator, maximum_records=1
        )

    _write_export(export_path, [first], opened=True)
    with pytest.raises(ValueError, match="eligibility"):
        RustPromotionBridge(tmp_path / "opened.sqlite").import_incremental(
            export_path, GENERATOR, orchestrator, maximum_records=1
        )

    _write_export(export_path, [first])
    block_path = tmp_path / first["file"]
    tampered = bytearray(block_path.read_bytes())
    tampered[-1] ^= 1
    block_path.write_bytes(tampered)
    with pytest.raises(ValueError, match="file hash"):
        RustPromotionBridge(tmp_path / "tampered.sqlite").import_incremental(
            export_path, GENERATOR, orchestrator, maximum_records=1
        )


def test_changed_ordinal_identity_and_unbounded_batches_are_rejected(tmp_path: Path) -> None:
    block = _write_block(tmp_path, 0, 0, 4, [(0, 1)])
    path = tmp_path / block["file"]
    raw = bytearray(path.read_bytes())
    record_offset = PROMOTION_HEADER.size
    fields = list(PROMOTION_RECORD.unpack(raw[record_offset:]))
    fields[5] = 49  # Valid-width term id, but not the identity decoded from ordinal zero.
    raw[record_offset:] = PROMOTION_RECORD.pack(*fields)
    path.write_bytes(raw)
    block["sha256"] = hashlib.sha256(raw).hexdigest()
    export_path = tmp_path / "portable-export.json"
    _write_export(export_path, [block])
    orchestrator = PromotionOrchestrator(tmp_path / "promotion.sqlite", _pipeline())
    bridge = RustPromotionBridge(tmp_path / "bridge.sqlite")
    with pytest.raises(ValueError, match="ordinal decoder"):
        bridge.import_incremental(export_path, GENERATOR, orchestrator, maximum_records=1)
    with pytest.raises(ValueError, match="maximum records"):
        bridge.import_incremental(export_path, GENERATOR, orchestrator, maximum_records=0)


def test_empty_snapshot_is_safe_and_does_not_guess_a_cursor(tmp_path: Path) -> None:
    export_path = tmp_path / "portable-export.json"
    _write_export(export_path, [])
    orchestrator = PromotionOrchestrator(tmp_path / "promotion.sqlite", _pipeline())
    result = RustPromotionBridge(tmp_path / "bridge.sqlite").import_incremental(
        export_path, GENERATOR, orchestrator, maximum_records=1
    )
    assert result["snapshot_exhausted"]
    assert result["cursor"] is None
    assert not RustPromotionBridge(tmp_path / "bridge.sqlite").status()["initialized"]
