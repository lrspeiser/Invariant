from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .composite_covariant_lift_campaign import compile_composite_aether_action
from .composite_negative_local_kinetic_campaign import (
    evaluate_negative_local_kinetic_family,
)
from .composite_positive_qx_tilt_campaign import evaluate_positive_qx_tilt_family
from .composite_q_degenerate_formal_campaign import (
    evaluate_zero_local_acceleration_family,
)
from .formal_backend import load_field_contract
from .production_covariant_provenance import (
    map_candidate_to_covariant_action,
    production_blocked_candidates,
)
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-composite-promotion-overlay-config-1.0"
STATUS_SCHEMA = "sigma-composite-promotion-overlay-status-1.0"
DATABASE_SCHEMA = "sigma-composite-promotion-overlay-db-1.0"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _content_valid(value: dict[str, Any]) -> bool:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return value.get("content_sha256") == _sha(body)


def validate_overlay_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "expected_source_candidate_count",
        "expected_overlay_candidate_count",
        "expected_formal_rejected_count",
        "expected_remaining_formal_blocked_count",
        "expected_formal_passed_count",
        "maximum_candidates",
        "maximum_disk_bytes",
        "maximum_wall_seconds",
        "external_paid_llm_calls",
        "data_eligibility",
        "file_sha256",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("composite overlay config fields or schema are invalid")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("composite overlay paid LLM calls must remain disabled")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("composite overlay eligibility is not fail-closed")
    integers = (
        "expected_source_candidate_count",
        "expected_overlay_candidate_count",
        "expected_formal_rejected_count",
        "expected_remaining_formal_blocked_count",
        "expected_formal_passed_count",
        "maximum_candidates",
        "maximum_disk_bytes",
    )
    if any(not isinstance(config.get(key), int) or int(config[key]) < 0 for key in integers):
        raise ValueError("composite overlay integer bounds are invalid")
    if not 1 <= int(config["maximum_candidates"]) <= 10_000:
        raise ValueError("composite overlay candidate budget is invalid")
    if int(config["expected_overlay_candidate_count"]) > int(config["maximum_candidates"]):
        raise ValueError("composite overlay expected count exceeds its budget")
    if (
        int(config["expected_formal_rejected_count"])
        + int(config["expected_remaining_formal_blocked_count"])
        + int(config["expected_formal_passed_count"])
        != int(config["expected_overlay_candidate_count"])
        or int(config["expected_formal_passed_count"]) != 0
    ):
        raise ValueError("composite overlay final formal counts are inconsistent")
    if int(config["maximum_disk_bytes"]) < 4096:
        raise ValueError("composite overlay disk budget is invalid")
    if not 0 < float(config.get("maximum_wall_seconds", 0)) <= 3600:
        raise ValueError("composite overlay wall budget is invalid")
    hashes = config.get("file_sha256")
    expected_hash_names = {
        "source_database",
        "source_dossier",
        "lift_campaign_config",
        "lift_campaign_artifact",
        "formal_campaign_config",
        "formal_campaign_artifact",
        "negative_formal_campaign_config",
        "negative_formal_campaign_artifact",
        "positive_tilt_campaign_config",
        "positive_tilt_campaign_artifact",
        "generator",
        "grammar",
        "field_contract",
        "static_dictionary",
    }
    if not isinstance(hashes, dict) or set(hashes) != expected_hash_names:
        raise ValueError("composite overlay file hash allowlist is invalid")
    for value in hashes.values():
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("composite overlay file hash is invalid")
        try:
            bytes.fromhex(value)
        except ValueError as error:
            raise ValueError("composite overlay file hash is invalid") from error


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS overlay_metadata (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  schema_version TEXT NOT NULL,
  config_json TEXT NOT NULL,
  config_sha256 TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('building','completed')),
  overlay_root_sha256 TEXT
);
CREATE TABLE IF NOT EXISTS candidate_overlay (
  candidate_id TEXT PRIMARY KEY,
  ordinal INTEGER NOT NULL,
  candidate_payload_sha256 TEXT NOT NULL,
  upstream_initial_lineage_sha256 TEXT NOT NULL,
  upstream_covariant_stage_sha256 TEXT NOT NULL,
  dossier_candidate_sha256 TEXT NOT NULL,
  lift_evidence_json TEXT NOT NULL,
  lift_input_lineage_sha256 TEXT NOT NULL,
  lift_result_sha256 TEXT NOT NULL,
  lift_output_lineage_sha256 TEXT NOT NULL,
  formal_layer_one_decision TEXT NOT NULL CHECK(formal_layer_one_decision IN ('rejected','blocked')),
  formal_layer_one_reason TEXT NOT NULL,
  formal_layer_one_evidence_json TEXT NOT NULL,
  formal_layer_one_input_lineage_sha256 TEXT NOT NULL,
  formal_layer_one_result_sha256 TEXT NOT NULL,
  formal_layer_one_output_lineage_sha256 TEXT NOT NULL,
  formal_layer_two_decision TEXT NOT NULL CHECK(formal_layer_two_decision IN ('not_run','rejected','blocked')),
  formal_layer_two_reason TEXT NOT NULL,
  formal_layer_two_evidence_json TEXT,
  formal_layer_two_input_lineage_sha256 TEXT,
  formal_layer_two_result_sha256 TEXT,
  formal_layer_two_output_lineage_sha256 TEXT,
  formal_layer_three_decision TEXT NOT NULL CHECK(formal_layer_three_decision IN ('not_run','rejected','blocked')),
  formal_layer_three_reason TEXT NOT NULL,
  formal_layer_three_evidence_json TEXT,
  formal_layer_three_input_lineage_sha256 TEXT,
  formal_layer_three_result_sha256 TEXT,
  formal_layer_three_output_lineage_sha256 TEXT,
  formal_decision TEXT NOT NULL CHECK(formal_decision IN ('rejected','blocked')),
  formal_reason TEXT NOT NULL,
  formal_output_lineage_sha256 TEXT NOT NULL,
  solar_state TEXT NOT NULL CHECK(solar_state='blocked'),
  galaxy_state TEXT NOT NULL CHECK(galaxy_state='blocked'),
  overlay_record_sha256 TEXT NOT NULL UNIQUE
);
"""


class CompositePromotionOverlay:
    """One-way, restart-safe overlay for exact reviewed composite campaign decisions."""

    def __init__(
        self,
        database: str | Path,
        config: dict[str, Any],
        *,
        source_database: str | Path,
        source_dossier: str | Path,
        lift_campaign_config: str | Path,
        lift_campaign_artifact: str | Path,
        formal_campaign_config: str | Path,
        formal_campaign_artifact: str | Path,
        negative_formal_campaign_config: str | Path,
        negative_formal_campaign_artifact: str | Path,
        positive_tilt_campaign_config: str | Path,
        positive_tilt_campaign_artifact: str | Path,
        generator: str | Path,
        grammar: str | Path,
        field_contract: str | Path,
        static_dictionary: str | Path,
    ) -> None:
        validate_overlay_config(config)
        self.database = Path(database).resolve()
        self.config = config
        self.config_sha256 = _sha(config)
        self.paths = {
            "source_database": Path(source_database).resolve(),
            "source_dossier": Path(source_dossier).resolve(),
            "lift_campaign_config": Path(lift_campaign_config).resolve(),
            "lift_campaign_artifact": Path(lift_campaign_artifact).resolve(),
            "formal_campaign_config": Path(formal_campaign_config).resolve(),
            "formal_campaign_artifact": Path(formal_campaign_artifact).resolve(),
            "negative_formal_campaign_config": Path(
                negative_formal_campaign_config
            ).resolve(),
            "negative_formal_campaign_artifact": Path(
                negative_formal_campaign_artifact
            ).resolve(),
            "positive_tilt_campaign_config": Path(
                positive_tilt_campaign_config
            ).resolve(),
            "positive_tilt_campaign_artifact": Path(
                positive_tilt_campaign_artifact
            ).resolve(),
            "generator": Path(generator).resolve(),
            "grammar": Path(grammar).resolve(),
            "field_contract": Path(field_contract).resolve(),
            "static_dictionary": Path(static_dictionary).resolve(),
        }
        for name, path in self.paths.items():
            if not path.is_file() or _file_sha(path) != config["file_sha256"][name]:
                raise ValueError(f"hash-bound {name} file mismatch")
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database) as connection:
            connection.executescript(SCHEMA)
            row = connection.execute(
                "SELECT config_sha256 FROM overlay_metadata WHERE singleton=1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO overlay_metadata VALUES (1,?,?,?,'building',NULL)",
                    (DATABASE_SCHEMA, _canonical(config), self.config_sha256),
                )
            elif row[0] != self.config_sha256:
                raise ValueError("refusing to resume a changed composite overlay config")

    def _source_snapshot(self) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        dossier = _load(self.paths["source_dossier"])
        if not _content_valid(dossier):
            raise ValueError("source dossier content hash mismatch")
        expected_dossier_eligibility = {**ELIGIBILITY, "passed": True}
        if dossier.get("data_eligibility") != expected_dossier_eligibility:
            raise ValueError("source dossier eligibility is not fail-closed")
        if int(dossier.get("candidate_count", -1)) != int(
            self.config["expected_source_candidate_count"]
        ):
            raise ValueError("source dossier candidate count mismatch")
        dossier_rows = dossier.get("candidate_dossiers")
        if not isinstance(dossier_rows, list) or len(dossier_rows) != int(
            self.config["expected_source_candidate_count"]
        ):
            raise ValueError("source dossier candidate records are incomplete")
        dossier_by_id = {str(item["candidate_id"]): item for item in dossier_rows}
        if len(dossier_by_id) != len(dossier_rows):
            raise ValueError("source dossier contains duplicate candidate ids")

        source = self.paths["source_database"]
        connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only=ON")
            candidate_count = connection.execute("SELECT count(*) FROM candidates").fetchone()[0]
            if candidate_count != int(self.config["expected_source_candidate_count"]):
                raise ValueError("source registry candidate count mismatch")
            rows = connection.execute(
                "SELECT c.candidate_id,c.ordinal,c.payload_json,c.initial_lineage_sha256,"
                "s.stage_index,s.stage_name,s.category,s.state,s.blocker,s.attempt,"
                "s.evaluator_binding_sha256,s.input_lineage_sha256,s.result_sha256,"
                "s.output_lineage_sha256 FROM candidates c JOIN candidate_stages s "
                "USING(candidate_id) WHERE s.stage_name='covariant_symbolic_health' "
                "AND s.state='blocked' ORDER BY c.ordinal"
            ).fetchall()
        finally:
            connection.close()
        upstream: dict[str, dict[str, Any]] = {}
        for row in rows:
            item = dict(row)
            payload = json.loads(item.pop("payload_json"))
            item["payload"] = payload
            upstream[str(item["candidate_id"])] = item
        return upstream, {"report": dossier, "by_id": dossier_by_id}

    def _derive_records(self, deadline: float) -> list[dict[str, Any]]:
        lift_config = _load(self.paths["lift_campaign_config"])
        formal_config = _load(self.paths["formal_campaign_config"])
        negative_config = _load(self.paths["negative_formal_campaign_config"])
        positive_config = _load(self.paths["positive_tilt_campaign_config"])
        lift_artifact = _load(self.paths["lift_campaign_artifact"])
        formal_artifact = _load(self.paths["formal_campaign_artifact"])
        negative_artifact = _load(self.paths["negative_formal_campaign_artifact"])
        positive_artifact = _load(self.paths["positive_tilt_campaign_artifact"])
        if lift_artifact.get("schema_version") != (
            "sigma-composite-covariant-lift-campaign-1.0"
        ) or formal_artifact.get("schema_version") != (
            "sigma-composite-q-degenerate-formal-campaign-1.0"
        ):
            raise ValueError("composite campaign artifact schema mismatch")
        if negative_artifact.get("schema_version") != (
            "sigma-composite-negative-local-kinetic-campaign-1.0"
        ):
            raise ValueError("negative local kinetic campaign artifact schema mismatch")
        if positive_artifact.get("schema_version") != (
            "sigma-composite-positive-qx-tilt-campaign-1.0"
        ):
            raise ValueError("positive Q/X tilt campaign artifact schema mismatch")
        if not all(
            _content_valid(artifact)
            for artifact in (
                lift_artifact,
                formal_artifact,
                negative_artifact,
                positive_artifact,
            )
        ):
            raise ValueError("composite campaign artifact content hash mismatch")
        for campaign in (
            lift_artifact,
            formal_artifact,
            negative_artifact,
            positive_artifact,
        ):
            if campaign.get("data_eligibility") != ELIGIBILITY:
                raise ValueError("composite campaign eligibility is not fail-closed")
        if (
            lift_artifact.get("source_database_file_sha256")
            != self.config["file_sha256"]["source_database"]
            or formal_artifact.get("source_database_file_sha256")
            != self.config["file_sha256"]["source_database"]
            or negative_artifact.get("source_database_file_sha256")
            != self.config["file_sha256"]["source_database"]
            or positive_artifact.get("source_database_file_sha256")
            != self.config["file_sha256"]["source_database"]
        ):
            raise ValueError("composite campaigns target a different source registry")
        for config in (lift_config, formal_config, negative_config, positive_config):
            if config.get("data_eligibility") != ELIGIBILITY:
                raise ValueError("composite campaign config eligibility is not fail-closed")
            bindings = {
                "database_file_sha256": "source_database",
                "generator_file_sha256": "generator",
                "grammar_file_sha256": "grammar",
                "field_contract_file_sha256": "field_contract",
                "static_dictionary_file_sha256": "static_dictionary",
            }
            if any(
                config.get(field) != self.config["file_sha256"][path_name]
                for field, path_name in bindings.items()
            ):
                raise ValueError("composite campaign input binding mismatch")
        if lift_config.get("source_summary_file_sha256") != formal_config.get(
            "source_summary_file_sha256"
        ) or lift_config.get("source_summary_file_sha256") != negative_config.get(
            "source_summary_file_sha256"
        ) or lift_config.get("source_summary_file_sha256") != positive_config.get(
            "source_summary_file_sha256"
        ):
            raise ValueError("composite campaigns target different source summaries")
        if (
            negative_config.get("prior_campaign_file_sha256")
            != self.config["file_sha256"]["formal_campaign_artifact"]
            or negative_config.get("prior_campaign_content_sha256")
            != formal_artifact.get("content_sha256")
            or negative_artifact.get("prior_campaign_content_sha256")
            != formal_artifact.get("content_sha256")
        ):
            raise ValueError("negative local kinetic campaign prior-layer binding mismatch")
        if (
            int(negative_config.get("expected_input_candidate_count", -1))
            != int(formal_artifact.get("remaining_formal_blocked_count", -2))
            or int(negative_artifact.get("input_candidate_count", -1))
            != int(formal_artifact.get("remaining_formal_blocked_count", -2))
        ):
            raise ValueError("negative local kinetic campaign input count binding mismatch")
        if (
            positive_config.get("prior_campaign_file_sha256")
            != self.config["file_sha256"]["negative_formal_campaign_artifact"]
            or positive_config.get("prior_campaign_content_sha256")
            != negative_artifact.get("content_sha256")
            or positive_artifact.get("prior_campaign_content_sha256")
            != negative_artifact.get("content_sha256")
        ):
            raise ValueError("positive Q/X tilt campaign prior-layer binding mismatch")
        if (
            int(positive_config.get("expected_input_candidate_count", -1))
            != int(negative_artifact.get("remaining_formal_blocked_count", -2))
            or int(positive_artifact.get("input_candidate_count", -1))
            != int(negative_artifact.get("remaining_formal_blocked_count", -2))
        ):
            raise ValueError("positive Q/X tilt campaign input count binding mismatch")
        source_summary = lift_config.get("source_summary_file_sha256")
        if any(
            artifact.get("source_summary_file_sha256") != source_summary
            for artifact in (
                lift_artifact,
                formal_artifact,
                negative_artifact,
                positive_artifact,
            )
        ):
            raise ValueError("composite campaign artifact source summary binding mismatch")

        generator = _load(self.paths["generator"])
        grammar = _load(self.paths["grammar"])
        field_contract = load_field_contract(self.paths["field_contract"])
        static_dictionary = _load(self.paths["static_dictionary"])
        if static_dictionary.get("content_sha256") != lift_config.get(
            "static_dictionary_content_sha256"
        ) or static_dictionary.get("content_sha256") != formal_config.get(
            "static_dictionary_content_sha256"
        ) or static_dictionary.get("content_sha256") != negative_config.get(
            "static_dictionary_content_sha256"
        ) or static_dictionary.get("content_sha256") != positive_config.get(
            "static_dictionary_content_sha256"
        ):
            raise ValueError("static dictionary content binding mismatch")
        candidates = production_blocked_candidates(self.paths["source_database"])
        if len(candidates) != int(self.config["expected_overlay_candidate_count"]):
            raise ValueError("source registry overlay candidate count mismatch")

        records: list[dict[str, Any]] = []
        lift_identities: list[dict[str, Any]] = []
        formal_identities: list[dict[str, Any]] = []
        negative_formal_identities: list[dict[str, Any]] = []
        positive_tilt_identities: list[dict[str, Any]] = []
        for candidate in candidates:
            if time.monotonic() >= deadline:
                raise TimeoutError("composite overlay derivation wall budget exhausted")
            existing = map_candidate_to_covariant_action(
                candidate,
                generator,
                grammar,
                field_contract,
                source_sha256=str(lift_config["source_summary_file_sha256"]),
            )
            if existing["decision"] == "mapped":
                lift = {
                    "decision": "mapped_existing_typed_action",
                    "candidate_id": existing["candidate_id"],
                    "ordinal": existing["ordinal"],
                    "correction_expression": existing["correction_expression"],
                    "covariant_action_provenance": existing["covariant_action_provenance"],
                    "formal_outcome": {
                        "decision": "reject",
                        "reason": existing["formal_preflight"]["q_operator_conclusion"],
                    },
                    "data_eligibility": dict(ELIGIBILITY),
                }
                formal_layer_one = {
                    "schema_version": "sigma-existing-typed-action-formal-overlay-1.0",
                    "candidate_id": existing["candidate_id"],
                    "ordinal": existing["ordinal"],
                    "decision": "reject",
                    "reason": existing["formal_preflight"]["q_operator_conclusion"],
                    "input_action_sha256": existing["covariant_action_provenance"][
                        "input_action_sha256"
                    ],
                    "provenance_binding_sha256": existing["covariant_action_provenance"][
                        "provenance_binding_sha256"
                    ],
                    "data_eligibility": dict(ELIGIBILITY),
                }
                formal_layer_two = None
                formal_layer_three = None
            else:
                lift = compile_composite_aether_action(
                    candidate,
                    field_contract,
                    field_contract_file_sha256=str(
                        lift_config["field_contract_file_sha256"]
                    ),
                    static_dictionary_file_sha256=str(
                        lift_config["static_dictionary_file_sha256"]
                    ),
                    static_dictionary_content_sha256=str(
                        lift_config["static_dictionary_content_sha256"]
                    ),
                    source_sha256=str(lift_config["source_summary_file_sha256"]),
                )
                if lift.get("decision") != "mapped":
                    raise ValueError("composite campaign lift did not map a blocked candidate")
                formal_layer_one = evaluate_zero_local_acceleration_family(lift)
                formal_identities.append(
                    {
                        "candidate_id": formal_layer_one["candidate_id"],
                        "ordinal": formal_layer_one["ordinal"],
                        "decision": formal_layer_one["decision"],
                        "reason_or_blocker": formal_layer_one.get(
                            "reason", formal_layer_one.get("blocker")
                        ),
                        "input_action_sha256": formal_layer_one["input_action_sha256"],
                        "provenance_binding_sha256": formal_layer_one[
                            "provenance_binding_sha256"
                        ],
                        "formal_evidence_sha256": formal_layer_one.get("content_sha256"),
                    }
                )
                formal_layer_two = (
                    evaluate_negative_local_kinetic_family(lift)
                    if formal_layer_one["decision"] == "blocked"
                    else None
                )
                if formal_layer_two is not None:
                    negative_formal_identities.append(
                        {
                            "candidate_id": formal_layer_two["candidate_id"],
                            "ordinal": formal_layer_two["ordinal"],
                            "decision": formal_layer_two["decision"],
                            "reason_or_blocker": formal_layer_two.get(
                                "reason", formal_layer_two.get("blocker")
                            ),
                            "input_action_sha256": formal_layer_two[
                                "input_action_sha256"
                            ],
                            "provenance_binding_sha256": formal_layer_two[
                                "provenance_binding_sha256"
                            ],
                            "formal_evidence_sha256": formal_layer_two.get(
                                "content_sha256"
                            ),
                        }
                    )
                formal_layer_three = (
                    evaluate_positive_qx_tilt_family(lift)
                    if formal_layer_two is not None
                    and formal_layer_two["decision"] == "blocked"
                    else None
                )
                if formal_layer_three is not None:
                    positive_tilt_identities.append(
                        {
                            "candidate_id": formal_layer_three["candidate_id"],
                            "ordinal": formal_layer_three["ordinal"],
                            "decision": formal_layer_three["decision"],
                            "reason_or_blocker": formal_layer_three.get(
                                "reason", formal_layer_three.get("blocker")
                            ),
                            "input_action_sha256": formal_layer_three[
                                "input_action_sha256"
                            ],
                            "provenance_binding_sha256": formal_layer_three[
                                "provenance_binding_sha256"
                            ],
                            "formal_evidence_sha256": formal_layer_three.get(
                                "content_sha256"
                            ),
                        }
                    )
            provenance = lift["covariant_action_provenance"]
            lift_identities.append(
                {
                    "candidate_id": lift["candidate_id"],
                    "ordinal": lift["ordinal"],
                    "decision": lift["decision"],
                    "action_sha256": provenance["input_action_sha256"],
                    "provenance_binding_sha256": provenance["provenance_binding_sha256"],
                    "formal_outcome": lift["formal_outcome"],
                }
            )
            records.append(
                {
                    "candidate": candidate,
                    "lift": lift,
                    "formal_layer_one": formal_layer_one,
                    "formal_layer_two": formal_layer_two,
                    "formal_layer_three": formal_layer_three,
                }
            )
            if time.monotonic() >= deadline:
                raise TimeoutError("composite overlay derivation wall budget exhausted")

        if _sha(lift_identities) != lift_artifact.get("candidate_provenance_root_sha256"):
            raise ValueError("derived lift decisions differ from the reviewed campaign root")
        if _sha(formal_identities) != formal_artifact.get("candidate_evidence_root_sha256"):
            raise ValueError("derived formal decisions differ from the reviewed campaign root")
        if _sha(negative_formal_identities) != negative_artifact.get(
            "candidate_evidence_root_sha256"
        ):
            raise ValueError(
                "derived negative local kinetic decisions differ from the reviewed campaign root"
            )
        if _sha(positive_tilt_identities) != positive_artifact.get(
            "candidate_evidence_root_sha256"
        ):
            raise ValueError(
                "derived positive Q/X tilt decisions differ from the reviewed campaign root"
            )
        counts = Counter(
            str(
                (
                    record["formal_layer_three"]
                    or record["formal_layer_two"]
                    or record["formal_layer_one"]
                )["decision"]
            )
            for record in records
        )
        if counts != Counter(
            {
                "reject": int(self.config["expected_formal_rejected_count"]),
                "blocked": int(self.config["expected_remaining_formal_blocked_count"]),
            }
        ):
            raise ValueError("derived formal decision counts differ from the overlay contract")
        return records

    def apply(self, *, maximum_new_records: int | None = None) -> dict[str, Any]:
        started = time.monotonic()
        deadline = started + float(self.config["maximum_wall_seconds"])
        source_hash_before = _file_sha(self.paths["source_database"])
        dossier_hash_before = _file_sha(self.paths["source_dossier"])
        upstream, dossier = self._source_snapshot()
        records = self._derive_records(deadline)
        maximum_new = len(records) if maximum_new_records is None else maximum_new_records
        if not isinstance(maximum_new, int) or maximum_new < 0:
            raise ValueError("maximum new overlay records must be a nonnegative integer")
        inserted = replayed = 0
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            for source_record in records:
                if time.monotonic() - started > float(self.config["maximum_wall_seconds"]):
                    break
                if self.database.stat().st_size > int(self.config["maximum_disk_bytes"]):
                    raise ValueError("composite overlay disk budget exhausted")
                candidate = source_record["candidate"]
                candidate_id = str(candidate["candidate_id"])
                upstream_row = upstream.get(candidate_id)
                dossier_row = dossier["by_id"].get(candidate_id)
                if upstream_row is None or dossier_row is None:
                    raise ValueError("overlay candidate is absent from source registry or dossier")
                if (
                    dossier_row.get("initial_lineage_sha256")
                    != upstream_row["initial_lineage_sha256"]
                    or dossier_row.get("first_nonpass", {}).get("stage_name")
                    != "covariant_symbolic_health"
                ):
                    raise ValueError("source dossier candidate lineage or gate state mismatch")
                upstream_stage = {
                    key: upstream_row[key]
                    for key in (
                        "candidate_id",
                        "stage_index",
                        "stage_name",
                        "category",
                        "state",
                        "blocker",
                        "attempt",
                        "evaluator_binding_sha256",
                        "input_lineage_sha256",
                        "result_sha256",
                        "output_lineage_sha256",
                    )
                }
                lift_evidence = source_record["lift"]
                formal_layer_one = source_record["formal_layer_one"]
                formal_layer_two = source_record["formal_layer_two"]
                formal_layer_three = source_record["formal_layer_three"]
                lift_result = _sha(lift_evidence)
                lift_input = _sha(
                    {
                        "candidate_id": candidate_id,
                        "upstream_initial_lineage_sha256": upstream_row[
                            "initial_lineage_sha256"
                        ],
                        "upstream_covariant_stage_sha256": _sha(upstream_stage),
                        "lift_campaign_file_sha256": self.config["file_sha256"][
                            "lift_campaign_artifact"
                        ],
                    }
                )
                lift_output = _sha(
                    {"input_lineage_sha256": lift_input, "result_sha256": lift_result}
                )
                layer_one_result = _sha(formal_layer_one)
                layer_one_campaign_hash = (
                    self.config["file_sha256"]["lift_campaign_artifact"]
                    if lift_evidence["decision"] == "mapped_existing_typed_action"
                    else self.config["file_sha256"]["formal_campaign_artifact"]
                )
                layer_one_input = _sha(
                    {
                        "candidate_id": candidate_id,
                        "prior_lineage_sha256": lift_output,
                        "formal_campaign_file_sha256": layer_one_campaign_hash,
                    }
                )
                layer_one_output = _sha(
                    {
                        "input_lineage_sha256": layer_one_input,
                        "result_sha256": layer_one_result,
                    }
                )
                layer_one_decision = str(formal_layer_one["decision"])
                layer_one_state = (
                    "rejected" if layer_one_decision == "reject" else "blocked"
                )
                layer_one_reason = str(
                    formal_layer_one.get("reason", formal_layer_one.get("blocker", ""))
                )
                if formal_layer_two is None:
                    layer_two_state = "not_run"
                    layer_two_reason = "prior_formal_layer_rejected"
                    layer_two_json = None
                    layer_two_input = None
                    layer_two_result = None
                    layer_two_output = None
                    final_evidence = formal_layer_one
                    formal_output = layer_one_output
                else:
                    layer_two_decision = str(formal_layer_two["decision"])
                    layer_two_state = (
                        "rejected" if layer_two_decision == "reject" else "blocked"
                    )
                    layer_two_reason = str(
                        formal_layer_two.get(
                            "reason", formal_layer_two.get("blocker", "")
                        )
                    )
                    layer_two_json = _canonical(formal_layer_two)
                    layer_two_result = _sha(formal_layer_two)
                    layer_two_input = _sha(
                        {
                            "candidate_id": candidate_id,
                            "prior_lineage_sha256": layer_one_output,
                            "formal_campaign_file_sha256": self.config["file_sha256"][
                                "negative_formal_campaign_artifact"
                            ],
                        }
                    )
                    layer_two_output = _sha(
                        {
                            "input_lineage_sha256": layer_two_input,
                            "result_sha256": layer_two_result,
                        }
                    )
                    final_evidence = formal_layer_two
                    formal_output = layer_two_output
                if formal_layer_three is None:
                    layer_three_state = "not_run"
                    layer_three_reason = "prior_formal_layer_rejected"
                    layer_three_json = None
                    layer_three_input = None
                    layer_three_result = None
                    layer_three_output = None
                else:
                    if layer_two_output is None:
                        raise ValueError("third formal layer lacks prior output lineage")
                    layer_three_decision = str(formal_layer_three["decision"])
                    layer_three_state = (
                        "rejected" if layer_three_decision == "reject" else "blocked"
                    )
                    layer_three_reason = str(
                        formal_layer_three.get(
                            "reason", formal_layer_three.get("blocker", "")
                        )
                    )
                    layer_three_json = _canonical(formal_layer_three)
                    layer_three_result = _sha(formal_layer_three)
                    layer_three_input = _sha(
                        {
                            "candidate_id": candidate_id,
                            "prior_lineage_sha256": layer_two_output,
                            "formal_campaign_file_sha256": self.config["file_sha256"][
                                "positive_tilt_campaign_artifact"
                            ],
                        }
                    )
                    layer_three_output = _sha(
                        {
                            "input_lineage_sha256": layer_three_input,
                            "result_sha256": layer_three_result,
                        }
                    )
                    final_evidence = formal_layer_three
                    formal_output = layer_three_output
                final_decision = str(final_evidence["decision"])
                formal_state = "rejected" if final_decision == "reject" else "blocked"
                reason = str(
                    final_evidence.get("reason", final_evidence.get("blocker", ""))
                )
                body = {
                    "candidate_id": candidate_id,
                    "ordinal": int(candidate["ordinal"]),
                    "candidate_payload_sha256": _sha(candidate),
                    "upstream_initial_lineage_sha256": upstream_row[
                        "initial_lineage_sha256"
                    ],
                    "upstream_covariant_stage_sha256": _sha(upstream_stage),
                    "dossier_candidate_sha256": _sha(dossier_row),
                    "lift_evidence_json": _canonical(lift_evidence),
                    "lift_input_lineage_sha256": lift_input,
                    "lift_result_sha256": lift_result,
                    "lift_output_lineage_sha256": lift_output,
                    "formal_layer_one_decision": layer_one_state,
                    "formal_layer_one_reason": layer_one_reason,
                    "formal_layer_one_evidence_json": _canonical(formal_layer_one),
                    "formal_layer_one_input_lineage_sha256": layer_one_input,
                    "formal_layer_one_result_sha256": layer_one_result,
                    "formal_layer_one_output_lineage_sha256": layer_one_output,
                    "formal_layer_two_decision": layer_two_state,
                    "formal_layer_two_reason": layer_two_reason,
                    "formal_layer_two_evidence_json": layer_two_json,
                    "formal_layer_two_input_lineage_sha256": layer_two_input,
                    "formal_layer_two_result_sha256": layer_two_result,
                    "formal_layer_two_output_lineage_sha256": layer_two_output,
                    "formal_layer_three_decision": layer_three_state,
                    "formal_layer_three_reason": layer_three_reason,
                    "formal_layer_three_evidence_json": layer_three_json,
                    "formal_layer_three_input_lineage_sha256": layer_three_input,
                    "formal_layer_three_result_sha256": layer_three_result,
                    "formal_layer_three_output_lineage_sha256": layer_three_output,
                    "formal_decision": formal_state,
                    "formal_reason": reason,
                    "formal_output_lineage_sha256": formal_output,
                    "solar_state": "blocked",
                    "galaxy_state": "blocked",
                }
                record_sha = _sha(body)
                existing = connection.execute(
                    "SELECT overlay_record_sha256 FROM candidate_overlay WHERE candidate_id=?",
                    (candidate_id,),
                ).fetchone()
                if existing is not None:
                    if existing[0] != record_sha:
                        raise ValueError("refusing changed overlay evidence on replay")
                    replayed += 1
                    continue
                if inserted >= maximum_new:
                    continue
                connection.execute(
                    "INSERT INTO candidate_overlay VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (*body.values(), record_sha),
                )
                connection.commit()
                inserted += 1

            rows = connection.execute(
                "SELECT candidate_id,overlay_record_sha256 FROM candidate_overlay ORDER BY ordinal"
            ).fetchall()
            complete = len(rows) == int(self.config["expected_overlay_candidate_count"])
            overlay_root = (
                _sha(
                    {
                        "config_sha256": self.config_sha256,
                        "records": [dict(row) for row in rows],
                    }
                )
                if complete
                else None
            )
            connection.execute(
                "UPDATE overlay_metadata SET state=?,overlay_root_sha256=? WHERE singleton=1",
                ("completed" if complete else "building", overlay_root),
            )
            connection.commit()
        if (
            _file_sha(self.paths["source_database"]) != source_hash_before
            or _file_sha(self.paths["source_dossier"]) != dossier_hash_before
        ):
            raise ValueError("upstream registry or dossier changed during overlay import")
        return self.status(inserted=inserted, replayed=replayed)

    def status(self, *, inserted: int = 0, replayed: int = 0) -> dict[str, Any]:
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            metadata = connection.execute("SELECT * FROM overlay_metadata WHERE singleton=1").fetchone()
            counts = {
                row[0]: row[1]
                for row in connection.execute(
                    "SELECT formal_decision,count(*) FROM candidate_overlay GROUP BY formal_decision"
                )
            }
            layer_one_counts = {
                row[0]: row[1]
                for row in connection.execute(
                    "SELECT formal_layer_one_decision,count(*) FROM candidate_overlay "
                    "GROUP BY formal_layer_one_decision"
                )
            }
            layer_two_counts = {
                row[0]: row[1]
                for row in connection.execute(
                    "SELECT formal_layer_two_decision,count(*) FROM candidate_overlay "
                    "GROUP BY formal_layer_two_decision"
                )
            }
            layer_three_counts = {
                row[0]: row[1]
                for row in connection.execute(
                    "SELECT formal_layer_three_decision,count(*) FROM candidate_overlay "
                    "GROUP BY formal_layer_three_decision"
                )
            }
            imported = connection.execute("SELECT count(*) FROM candidate_overlay").fetchone()[0]
        status = {
            "schema_version": STATUS_SCHEMA,
            "state": metadata["state"],
            "source_candidate_count": int(self.config["expected_source_candidate_count"]),
            "upstream_terminal_candidate_count": int(
                self.config["expected_source_candidate_count"]
            )
            - int(self.config["expected_overlay_candidate_count"]),
            "overlay_candidate_count": imported,
            "lift_passed_count": imported,
            "formal_rejected_count": int(counts.get("rejected", 0)),
            "remaining_formal_blocked_count": int(counts.get("blocked", 0)),
            "formal_passed_count": int(self.config["expected_formal_passed_count"]),
            "formal_layer_one_counts": dict(sorted(layer_one_counts.items())),
            "formal_layer_two_counts": dict(sorted(layer_two_counts.items())),
            "formal_layer_three_counts": dict(sorted(layer_three_counts.items())),
            "solar_opened_count": 0,
            "galaxy_opened_count": 0,
            "inserted_this_run": inserted,
            "replayed_this_run": replayed,
            "overlay_root_sha256": metadata["overlay_root_sha256"],
            "source_database_file_sha256": self.config["file_sha256"]["source_database"],
            "source_dossier_file_sha256": self.config["file_sha256"]["source_dossier"],
            "lift_campaign_file_sha256": self.config["file_sha256"][
                "lift_campaign_artifact"
            ],
            "formal_campaign_file_sha256": self.config["file_sha256"][
                "formal_campaign_artifact"
            ],
            "negative_formal_campaign_file_sha256": self.config["file_sha256"][
                "negative_formal_campaign_artifact"
            ],
            "positive_tilt_campaign_file_sha256": self.config["file_sha256"][
                "positive_tilt_campaign_artifact"
            ],
            "upstream_mutation_contract": "read-only source registry and dossier",
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
        }
        status["content_sha256"] = _sha(status)
        return status

    def export(self) -> dict[str, Any]:
        """Return a deterministic dossier overlay without changing the source dossier."""

        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            metadata = connection.execute(
                "SELECT state,overlay_root_sha256 FROM overlay_metadata WHERE singleton=1"
            ).fetchone()
            if metadata["state"] != "completed":
                raise ValueError("cannot export an incomplete composite overlay")
            rows = [
                dict(row)
                for row in connection.execute(
                    "SELECT candidate_id,ordinal,candidate_payload_sha256,"
                    "upstream_initial_lineage_sha256,upstream_covariant_stage_sha256,"
                    "dossier_candidate_sha256,lift_input_lineage_sha256,lift_result_sha256,"
                    "lift_output_lineage_sha256,formal_layer_one_decision,"
                    "formal_layer_one_reason,formal_layer_one_input_lineage_sha256,"
                    "formal_layer_one_result_sha256,formal_layer_one_output_lineage_sha256,"
                    "formal_layer_two_decision,formal_layer_two_reason,"
                    "formal_layer_two_input_lineage_sha256,formal_layer_two_result_sha256,"
                    "formal_layer_two_output_lineage_sha256,formal_layer_three_decision,"
                    "formal_layer_three_reason,formal_layer_three_input_lineage_sha256,"
                    "formal_layer_three_result_sha256,formal_layer_three_output_lineage_sha256,"
                    "formal_decision,formal_reason,"
                    "formal_output_lineage_sha256,solar_state,galaxy_state,"
                    "overlay_record_sha256 FROM candidate_overlay ORDER BY ordinal"
                )
            ]
        report = {
            "schema_version": "sigma-composite-promotion-overlay-export-1.0",
            "source_database_file_sha256": self.config["file_sha256"]["source_database"],
            "source_dossier_file_sha256": self.config["file_sha256"]["source_dossier"],
            "overlay_root_sha256": metadata["overlay_root_sha256"],
            "candidate_count": len(rows),
            "candidate_overlays": rows,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "interpretation": (
                "This is a one-way decision overlay. It does not alter upstream Rust evidence, "
                "open observations, or convert blocked formal families into passes."
            ),
        }
        report["content_sha256"] = _sha(report)
        return report
