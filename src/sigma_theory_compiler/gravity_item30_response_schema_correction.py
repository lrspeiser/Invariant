"""Target-blind SDSS comment-line correction for Item 30 response acquisition.

The frozen Item 30 acquisition adapter expected the CSV header on the first line.
SkyServer prepends ``#Table1`` metadata, so the original adapter failed closed after
the first exploration-only payload.  This narrow adapter removes only blank and
comment lines, preserves the frozen query/schema/sample, and writes an explicit
incident receipt.  It never queries reserved confirmations.
"""

from __future__ import annotations

import argparse
import csv
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sigma_theory_compiler.gravity_item30_screening_mechanisms import (
    GravityItem30Error,
    _content_hashed,
    _read_json,
    _response_query,
    _sha256_bytes,
    _sha256_file,
    _skyserver_query,
    _source_paths,
    _verify_content_hash,
    _write_json,
    _write_tsv,
    load_config,
    verify_sample_freeze,
    verify_science_freeze,
)


def parse_corrected_skyserver_csv(payload: bytes) -> tuple[list[dict[str, str]], list[str]]:
    lines = payload.decode("utf-8-sig", errors="strict").splitlines()
    comments = [line.strip() for line in lines if line.strip().startswith("#")]
    table_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
    if not table_lines:
        raise GravityItem30Error("empty SkyServer CSV after comment filtering")
    reader = csv.DictReader(io.StringIO("\n".join(table_lines)))
    rows = [
        {str(key): "" if value is None else str(value).strip() for key, value in row.items()}
        for row in reader
    ]
    if reader.fieldnames == ["error_message"]:
        message = rows[0]["error_message"] if rows else "unknown SkyServer error"
        raise GravityItem30Error(message)
    return rows, comments


def acquire_corrected_responses(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    sample = _read_json(paths["sample_manifest"])
    _verify_content_hash(sample, "Item 30 sample manifest")
    exploration = sorted(
        str(row["plateifu"]) for row in sample["objects"] if row["role"] == "exploration"
    )
    confirmations = {
        str(row["plateifu"]) for row in sample["objects"] if row["role"] == "reserved_confirmation"
    }
    if len(exploration) != int(config["sample"]["expected_exploration"]):
        raise GravityItem30Error("Item 30 exploration role count changed before corrected query")

    chunks: list[dict[str, Any]] = []
    all_rows: list[dict[str, str]] = []
    observed_comments: set[str] = set()
    chunk_size = int(config["sources"]["response_chunk_size"])
    expected_columns = tuple(str(value) for value in config["sources"]["response_columns"])
    for begin in range(0, len(exploration), chunk_size):
        identities = exploration[begin : begin + chunk_size]
        query = _response_query(config, identities)
        payload, url = _skyserver_query(config, query)
        rows, comments = parse_corrected_skyserver_csv(payload)
        observed_comments.update(comments)
        if rows and tuple(rows[0].keys()) != expected_columns:
            raise GravityItem30Error("MaNGA response schema changed after comment filtering")
        returned = {row["plateifu"] for row in rows}
        if returned & confirmations:
            raise GravityItem30Error("confirmation response entered corrected Item 30 acquisition")
        if not returned <= set(identities):
            raise GravityItem30Error("unrequested MaNGA response entered corrected acquisition")
        all_rows.extend(rows)
        chunks.append(
            {
                "begin": begin,
                "requested": len(identities),
                "returned": len(rows),
                "comment_lines": comments,
                "query_sha256": _sha256_bytes(query.encode()),
                "payload_sha256": _sha256_bytes(payload),
                "url_sha256": _sha256_bytes(url.encode()),
            }
        )
    if len({row["plateifu"] for row in all_rows}) != len(all_rows):
        raise GravityItem30Error("duplicate MaNGA response row")
    all_rows = sorted(all_rows, key=lambda row: row["plateifu"])
    _write_tsv(paths["exploration_responses"], all_rows, expected_columns)

    correction_path = paths["response_source_manifest"].with_name(
        "response-source-schema-correction.json"
    )
    correction = _content_hashed(
        {
            "schema_version": "invariant-gravity-item30-response-schema-correction-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "sample_freeze_commit": config["sample_freeze_commit"],
            "failure": "The frozen parser treated the leading #Table1 metadata line as the CSV header and failed closed before writing a response table.",
            "correction": "Remove only blank lines and lines whose stripped text begins with # before csv.DictReader; preserve the frozen query, columns, identities, and ordering.",
            "observed_comment_lines": sorted(observed_comments),
            "first_failed_acquisition_payloads": 1,
            "header_diagnostic_payloads": 1,
            "counts": {
                "corrected_exploration_identities_requested": len(exploration),
                "corrected_response_rows_returned": len(all_rows),
                "confirmation_identities_requested": 0,
                "confirmation_values_read": 0,
                "candidate_cells_changed": 0,
                "sample_roles_changed": 0,
                "quality_gates_changed": 0,
                "paid_api_calls": 0,
            },
            "claims": {
                "response_values_displayed_during_diagnosis": False,
                "response_values_used_to_change_science": False,
                "confirmation_opened": False,
            },
        }
    )
    _write_json(correction_path, correction)

    manifest: Mapping[str, Any] = _content_hashed(
        {
            "schema_version": "invariant-gravity-item30-response-source-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "sample_freeze_commit": config["sample_freeze_commit"],
            "endpoint": config["sources"]["skyserver_endpoint"],
            "daptype": config["sources"]["daptype"],
            "response_columns": list(expected_columns),
            "counts": {
                "exploration_identities_requested": len(exploration),
                "response_rows_returned": len(all_rows),
                "confirmation_identities_requested": 0,
                "confirmation_values_read": 0,
                "paid_api_calls": 0,
            },
            "chunks": chunks,
            "schema_correction": {
                "path": correction_path.relative_to(root).as_posix(),
                "sha256": _sha256_file(correction_path),
                "content_sha256": correction["content_sha256"],
            },
            "response_file": {
                "path": paths["exploration_responses"].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths["exploration_responses"]),
            },
        }
    )
    _write_json(paths["response_source_manifest"], manifest)
    return paths["response_source_manifest"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire",))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(acquire_corrected_responses(args.root).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
