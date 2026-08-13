from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA = "sigma-quartic-tc2-d4-coordinate-free-candidate-normalization-registration-1.0"
CONFIG_SCHEMA = (
    "sigma-quartic-tc2-d4-coordinate-free-candidate-normalization-registration-config-1.0"
)
STATUS = "block_coordinate_free_D4_recurrence_emitter_missing_286_symbolic_packets"
CONFIG_PATH = (
    "configs/backgrounds/quartic_tc2_d4_coordinate_free_candidate_normalization_registration.json"
)
SOURCE_PATH = (
    "src/sigma_theory_compiler/"
    "quartic_tc2_d4_coordinate_free_candidate_normalization_registration.py"
)
TEST_PATH = "tests/test_quartic_tc2_d4_coordinate_free_candidate_normalization_registration.py"
REQUIRED_PACKETS = 304
PREDECESSOR_REGISTERED = 6
NEW_PACKETS = 12
REGISTERED_PACKETS = 18
MISSING_PACKETS = 286
REQUIRED_ROWS = 117_180
EXPECTED_UPSTREAMS = {
    "P55_recurrence_registration": {
        "file_sha256": "4a940de0726fa319a311d3a1d311c912afe09d00e2f6dfc62eeef1a2b6d5f149",
        "content_sha256": "2dfab5867211cf5052b4b43df2179cb71d9bdc239daa4f522343fea42b09dc73",
        "status": "block_coordinate_free_D4_recurrence_emitter_missing_298_symbolic_packets",
    },
    "canonical_D4_obstruction": {
        "file_sha256": "73cb76e16153b1f7f781b6fa1bdc6659a03553b583456f7aebbd46516f9abfe5",
        "content_sha256": "bef3246a17942c74e8f3cdb09bc14a36c6bdc44d030a9a70ce833c30ec04bc65",
        "status": "pass_exact_canonical_d4_obstruction_cokernel_classification",
    },
    "curl_constraint_admission": {
        "file_sha256": "61beec6a3be333eba6858e04339cd978acb42dfb049b055ddcbf3e5edb732401",
        "content_sha256": "df37680b05b3d6da32e107bee16397a1a37b77d97c3da5a26c0bd49ad194cb6a",
        "status": "pass_exact_gauge_fixed_curl_constraint_admission_for_minimal_V",
    },
}

FALSE_CLAIMS = {
    "B7_closed",
    "CK1_closed",
    "CK3_closed",
    "TC2_closed",
    "complete_coordinate_free_coefficient_map_emitted",
    "complete_coordinate_free_rhs_emitted",
    "full_direction_sphere_D4_compatibility_proved",
    "full_high_atom_identity_proved",
    "global_H7_closed",
    "lifespan_proved",
    "matrix_projectors_evaluated",
    "nonlinear_PDE_closure_proved",
    "phase_two_exact_solve_admitted",
    "theory_candidate_rejected",
}


class CandidateNormalizationRegistrationError(ValueError):
    """Raised when exact candidate-normalization registration fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _content_hash(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _hash_matches(value: dict[str, Any]) -> bool:
    return value.get("content_sha256") == _content_hash(value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CandidateNormalizationRegistrationError(f"expected JSON object: {path}")
    return value


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise CandidateNormalizationRegistrationError("bound path escaped project root")
    return path


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else str(value)


def _validate_config(config: dict[str, Any]) -> None:
    target = config.get("target", {})
    caps = config.get("resource_caps", {})
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy") != "register_largest_complete_exact_serialized_family_fail_closed"
        or not _hash_matches(config)
        or target
        != {
            "required_symbolic_input_packets": REQUIRED_PACKETS,
            "predecessor_registered_packets": PREDECESSOR_REGISTERED,
            "candidate_normalization_packets": NEW_PACKETS,
            "expected_registered_packets": REGISTERED_PACKETS,
            "expected_missing_packets": MISSING_PACKETS,
            "required_output_rows": REQUIRED_ROWS,
        }
        or config.get("normalization_identity") != "eta=-(34816/15)*a10^5"
        or caps
        != {
            "maximum_family_audit_records": 6,
            "maximum_candidate_packets": 12,
            "maximum_output_rows_emitted": 0,
        }
        or set(config.get("upstreams", {})) != set(EXPECTED_UPSTREAMS)
    ):
        raise CandidateNormalizationRegistrationError("invalid normalization config")
    for name, expected in EXPECTED_UPSTREAMS.items():
        binding = config["upstreams"][name]
        if (
            binding.get("file_sha256") != expected["file_sha256"]
            or binding.get("content_sha256") != expected["content_sha256"]
        ):
            raise CandidateNormalizationRegistrationError(f"config upstream seal mismatch: {name}")


def _load_upstream(root: Path, name: str, binding: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_under(root, binding["path"])
    document = _load_json(path)
    expected = EXPECTED_UPSTREAMS[name]
    if (
        _file_sha256(path) != expected["file_sha256"]
        or not _hash_matches(document)
        or document.get("content_sha256") != expected["content_sha256"]
        or document.get("status") != expected["status"]
        or document.get("errors") != []
    ):
        raise CandidateNormalizationRegistrationError(f"upstream mismatch: {name}")
    return document


def _candidate_packets(obstruction: dict[str, Any], curl: dict[str, Any]) -> list[dict[str, Any]]:
    obstruction_rows = obstruction.get("exact_symbolic_certificate", {}).get(
        "candidate_classification", []
    )
    curl_binding = curl.get("exact_admission", {}).get("reference_D4_binding", {})
    curl_rows = curl_binding.get("candidate_specializations", [])
    if (
        curl_binding.get("unique_tuning") != "eta=-(34816/15)*alpha^5"
        or len(obstruction_rows) != NEW_PACKETS
        or len(curl_rows) != NEW_PACKETS
    ):
        raise CandidateNormalizationRegistrationError("candidate source table mismatch")
    obstruction_map = {row.get("candidate_id"): row for row in obstruction_rows}
    curl_map = {row.get("candidate_id"): row for row in curl_rows}
    if len(obstruction_map) != NEW_PACKETS or set(obstruction_map) != set(curl_map):
        raise CandidateNormalizationRegistrationError("candidate identity set mismatch")
    factor = Fraction(34_816, 15)
    packets: list[dict[str, Any]] = []
    for candidate_id in sorted(obstruction_map):
        obstruction_row = obstruction_map[candidate_id]
        curl_row = curl_map[candidate_id]
        a10 = Fraction(obstruction_row["a10"])
        c20 = Fraction(obstruction_row["c20"])
        eta = Fraction(curl_row["eta_unique_tuning"])
        residual = eta + factor * a10**5
        if (
            Fraction(curl_row["a10"]) != a10
            or residual != 0
            or curl_row.get("reference_direction_D4_Sylvester_solvable") is not True
            or obstruction_row.get("compatible") is not False
        ):
            raise CandidateNormalizationRegistrationError(
                f"candidate normalization residual mismatch: {candidate_id}"
            )
        body = {
            "schema_version": "sigma-D4-common-shape-candidate-normalization-1.0",
            "candidate_id": candidate_id,
            "a10": _fraction_text(a10),
            "c20": _fraction_text(c20),
            "eta": _fraction_text(eta),
            "common_shape_formula": "eta=-(34816/15)*a10^5",
            "common_shape_factorization_residual": _fraction_text(residual),
            "common_shape_factorization_residual_zero": True,
            "obstruction_compression_sha256": obstruction_row["compression_sha256"],
            "reference_direction_deltaK_sha256": curl_row["reference_direction_deltaK_sha256"],
        }
        packets.append({**body, "content_sha256": _content_hash(body)})
    return packets


def _family_evidence_audit() -> list[dict[str, Any]]:
    return [
        {
            "input_id": "polarized_P55_Taylor_packets",
            "required_packets": 75,
            "exact_complete_packets_found": 0,
            "blocker": "five state-Taylor orders at 15 polarization evaluations are not serialized as exact 55x55 matrices",
        },
        {
            "input_id": "polarized_K55_Taylor_packets",
            "required_packets": 75,
            "exact_complete_packets_found": 0,
            "blocker": "available K55 reference matrices and derivative envelopes do not serialize the 75 exact 55x55 Taylor matrices",
        },
        {
            "input_id": "polarized_TC2_Taylor_packets",
            "required_packets": 75,
            "exact_complete_packets_found": 0,
            "blocker": "directional recurrence evaluations do not expose the 75 coordinate-free TC2 Taylor matrices",
        },
        {
            "input_id": "lower_Sylvester_correction_recurrence",
            "required_packets": 60,
            "exact_complete_packets_found": 0,
            "blocker": "pointwise lower-order corrections are not a 15-polarization by four-order coordinate-free matrix packet family",
        },
        {
            "input_id": "candidate_normalization_table",
            "required_packets": 12,
            "exact_complete_packets_found": 12,
            "decision": "REGISTER_LARGEST_COMPLETE_DERIVABLE_FAMILY",
        },
        {
            "input_id": "sphere_mode_normal_form_reducer",
            "required_packets": 1,
            "exact_complete_packets_found": 0,
            "blocker": "degree-19 odd-mode ordering and replay certificate remain unregistered",
        },
    ]


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    upstreams = {
        name: _load_upstream(root, name, config["upstreams"][name]) for name in EXPECTED_UPSTREAMS
    }
    predecessor = upstreams["P55_recurrence_registration"]
    if predecessor.get("counts", {}).get("registered_symbolic_input_packets") != 6:
        raise CandidateNormalizationRegistrationError("predecessor registered count mismatch")
    manifest = json.loads(json.dumps(predecessor["required_symbolic_input_manifest"]))
    records = {record.get("input_id"): record for record in manifest}
    normalization = records.get("candidate_normalization_table")
    if (
        len(records) != 8
        or normalization is None
        or normalization.get("required_packets") != NEW_PACKETS
        or normalization.get("registered_packets") != 0
    ):
        raise CandidateNormalizationRegistrationError("predecessor manifest boundary mismatch")
    packets = _candidate_packets(
        upstreams["canonical_D4_obstruction"], upstreams["curl_constraint_admission"]
    )
    normalization.update(
        {
            "registered_packets": NEW_PACKETS,
            "status": "registered_exact_common_shape_candidate_normalizations",
            "packet_content_sha256": [packet["content_sha256"] for packet in packets],
            "normalization_identity": "eta=-(34816/15)*a10^5",
            "all_common_shape_factorization_residuals_zero": True,
        }
    )
    if (
        sum(record["required_packets"] for record in manifest) != REQUIRED_PACKETS
        or sum(record["registered_packets"] for record in manifest) != REGISTERED_PACKETS
    ):
        raise CandidateNormalizationRegistrationError("updated manifest total mismatch")
    missing_records = [
        {
            "input_id": record["input_id"],
            "required_packets": record["required_packets"],
            "registered_packets": record["registered_packets"],
            "missing_packets": record["required_packets"] - record["registered_packets"],
        }
        for record in manifest
        if record["registered_packets"] < record["required_packets"]
    ]
    if sum(record["missing_packets"] for record in missing_records) != MISSING_PACKETS:
        raise CandidateNormalizationRegistrationError("updated missing total mismatch")
    audit = _family_evidence_audit()
    claims = {claim: False for claim in sorted(FALSE_CLAIMS)}
    claims.update(
        {
            "largest_complete_serialized_missing_family_selected": True,
            "all_12_candidate_normalization_packets_registered": True,
            "all_12_common_shape_factorization_residuals_zero": True,
            "manifest_recomputed_from_exact_packets": True,
        }
    )
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "errors": [],
        "config_sha256": config["content_sha256"],
        "upstream_bindings": {
            name: {
                "path": config["upstreams"][name]["path"],
                "file_sha256": EXPECTED_UPSTREAMS[name]["file_sha256"],
                "content_sha256": EXPECTED_UPSTREAMS[name]["content_sha256"],
                "verified": True,
            }
            for name in EXPECTED_UPSTREAMS
        },
        "missing_family_evidence_audit": audit,
        "registered_candidate_normalization_packets": packets,
        "required_symbolic_input_manifest": manifest,
        "remaining_missing_inputs": missing_records,
        "bounded_emitter_checkpoint": {
            "complete": False,
            "first_missing_input": "polarized_P55_Taylor_packets",
            "required_output_rows": REQUIRED_ROWS,
            "emitted_output_rows": 0,
            "emitted_rhs_rows": 0,
            "emitted_sparse_entries": 0,
        },
        "phase_two": {
            "decision": "BLOCK",
            "admitted": False,
            "attempted": False,
            "blocker": "286 required symbolic input packets remain unregistered",
        },
        "counts": {
            "missing_families_audited": 6,
            "complete_constructive_families_found": 1,
            "upstream_seals_verified": 3,
            "required_symbolic_input_packets": REQUIRED_PACKETS,
            "predecessor_registered_symbolic_input_packets": PREDECESSOR_REGISTERED,
            "new_candidate_normalization_packets_registered": NEW_PACKETS,
            "registered_symbolic_input_packets": REGISTERED_PACKETS,
            "missing_symbolic_input_packets": MISSING_PACKETS,
            "common_shape_factorization_residuals_checked": NEW_PACKETS,
            "common_shape_factorization_nonzero_residuals": 0,
            "required_output_rows": REQUIRED_ROWS,
            "emitted_output_rows": 0,
            "emitted_rhs_rows": 0,
            "phase_two_solve_attempts": 0,
        },
        "claims": claims,
        "negative_controls": {
            "infer_Taylor_matrices_from_directional_evaluation_counts": {"rejected": True},
            "substitute_operator_norm_envelopes_for_exact_matrix_packets": {"rejected": True},
            "register_candidate_without_matching_a10_across_upstreams": {"rejected": True},
            "register_nonzero_common_shape_residual": {"rejected": True},
            "emit_rows_with_286_missing_packets": {"rejected": True},
            "promote_normalization_table_to_full_D4_or_H7": {"rejected": True},
        },
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {
                "path": SOURCE_PATH,
                "file_sha256": _file_sha256(root / SOURCE_PATH),
            },
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
        },
        "scope": (
            "Registers the 12 exact common-shape candidate normalization packets already "
            "determined by sealed D4 obstruction and curl-admission artifacts. The 225 "
            "Taylor packets, 60 lower-recurrence packets, and one sphere reducer remain "
            "missing; no coefficient row, D4 theorem, H7 closure, PDE theorem, lifespan, "
            "or candidate rejection follows."
        ),
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], project_root: Path) -> None:
    expected = build_campaign(project_root.resolve(), project_root.resolve() / CONFIG_PATH)
    if document != expected or not _hash_matches(document):
        raise CandidateNormalizationRegistrationError("campaign replay mismatch")


def write_campaign(document: dict[str, Any], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    document = build_campaign(args.project_root.resolve(), args.config.resolve())
    print(write_campaign(document, args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
