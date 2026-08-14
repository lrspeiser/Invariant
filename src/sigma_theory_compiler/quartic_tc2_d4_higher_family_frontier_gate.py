from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "sigma-quartic-tc2-d4-higher-family-frontier-gate-1.0"
CONFIG_SCHEMA = f"{SCHEMA.removesuffix('-1.0')}-config-1.0"
STATUS = "block_higher_K55_at_first_missing_physical_H_star_order_two_packet"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_higher_family_frontier_gate.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_tc2_d4_higher_family_frontier_gate.py"
TEST_PATH = "tests/test_quartic_tc2_d4_higher_family_frontier_gate.py"
OUTPUT_PATH = "runs/physics-language/quartic-tc2-d4-higher-family-frontier-gate/campaign.json"


class HigherFamilyFrontierError(ValueError):
    """Raised when the higher-family fail-closed boundary changes."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _content_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes({key: item for key, item in value.items() if key != "content_sha256"})).hexdigest()


def _hash_matches(value: dict[str, Any]) -> bool:
    return value.get("content_sha256") == _content_hash(value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HigherFamilyFrontierError(f"expected object: {path}")
    return value


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = (root / binding["path"]).resolve()
    if root != path and root not in path.parents:
        raise HigherFamilyFrontierError("bound path escaped root")
    value = _load_json(path)
    if _file_sha256(path) != binding["file_sha256"] or value.get("content_sha256") != binding["content_sha256"] or not _hash_matches(value):
        raise HigherFamilyFrontierError(f"upstream seal mismatch: {binding['path']}")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy") != "advance_no_downstream_family_without_every_exact_physical_primitive"
        or set(config.get("upstreams", {})) != {"predecessor", "higher_P55", "K55_order_one", "H_star_order_one"}
        or config.get("target") != {"registered_before": 109, "registered_after_P55": 154, "required_total": 304, "required_rows": 117180}
        or not _hash_matches(config)
    ):
        raise HigherFamilyFrontierError("invalid higher-family frontier config")


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    upstreams = {name: _load_bound(root, binding) for name, binding in config["upstreams"].items()}
    predecessor = upstreams["predecessor"]
    higher_p = upstreams["higher_P55"]
    k1 = upstreams["K55_order_one"]
    h1 = upstreams["H_star_order_one"]
    if (
        predecessor.get("counts", {}).get("manifest_registered_after") != 109
        or higher_p.get("counts", {}).get("P55_higher_packets_registered") != 45
        or higher_p.get("counts", {}).get("manifest_registered_after") != 154
        or len(k1.get("registered_coordinate_free_K55_Taylor_order_one_packets", [])) != 15
        or len(h1.get("packets", [])) != 15
    ):
        raise HigherFamilyFrontierError("higher-family predecessor boundary changed")
    manifest = json.loads(json.dumps(higher_p["required_symbolic_input_manifest"]))
    records = {row["input_id"]: row for row in manifest}
    if (
        records["polarized_P55_Taylor_packets"].get("registered_packets") != 75
        or records["polarized_K55_Taylor_packets"].get("registered_packets") != 30
        or records["polarized_TC2_Taylor_packets"].get("registered_packets") != 30
        or records["lower_Sylvester_correction_recurrence"].get("registered_packets") != 0
        or sum(row["registered_packets"] for row in manifest) != 154
    ):
        raise HigherFamilyFrontierError("154-packet manifest boundary changed")
    first_evaluation = h1["packets"][0]
    if first_evaluation.get("evaluation_id") != "subset_0" or first_evaluation.get("H_star_plus_order_one_matrix", {}).get("Taylor_order") != 1:
        raise HigherFamilyFrontierError("H-star evaluation ordering changed")
    claims = {
        "all_45_higher_P55_packets_registered": True,
        "higher_K55_packets_registered": False,
        "higher_TC2_packets_registered": False,
        "lower_Sylvester_packets_registered": False,
        "all_117180_rows_emitted": False,
    }
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "decision": "BLOCK_SERIALIZATION",
        "errors": [],
        "config_sha256": config["content_sha256"],
        "upstream_bindings": {name: {**binding, "verified": True} for name, binding in config["upstreams"].items()},
        "required_symbolic_input_manifest": manifest,
        "first_missing_primitive": {
            "family": "physical_H_star_plus_Taylor_packets",
            "evaluation_id": "subset_0",
            "Taylor_order": 2,
            "shape": [22, 22],
            "factorial_normalization": "1/2!",
            "registered_packets_at_this_order": 0,
            "required_before_K55_order_two": 15,
            "required_orders_before_complete_K55": [2, 3, 4],
            "required_total_packets": 45,
            "reason": "P55 determines the companion and lift derivatives but not the independent physical action inner-product derivative used by K55.",
        },
        "downstream_atomicity": {
            "K55_higher_family_attempted_for_registration": False,
            "K55_higher_packets_registered": 0,
            "TC2_higher_packets_registered": 0,
            "lower_Sylvester_packets_registered": 0,
            "manifest_advanced_beyond_154": False,
            "output_rows_emitted": 0,
        },
        "counts": {
            "required_symbolic_packets": 304,
            "registered_symbolic_packets": 154,
            "missing_symbolic_packets": 150,
            "P55_higher_packets_registered": 45,
            "required_H_star_higher_packets": 45,
            "registered_H_star_higher_packets": 0,
            "required_K55_higher_packets": 45,
            "registered_K55_higher_packets": 0,
            "required_TC2_higher_packets": 45,
            "registered_TC2_higher_packets": 0,
            "required_lower_Sylvester_packets": 60,
            "registered_lower_Sylvester_packets": 0,
            "required_output_rows": 117180,
            "emitted_output_rows": 0,
        },
        "claims": claims,
        "negative_controls": {
            "infer_absent_H_star_packet_as_zero": {"rejected": True},
            "derive_K55_from_symmetrizer_identity_without_physical_metric": {"rejected": True},
            "partially_advance_K55_family": {"rejected": True},
            "emit_TC2_or_lower_Sylvester_before_K55_closes": {"rejected": True},
            "emit_rows_with_150_missing_packets": {"rejected": True},
        },
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha256(root / SOURCE_PATH)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
        },
        "scope": "Closes all 45 higher-P55 packets and seals the exact next physical input boundary. It does not register K55, TC2, lower-Sylvester packets or coefficient rows.",
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], project_root: Path) -> None:
    if not _hash_matches(document) or document != build_campaign(project_root, project_root / CONFIG_PATH):
        raise HigherFamilyFrontierError("frontier campaign replay mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    document = build_campaign(args.project_root.resolve(), args.config.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
