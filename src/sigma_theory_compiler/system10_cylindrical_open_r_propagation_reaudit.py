from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .system10_cylindrical_open_r_twelve_candidate_rhs_jets import (
    _validate_config as _validate_jets_config,
)
from .system10_cylindrical_open_r_twelve_candidate_rhs_jets import (
    _verify_candidate,
)
from .system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    _canonical_lf_sha,
    _canonical_sha,
    _load_json,
    _resolve,
)


class System10OpenRPropagationReauditError(RuntimeError):
    """Raised when an all-candidate propagation re-audit input changes."""


SCHEMA = "invariant-system10-open-r-propagation-reaudit-1.0"
DECISION = "BLOCK_TANGENTIAL_RHS_JETS_UNREGISTERED_AFTER_RADIAL_JET_CLOSURE"


def _sealed(document: dict[str, Any]) -> bool:
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    return document.get("content_sha256") == _canonical_sha(body)


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, binding["path"])
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10OpenRPropagationReauditError(f"bound file mismatch: {path}")
    document = _load_json(path)
    if document.get("content_sha256") != binding.get("content_sha256") or not _sealed(document):
        raise System10OpenRPropagationReauditError(f"bound content mismatch: {path}")
    return path, document


def _authority_sha(config: dict[str, Any]) -> str:
    return _canonical_sha({key: value for key, value in config.items() if key != "source_evidence"})


def _validate_config(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != f"{SCHEMA}-config":
        raise System10OpenRPropagationReauditError("unsupported config schema")
    if config.get("caps") != {
        "candidates": 12,
        "state_dimension": 85,
        "physical_gravity_rows": 96,
        "divQ_rows": 4,
        "radial_rhs_jets": 132,
        "tangential_directions": [2, 3],
        "required_tangential_rhs_jets": 264,
        "maximum_output_bytes": 262144,
    }:
        raise System10OpenRPropagationReauditError("caps changed")

    jets = config["bindings"]["all_twelve_radial_jets"]
    jets_config_path = _resolve(root, jets["config_path"])
    if (
        _canonical_lf_sha(jets_config_path) != jets["config_canonical_lf_sha256"]
        or _canonical_sha(_load_json(jets_config_path)) != jets["config_content_sha256"]
    ):
        raise System10OpenRPropagationReauditError("radial-jet config mismatch")
    jets_config, _ = _validate_jets_config(jets_config_path, root)
    _, jets_receipt = _load_binding(root, jets["receipt"])
    if (
        jets_receipt.get("decision") != "BOUNDED_PASS_ALL_TWELVE_OPEN_R_RHS_ROWS_AND_RADIAL_JETS"
        or jets_receipt.get("counts", {}).get("open_r_radial_rhs_jets") != 132
        or jets_receipt.get("counts", {}).get("candidate_passes") != 12
    ):
        raise System10OpenRPropagationReauditError("radial-jet receipt mismatch")
    packet_dir = _resolve(root, jets["packet_dir"])
    seals = []
    for index in range(12):
        packet = _load_json(packet_dir / f"candidate-{index:02d}.json")
        _verify_candidate(packet, jets_config, index)
        if (
            packet["content_sha256"]
            != jets_receipt["candidate_results"][index]["packet_content_sha256"]
        ):
            raise System10OpenRPropagationReauditError("radial-jet packet receipt mismatch")
        seals.append(packet["content_sha256"])
    if (
        hashlib.sha256("".join(seals).encode("ascii")).hexdigest()
        != jets_receipt["source_bindings"]["ordered_candidate_packet_set_sha256"]
    ):
        raise System10OpenRPropagationReauditError("radial-jet packet set mismatch")

    _, constraints = _load_binding(root, config["bindings"]["physical_constraint_rows"])
    if constraints.get("counts", {}).get("physical_gravity_rows_closed") != 96:
        raise System10OpenRPropagationReauditError("constraint row authority changed")
    _, divq = _load_binding(root, config["bindings"]["divQ_rows"])
    if divq.get("counts", {}).get("divq_rows_registered") != 4:
        raise System10OpenRPropagationReauditError("divQ authority changed")
    _, identity = _load_binding(root, config["bindings"]["off_shell_identity"])
    if not identity.get("claims", {}).get("common_off_shell_covariant_sourced_identity_closed"):
        raise System10OpenRPropagationReauditError("off-shell identity changed")
    _, old_blocker = _load_binding(root, config["bindings"]["radial_jet_blocker"])
    if (
        old_blocker.get("decision")
        != "BLOCK_COMMON_TUBE_RHS_HAS_NO_RADIAL_JET_FOR_CONSTRAINT_PROPAGATION"
    ):
        raise System10OpenRPropagationReauditError("radial blocker changed")

    sources = {}
    for name, binding in config.get("source_evidence", {}).items():
        path = _resolve(root, binding["path"])
        if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
            raise System10OpenRPropagationReauditError(f"source evidence mismatch: {name}")
        sources[name] = path
    if (
        set(sources) != {"source", "test"}
        or sources["source"] != Path(__file__).resolve()
        or sources["test"] != root / "tests/test_system10_cylindrical_open_r_propagation_reaudit.py"
    ):
        raise System10OpenRPropagationReauditError("source evidence paths changed")
    return config, {"jets_receipt": jets_receipt, "constraints": constraints}


def _find_tangential_witness(constraints: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_id = constraints["materialization"]["candidate_results"][0]["candidate_id"]
    expected = {
        "coefficient": "-sqrt(2)/(16*r**2)",
        "denominator_r_power": 2,
        "factors": [
            {"atom": "v_10", "power": 2},
            {"atom": "partial_2_v_5", "power": 1},
        ],
    }
    matches = []
    for row in constraints["materialization"]["sourced_rational_rows"]:
        if row["candidate_id"] == candidate_id and row["row"] == "momentum_E_n1":
            matches.extend((row, term) for term in row["terms"] if term == expected)
    if len(matches) != 1:
        raise System10OpenRPropagationReauditError("tangential chain-rule witness changed")
    return matches[0]


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, bound = _validate_config(config_path.resolve(), repository)
    row, term = _find_tangential_witness(bound["constraints"])
    witness = {
        "candidate_index": 0,
        "candidate_id": row["candidate_id"],
        "constraint_row": row["row"],
        "constraint_row_sha256": row["rational_row_sha256"],
        "sensitive_term": term,
        "sensitive_term_sha256": _canonical_sha(term),
        "chain_rule": "partial_0(partial_2 v_5)=partial_2 F_5",
        "evaluation": {"r": "1", "v_10": "1/4"},
        "exact_unreplayed_partial_2_F_5_coefficient": "-sqrt(2)/256",
        "nonzero": True,
        "registered_partial_1_F_5": True,
        "registered_partial_2_F_5": False,
    }
    witness["witness_sha256"] = _canonical_sha(witness)
    missing = {
        "primitive": "all_twelve_candidate_bound_tangential_total_derivative_rhs_DAGs",
        "required_directions": [2, 3],
        "required_outputs": [
            f"partial_{direction} F_{row}" for direction in (2, 3) for row in range(11)
        ],
        "required_candidate_instances": 264,
        "required_W_derivative_nodes": ["D2_W_0..D2_W_10", "D3_W_0..D3_W_10"],
        "status": "BLOCK_TANGENTIAL_TOTAL_DERIVATIVE_DAGS_UNREGISTERED",
        "witness_sha256": witness["witness_sha256"],
    }
    missing["primitive_sha256"] = _canonical_sha(missing)
    body = {
        "schema_version": SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": DECISION,
        "scope": (
            "Propagation re-audit after exact all-twelve open-r radial RHS-jet closure. The old "
            "partial_1 F blocker is closed, but the 96 physical constraint rows also contain "
            "tangential partial_2 v and partial_3 v atoms. Their time derivatives require "
            "partial_2 F and partial_3 F linked derivative DAGs, which are not registered. "
            "No subsidiary, propagation, energy, or hyperbolicity claim is made."
        ),
        "source_bindings": {"reaudit_authority_sha256": _authority_sha(config)},
        "counts": {
            "candidates": 12,
            "physical_gravity_rows_bound": 96,
            "divQ_rows_bound": 4,
            "open_r_rhs_rows_bound": 132,
            "radial_rhs_jets_bound": 132,
            "tangential_rhs_jets_required": 264,
            "tangential_rhs_jets_bound": 0,
            "candidate_subsidiary_systems_closed": 0,
            "constraint_propagation_proofs": 0,
        },
        "materialization": {
            "closed_predecessor_block": {
                "old_missing_primitive": "candidate_bound_radial_first_jet_of_all_11_solved_dynamic_rhs_rows",
                "closed_candidate_instances": 132,
                "status": "PASS_ALL_TWELVE_RADIAL_RHS_JETS",
                "source_receipt_content_sha256": bound["jets_receipt"]["content_sha256"],
            },
            "tangential_chain_rule_witness": witness,
            "first_missing_primitive": missing,
            "negative_controls": {
                "reuse_partial_1_as_partial_2": {
                    "mutation": "identify partial_2 F_5 with registered partial_1 F_5",
                    "rejected": True,
                    "reason": "coordinate derivative directions are independent",
                },
                "zero_fill_tangential_jets": {
                    "mutation": "set all unregistered partial_2 F and partial_3 F rows to zero",
                    "nonzero_witness_coefficient": "-sqrt(2)/256",
                    "rejected": True,
                },
            },
        },
        "claims": {
            "all_twelve_radial_rhs_jets_closed": True,
            "all_twelve_tangential_rhs_jets_closed": False,
            "candidate_bound_subsidiary_system_closed": False,
            "constraint_propagation_closed": False,
            "subsidiary_energy_estimate_closed": False,
            "hyperbolicity_closed": False,
            "global_theorem_established": False,
            "promotion_authorized": False,
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > config["caps"]["maximum_output_bytes"]:
        raise System10OpenRPropagationReauditError("receipt cap exceeded")
    return receipt


def write_receipt(config_path: Path, output_path: Path, *, root: Path | None = None) -> Path:
    receipt = build_receipt(config_path, root=root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-audit propagation after radial jet closure")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_receipt(args.config, args.output, root=args.config.resolve().parents[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
