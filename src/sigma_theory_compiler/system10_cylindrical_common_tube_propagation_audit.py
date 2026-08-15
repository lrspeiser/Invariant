from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .system10_cylindrical_common_tube_full_rhs import (
    _validate_config as _validate_full_rhs_config,
)
from .system10_cylindrical_common_tube_full_rhs import (
    _verify_candidate_packet,
)
from .system10_cylindrical_r_positive_gravity_scalar_aw_materializer import (
    _canonical_lf_sha,
    _canonical_sha,
    _load_json,
    _resolve,
)


class System10CommonTubePropagationAuditError(RuntimeError):
    """Raised when a bound propagation input or blocker witness changes."""


SCHEMA = "invariant-system10-cylindrical-common-tube-propagation-audit-1.0"
DECISION = "BLOCK_COMMON_TUBE_RHS_HAS_NO_RADIAL_JET_FOR_CONSTRAINT_PROPAGATION"


def _sealed(document: dict[str, Any]) -> bool:
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    return document.get("content_sha256") == _canonical_sha(body)


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
        raise System10CommonTubePropagationAuditError(f"bound file hash mismatch: {path}")
    document = _load_json(path)
    if document.get("content_sha256") != binding.get("content_sha256") or not _sealed(document):
        raise System10CommonTubePropagationAuditError(f"bound content mismatch: {path}")
    return path, document


def _authority_sha(config: dict[str, Any]) -> str:
    return _canonical_sha({key: value for key, value in config.items() if key != "source_evidence"})


def _validate_config(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != f"{SCHEMA}-config":
        raise System10CommonTubePropagationAuditError("unsupported config schema")
    if config.get("caps") != {
        "candidate_count": 12,
        "state_dimension": 85,
        "physical_gravity_rows": 96,
        "divQ_rows": 4,
        "full_rhs_rows_per_candidate": 85,
        "r": "1",
        "real_v_10_interval": ["-1/4", "1/4"],
        "maximum_output_bytes": 262144,
    }:
        raise System10CommonTubePropagationAuditError("caps changed")

    _, domain = _load_binding(root, config["bindings"]["physical_constraint_rows"])
    if (
        domain.get("decision")
        != "BOUNDED_PASS_FIXED_CYLINDRICAL_R_POSITIVE_1010_JETS_AND_96_ROWS_NO_PROPAGATION_CLAIM"
        or domain.get("counts", {}).get("physical_gravity_rows_closed") != 96
        or len(domain.get("materialization", {}).get("sourced_rational_rows", [])) != 48
        or len(domain.get("materialization", {}).get("shared_symbolic_gauge_rows", [])) != 4
    ):
        raise System10CommonTubePropagationAuditError("physical constraint authority changed")
    _, divq = _load_binding(root, config["bindings"]["divQ_rows"])
    if (
        divq.get("decision") != "BOUNDED_PASS_FOUR_R_POSITIVE_DIVQ_ROWS_BLOCK_FULL_EVOLUTION_RHS"
        or divq.get("counts", {}).get("divq_rows_registered") != 4
        or len(divq.get("materialization", {}).get("rows", [])) != 4
    ):
        raise System10CommonTubePropagationAuditError("divQ authority changed")
    _, identity = _load_binding(root, config["bindings"]["off_shell_identity"])
    formula = identity.get("materialization", {}).get("common_formula", {})
    if not identity.get("claims", {}).get(
        "common_off_shell_covariant_sourced_identity_closed"
    ) or "2*nabla^mu Q_mu_nu" not in formula.get("maximal_common_off_shell_identity", ""):
        raise System10CommonTubePropagationAuditError("off-shell identity changed")

    full = config["bindings"]["full_rhs"]
    full_config_path = _resolve(root, full["config_path"])
    if (
        _canonical_lf_sha(full_config_path) != full["config_canonical_lf_sha256"]
        or _canonical_sha(_load_json(full_config_path)) != full["config_content_sha256"]
    ):
        raise System10CommonTubePropagationAuditError("full RHS config changed")
    full_config, _ = _validate_full_rhs_config(full_config_path, root)
    _, full_receipt = _load_binding(root, full["receipt"])
    if (
        full_receipt.get("decision")
        != "BOUNDED_PASS_12_CANDIDATES_EXACT_85_OF_85_LINKED_RHS_ON_COMMON_TUBE"
        or full_receipt.get("counts", {}).get("full_rhs_candidates_closed_on_common_tube") != 12
        or full_receipt.get("claims", {}).get("fixed_r_positive_domain_full_rhs_closed")
    ):
        raise System10CommonTubePropagationAuditError("full RHS authority changed")
    packet_dir = _resolve(root, full["packet_dir"])
    seals = []
    for index in range(12):
        packet = _load_json(packet_dir / f"candidate-{index:02d}.json")
        _verify_candidate_packet(packet, index, full_config)
        if (
            packet["content_sha256"]
            != full_receipt["candidate_results"][index]["packet_content_sha256"]
        ):
            raise System10CommonTubePropagationAuditError("full RHS packet receipt mismatch")
        seals.append(packet["content_sha256"])
    if (
        hashlib.sha256("".join(seals).encode("ascii")).hexdigest()
        != full_receipt["source_bindings"]["ordered_candidate_packet_set_sha256"]
    ):
        raise System10CommonTubePropagationAuditError("full RHS packet set changed")

    sources = {}
    for name, binding in config.get("source_evidence", {}).items():
        path = _resolve(root, binding["path"])
        if _canonical_lf_sha(path) != binding.get("canonical_lf_sha256"):
            raise System10CommonTubePropagationAuditError(f"source evidence changed: {name}")
        sources[name] = path
    if (
        set(sources) != {"source", "test"}
        or sources["source"] != Path(__file__).resolve()
        or sources["test"]
        != root / "tests/test_system10_cylindrical_common_tube_propagation_audit.py"
    ):
        raise System10CommonTubePropagationAuditError("source evidence paths changed")
    return config, {"domain": domain, "divq": divq, "full_receipt": full_receipt}


def _find_witness_term(domain: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = domain["materialization"]["sourced_rational_rows"]
    candidate_id = domain["materialization"]["candidate_results"][0]["candidate_id"]
    matches = []
    for row in rows:
        if row["candidate_id"] != candidate_id or row["row"] != "momentum_E_n1":
            continue
        for term in row["terms"]:
            if term == {
                "coefficient": "1/(8*r**2)",
                "denominator_r_power": 2,
                "factors": [
                    {"atom": "v_10", "power": 2},
                    {"atom": "partial_1_v_7", "power": 1},
                ],
            }:
                matches.append((row, term))
    if len(matches) != 1:
        raise System10CommonTubePropagationAuditError("radial differentiation witness changed")
    return matches[0]


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config, bound = _validate_config(config_path.resolve(), repository)
    row, term = _find_witness_term(bound["domain"])
    full = config["bindings"]["full_rhs"]
    packet = _load_json(_resolve(repository, full["packet_dir"]) / "candidate-00.json")
    dynamic_row = packet["dynamic_11_rows"][7]
    if (
        dynamic_row["row_id"] != "evolution_v[7]"
        or dynamic_row["lhs_state_index"] != 24
        or packet["common_tube"]["r"] != "1"
        or packet["common_tube"]["real_v_10_interval"] != ["-1/4", "1/4"]
        or "v_7" not in packet["common_tube"]["zeroed_A_symbols"]
    ):
        raise System10CommonTubePropagationAuditError("tube witness row changed")

    witness = {
        "candidate_index": 0,
        "candidate_id": packet["candidate_id"],
        "constraint_row": row["row"],
        "constraint_row_sha256": row["rational_row_sha256"],
        "sensitive_term": term,
        "sensitive_term_sha256": _canonical_sha(term),
        "dynamic_row": dynamic_row["row_id"],
        "dynamic_row_sha256": dynamic_row["row_sha256"],
        "evaluation": {"r": "1", "v_10": "1/4"},
        "extension_0": "F_7^(0)(r)=F_7^tube",
        "extension_1": "F_7^(1)(r)=F_7^tube+(r-1)",
        "same_registered_tube_value": True,
        "radial_derivative_delta_at_r_1": "1",
        "constraint_time_derivative_rule": (
            "partial_0(partial_1 v_7)=partial_1 F_7; the displayed constraint term "
            "therefore contributes (v_10**2/8)*partial_1 F_7 at r=1"
        ),
        "exact_constraint_time_derivative_delta": "1/128",
        "nonzero": True,
    }
    witness["witness_sha256"] = _canonical_sha(witness)
    first_missing = {
        "primitive": "candidate_bound_radial_first_jet_of_all_11_solved_dynamic_rhs_rows",
        "required_domain": "an open radial neighborhood of r=1 with the same state tube",
        "required_outputs": [f"partial_1 F_{index}" for index in range(11)],
        "reason": (
            "a point-slice RHS determines F_A at r=1 but cannot determine partial_1 F_A; "
            "constraint chain-rule differentiation contains partial_1 F_A atoms"
        ),
        "status": "BLOCK_RADIAL_RHS_JET_UNREGISTERED",
        "witness_sha256": witness["witness_sha256"],
    }
    first_missing["primitive_sha256"] = _canonical_sha(first_missing)
    body = {
        "schema_version": SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": DECISION,
        "scope": (
            "Exact fail-closed propagation audit combining the four r>0 divQ rows, the 96 "
            "physical gravity constraint rows, the off-shell identity, and all twelve 85/85 "
            "RHS packets. The RHS is complete only on the r=1 tube. Constraint time "
            "differentiation requires a radial RHS jet not determined by point data, so no "
            "subsidiary system or propagation claim is made."
        ),
        "source_bindings": {"audit_authority_sha256": _authority_sha(config)},
        "counts": {
            "candidates": 12,
            "physical_gravity_rows_bound": 96,
            "divQ_rows_bound": 4,
            "full_rhs_rows_bound_per_candidate": 85,
            "full_rhs_candidate_packets_bound": 12,
            "exact_nonidentifiability_witnesses": 1,
            "candidate_subsidiary_systems_closed": 0,
            "constraint_propagation_proofs": 0,
            "subsidiary_energy_estimates": 0,
        },
        "materialization": {
            "closed_inputs": {
                "physical_constraint_row_count": 96,
                "divQ_row_count": 4,
                "full_rhs_row_instances": 1020,
                "full_rhs_domain": "r=1, real |v_10|<=1/4",
                "off_shell_identity_closed": True,
            },
            "radial_jet_nonidentifiability_witness": witness,
            "first_missing_primitive": first_missing,
            "negative_controls": {
                "differentiate_point_value_as_constant": {
                    "mutation": "set partial_1 F_7=0 because only F_7(r=1) is registered",
                    "exact_missed_delta": "1/128",
                    "rejected": True,
                },
                "claim_propagation_from_85_rows": {
                    "registered_rhs_domain": "r=1 point slice",
                    "required_operation": "radial differentiation",
                    "rejected": True,
                },
            },
        },
        "claims": {
            "all_inputs_hash_bound": True,
            "full_85_state_rhs_closed_on_common_tube": True,
            "radial_rhs_first_jet_identifiable": False,
            "candidate_bound_subsidiary_system_closed": False,
            "constraint_propagation_closed_on_common_tube": False,
            "fixed_r_positive_constraint_propagation_closed": False,
            "subsidiary_energy_estimate_closed": False,
            "hyperbolicity_closed": False,
            "global_theorem_established": False,
            "promotion_authorized": False,
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > config["caps"]["maximum_output_bytes"]:
        raise System10CommonTubePropagationAuditError("receipt cap exceeded")
    return receipt


def write_receipt(config_path: Path, output_path: Path, *, root: Path | None = None) -> Path:
    receipt = build_receipt(config_path, root=root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit common-tube constraint propagation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_receipt(args.config, args.output, root=args.config.resolve().parents[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
