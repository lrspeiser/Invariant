"""Build sealed Markdown/notebook twins for bounded Goal 25/26 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "invariant-goal25-26-progress-evidence-report-1.0"
MARKDOWN_PATH = "docs/notebooks/generated/goal25-26-progress-evidence.md"
IPYNB_PATH = "docs/notebooks/generated/goal25-26-progress-evidence.ipynb"

CLAIMS = {
    "native_receipt_validation_closed": True,
    "bounded_failure_construction_success_story_documented": True,
    "directional_k0_formula_closed": True,
    "k0_polynomial_packet_closed": True,
    "k55_order_one_packets_closed": True,
    "tc2_order_one_packets_closed": True,
    "flat_reference_bounded_B_symmetrizer_closed": True,
    "flat_reference_gauge_scalar_rows_closed": True,
    "full_d4_closed": False,
    "global_h7_closed": False,
    "nonlinear_global_closure_established": False,
    "constraint_propagation_closed": False,
    "universal_all_matter_closure_established": False,
    "promotion_authorized": False,
}

_SPECS = {
    "directional_k0": {
        "path": (
            "runs/physics-language/quartic-tc2-d4-coordinate-free-k0-directional-lift/campaign.json"
        ),
        "terminal_key": "status",
        "terminal_value": "pass_exact_coordinate_free_K0_directional_lift_formula",
        "counts": {
            "exact_direction_controls": 6,
            "exact_direction_controls_passed": 6,
            "e1_reference_matrix_mismatches": 0,
            "full_symbol_build_calls": 0,
        },
        "true_claims": ["coordinate_free_K0_directional_lift_formula_constructed"],
        "false_claims": ["full_direction_sphere_D4_compatibility_proved", "global_H7_closed"],
    },
    "k0_polynomial": {
        "path": (
            "runs/physics-language/quartic-tc2-d4-coordinate-free-"
            "k0-polynomial-packet/campaign.json"
        ),
        "terminal_key": "status",
        "terminal_value": "pass_exact_coordinate_free_K0_polynomial_packet",
        "counts": {
            "K0_polynomial_nonzero_entries": 847,
            "K0_polynomial_normal_form_terms": 2732,
            "sphere_identity_nonzero_remainders": 0,
            "K55_order_one_packets_registered": 0,
        },
        "true_claims": ["expanded_55x55_polynomial_K0_packet_emitted"],
        "false_claims": ["K55_Taylor_order_one_registered", "global_H7_closed"],
    },
    "k55_order_one": {
        "path": (
            "runs/physics-language/quartic-tc2-d4-coordinate-free-"
            "k55-order-one-registration/campaign.json"
        ),
        "terminal_key": "status",
        "terminal_value": ("pass_exact_15_coordinate_free_K55_Taylor_order_one_packets_registered"),
        "counts": {
            "K55_order_one_packets_registered": 15,
            "K55_order_one_packets_required": 15,
            "K55_order_one_normal_form_terms_total": 17704,
            "differentiated_identity_matrix_entries_reduced": 45375,
            "differentiated_identity_nonzero_remainders": 0,
            "manifest_missing_after": 210,
        },
        "true_claims": ["all_15_coordinate_free_K55_Taylor_order_one_packets_registered"],
        "false_claims": ["full_direction_sphere_D4_compatibility_proved", "global_H7_closed"],
    },
    "tc2_order_one": {
        "path": (
            "runs/physics-language/quartic-tc2-d4-coordinate-free-"
            "tc2-order-one-registration/campaign.json"
        ),
        "terminal_key": "status",
        "terminal_value": ("pass_exact_15_coordinate_free_TC2_Taylor_order_one_packets_registered"),
        "counts": {
            "TC2_order_one_packets_registered": 15,
            "TC2_order_one_packets_required": 15,
            "TC2_order_one_zero_packets": 15,
            "product_rule_nonzero_remainders": 0,
            "manifest_missing_after": 195,
        },
        "true_claims": ["all_15_coordinate_free_TC2_Taylor_order_one_packets_registered"],
        "false_claims": ["full_direction_sphere_D4_compatibility_proved", "global_H7_closed"],
    },
    "bounded_b_symmetrizer": {
        "path": "runs/math/quartic-85-state-bounded-B-schur-symmetrizer-gate/receipt.json",
        "terminal_key": "decision",
        "terminal_value": "PASS_EXACT_FLAT_SPHERE_FULL_SYMMETRIZER_BOUNDED_B",
        "counts": {
            "flat_reference_full_symmetrizers": 1,
            "bounded_nonzero_potential_domains": 1,
            "full_85_state_symmetry_residual_nonzero_entries": 0,
            "sourced_constraint_propagation_claims": 0,
        },
        "true_claims": ["exact_flat_sphere_full_85_state_symmetrizer_closed"],
        "false_claims": [
            "gravity_h7_theorem_established",
            "universal_all_matter_closure_established",
        ],
    },
    "gauge_readiness": {
        "path": "runs/math/quartic-85-state-differentiated-gauge-map-readiness/receipt.json",
        "terminal_key": "decision",
        "terminal_value": "PASS_RESUMABLE_READINESS_CONTRACT_FIVE_PRIMITIVE_JET_BLOCKERS",
        "counts": {
            "missing_primitive_jet_families": 5,
            "missing_primitive_slots": 780,
            "primitive_resume_chunks": 48,
            "differentiated_gauge_maps_constructed": 0,
        },
        "true_claims": ["differentiated_gauge_map_readiness_contract_closed"],
        "false_claims": [
            "differentiated_gauge_map_in_85_state_coordinates_closed",
            "constraint_propagation_closed",
        ],
    },
    "gauge_materializer": {
        "path": "runs/math/quartic-85-state-differentiated-gauge-map-materializer/receipt.json",
        "terminal_key": "decision",
        "terminal_value": "PASS_EXACT_INDEXED_GAUGE_MAP_WITH_FORMAL_EXTERNAL_JET_PACKETS",
        "counts": {
            "total_primitive_slots": 780,
            "checkpoint_packets": 48,
            "formal_external_jet_atoms": 580,
            "physical_metric_third_operator_slots": 200,
            "indexed_formula_templates": 17,
            "output_divergence_components": 4,
        },
        "true_claims": ["exact_indexed_differentiated_gauge_map_closed"],
        "false_claims": [
            "external_formulation_jet_values_certified",
            "constraint_propagation_closed",
        ],
    },
    "flat_scalar_expansion": {
        "path": (
            "runs/math/quartic-85-state-gauge-map-scalar-coefficient-expansion-gate/receipt.json"
        ),
        "terminal_key": "decision",
        "terminal_value": "BOUNDED_PASS_FLAT_SCALAR_ROWS_TYPED_BLOCK_GENERAL_EXTERNAL_JETS",
        "counts": {
            "flat_gravity_constraint_rows_expanded": 4,
            "nonzero_scalar_coefficients_total": 112,
            "candidate_flat_row_manifests": 12,
            "required_general_scalar_values_before_domain": 1010,
            "general_external_jet_row_expansions": 0,
        },
        "true_claims": ["exact_flat_reference_scalar_coefficient_rows_closed"],
        "false_claims": [
            "general_external_jet_scalar_expansion_closed",
            "constraint_propagation_closed",
        ],
    },
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _content_sha(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return hashlib.sha256(_canonical(body)).hexdigest()


def _resolve(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative:
        raise ValueError("report path is not portable")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("report path escapes project root") from error
    return path


def _load_native(
    root: Path, name: str, spec: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _resolve(root, str(spec["path"]))
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} receipt is unreadable") from error
    if not isinstance(value, dict) or value.get("content_sha256") != _content_sha(value):
        raise ValueError(f"{name} receipt content seal changed")
    if value.get(spec["terminal_key"]) != spec["terminal_value"]:
        raise ValueError(f"{name} terminal result changed")
    counts = value.get("counts", {})
    if any(counts.get(key) != expected for key, expected in spec["counts"].items()):
        raise ValueError(f"{name} measured counts changed")
    claims = value.get("claims", {})
    if any(claims.get(key) is not True for key in spec["true_claims"]):
        raise ValueError(f"{name} success claim changed")
    if any(claims.get(key) is not False for key in spec["false_claims"]):
        raise ValueError(f"{name} bounded claim changed")
    binding = {
        "path": spec["path"],
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": value["content_sha256"],
        "terminal_result": value[spec["terminal_key"]],
    }
    return value, binding


def build_report(root: Path) -> dict[str, Any]:
    """Native-validate eight receipts and produce one sealed explanatory object."""

    root = root.resolve()
    bindings: dict[str, dict[str, Any]] = {}
    for name, spec in _SPECS.items():
        _, bindings[name] = _load_native(root, name, spec)

    body = {
        "schema_version": REPORT_SCHEMA,
        "title": "From a typed failure to two bounded constructions",
        "decision": "PASS_SEALED_BOUNDED_GOAL25_26_EVIDENCE_REPORT",
        "bindings": bindings,
        "summary_counts": {
            "native_receipts_validated": 8,
            "h7_lane_receipts": 4,
            "matter_lane_receipts": 4,
            "coordinate_free_order_one_packets_registered": 30,
            "flat_constraint_rows_expanded": 4,
            "flat_scalar_coefficients": 112,
            "candidate_manifests_bound": 12,
            "remaining_general_scalar_values_before_domain": 1010,
        },
        "h7_lane": {
            "failure_boundary": {
                "finding": (
                    "The directional K0 formula passed six exact controls, but its own "
                    "receipt left the expanded polynomial packet, D4, and H7 open."
                ),
                "exact_direction_controls_passed": 6,
                "e1_matrix_mismatches": 0,
            },
            "construction": {
                "k0_nonzero_entries": 847,
                "k0_normal_form_terms": 2732,
                "sphere_identity_nonzero_remainders": 0,
                "authorized_but_initially_unregistered_k55_packets": 15,
            },
            "bounded_success": {
                "k55_order_one_packets_registered": 15,
                "k55_normal_form_terms": 17704,
                "differentiated_identity_entries_reduced": 45375,
                "differentiated_identity_nonzero_remainders": 0,
                "tc2_order_one_packets_registered": 15,
                "tc2_zero_packets": 15,
                "tc2_product_rule_nonzero_remainders": 0,
            },
        },
        "matter_lane": {
            "first_success": {
                "flat_reference_full_symmetrizers": 1,
                "bounded_nonzero_potential_domains": 1,
                "symmetry_residual_nonzero_entries": 0,
                "scope": "flat reference over the registered direction sphere",
            },
            "typed_failure": {
                "missing_primitive_jet_families": 5,
                "missing_primitive_slots": 780,
                "resume_chunks": 48,
                "differentiated_maps_at_readiness_stage": 0,
            },
            "construction": {
                "primitive_slots_registered": 780,
                "formal_external_jet_atoms": 580,
                "physical_metric_operator_slots": 200,
                "indexed_formula_templates": 17,
                "divergence_components": 4,
            },
            "bounded_success": {
                "flat_constraint_rows_expanded": 4,
                "nonzero_exact_q_sqrt2_coefficients": 112,
                "candidate_flat_row_manifests": 12,
            },
        },
        "typed_blockers": [
            {
                "lane": "coordinate-free H7",
                "remaining": (
                    "K55 and TC2 orders two through four, recurrence rows, full D4, "
                    "global H7, PDE closure, and lifespan"
                ),
                "manifest_missing_after_order_one": 195,
            },
            {
                "lane": "coupled matter constraints",
                "remaining": (
                    "580 certified external-jet values, 280 lower formulation-jet "
                    "values, 150 physical metric jet values, sourced accelerations, "
                    "and a common domain"
                ),
                "exact_scalar_values_before_domain": 1010,
            },
        ],
        "claims": dict(CLAIMS),
        "scope": (
            "A human-readable projection of eight natively validated, content-addressed "
            "receipts. It documents bounded exact controls and typed blockers; it does "
            "not establish full D4, global H7, nonlinear/global closure, sourced "
            "constraint propagation, universal matter closure, or promotion."
        ),
    }
    return {**body, "content_sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def validate_report(report: Mapping[str, Any]) -> None:
    expected_keys = {
        "bindings",
        "claims",
        "content_sha256",
        "decision",
        "h7_lane",
        "matter_lane",
        "schema_version",
        "scope",
        "summary_counts",
        "title",
        "typed_blockers",
    }
    if (
        set(report) != expected_keys
        or report.get("schema_version") != REPORT_SCHEMA
        or report.get("content_sha256") != _content_sha(report)
    ):
        raise ValueError("Goal 25/26 report schema or seal changed")
    if report.get("claims") != CLAIMS:
        raise ValueError("Goal 25/26 report claim boundary changed")
    counts = report.get("summary_counts", {})
    h7 = report.get("h7_lane", {})
    matter = report.get("matter_lane", {})
    blockers = report.get("typed_blockers", [])
    if (
        counts.get("native_receipts_validated") != 8
        or counts.get("coordinate_free_order_one_packets_registered") != 30
        or h7.get("bounded_success", {}).get("differentiated_identity_nonzero_remainders") != 0
        or matter.get("construction", {}).get("indexed_formula_templates") != 17
        or matter.get("bounded_success", {}).get("nonzero_exact_q_sqrt2_coefficients") != 112
        or len(blockers) != 2
        or blockers[1].get("exact_scalar_values_before_domain") != 1010
    ):
        raise ValueError("Goal 25/26 report semantic boundary changed")


def render_markdown(report: Mapping[str, Any]) -> str:
    validate_report(report)
    h7 = report["h7_lane"]
    matter = report["matter_lane"]
    bindings = report["bindings"]
    lines = [
        "# Goal 25/26 evidence notebook: failure, construction, bounded success",
        "",
        "> This generated report is intentionally narrower than a theorem announcement.",
        "> It shows what failed, what was built next, what now passes, and what remains open.",
        "",
        "## How to read the evidence",
        "",
        "Eight checked JSON receipts were parsed, content-seal replayed, and checked against exact terminal results, measured counts, and claim boundaries. The same sealed report object generated this Markdown file and the notebook twin.",
        "",
        "## Goal 25 lane: coordinate-free coefficient construction",
        "",
        "### Failure boundary",
        "",
        h7["failure_boundary"]["finding"],
        "",
        "That was a useful failure: it isolated representation, rather than the six tested directions, as the next missing object. There were zero mismatches in the 3,025-entry e1 reference comparison, but no full D4 or H7 conclusion followed.",
        "",
        "### Construction",
        "",
        f"The next receipt serialized K0 as a 55×55 unit-sphere polynomial packet with **{h7['construction']['k0_nonzero_entries']} nonzero entries** and **{h7['construction']['k0_normal_form_terms']} normal-form terms**. All 3,025 sphere-identity entries reduced to zero. This authorized 15 K55 order-one packets while still recording that zero had yet been registered.",
        "",
        "### Bounded success",
        "",
        f"The K55 order-one gate then registered **{h7['bounded_success']['k55_order_one_packets_registered']} of 15 packets**, containing {h7['bounded_success']['k55_normal_form_terms']} normal-form terms. It reduced {h7['bounded_success']['differentiated_identity_entries_reduced']:,} differentiated matrix identities with **zero nonzero remainders**.",
        "",
        f"The TC2 order-one gate independently registered **{h7['bounded_success']['tc2_order_one_packets_registered']} of 15 packets**. All 15 are exact zero packets for the sealed fixed-coefficient jet basis, and the product-rule replay has zero nonzero remainders.",
        "",
        "This is an order-one registration success. Orders two through four, recurrence rows, full D4, global H7, PDE closure, and lifespan remain open.",
        "",
        "## Goal 26 lane: coupled matter and gravity constraints",
        "",
        "### First bounded success",
        "",
        f"At the flat reference, the 85-state Schur construction produced **{matter['first_success']['flat_reference_full_symmetrizers']} full symmetrizer** on **{matter['first_success']['bounded_nonzero_potential_domains']} bounded nonzero Maxwell-potential domain**, with zero symmetry-residual entries. This is not candidate-jet uniformity or a global result.",
        "",
        "### Typed failure",
        "",
        f"Differentiating the modified-harmonic gauge source initially stopped on **{matter['typed_failure']['missing_primitive_jet_families']} missing jet families**, totaling **{matter['typed_failure']['missing_primitive_slots']} primitive slots**. The readiness receipt divided them into {matter['typed_failure']['resume_chunks']} resumable chunks and claimed zero completed differentiated maps.",
        "",
        "### Construction",
        "",
        f"The materializer registered all {matter['construction']['primitive_slots_registered']} slots: {matter['construction']['formal_external_jet_atoms']} formal external-jet atoms and {matter['construction']['physical_metric_operator_slots']} physical-metric third-derivative operator slots. A **17-template** indexed tensor program now constructs all four divergence components. The external atoms are formal inputs, not certified values.",
        "",
        "### Bounded success",
        "",
        f"At the registered flat constant formulation reference, the scalar-expansion gate lowered all **{matter['bounded_success']['flat_constraint_rows_expanded']} gravity-constraint rows** into the 85-state ordering. It found **{matter['bounded_success']['nonzero_exact_q_sqrt2_coefficients']} exact Q(√2) coefficients**—28 per row—and bound the common row packet to **{matter['bounded_success']['candidate_flat_row_manifests']} candidate manifests**.",
        "",
        "The nonlinear/general row expansion remains blocked on 1,010 exact scalar values before a common domain: 580 external jets, 280 lower formulation jets, and 150 physical metric jets, including sourced acceleration data.",
        "",
        "## Receipt ledger",
        "",
        "| Evidence | Terminal result | File SHA-256 |",
        "|---|---|---|",
    ]
    for name, binding in bindings.items():
        lines.append(f"| `{name}` | `{binding['terminal_result']}` | `{binding['file_sha256']}` |")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "The receipts support the exact directional, polynomial, order-one, flat-reference, and indexed constructions described above. They do **not** establish full D4, global H7, nonlinear/global closure, sourced constraint propagation, universal all-matter closure, or promotion.",
            "",
            f"Sealed report content SHA-256: `{report['content_sha256']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_ipynb(report: Mapping[str, Any]) -> dict[str, Any]:
    markdown = render_markdown(report)
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"report_content_sha256": report["content_sha256"]},
                "source": markdown.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "language_info": {"name": "markdown"},
            "report_content_sha256": report["content_sha256"],
            "report_schema": REPORT_SCHEMA,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_outputs(root: Path) -> tuple[str, str]:
    report = build_report(root)
    validate_report(report)
    markdown = render_markdown(report)
    notebook = json.dumps(render_ipynb(report), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return markdown, notebook


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    markdown, notebook = build_outputs(root)
    _write(_resolve(root, MARKDOWN_PATH), markdown)
    _write(_resolve(root, IPYNB_PATH), notebook)
    print(
        json.dumps(
            {
                "markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
                "ipynb_sha256": hashlib.sha256(notebook.encode()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
