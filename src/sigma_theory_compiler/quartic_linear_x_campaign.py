from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

from .horndeski_principal import (
    quartic_horndeski_full_local_principal_control,
    quartic_horndeski_x2_kessence_extension_control,
)

SCHEMA_VERSION = "sigma-quartic-linear-x-symbol-campaign-1.0"
_QUARTIC_PREFIX = "quartic_linear_X_G4_only_"
_REQUIRED_ZERO_COEFFICIENTS = ("c11", "c02", "d01", "a01")


class QuarticLinearXCampaignError(ValueError):
    """Raised when an alleged exact symbol binding is outside the proven action."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _is_zero(value: Any) -> bool:
    return sp.sympify(value) == 0


def _bind_candidate(
    classification: dict[str, Any], fixed_coefficients: dict[str, Any]
) -> dict[str, Any]:
    assignment = classification.get("assignment")
    if not isinstance(assignment, dict):
        raise QuarticLinearXCampaignError("classification has no mutation assignment")
    coefficients = {**fixed_coefficients, **assignment}
    missing = sorted(
        {"m2", "c20", "d10", "a10", "a20", *_REQUIRED_ZERO_COEFFICIENTS}
        - set(coefficients)
    )
    if missing:
        raise QuarticLinearXCampaignError(
            "binding is missing coefficients: " + ", ".join(missing)
        )
    violated_zeroes = [
        name for name in _REQUIRED_ZERO_COEFFICIENTS if not _is_zero(coefficients[name])
    ]
    if violated_zeroes:
        raise QuarticLinearXCampaignError(
            "existing principal symbols require these coefficients fixed to zero: "
            + ", ".join(violated_zeroes)
        )
    if not _is_zero(coefficients["d10"]) or not _is_zero(coefficients["a20"]):
        raise QuarticLinearXCampaignError("binding is not G3-free and linear in X in G4")
    if _is_zero(coefficients["a10"]):
        raise QuarticLinearXCampaignError("binding has no quartic G4_X deformation")
    if sp.sympify(coefficients["m2"]).is_positive is not True:
        raise QuarticLinearXCampaignError("m2 must be provably positive")

    c20 = sp.sympify(coefficients["c20"])
    expected_subclass = (
        "quartic_linear_X_G4_only_G2_linear_X"
        if c20 == 0
        else "quartic_linear_X_G4_only_G2_nonlinear_X"
    )
    if classification.get("proof_subclass") != expected_subclass:
        raise QuarticLinearXCampaignError(
            "IR proof subclass does not match the bound G2_XX coefficient"
        )
    action = {
        "G2": "X" if c20 == 0 else f"X+({c20})*X**2",
        "G3": "0",
        "G4": f"({sp.sympify(coefficients['m2'])})/2+({sp.sympify(coefficients['a10'])})*X",
        "G5": "0",
    }
    identifier_payload = {
        "coefficients": {name: str(value) for name, value in sorted(coefficients.items())},
        "proof_subclass": expected_subclass,
    }
    candidate_id = "quartic-symbol-" + hashlib.sha256(
        _canonical_json(identifier_payload).encode()
    ).hexdigest()[:16]
    return {
        "candidate_id": candidate_id,
        "mutation_assignment": assignment,
        "coefficients": {name: str(value) for name, value in sorted(coefficients.items())},
        "proof_subclass": expected_subclass,
        "specialized_action": action,
        "principal_symbol_binding": (
            "canonical_linear_X_quartic_11x11"
            if c20 == 0
            else "linear_X_quartic_11x11_plus_exact_quadratic_kessence_scalar_block"
        ),
        "status": "pass_exact_11x11_symbol_binding_symmetrizer_unresolved",
        "remaining": [
            "declare a background local-jet domain satisfying the exact time-block condition",
            "construct and uniformly bound a positive direction-dependent symmetrizer",
            "compare the correction norm with the auxiliary-cone separation budget",
        ],
    }


def run_quartic_linear_x_symbol_campaign(
    ir: dict[str, Any], campaign_config: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if campaign_config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticLinearXCampaignError("unsupported campaign schema_version")
        if ir.get("status") not in {"compiled", "compiled_formal_adapters_unresolved"}:
            raise QuarticLinearXCampaignError("source scalar-tensor IR is not compiled")
        fixed_coefficients = campaign_config.get("fixed_coefficients")
        if not isinstance(fixed_coefficients, dict):
            raise QuarticLinearXCampaignError("fixed_coefficients must be an object")
        classifications = ir.get("formulation_classification", {}).get(
            "mutation_axis_partition", {}
        ).get("assignment_classifications")
        if not isinstance(classifications, list):
            raise QuarticLinearXCampaignError(
                "source IR has no exact assignment classifications"
            )
        selected = [
            item
            for item in classifications
            if str(item.get("proof_subclass", "")).startswith(_QUARTIC_PREFIX)
        ]
        expected = int(campaign_config.get("expected_candidate_count", len(selected)))
        if len(selected) != expected:
            raise QuarticLinearXCampaignError(
                f"expected {expected} linear-X quartic candidates, found {len(selected)}"
            )

        base_passed, base_evidence = quartic_horndeski_full_local_principal_control()
        extension_passed, extension_evidence = (
            quartic_horndeski_x2_kessence_extension_control()
        )
        if not base_passed or not extension_passed:
            raise QuarticLinearXCampaignError("a required principal-symbol control failed")
        bindings = [_bind_candidate(item, fixed_coefficients) for item in selected]

        corrupted_fixed = dict(fixed_coefficients)
        corrupted_fixed["a01"] = "1"
        negative_rejected = False
        negative_error = ""
        try:
            _bind_candidate(selected[0], corrupted_fixed)
        except QuarticLinearXCampaignError as error:
            negative_rejected = True
            negative_error = str(error)
        if not negative_rejected:
            raise QuarticLinearXCampaignError(
                "phi-dependent G4 negative control was not rejected"
            )

        linear_count = sum(
            item["proof_subclass"].endswith("G2_linear_X") for item in bindings
        )
        nonlinear_count = len(bindings) - linear_count
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_exact_symbol_binding_uniform_symmetrizer_unresolved",
            "errors": [],
            "source_ir_sha256": ir.get("content_sha256"),
            "config_sha256": hashlib.sha256(
                _canonical_json(campaign_config).encode()
            ).hexdigest(),
            "counts": {
                "selected": len(selected),
                "exactly_bound": len(bindings),
                "canonical_G2": linear_count,
                "quadratic_kessence_G2": nonlinear_count,
                "rejected": 0,
            },
            "proof_controls": {
                "canonical_linear_X_quartic": base_evidence,
                "quadratic_kessence_extension": extension_evidence,
            },
            "negative_controls": {
                "phi_dependent_G4_outside_extracted_symbol": {
                    "mutation": {"a01": "1"},
                    "rejected": negative_rejected,
                    "error": negative_error,
                }
            },
            "candidates": sorted(bindings, key=lambda item: item["candidate_id"]),
            "claim": (
                "All selected G3-free, linear-X G4 mutations under the declared fixed-coefficient "
                "specialization are bound to an exact local-frame 11-by-11 principal matrix."
            ),
            "scope": (
                "This is exact principal-symbol extraction and candidate binding, not a completed "
                "strong-hyperbolicity theorem. Uniform local-jet symmetrizer and correction-norm "
                "bounds remain unresolved and are reported per candidate."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticLinearXCampaignError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "source_ir_sha256": ir.get("content_sha256"),
            "counts": {
                "selected": 0,
                "exactly_bound": 0,
                "canonical_G2": 0,
                "quadratic_kessence_G2": 0,
                "rejected": 0,
            },
            "candidates": [],
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_linear_x_symbol_campaign(result: dict[str, Any], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
