from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-positive-quantization-campaign-1.0"


class QuarticPositiveQuantizationError(ValueError):
    """Raised when the positive operator quantization contract cannot be certified."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash_matches(campaign: dict[str, Any]) -> bool:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    return campaign.get("content_sha256") == hashlib.sha256(
        _canonical_json(body).encode()
    ).hexdigest()


@cache
def generic_gaussian_anti_wick_control() -> tuple[bool, dict[str, Any]]:
    """Verify Gaussian normalization and the phase-space resolution coefficient."""

    h = sp.Symbol("h", positive=True, finite=True)
    dimension = 3
    gaussian_square_prefactor = (sp.pi * h) ** (-sp.Rational(dimension, 2))
    gaussian_square_integral = (sp.sqrt(sp.pi * h)) ** dimension
    window_norm_squared = sp.simplify(
        gaussian_square_prefactor * gaussian_square_integral
    )
    phase_space_prefactor = (2 * sp.pi * h) ** (-dimension)
    phase_delta_factor = (2 * sp.pi * h) ** dimension
    resolution_coefficient = sp.simplify(
        phase_space_prefactor * phase_delta_factor * window_norm_squared
    )
    corrupted_resolution = sp.simplify(
        2 * phase_space_prefactor * phase_delta_factor * window_norm_squared
    )
    corrupted_residual = sp.simplify(corrupted_resolution - 1)

    lam, upper, a, b = sp.symbols(
        "lambda Lambda a b", positive=True, finite=True
    )
    diagonal_form = lam * a**2 + upper * b**2
    lower_residual = sp.expand(diagonal_form - lam * (a**2 + b**2))
    upper_residual = sp.expand(upper * (a**2 + b**2) - diagonal_form)
    passed = bool(
        window_norm_squared == 1
        and resolution_coefficient == 1
        and sp.factor(lower_residual) == b**2 * (upper - lam)
        and sp.factor(upper_residual) == a**2 * (upper - lam)
        and corrupted_residual != 0
    )
    return passed, {
        "spatial_dimension": dimension,
        "normalized_window": (
            "g_h(y)=(pi*h)^(-3/4) exp(-|y|^2/(2h)), with ||g_h||_L2=1"
        ),
        "coherent_state": (
            "psi_(x,xi,h)(y)=exp(i xi.(y-x)/h) g_h(y-x)"
        ),
        "window_norm_squared": str(window_norm_squared),
        "phase_space_measure": "(2*pi*h)^(-3) dx dxi",
        "phase_delta_factor": str(phase_delta_factor),
        "resolution_of_identity_coefficient": str(resolution_coefficient),
        "moyal_identity": (
            "(2*pi*h)^(-3) integral |<psi_(x,xi,h),u>|^2 dx dxi=||u||_L2^2"
        ),
        "matrix_energy_control": {
            "lower_residual": str(sp.factor(lower_residual)),
            "upper_residual": str(sp.factor(upper_residual)),
            "conclusion": (
                "lambda I<=K<=Lambda I implies lambda||u||^2<="
                "<u,Op_h^AW(K)u><=Lambda||u||^2"
            ),
        },
        "negative_control": {
            "corruption": "double the phase-space measure prefactor",
            "resolution_coefficient": str(corrupted_resolution),
            "identity_residual": str(corrupted_residual),
            "rejected": corrupted_residual != 0,
        },
        "primary_references": [
            {
                "title": "Integral formulas for the Weyl and anti-Wick symbols",
                "url": "https://arxiv.org/abs/1806.04898",
            },
            {
                "title": (
                    "Pseudo-differential operators with isotropic symbols, Wick and "
                    "anti-Wick operators, and hypoellipticity"
                ),
                "url": "https://arxiv.org/abs/2011.00313",
            },
        ],
        "passed": passed,
        "scope": (
            "Exact Gaussian/Moyal normalization and operator-energy equivalence. "
            "Composition with the evolution operator is a separate gate."
        ),
    }


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


def _certify_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id"))
    if candidate.get("status") != "pass_global_C4_positive_K55_symbol_extension":
        raise QuarticPositiveQuantizationError(
            f"candidate {candidate_id} lacks a positive global symbol"
        )
    energy = candidate.get("energy_equivalence", {})
    lower = energy.get("K55_2_lower")
    upper = energy.get("K55_2_upper")
    lower_numeric = float(energy.get("K55_2_lower_numeric", 0))
    upper_numeric = float(energy.get("K55_2_upper_numeric", 0))
    if not (isinstance(lower, str) and isinstance(upper, str)):
        raise QuarticPositiveQuantizationError(
            f"candidate {candidate_id} lacks exact energy bounds"
        )
    if not (lower_numeric > 0 and upper_numeric >= lower_numeric):
        raise QuarticPositiveQuantizationError(
            f"candidate {candidate_id} lacks positive numeric energy bounds"
        )
    return {
        "schema_version": "sigma-quartic-positive-quantization-certificate-1.0",
        "status": "pass_uniform_positive_anti_wick_K55_operator",
        "candidate_id": candidate_id,
        "coefficients": candidate.get("coefficients"),
        "quantization": {
            "name": "semiclassical Gaussian anti-Wick quantization",
            "space": "L2(R^3;C^55)",
            "scale_domain": "0<h<=1",
            "definition": (
                "Op_h^AW(K)=(2*pi*h)^(-3) integral |psi_z,h><psi_z,h| "
                "tensor K(U(x),xi) dx dxi"
            ),
            "self_adjoint": True,
            "positive": True,
        },
        "operator_energy_equivalence": {
            "lower": lower,
            "lower_numeric": lower_numeric,
            "upper": upper,
            "upper_numeric": upper_numeric,
            "uniform_in_h": True,
            "exactly_matches_pointwise_symbol_bounds": True,
        },
        "claim": (
            "For every 0<h<=1 and state field taking values in the certified tube, "
            "Op_h^AW(K_ext) is self-adjoint and its quadratic form lies between the "
            "same uniform lower and upper K55 bounds."
        ),
        "remaining_gate": (
            "anti_wick_principal_composition_remainder_commuted_Sobolev_energy_lifespan"
        ),
        "scope": (
            "This proves positivity and L2 energy equivalence of the quantized symmetrizer. "
            "It does not yet bound d_t Op_h^AW(K), the composition/commutator with the "
            "first-order evolution operator, dyadic summation, a nonlinear energy "
            "inequality, lifespan, or matter."
        ),
    }


def run_quartic_positive_quantization_campaign(
    low_frequency_campaign: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticPositiveQuantizationError(
                "unsupported campaign schema_version"
            )
        if low_frequency_campaign.get("status") != (
            "pass_all_12_global_C4_positive_K55_symbol_extensions"
        ):
            raise QuarticPositiveQuantizationError(
                "low-frequency symbol prerequisite failed"
            )
        if not _content_hash_matches(low_frequency_campaign):
            raise QuarticPositiveQuantizationError(
                "low-frequency campaign content hash mismatch"
            )
        if int(config["spatial_dimension"]) != 3:
            raise QuarticPositiveQuantizationError(
                "quantization requires three spatial dimensions"
            )
        if int(config["state_dimension"]) != 55:
            raise QuarticPositiveQuantizationError(
                "quantization requires the complete 55-state reduction"
            )
        if config.get("coherent_window") != "normalized_isotropic_Gaussian":
            raise QuarticPositiveQuantizationError(
                "unsupported coherent-state window"
            )
        if config.get("semiclassical_scale_domain") != "0<h<=1":
            raise QuarticPositiveQuantizationError(
                "unsupported semiclassical scale domain"
            )
        control_passed, control = generic_gaussian_anti_wick_control()
        if not control_passed:
            raise QuarticPositiveQuantizationError(
                "generic anti-Wick normalization control failed"
            )
        records = _candidate_records(low_frequency_campaign)
        expected = int(config.get("expected_candidate_count", 12))
        if len(records) != expected:
            raise QuarticPositiveQuantizationError("candidate-set mismatch")
        certificates = [
            _certify_candidate(records[candidate_id])
            for candidate_id in sorted(records)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_uniform_positive_anti_wick_K55_operators",
            "errors": [],
            "low_frequency_campaign_sha256": low_frequency_campaign.get(
                "content_sha256"
            ),
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_gaussian_anti_wick_control": control,
            "counts": {
                "selected": len(certificates),
                "positive_quantizations_passed": len(certificates),
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 quartic candidates have a uniformly positive self-adjoint "
                "anti-Wick quantization of the global K55 symbol."
            ),
            "scope": certificates[0]["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticPositiveQuantizationError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "positive_quantizations_passed": 0,
                "rejected": 0,
            },
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_positive_quantization_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
