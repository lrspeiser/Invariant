from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import sympy as sp

from .gates import algebraic_gates, sampled_static_convexity
from .grammar import Q, X, Z


TRANSFORMS = ("Identity", "Sqrt1pMinus1", "Saturate")


def choose(n: int, k: int) -> int:
    return math.comb(n, k)


def family_count(basis_count: int, terms: int) -> int:
    return choose(basis_count, terms) * 2**terms


def total_search_count(basis_count: int, max_action_terms: int) -> int:
    return sum(family_count(basis_count, terms) for terms in range(1, max_action_terms + 1))


def unrank_combination(n: int, k: int, rank: int) -> list[int]:
    if not 0 <= rank < choose(n, k):
        raise ValueError("combination rank outside family")
    result: list[int] = []
    start = 0
    for position in range(k):
        remaining = k - position - 1
        for value in range(start, n):
            count = 1 if remaining == 0 else choose(n - value - 1, remaining)
            if rank < count:
                result.append(value)
                start = value + 1
                break
            rank -= count
    return result


def decode_ordinal(basis_count: int, max_action_terms: int, ordinal: int) -> dict[str, Any]:
    offset = 0
    for terms in range(1, max_action_terms + 1):
        width = family_count(basis_count, terms)
        if ordinal < offset + width:
            local = ordinal - offset
            sign_mask = local & (2**terms - 1)
            combination_rank = local >> terms
            return {
                "ordinal": ordinal,
                "term_ids": unrank_combination(basis_count, terms, combination_rank),
                "signs": [1 if sign_mask & (1 << position) else -1 for position in range(terms)],
            }
        offset += width
    raise ValueError(f"ordinal {ordinal} outside search space")


def _monomial(px: int, pq: int, pz: int) -> str:
    factors: list[str] = []
    for name, power in (("x", px), ("q", pq), ("z", pz)):
        if power == 1:
            factors.append(name)
        elif power > 1:
            factors.append(f"{name}**{power}")
    return "*".join(factors) or "1"


def build_basis(count: int) -> list[dict[str, Any]]:
    basis: list[dict[str, Any]] = []
    for degree in range(1, 21):
        for px in range(degree + 1):
            for pq in range(degree - px + 1):
                pz = degree - px - pq
                monomial = _monomial(px, pq, pz)
                for transform in TRANSFORMS:
                    if transform == "Identity":
                        expression = monomial
                        growth = (px, 1)
                    elif transform == "Sqrt1pMinus1":
                        expression = f"sqrt(1+({monomial}))-1"
                        growth = (px, 2)
                    else:
                        expression = f"({monomial})/(1+({monomial}))"
                        growth = (0, 1)
                    basis.append(
                        {
                            "id": len(basis),
                            "px": px,
                            "pq": pq,
                            "pz": pz,
                            "transform": transform,
                            "dimension_l": 0,
                            "dimension_t": 0,
                            "derivative_order_in_h": 1 if pq > 0 else 0,
                            "has_measured_state": pq > 0 or pz > 0,
                            "high_field_growth_numerator": growth[0],
                            "high_field_growth_denominator": growth[1],
                            "expression": expression,
                        }
                    )
                    if len(basis) == count:
                        return basis
    raise ValueError("requested basis exceeds reference library")


def correction_expression(decoded: dict[str, Any], basis: list[dict[str, Any]]) -> str:
    return "".join(
        f"{'+' if sign > 0 else '-'}({basis[term_id]['expression']})"
        for term_id, sign in zip(decoded["term_ids"], decoded["signs"], strict=True)
    )


def candidate_id(protocol: str, decoded: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(b"SIGMA-GENERATOR-V2\0")
    digest.update(protocol.encode("utf-8"))
    digest.update(b"\0")
    digest.update(bytes([len(decoded["term_ids"])]))
    for term_id, sign in zip(decoded["term_ids"], decoded["signs"], strict=True):
        digest.update(int(term_id).to_bytes(2, "little"))
        digest.update(bytes([1 if sign > 0 else 0]))
    return f"STC2-{digest.hexdigest()[:24]}"


def _sympy_expression(text: str) -> sp.Expr:
    return sp.sympify(text, locals={"x": X, "q": Q, "z": Z, "sqrt": sp.sqrt})


def crosscheck_manifest(
    manifest_path: str | Path,
    config_path: str | Path,
    coupling_magnitude: float = 0.1,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    config_bytes = Path(config_path).read_bytes()
    config = json.loads(config_bytes)
    basis = build_basis(config["basis_count"])
    expected_total = total_search_count(config["basis_count"], config["max_action_terms"])
    basis_payload = json.dumps(basis, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    root = hashlib.sha256()
    root.update(b"SIGMA-GENERATOR-V2-ROOT\0")
    for block in manifest["blocks"]:
        root.update(int(block["block_index"]).to_bytes(8, "little"))
        root.update(int(block["start_ordinal"]).to_bytes(8, "little"))
        root.update(int(block["end_ordinal_exclusive"]).to_bytes(8, "little"))
        root.update(block["digest_sha256"].encode("ascii"))

    accounting = {
        "declared_count_matches": manifest["total_declared_actions"] == expected_total,
        "config_hash_matches": manifest["config_sha256"]
        == hashlib.sha256(config_bytes).hexdigest(),
        "basis_hash_matches": manifest["basis_library_sha256"]
        == hashlib.sha256(basis_payload).hexdigest(),
        "blocks_root_matches": manifest["blocks_root_sha256"] == root.hexdigest(),
        "gate_counts_sum_matches": sum(manifest["gate_counts"].values())
        == manifest["processed_actions"],
        "block_counts_sum_matches": sum(block["processed"] for block in manifest["blocks"])
        == manifest["processed_actions"],
        "observational_data_closed": manifest["observational_data_opened"] is False,
        "coefficient_alphabet_matches": manifest["coefficient_alphabet"]
        == config["coefficient_alphabet"],
        "survivor_count_matches": manifest["survivor_count"]
        == manifest["gate_counts"].get("survive_sampled_static", 0),
    }

    sample_rows: list[dict[str, Any]] = []
    for sample in manifest["survivor_samples"]:
        decoded = decode_ordinal(
            config["basis_count"], config["max_action_terms"], sample["ordinal"]
        )
        expected_expression = correction_expression(decoded, basis)
        expected_id = candidate_id(config["protocol_version"], decoded)
        structural_agreement = {
            "term_ids": decoded["term_ids"] == sample["term_ids"],
            "signs": decoded["signs"] == sample["signs"],
            "expression": expected_expression == sample["correction_expression"],
            "candidate_id": expected_id == sample["candidate_id"],
        }
        expression = _sympy_expression(expected_expression)
        gates = algebraic_gates(
            expression,
            constants_count=len(config["universal_constants"]),
            maximum_constants=config["maximum_universal_constants"],
        )
        tier0_python_pass = all(
            gate.status == "pass"
            for gate in gates
            if gate.name
            in {
                "finite_origin",
                "vacuum_zero",
                "high_field_newtonian_limit",
                "new_spatial_state_information",
                "universal_constant_cap",
                "derivative_order",
                "one_metric_no_private_lensing_law",
            }
        )
        convexity = sampled_static_convexity(
            expression,
            coupling_magnitude,
            samples={"d": [0.1, 1.0, 10.0], "p": [0.0, 0.5, 1.0], "state": [0.0, 0.5, 1.0]},
            tolerance=1e-9,
        )
        sample_rows.append(
            {
                "candidate_id": sample["candidate_id"],
                "ordinal": sample["ordinal"],
                "structural_agreement": structural_agreement,
                "rust_tier0_gate": sample["gate"],
                "python_tier0_pass": tier0_python_pass,
                "python_sampled_static_convexity": convexity.as_dict(),
            }
        )

    rejection_rows: list[dict[str, Any]] = []
    for rust_gate, sample in manifest.get("rejection_witnesses", {}).items():
        decoded = decode_ordinal(
            config["basis_count"], config["max_action_terms"], sample["ordinal"]
        )
        expected_expression = correction_expression(decoded, basis)
        expression = _sympy_expression(expected_expression)
        gates = algebraic_gates(
            expression,
            constants_count=len(config["universal_constants"]),
            maximum_constants=config["maximum_universal_constants"],
        )
        statuses = {gate.name: gate.status for gate in gates}
        convexity = sampled_static_convexity(
            expression,
            coupling_magnitude,
            samples={
                "d": config["convexity_samples"]["d"],
                "p": config["convexity_samples"]["p"],
                "state": config["convexity_samples"]["state"],
            },
            tolerance=float(config["convexity_tolerance"]),
        )
        expected_rejection = {
            "reject_flux_only": statuses["new_spatial_state_information"] == "reject",
            "reject_high_field": statuses["high_field_newtonian_limit"] == "reject",
            "reject_no_gradient_sector": convexity.status == "reject",
            "reject_negative_elasticity": convexity.status == "reject",
            "reject_sampled_static_convexity": convexity.status == "reject",
        }[rust_gate]
        identity_agreement = (
            decoded["term_ids"] == sample["term_ids"]
            and decoded["signs"] == sample["signs"]
            and expected_expression == sample["correction_expression"]
            and candidate_id(config["protocol_version"], decoded) == sample["candidate_id"]
        )
        rejection_rows.append(
            {
                "rust_gate": rust_gate,
                "candidate_id": sample["candidate_id"],
                "ordinal": sample["ordinal"],
                "identity_agreement": identity_agreement,
                "python_reproduces_rejection": expected_rejection,
            }
        )

    all_accounting = all(accounting.values())
    all_samples_agree = all(
        all(row["structural_agreement"].values()) and row["python_tier0_pass"]
        for row in sample_rows
    )
    convexity_counts = Counter(
        row["python_sampled_static_convexity"]["status"] for row in sample_rows
    )
    all_rejections_agree = len(rejection_rows) == 5 and all(
        row["identity_agreement"] and row["python_reproduces_rejection"]
        for row in rejection_rows
    )
    return {
        "schema_version": "sigma-generator-v2-crosscheck-1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "config_path": str(config_path),
        "expected_total_declared_actions": expected_total,
        "accounting": accounting,
        "all_accounting_checks_pass": all_accounting,
        "sample_count": len(sample_rows),
        "all_cross_language_samples_agree": all_samples_agree,
        "all_recorded_survivors_pass_python_sampled_convexity": all(
            row["python_sampled_static_convexity"]["status"] == "pass"
            for row in sample_rows
        ),
        "sampled_static_convexity_counts": dict(convexity_counts),
        "samples": sample_rows,
        "rejection_witness_count": len(rejection_rows),
        "all_rejection_witnesses_agree": all_rejections_agree,
        "rejection_witnesses": rejection_rows,
        "interpretation": (
            "Cross-language agreement verifies ordinal decoding, basis construction, signs, stable IDs, "
            "and cheap gate semantics on the recorded sample. Static convexity remains a sampled kill "
            "gate and a pass is not a covariant-health claim."
        ),
    }


def write_crosscheck(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
