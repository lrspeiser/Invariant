from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import sympy as sp

from .static_dictionary import _parse_generator_expression

SCHEMA_VERSION = "sigma-grammar-exhaustion-1.0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_static_grammar_exhaustion(
    priority_path: str | Path,
    formal_controls_path: str | Path,
    q_operator_path: str | Path,
) -> dict[str, Any]:
    """Prove whether the current x/q/z static queue has any admissible covariant lane left."""

    priority_path = Path(priority_path).resolve()
    formal_controls_path = Path(formal_controls_path).resolve()
    q_operator_path = Path(q_operator_path).resolve()
    priority = json.loads(priority_path.read_text(encoding="utf-8"))
    formal = json.loads(formal_controls_path.read_text(encoding="utf-8"))
    q_operator = json.loads(q_operator_path.read_text(encoding="utf-8"))
    checks = {item["name"]: item for item in formal.get("checks", [])}
    tilt_control = checks.get("projected_aether_q_constant_tilt_root_audit", {})
    tilt_evidence = tilt_control.get("evidence", {})
    q_rejection_verified = (
        tilt_control.get("status") == "pass"
        and tilt_evidence.get("generic_tilt_hyperbolicity_status") == "reject"
        and tilt_evidence.get("expanded_quartic_nonreal_root_count") == 2
        and q_operator.get("status") == "reject"
        and q_operator.get("constant_background_covariant_principal", {}).get("status")
        == "reject"
    )

    x, q, z = sp.symbols("x q z", real=True)
    del x
    decisions: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for candidate in priority.get("work_queue", []):
        parsed = _parse_generator_expression(candidate["correction_expression"])
        if z in parsed.free_symbols:
            decision = "reject_forbidden_baryonic_z"
            reason = (
                "z is a matter-current diagnostic and cannot enter the universal minimally "
                "coupled gravitational action"
            )
        elif q in parsed.free_symbols and q_rejection_verified:
            decision = "reject_registered_projected_q_completion"
            reason = (
                "the only registered exact q lift in this grammar, Q_a_u, fails the complete "
                "constant-tilt root audit with a nonreal conjugate pair"
            )
        elif q in parsed.free_symbols:
            decision = "unresolved_q_rejection_evidence_missing"
            reason = "the Q structural rejection artifacts are missing or inconsistent"
        else:
            decision = "unresolved_outside_q_z_partition"
            reason = "candidate is not covered by the current q/z exhaustion proof"
        decisions[decision] += 1
        records.append(
            {
                "family_id": candidate["family_id"],
                "ordinal": candidate["ordinal"],
                "pareto_front": candidate.get("pareto_front"),
                "decision": decision,
                "reason": reason,
            }
        )

    unresolved = sum(count for name, count in decisions.items() if name.startswith("unresolved"))
    rejected = sum(count for name, count in decisions.items() if name.startswith("reject"))
    exhausted = q_rejection_verified and unresolved == 0 and rejected == len(records)
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": "exhausted_no_admissible_family" if exhausted else "unresolved",
        "current_grammar_scope": (
            "the frozen dense x/q/z static-survivor queue and its registered universal-metric "
            "covariant completions only"
        ),
        "input_priority": str(priority_path),
        "input_priority_sha256": _sha256(priority_path),
        "formal_controls": str(formal_controls_path),
        "formal_controls_sha256": _sha256(formal_controls_path),
        "q_operator_ir": str(q_operator_path),
        "q_operator_ir_sha256": _sha256(q_operator_path),
        "q_rejection_verified": q_rejection_verified,
        "queue_count": len(records),
        "rejected_count": rejected,
        "unresolved_count": unresolved,
        "decision_counts": dict(sorted(decisions.items())),
        "records": records,
        "hard_lessons": [
            {
                "pattern": "contains legacy z",
                "scope": "all current static families",
                "decision": "reject",
                "reason": "violates universal minimal matter coupling",
            },
            {
                "pattern": "uses the registered Q_a_u lift of legacy q",
                "scope": "current generic unit-vector covariant grammar",
                "decision": "reject",
                "reason": "nonzero-tilt quartic has two nonreal lab-frequency roots",
            },
        ],
        "next_grammar": {
            "strategy": "covariant-first rather than another static q/z mutation pass",
            "admissible_lanes": [
                "nonlinear first-derivative F(X_a_u) with independently derived gradient and constraint completion",
                "healthy Einstein-Aether K1..K4 functions restricted by known mode and energy domains",
                "explicitly degenerate scalar-tensor/DHOST combinations with symbolic degeneracy relations",
                "a separately declared hypersurface-orthogonal khronon theory only if its field and Cauchy-surface contract is explicit",
            ],
            "prohibited_carryovers": [
                "baryonic z as a gravitational action atom",
                "Q_a_u or algebraic repackagings of it in the generic unit-vector lane",
                "dark-matter or halo-derived target terms",
                "redshift-derived or supernova-derived distance targets",
            ],
            "first_required_gates": [
                "exact covariant field/action grammar and static dictionary",
                "generic-background kinetic and degeneracy screen before large enumeration",
                "complete frozen-coefficient root count on an open time-covector neighborhood",
                "ADM/Dirac closure and reduced Hamiltonian before any observation",
            ],
        },
        "observational_data_opened": False,
        "interpretation": (
            "This proves exhaustion of the current registered grammar, not impossibility of all "
            "covariant completions sharing a similar static phenomenology."
        ),
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {**body, "content_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def write_grammar_exhaustion(report: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
