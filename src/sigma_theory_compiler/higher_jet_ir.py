from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "sigma-higher-jet-auxiliary-ir-1.0"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compile_higher_jet_auxiliary_ir(action_ir: dict[str, Any]) -> dict[str, Any]:
    """Construct the exact constrained first-derivative lift of projected Q actions."""

    if not action_ir.get("valid"):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "promotion_allowed": False,
            "input_action_sha256": action_ir.get("content_sha256"),
            "errors": ["covariant action IR is invalid"],
        }
    q_terms = sorted(
        term["id"]
        for term in action_ir["canonical"]["terms"]
        if term["id"].startswith("AETHER_Q")
    )
    applicable = bool(q_terms)
    if applicable:
        status = "unresolved"
        reason = (
            "the exact auxiliary lift is available, but its complete primary/secondary "
            "constraint closure and reduced Hamiltonian have not been derived"
        )
    else:
        status = "pass"
        reason = "action contains no projected acceleration-gradient higher-jet term"
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "promotion_allowed": False,
        "input_action_sha256": action_ir["content_sha256"],
        "applicable": applicable,
        "q_term_ids": q_terms,
        "auxiliary_fields": ["b_mu", "r^mu"] if applicable else [],
        "lift": (
            {
                "replacement_definition": "B_mu_nu=nabla_mu b_nu",
                "replacement_invariant": (
                    "Q_aux=(L_sigma^2/a_sigma^2) P^{mu rho}P^{nu sigma} "
                    "B_mu_nu B_rho_sigma"
                ),
                "constraint_density": "sqrt(-g) r^mu (b_mu-u^alpha nabla_alpha u_mu)",
                "multiplier_euler_equation": "b_mu-u^alpha nabla_alpha u_mu=0",
                "on_constraint_substitution": "b_mu -> a_mu=u^alpha nabla_alpha u_mu",
                "on_constraint_result": "Q_aux -> Q_a_u exactly for every admitted Q power",
                "maximum_derivative_order_per_independent_field": {
                    "b_mu": 1,
                    "u_mu": 1,
                    "r^mu": 0,
                },
            }
            if applicable
            else None
        ),
        "equivalence_certificate": {
            "status": "pass" if applicable else "not_applicable",
            "route": "algebraic multiplier equation followed by literal invariant substitution",
            "quantum_equivalence_claimed": False,
            "boundary_equivalence_claimed": False,
        },
        "required_dirac_work": (
            [
                "3+1 decompose b_mu and r^mu on the positive unit-Aether branch",
                "derive all lapse, shift, multiplier, and projected-b kinetic primaries",
                "preserve every primary and iterate secondary/higher constraints",
                "compute constraint-surface Poisson rank on generic-tilt parameter strata",
                "remove gauge and second-class directions before Hamiltonian boundedness",
            ]
            if applicable
            else []
        ),
        "reason": reason,
        "observational_data_opened": False,
        "proof_scope": (
            "exact classical constrained first-derivative rewriting only; it is the input to, "
            "not a substitute for, the higher-jet Dirac and physical-Hamiltonian calculation"
        ),
    }
    content = _canonical_json(body)
    return {**body, "content_sha256": hashlib.sha256(content.encode()).hexdigest()}


def write_higher_jet_auxiliary_ir(payload: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
