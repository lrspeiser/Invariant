from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

from .dimensions import ACCELERATION, assert_dimensionless_invariants, normalized_invariant_dimensions
from .gates import algebraic_gates, deferred_gates, sampled_static_convexity
from .grammar import Q, X, Z, GrammarExpression, enumerate_expressions


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    sign: int
    grammar_expression: GrammarExpression


class TheoryCompiler:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        assert_dimensionless_invariants(normalized_invariant_dimensions())

    @classmethod
    def from_path(cls, path: str | Path) -> "TheoryCompiler":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def enumerate(self) -> tuple[list[Candidate], dict[str, int]]:
        grammar = self.config["grammar"]
        expressions, counts = enumerate_expressions(
            atoms=grammar["atoms"],
            unary_operators=grammar["unary_operators"],
            binary_operators=grammar["binary_operators"],
            max_complexity=grammar["max_complexity"],
        )
        candidates: list[Candidate] = []
        for item in expressions:
            for sign in self.config["search"]["coupling_signs"]:
                payload = f"{item.canonical}|{sign:+d}|{self.config['protocol_version']}"
                digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
                candidates.append(Candidate(f"STC-{digest}", sign, item))
        candidates.sort(key=lambda item: item.candidate_id)
        counts["signed_candidates"] = len(candidates)
        return candidates, counts

    @staticmethod
    def field_equation(expression: sp.Expr, signed_coupling: float) -> sp.Equality:
        radius = sp.symbols("r", real=True)
        a_sigma, length_sigma, z_zero = sp.symbols(
            "a_sigma L_sigma Z_0", positive=True, finite=True
        )
        displacement = sp.Function("D")(radius)
        state = sp.Function("Z_b")(radius)
        substituted = expression.subs(
            {
                X: displacement**2 / a_sigma**2,
                Q: length_sigma**2 * sp.diff(displacement, radius) ** 2 / a_sigma**2,
                Z: state**2 / z_zero**2,
            }
        )
        hamiltonian = displacement**2 / 2 + sp.Float(signed_coupling) * a_sigma**2 * substituted
        displacement_gradient = sp.diff(displacement, radius)
        # SymPy can differentiate with respect to Derivative(D(r), r)
        # directly. Keeping it as a derivative object is essential: the outer
        # total r derivative must produce the D'' Euler-Lagrange term.
        euler = sp.diff(hamiltonian, displacement) - sp.diff(
            sp.diff(hamiltonian, displacement_gradient), radius
        )
        euler = sp.simplify(euler.doit())
        return sp.Eq(sp.Symbol("dW_dr"), euler)

    def compile_candidate(self, candidate: Candidate) -> dict[str, Any]:
        search = self.config["search"]
        signed_coupling = candidate.sign * float(search["coupling_magnitude"])
        expression = candidate.grammar_expression.expression
        constants = self.config["theory_contract"]["universal_constants"]
        gates = algebraic_gates(
            expression,
            constants_count=len(constants),
            maximum_constants=self.config["theory_contract"]["maximum_universal_constants"],
        )
        gates.append(
            sampled_static_convexity(
                expression,
                signed_coupling,
                samples=search["convexity_samples"],
                tolerance=float(search["convexity_tolerance"]),
            )
        )
        gates.extend(deferred_gates())
        statuses = [gate.status for gate in gates]
        status = "rejected_pre_covariant" if "reject" in statuses else "requires_covariant_lift"
        equation = self.field_equation(expression, signed_coupling)
        return {
            "candidate_id": candidate.candidate_id,
            "status": status,
            "complexity": candidate.grammar_expression.complexity,
            "canonical_expression": str(expression),
            "canonical_key": candidate.grammar_expression.canonical,
            "coupling": signed_coupling,
            "static_action": f"H = D**2/2 + ({signed_coupling})*a_sigma**2*({expression})",
            "radial_constitutive_equation": str(equation),
            "radial_constitutive_equation_latex": sp.latex(equation),
            "dimensions": {
                "x": {"L": 0, "T": 0},
                "q": {"L": 0, "T": 0},
                "z": {"L": 0, "T": 0},
                "H": (ACCELERATION**2).as_dict(),
            },
            "gates": [gate.as_dict() for gate in gates],
        }

    def run(self, config_path: str | Path | None = None) -> dict[str, Any]:
        candidates, enumeration = self.enumerate()
        compiled = [self.compile_candidate(candidate) for candidate in candidates]
        counts = {
            "total": len(compiled),
            "rejected_pre_covariant": sum(
                row["status"] == "rejected_pre_covariant" for row in compiled
            ),
            "requires_covariant_lift": sum(
                row["status"] == "requires_covariant_lift" for row in compiled
            ),
            "fully_validated_theories": 0,
        }
        config_hash = hashlib.sha256(
            json.dumps(self.config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": "sigma-theory-compiler-registry-1.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_version": self.config["protocol_version"],
            "config_path": str(config_path) if config_path else None,
            "config_sha256": config_hash,
            "scope_claim": self.config["scope_claim"],
            "parent_protocol": self.config["parent_protocol"],
            "enumeration": enumeration,
            "counts": counts,
            "scientific_warning": (
                "A requires_covariant_lift row is only a static-sector survivor. It is not a healthy "
                "relativistic theory and is not authorized for observational fitting."
            ),
            "candidates": compiled,
        }
