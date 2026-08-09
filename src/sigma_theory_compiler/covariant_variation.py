from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sympy as sp

from .action_ir import compile_action_file
from .formal_backend import probe_cadabra, run_cadabra_script


def _safe_coefficient(value: str | int | float) -> str:
    parsed = sp.sympify(value)
    if any(not isinstance(symbol, sp.Symbol) for symbol in parsed.free_symbols):
        raise ValueError(f"unsupported coefficient: {value}")
    return str(parsed).replace("**", "^")


def render_scalar_cadabra_variation(action_ir: dict[str, Any]) -> str:
    if not action_ir.get("valid"):
        raise ValueError("cannot export an invalid action IR")
    canonical = action_ir["canonical"]
    if "phi" not in canonical["fields"]:
        raise ValueError("scalar variation requires phi in the action field list")
    terms = {item["id"]: item for item in canonical["terms"]}
    supported = {"EH_R", "SCALAR_X", "SCALAR_MASS"}
    unsupported = sorted(set(terms) - supported)
    if unsupported:
        raise ValueError("unsupported scalar-export terms: " + ", ".join(unsupported))
    pieces: list[str] = []
    if "SCALAR_X" in terms:
        coefficient = _safe_coefficient(terms["SCALAR_X"]["coefficient"])
        pieces.append(f"({coefficient})/2 \\partial_{{\\mu}}{{\\phi}} \\partial^{{\\mu}}{{\\phi}}")
    if "SCALAR_MASS" in terms:
        # The library fixes this coefficient to -1/2 in the action density.
        coefficient = sp.sympify(terms["SCALAR_MASS"]["coefficient"])
        if sp.simplify(coefficient + sp.Rational(1, 2)) != 0:
            raise ValueError("SCALAR_MASS exporter currently requires its grammar coefficient -1/2")
        # Cadabra treats an underscore as an index marker, so scalar identifiers stay ASCII-flat.
        pieces.append("mphi^2/2 \\phi^2")
    if not pieces:
        raise ValueError("the action contains no supported phi-dependent term")
    integrand = " + ".join(pieces)
    return "\n".join(
        [
            "{\\mu,\\nu}::Indices.",
            "\\partial{#}::PartialDerivative;",
            "",
            f"action := -\\int{{{integrand}}}{{x}};",
            "vary(action, $\\phi -> \\delta{\\phi}$);",
            "integrate_by_parts(action, $\\delta{\\phi}$);",
            "canonicalise(action);",
            "sort_product(action);",
            "factor_out(action, $\\delta{\\phi}$);",
            "print(action);",
            "",
        ]
    )


def render_proca_cadabra_variation(action_ir: dict[str, Any]) -> str:
    if not action_ir.get("valid"):
        raise ValueError("cannot export an invalid action IR")
    canonical = action_ir["canonical"]
    if "A_mu" not in canonical["fields"]:
        raise ValueError("Proca variation requires A_mu in the action field list")
    terms = {item["id"]: item for item in canonical["terms"]}
    supported = {"EH_R", "PROCA_F2", "PROCA_MASS"}
    unsupported = sorted(set(terms) - supported)
    if unsupported:
        raise ValueError("unsupported Proca-export terms: " + ", ".join(unsupported))
    if not {"PROCA_F2", "PROCA_MASS"}.issubset(terms):
        raise ValueError("the Proca exporter requires both F^2 and mass terms")
    if sp.simplify(sp.sympify(terms["PROCA_F2"]["coefficient"]) + sp.Rational(1, 4)) != 0:
        raise ValueError("PROCA_F2 exporter requires the grammar coefficient -1/4")
    if sp.simplify(sp.sympify(terms["PROCA_MASS"]["coefficient"]) + sp.Rational(1, 2)) != 0:
        raise ValueError("PROCA_MASS exporter requires the grammar coefficient -1/2")
    return "\n".join(
        [
            "{a,b,c,d}::Indices(position=free).",
            "x::Coordinate.",
            "\\partial{#}::Derivative.",
            "F_{a b}::AntiSymmetric.",
            "F_{a b}::Depends(x).",
            "A_{a}::Depends(x,\\partial{#}).",
            "\\delta{#}::Accent.",
            "",
            "action := -\\int{F_{a b} F^{a b}/4 + mA^2 A_{a} A^{a}/2}{x};",
            "field_strength := F_{a b} = \\partial_{a}{A_{b}} - \\partial_{b}{A_{a}};",
            "substitute(action, field_strength);",
            "vary(action, $A_{a} -> \\delta{A_{a}}$);",
            "distribute(action);",
            "integrate_by_parts(action, $\\delta{A_{a}}$);",
            "canonicalise(action);",
            "sort_product(action);",
            'print("SIGMA_PROCA_VARIATION_FINAL");',
            "print(action);",
            "",
        ]
    )


def vary_scalar_action_file(
    spec_path: str | Path,
    grammar_path: str | Path,
    contract_path: str | Path,
    output_directory: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    action_ir = compile_action_file(spec_path, grammar_path, contract_path)
    ir_path = output / "action-ir.json"
    ir_path.write_text(json.dumps(action_ir, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    script_text = render_scalar_cadabra_variation(action_ir)
    script_path = output / "scalar-variation.cdb"
    script_path.write_text(script_text, encoding="utf-8")
    backend = probe_cadabra(project_root)
    if not backend["available"]:
        return {
            "status": "unresolved",
            "reason": "Cadabra backend unavailable",
            "backend": backend,
            "action_ir": str(ir_path),
            "script": str(script_path),
        }
    passed, execution = run_cadabra_script(
        project_root,
        backend,
        script_path,
        ["δ(φ)", "\\partial", "mphi"],
    )
    result = {
        "schema_version": "sigma-covariant-variation-result-1.0",
        "status": "pass" if passed else "fail",
        "scope": "variation with respect to phi; EH is independent of phi; metric variation remains separate",
        "input_action_sha256": action_ir["content_sha256"],
        "action_ir": str(ir_path),
        "script": str(script_path),
        "backend": backend,
        "execution": execution,
        "expected_euler_lagrange_structure": "c_X box(phi) - m_phi^2 phi = 0",
    }
    result_path = output / "variation-result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["result_path"] = str(result_path)
    return result


def vary_proca_action_file(
    spec_path: str | Path,
    grammar_path: str | Path,
    contract_path: str | Path,
    output_directory: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    action_ir = compile_action_file(spec_path, grammar_path, contract_path)
    ir_path = output / "action-ir.json"
    ir_path.write_text(json.dumps(action_ir, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    script_path = output / "proca-variation.cdb"
    script_path.write_text(render_proca_cadabra_variation(action_ir), encoding="utf-8")
    backend = probe_cadabra(project_root)
    if not backend["available"]:
        return {
            "status": "unresolved",
            "reason": "Cadabra backend unavailable",
            "backend": backend,
            "action_ir": str(ir_path),
            "script": str(script_path),
        }
    passed, execution = run_cadabra_script(
        project_root,
        backend,
        script_path,
        ["SIGMA_PROCA_VARIATION_FINAL", "δ{A", "\\partial", "mA"],
    )
    result = {
        "schema_version": "sigma-covariant-variation-result-1.0",
        "status": "pass" if passed else "fail",
        "scope": "variation with respect to A_mu; EH is independent of A_mu; metric variation remains separate",
        "input_action_sha256": action_ir["content_sha256"],
        "action_ir": str(ir_path),
        "script": str(script_path),
        "backend": backend,
        "execution": execution,
        "expected_euler_lagrange_structure": "partial_mu F^{mu nu} - m_A^2 A^nu = 0 up to the declared sign convention",
        "divergence_identity": "antisymmetry and commuting derivatives imply m_A^2 partial_mu A^mu = 0",
    }
    result_path = output / "variation-result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["result_path"] = str(result_path)
    return result
