from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class MaxwellHilbertNoetherGateError(RuntimeError):
    """Raised when registered Maxwell interface evidence is absent or altered."""


_RECEIPT_KEYS = {
    "arbitrary_background_block",
    "campaign_id",
    "claims",
    "content_sha256",
    "counts",
    "decision",
    "gate_id",
    "gate_results",
    "massless_specialization",
    "noether_interface",
    "registered_controls",
    "schema_version",
    "scope",
    "source_bindings",
}


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise MaxwellHilbertNoetherGateError(f"cannot read evidence: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise MaxwellHilbertNoetherGateError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise MaxwellHilbertNoetherGateError(f"JSON evidence is not an object: {path}")
    return value


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise MaxwellHilbertNoetherGateError("evidence path escapes repository root")
    return path


def _bound_json(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _sha256(path) != binding.get("file_sha256"):
        raise MaxwellHilbertNoetherGateError(f"file hash mismatch: {path}")
    value = _load_json(path)
    expected_content = binding.get("content_sha256")
    if expected_content is not None and value.get("content_sha256") != expected_content:
        raise MaxwellHilbertNoetherGateError(f"content hash mismatch: {path}")
    return path, value


def _control_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    checks = report.get("semantic_report", {}).get("checks")
    if not isinstance(checks, list):
        raise MaxwellHilbertNoetherGateError("formal report has no checks list")
    return {
        str(item["name"]): item
        for item in checks
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def _zero_residuals(value: Any, expected: int, label: str) -> list[str]:
    if not isinstance(value, list) or value != ["0"] * expected:
        raise MaxwellHilbertNoetherGateError(f"{label} residuals are not exactly zero")
    return value


def _antisymmetric_symmetric_contraction() -> dict[str, Any]:
    # The gauge identity is structural: R_{rho nu} F^{rho nu}=0 because swapping
    # the two dummy indices preserves symmetric R and negates antisymmetric F.
    # This integer replay checks every ordered pair and includes a false control.
    symmetric = [[2, 3, -1, 4], [3, 5, 6, 0], [-1, 6, 7, 8], [4, 0, 8, 9]]
    antisymmetric = [[0, 2, -3, 1], [-2, 0, 5, 4], [3, -5, 0, -6], [-1, -4, 6, 0]]
    residual = sum(symmetric[mu][nu] * antisymmetric[mu][nu] for mu in range(4) for nu in range(4))
    corrupted = [row[:] for row in symmetric]
    corrupted[0][1] += 1
    negative_residual = sum(
        corrupted[mu][nu] * antisymmetric[mu][nu] for mu in range(4) for nu in range(4)
    )
    if residual != 0 or negative_residual == 0:
        raise MaxwellHilbertNoetherGateError("gauge-divergence structural replay failed")
    return {
        "dimension": 4,
        "ricci_symmetry": "R_rho_nu=R_nu_rho",
        "field_strength_antisymmetry": "F^rho_nu=-F^nu_rho",
        "contraction_residual": str(residual),
        "corrupted_symmetry_negative_residual": str(negative_residual),
        "conclusion": "nabla_rho(nabla_mu F^mu_rho)=0",
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != "invariant-maxwell-hilbert-noether-interface-config-1.0":
        raise MaxwellHilbertNoetherGateError("unsupported config schema")
    expected_policy = {
        "allow_bounded_maxwell_interface_pass": True,
        "allow_universal_matter_claim": False,
        "allow_gravity_h7_claim": False,
        "allow_global_or_boundary_claim": False,
        "allow_promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise MaxwellHilbertNoetherGateError("claims policy is absent or broadened")

    predecessor_path, predecessor = _bound_json(repository, config["predecessor"])
    blocker = predecessor.get("first_blocker", {}).get("reason_codes", [])
    if config["predecessor"]["required_first_blocker"] not in blocker:
        raise MaxwellHilbertNoetherGateError("predecessor Maxwell blocker is absent")

    bindings = config["evidence_bindings"]
    formal_path, formal = _bound_json(repository, bindings["formal_controls"])
    action_path, action_ir = _bound_json(repository, bindings["proca_action_ir"])
    identity_path = _resolve(repository, bindings["covariant_identity_source"]["path"])
    if _sha256(identity_path) != bindings["covariant_identity_source"]["file_sha256"]:
        raise MaxwellHilbertNoetherGateError("covariant identity source hash mismatch")

    controls = _control_map(formal)
    selected: dict[str, dict[str, Any]] = {}
    for name in config["required_controls"]:
        control = controls.get(name)
        if control is None or control.get("status") != "pass":
            raise MaxwellHilbertNoetherGateError(f"required PASS control absent: {name}")
        selected[name] = control

    terms = action_ir.get("canonical", {}).get("terms", [])
    term_ids = {item.get("id") for item in terms if isinstance(item, dict)}
    specialization = config["massless_specialization"]
    if not {specialization["retained_term_id"], specialization["removed_term_id"]} <= term_ids:
        raise MaxwellHilbertNoetherGateError("Proca action lacks the massless specialization terms")

    flat = selected["proca_stress_noether_identity"]["evidence"]
    curved = selected["proca_curved_background_noether_identity"]["evidence"]
    flat_residuals = _zero_residuals(flat.get("residuals"), 4, "Minkowski")
    flrw_residuals = _zero_residuals(curved.get("flrw", {}).get("residuals"), 4, "FLRW")
    spherical_residuals = _zero_residuals(
        curved.get("static_spherical", {}).get("residuals"), 4, "static-spherical"
    )
    gauge_identity = _antisymmetric_symmetric_contraction()
    source_path = Path(__file__).resolve()
    test_path = repository / "tests/test_maxwell_hilbert_noether_interface_gate.py"

    body: dict[str, Any] = {
        "schema_version": "invariant-maxwell-hilbert-noether-interface-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "BOUNDED_PASS_WITH_TYPED_BLOCK",
        "gate_id": "dedicated_maxwell_hilbert_stress_noether_identity",
        "registered_controls": [
            {
                "name": name,
                "status": selected[name]["status"],
                "claim": selected[name]["claim"],
                "scope": selected[name]["scope"],
            }
            for name in config["required_controls"]
        ],
        "gate_results": [
            {
                "scope": (
                    "Minkowski arbitrary vector profile plus registered FLRW-homogeneous "
                    "and static-spherical-radial profiles"
                ),
                "outcome": "PASS",
                "reason_codes": [],
            },
            {
                "scope": "arbitrary curved metric and arbitrary vector profile",
                "outcome": "BLOCK",
                "reason_codes": [
                    "missing_arbitrary_background_maxwell_stress_divergence_derivation"
                ],
            },
        ],
        "counts": {
            "registered_controls": 4,
            "action_terms_specialized": 2,
            "exact_noether_residuals": len(flat_residuals + flrw_residuals + spherical_residuals),
            "exact_structural_residuals": 1,
            "negative_controls": 1,
            "blocks": 1,
            "rejects": 0,
        },
        "massless_specialization": {
            "operation": "set m_A=0 before drawing the Maxwell conclusion",
            "retained_action_term": specialization["retained_term_id"],
            "removed_action_term": specialization["removed_term_id"],
            "hilbert_stress": "T_mu_nu=F_mu_rho F_nu^rho-g_mu_nu F_rho_sigma F^rho_sigma/4",
            "vector_euler": "E^rho=nabla_mu F^(mu rho)",
        },
        "noether_interface": {
            "off_shell_identity": "nabla^mu T_mu_nu-F_nu_rho E^rho=0",
            "on_shell_conservation": "nabla^mu T_mu_nu=0 when E^rho=0",
            "minkowski_arbitrary_profile_residuals": flat_residuals,
            "curved_profile_residuals": {
                "flrw_homogeneous": flrw_residuals,
                "static_spherical_radial": spherical_residuals,
            },
            "gauge_divergence_identity": gauge_identity,
        },
        "claims": {
            "dedicated_maxwell_registered_profile_interface_closed": True,
            "dedicated_maxwell_arbitrary_background_interface_closed": False,
            "curved_executable_profiles_are_universal_proof": False,
            "universal_matter_closure_established": False,
            "gravity_h7_theorem_established": False,
            "global_boundary_control_established": False,
            "promotion_authorized": False,
        },
        "arbitrary_background_block": {
            "outcome": "BLOCK",
            "reason_code": "missing_arbitrary_background_maxwell_stress_divergence_derivation",
            "why_gauge_identity_is_insufficient": (
                "nabla_rho E^rho=0 removes the Proca gauge term after m_A=0 but does not "
                "independently derive nabla^mu T_mu_nu-F_nu_rho E^rho=0"
            ),
            "minimal_registration_contract": [
                "source-bound Maxwell action and Hilbert metric variation on one action hash",
                (
                    "exact abstract-index or full local-jet derivation of "
                    "nabla^mu T_mu_nu-F_nu_rho E^rho"
                ),
                ("explicit metric-compatibility and covariant-derivative commutator reduction"),
                (
                    "at least one corrupted-sign or omitted-connection negative "
                    "with nonzero residual"
                ),
                "path-free immutable receipt binding source, test, action, and formal evidence hashes",
            ],
        },
        "scope": (
            "bounded source-free Maxwell Hilbert stress/Noether interface on arbitrary-profile "
            "Minkowski and two registered curved profile families; arbitrary curved metrics/profiles "
            "remain typed BLOCK; coupled-gravity, boundary, and universal-matter claims excluded"
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "file_sha256": _sha256(config_path),
            },
            "predecessor": {
                "path": predecessor_path.relative_to(repository).as_posix(),
                "file_sha256": _sha256(predecessor_path),
                "content_sha256": predecessor["content_sha256"],
            },
            "formal_controls": {
                "path": formal_path.relative_to(repository).as_posix(),
                "file_sha256": _sha256(formal_path),
            },
            "action_ir": {
                "path": action_path.relative_to(repository).as_posix(),
                "file_sha256": _sha256(action_path),
                "content_sha256": action_ir["content_sha256"],
            },
            "covariant_identity_source": {
                "path": identity_path.relative_to(repository).as_posix(),
                "file_sha256": _sha256(identity_path),
            },
            "source": {
                "path": source_path.relative_to(repository).as_posix(),
                "file_sha256": _sha256(source_path),
            },
            "test": {
                "path": test_path.relative_to(repository).as_posix(),
                "file_sha256": _sha256(test_path),
            },
        },
    }
    return {**body, "content_sha256": _canonical_sha(body)}


def write_receipt(
    config_path: Path, output_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    receipt = build_receipt(config_path, root=root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def validate_receipt(
    receipt: dict[str, Any], config_path: Path, *, root: Path | None = None
) -> None:
    """Rebuild and compare the closed immutable Maxwell receipt exactly."""
    if set(receipt) != _RECEIPT_KEYS:
        raise MaxwellHilbertNoetherGateError("Maxwell receipt schema changed")
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if receipt.get("content_sha256") != _canonical_sha(body):
        raise MaxwellHilbertNoetherGateError("Maxwell receipt content seal changed")
    if receipt != build_receipt(config_path, root=root):
        raise MaxwellHilbertNoetherGateError("Maxwell receipt immutable replay changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_receipt(args.config.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
