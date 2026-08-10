from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-component-jacobian-contract-campaign-1.0"


class QuarticComponentJacobianContractError(ValueError):
    """Raised when the component-Jacobian contract audit cannot be certified."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _content_hash_matches(campaign: dict[str, Any]) -> bool:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    return campaign.get("content_sha256") == _content_hash(body)


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


def _state_basis() -> list[str]:
    return [
        *[f"q[{field}]" for field in range(11)],
        *[f"v0[{field}]" for field in range(11)],
        *[
            f"w{spatial}[{field}]"
            for spatial in range(1, 4)
            for field in range(11)
        ],
    ]


def _atom_basis() -> list[str]:
    spatial_pairs = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
    return [
        # The shift-symmetric scalar value is absent: 10 metric deviations,
        # 44 first partials, and 99 acceleration-free second partials total 153.
        *[f"q[{field}]" for field in range(10)],
        *[
            f"p{derivative}[{field}]"
            for derivative in range(4)
            for field in range(11)
        ],
        *[
            f"s0{spatial}[{field}]"
            for spatial in range(1, 4)
            for field in range(11)
        ],
        *[
            f"s{left}{right}[{field}]"
            for left, right in spatial_pairs
            for field in range(11)
        ],
    ]


@cache
def generic_component_jacobian_contract_control() -> tuple[bool, dict[str, Any]]:
    """Build the exact principal jet injection and prove what norm data cannot do."""

    state_basis = _state_basis()
    atom_basis = _atom_basis()
    state_index = {label: index for index, label in enumerate(state_basis)}
    atom_index = {label: index for index, label in enumerate(atom_basis)}
    xi = sp.symbols("xi1:4", real=True, finite=True)
    injection_entries: list[dict[str, Any]] = []
    for field in range(11):
        for spatial in range(1, 4):
            injection_entries.append(
                {
                    "row": atom_index[f"s0{spatial}[{field}]"],
                    "column": state_index[f"v0[{field}]"],
                    "coefficient": str(sp.I * xi[spatial - 1]),
                }
            )
        for left, right in ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)):
            if left == right:
                injection_entries.append(
                    {
                        "row": atom_index[f"s{left}{right}[{field}]"],
                        "column": state_index[f"w{left}[{field}]"],
                        "coefficient": str(sp.I * xi[left - 1]),
                    }
                )
            else:
                injection_entries.extend(
                    (
                        {
                            "row": atom_index[f"s{left}{right}[{field}]"],
                            "column": state_index[f"w{right}[{field}]"],
                            "coefficient": str(sp.I * xi[left - 1] / 2),
                        },
                        {
                            "row": atom_index[f"s{left}{right}[{field}]"],
                            "column": state_index[f"w{left}[{field}]"],
                            "coefficient": str(sp.I * xi[right - 1] / 2),
                        },
                    )
                )

    positions = {(item["row"], item["column"]) for item in injection_entries}
    duplicate_count = len(injection_entries) - len(positions)
    expected_entry_count = 11 * (3 + 3 + 2 * 3)

    kinematic_product: dict[tuple[int, int], sp.Expr] = {}
    for field in range(11):
        for spatial in range(1, 4):
            output_row = state_index[f"w{spatial}[{field}]"
            ]
            atom_row = atom_index[f"s0{spatial}[{field}]"
            ]
            for entry in injection_entries:
                if entry["row"] == atom_row:
                    key = (output_row, int(entry["column"]))
                    kinematic_product[key] = kinematic_product.get(
                        key, sp.Integer(0)
                    ) + sp.sympify(
                        entry["coefficient"],
                        locals={str(symbol): symbol for symbol in xi} | {"I": sp.I},
                    )
    kinematic_residuals = {
        f"{state_index[f'w{spatial}[{field}]']},"
        f"{state_index[f'v0[{field}]']}": str(
            sp.expand(
                kinematic_product.get(
                    (
                        state_index[f"w{spatial}[{field}]"],
                        state_index[f"v0[{field}]"],
                    ),
                    sp.Integer(0),
                )
                - sp.I * xi[spatial - 1]
            )
        )
        for field in range(11)
        for spatial in range(1, 4)
    }

    amplitude = sp.Symbol("amplitude", real=True, finite=True)
    curl_compatible_residuals: dict[str, str] = {}
    corrupted_off_diagonal_residuals: dict[str, str] = {}
    for left, right in ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)):
        w_left = sp.I * xi[left - 1] * amplitude
        w_right = sp.I * xi[right - 1] * amplitude
        injected = (
            sp.I * xi[left - 1] * w_left
            if left == right
            else sp.I
            * (xi[left - 1] * w_right + xi[right - 1] * w_left)
            / 2
        )
        expected = -xi[left - 1] * xi[right - 1] * amplitude
        curl_compatible_residuals[f"{left},{right}"] = str(
            sp.expand(injected - expected)
        )
        if left != right:
            corrupted = sp.I * (
                xi[left - 1] * w_right + xi[right - 1] * w_left
            )
            corrupted_off_diagonal_residuals[f"{left},{right}"] = str(
                sp.expand(corrupted - expected)
            )

    coefficient_a = sp.Symbol("A", nonzero=True, real=True, finite=True)
    mixed_atoms = sp.symbols("m1:4", real=True, finite=True)
    mixed_blocks = sp.symbols("B1:4", real=True, finite=True)
    solved = -2 * sum(
        mixed_blocks[index] * mixed_atoms[index] for index in range(3)
    ) / coefficient_a
    mixed_derivative_residuals = {
        str(index + 1): str(
            sp.expand(
                sp.diff(solved, mixed_atoms[index])
                + 2 * mixed_blocks[index] / coefficient_a
            )
        )
        for index in range(3)
    }
    omitted_two_residual = sp.expand(
        sp.diff(solved, mixed_atoms[0]) + mixed_blocks[0] / coefficient_a
    )

    same_norm_left = sp.Matrix([[1, 0], [0, 0]])
    same_norm_right = sp.Matrix([[0, 1], [0, 0]])
    probe = sp.Matrix([1, 0])
    norm_orientation_witness = same_norm_left * probe - same_norm_right * probe
    norm_equal = same_norm_left.norm(2) == same_norm_right.norm(2)

    state_hash = _content_hash(state_basis)
    atom_hash = _content_hash(atom_basis)
    injection_packet = {
        "shape": [153, 55],
        "state_basis_sha256": state_hash,
        "coordinate_atom_basis_sha256": atom_hash,
        "entries": injection_entries,
    }
    injection_hash = _content_hash(injection_packet)
    passed = bool(
        len(state_basis) == 55
        and len(atom_basis) == 153
        and len(injection_entries) == expected_entry_count == 132
        and duplicate_count == 0
        and set(kinematic_residuals.values()) == {"0"}
        and set(curl_compatible_residuals.values()) == {"0"}
        and all(value != "0" for value in corrupted_off_diagonal_residuals.values())
        and set(mixed_derivative_residuals.values()) == {"0"}
        and omitted_two_residual != 0
        and norm_equal
        and not norm_orientation_witness.is_zero_matrix
    )
    return passed, {
        "control": "exact 153x55 principal jet-injection and missing-schema audit",
        "canonical_bases": {
            "state_dimension": len(state_basis),
            "coordinate_atom_dimension": len(atom_basis),
            "state_basis": state_basis,
            "coordinate_atom_basis": atom_basis,
            "state_basis_sha256": state_hash,
            "coordinate_atom_basis_sha256": atom_hash,
        },
        "principal_jet_injection": {
            **injection_packet,
            "content_sha256": injection_hash,
            "nonzero_entry_count": len(injection_entries),
            "duplicate_position_count": duplicate_count,
            "definition": (
                "delta s_0i=i xi_i delta v0; delta s_ij="
                "i(xi_i delta w_j+xi_j delta w_i)/2"
            ),
        },
        "kinematic_evolution_rows": {
            "identity": "D_Y(dot w_i=s_0i) J=i xi_i delta v0",
            "residuals": kinematic_residuals,
            "rows_proved": len(kinematic_residuals),
        },
        "curl_compatible_spatial_Hessian": {
            "identity": "delta w_i=i xi_i delta q implies delta s_ij=-xi_i xi_j delta q",
            "residuals": curl_compatible_residuals,
            "omitted_symmetrization_half_residuals": corrupted_off_diagonal_residuals,
            "negative_rejected": all(
                value != "0" for value in corrupted_off_diagonal_residuals.values()
            ),
        },
        "solved_acceleration_mixed_derivative": {
            "identity": "D_s0i F=-2 A^-1 P^0i",
            "residuals": mixed_derivative_residuals,
            "omitted_factor_two_residual": str(omitted_two_residual),
            "negative_rejected": omitted_two_residual != 0,
        },
        "norm_envelope_insufficiency_negative": {
            "left_matrix": str(same_norm_left),
            "right_matrix": str(same_norm_right),
            "equal_operator_norms": norm_equal,
            "probe": str(probe),
            "different_action_witness": str(norm_orientation_witness),
            "rejected_claim": (
                "a Frechet operator-norm envelope determines the component Jacobian"
            ),
            "rejected": norm_equal and not norm_orientation_witness.is_zero_matrix,
        },
        "required_component_packet_schema": {
            "schema_version": "sigma-source-P55-component-linearization-packet-1.0",
            "candidate_id": "required",
            "coefficients": "required exact mapping",
            "state_basis_sha256": state_hash,
            "coordinate_atom_basis_sha256": atom_hash,
            "principal_jet_injection_sha256": injection_hash,
            "source_formula_contract_sha256": "required",
            "physical_pencil_source_block_sha256": "required",
            "source_jacobian": {
                "shape": [11, 153],
                "entry_encoding": "exact row-major expressions or exact sparse entries",
                "principal_second_atom_column_count": 99,
            },
            "full_evolution_jacobian": {
                "shape": [55, 153],
                "fixed_kinematic_rows": 44,
                "dynamic_solved_source_rows": 11,
            },
            "composed_principal_residual": {
                "identity": "D_Y E55 J_153x55(xi)-i P55(Y,xi)=0",
                "shape": [55, 55],
                "required_zero_entry_count": 3025,
            },
            "remainder_prerequisite": {
                "source_Frechet_orders_required": [2, 3, 4],
                "required_bound": (
                    "explicit paraproduct remainder polynomial in ||Y||H6 and ||U||H7"
                ),
                "uniform_on_coordinate_tube": True,
            },
        },
        "passed": passed,
        "scope": (
            "The jet injection and all kinematic rows are exact. The eleven dynamic rows "
            "still require component expressions for the solved-source Jacobian; norm "
            "envelopes provably cannot determine them."
        ),
    }


def _certify_candidate(
    good_unknown: dict[str, Any],
    solved_source: dict[str, Any],
    nonlinear_evolution: dict[str, Any],
    nonquasilinear: dict[str, Any],
    generic: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(good_unknown.get("candidate_id"))
    others = (solved_source, nonlinear_evolution, nonquasilinear)
    if any(
        item.get("candidate_id") != candidate_id
        or item.get("coefficients") != good_unknown.get("coefficients")
        for item in others
    ):
        raise QuarticComponentJacobianContractError("candidate identity mismatch")
    expected_statuses = (
        "audit_paradifferential_good_unknown_binding_fail_closed",
        "pass_coordinate_atom_C4_solved_source_moser_envelopes",
        "pass_exact_local_nonlinear_time_acceleration_elimination",
        "pass_full_55_state_nonquasilinear_strong_hyperbolicity_lift",
    )
    if tuple(
        item.get("status")
        for item in (good_unknown, solved_source, nonlinear_evolution, nonquasilinear)
    ) != expected_statuses:
        raise QuarticComponentJacobianContractError(
            "candidate prerequisite status mismatch"
        )
    evolution_formula_hash = nonlinear_evolution.get(
        "evolution_formula_contract_sha256"
    )
    if nonquasilinear.get("nonquasilinear_equation", {}).get(
        "evolution_formula_contract_sha256"
    ) != evolution_formula_hash:
        raise QuarticComponentJacobianContractError(
            "nonlinear evolution formula provenance mismatch"
        )
    frechet = solved_source["solved_source_Frechet_derivatives"]
    component_packet = solved_source.get("source_P55_component_linearization_packet")
    packet_present = isinstance(component_packet, dict)
    packet_valid = bool(
        packet_present
        and component_packet.get("schema_version")
        == "sigma-source-P55-component-linearization-packet-1.0"
        and component_packet.get("state_basis_sha256")
        == generic["canonical_bases"]["state_basis_sha256"]
        and component_packet.get("coordinate_atom_basis_sha256")
        == generic["canonical_bases"]["coordinate_atom_basis_sha256"]
        and component_packet.get("principal_jet_injection_sha256")
        == generic["principal_jet_injection"]["content_sha256"]
    )
    return {
        "schema_version": "sigma-quartic-component-jacobian-contract-certificate-1.0",
        "status": "audit_component_jacobian_packet_missing_fail_closed",
        "candidate_id": candidate_id,
        "coefficients": good_unknown.get("coefficients"),
        "generator_provenance": {
            "evolution_formula_contract_sha256": evolution_formula_hash,
            "state_to_covariant_jet_formula_sha256": nonquasilinear[
                "nonquasilinear_equation"
            ]["state_to_covariant_jet_formula_sha256"],
            "physical_pencil_source_block_sha256": good_unknown[
                "physical_pencil_provenance"
            ]["source_spatial_block_sha256"],
        },
        "principal_jet_injection": {
            "shape": [153, 55],
            "content_sha256": generic["principal_jet_injection"]["content_sha256"],
            "nonzero_entries": generic["principal_jet_injection"][
                "nonzero_entry_count"
            ],
            "kinematic_rows_proved": generic["kinematic_evolution_rows"][
                "rows_proved"
            ],
        },
        "source_data_currently_available": {
            "Frechet_orders": frechet["orders"],
            "operator_norm_envelopes_only": True,
            "component_packet_present": packet_present,
        },
        "component_packet_validation": {
            "present": packet_present,
            "valid": packet_valid,
            "required_schema": generic["required_component_packet_schema"],
        },
        "D_Y_E55_times_J_equals_iP55_proved": packet_valid,
        "paralinearization_remainder_bound_proved": False,
        "H7_derivative_loss_resolved": False,
        "global_dyadic_summation_applied": False,
        "precise_blocker": (
            "emit the exact 11x153 solved-source Jacobian in the canonical atom basis, "
            "assemble the fixed 44 kinematic rows, multiply by the certified sparse J, "
            "and compare all 3025 entries with iP55; then use Frechet orders 2-4 to "
            "bound the Bony remainder"
        ),
        "scope": (
            "This proves the universal injection and the 33 derivative-bearing kinematic "
            "rows without inventing dynamics. The existing witness generator and Moser "
            "campaign do not publish the dynamic component Jacobian, so the decisive "
            "identity remains fail-closed."
        ),
    }


def run_quartic_component_jacobian_contract_campaign(
    good_unknown_campaign: dict[str, Any],
    solved_source_campaign: dict[str, Any],
    nonlinear_evolution_campaign: dict[str, Any],
    nonquasilinear_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticComponentJacobianContractError(
                "unsupported campaign schema_version"
            )
        campaigns = (
            good_unknown_campaign,
            solved_source_campaign,
            nonlinear_evolution_campaign,
            nonquasilinear_campaign,
        )
        expected_statuses = (
            "pass_all_12_paradifferential_good_unknown_audits_component_binding_fail_closed",
            "pass_all_12_coordinate_atom_C4_solved_source_moser_envelopes",
            "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations",
            "pass_all_12_full_55_state_nonquasilinear_strong_hyperbolicity_lifts",
        )
        if tuple(campaign.get("status") for campaign in campaigns) != expected_statuses:
            raise QuarticComponentJacobianContractError(
                "campaign prerequisite status mismatch"
            )
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticComponentJacobianContractError(
                "campaign content hash mismatch"
            )
        if good_unknown_campaign.get("upstream_sha256", {}).get(
            "solved_source"
        ) != solved_source_campaign.get("content_sha256"):
            raise QuarticComponentJacobianContractError(
                "good-unknown solved-source provenance mismatch"
            )
        if solved_source_campaign.get("upstream_sha256", {}).get(
            "nonquasilinear_pde"
        ) != nonquasilinear_campaign.get("content_sha256"):
            raise QuarticComponentJacobianContractError(
                "solved-source PDE provenance mismatch"
            )
        if (
            int(config["spatial_dimension"]) != 3
            or int(config["state_dimension"]) != 55
            or int(config["coordinate_atom_dimension"]) != 153
            or int(config["dynamic_source_row_count"]) != 11
        ):
            raise QuarticComponentJacobianContractError(
                "unsupported component-Jacobian dimensions"
            )
        if bool(config.get("declare_component_identity_proved", False)):
            raise QuarticComponentJacobianContractError(
                "component identity cannot be declared without the packet"
            )
        generic_passed, generic = generic_component_jacobian_contract_control()
        if not generic_passed:
            raise QuarticComponentJacobianContractError(
                "generic component-Jacobian control failed"
            )
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        candidate_ids = set(maps[0])
        expected = int(config.get("expected_candidate_count", 12))
        if len(candidate_ids) != expected or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticComponentJacobianContractError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                *(records[candidate_id] for records in maps), generic
            )
            for candidate_id in sorted(candidate_ids)
        ]
        proved = sum(
            int(item["D_Y_E55_times_J_equals_iP55_proved"])
            for item in certificates
        )
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_12_component_jacobian_schema_audits_packet_missing_"
                "fail_closed"
            ),
            "errors": [],
            "upstream_sha256": {
                "good_unknown": good_unknown_campaign.get("content_sha256"),
                "solved_source": solved_source_campaign.get("content_sha256"),
                "nonlinear_evolution": nonlinear_evolution_campaign.get(
                    "content_sha256"
                ),
                "nonquasilinear_pde": nonquasilinear_campaign.get("content_sha256"),
            },
            "config_sha256": _content_hash(config),
            "generic_component_jacobian_contract_control": generic,
            "counts": {
                "selected": len(certificates),
                "jet_injections_certified": len(certificates),
                "component_identities_proved": proved,
                "remainder_bounds_proved": 0,
                "global_H7_summations_applied": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 candidates share an exact canonical 153x55 principal jet "
                "injection and generator-provenance contract. None publishes the 11x153 "
                "component source Jacobian required for the dynamic identity."
            ),
            "scope": certificates[0]["scope"],
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        QuarticComponentJacobianContractError,
    ) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "jet_injections_certified": 0,
                "component_identities_proved": 0,
                "remainder_bounds_proved": 0,
                "global_H7_summations_applied": 0,
                "rejected": 0,
            },
        }
    return {
        **body,
        "content_sha256": _content_hash(body),
    }


def write_quartic_component_jacobian_contract_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
