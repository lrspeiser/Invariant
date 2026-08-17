"""Gates for the formal ladder over the 71 screened-gravity families of v3.

The tests pin: that the known-answer controls actually fire *through this module's own
ladder* (a canonical scalar clears every rung, a wrong-sign kinetic term rejects on ghost
freedom, a negative Einstein-Hilbert coefficient rejects on the tensor sector, and a
concave k-essence rejects on the cone policy); that every verdict kind is reachable and
structurally validated; that the nonlocal arm of every real family is BLOCKED and can
never be reported as a pass; that a blocked family can never be a FORMAL_PASS; that the
declared sector Lagrangians are the ones the lift text pins, in the pack's own sign
convention; receipt determinism, seal-tamper behaviour, binding to the v3 receipt hash and
to the derived representative set, the no-float rule, and the CLI.

The sealed run over all 71 families is shared through a session fixture because a single
scalar-tensor pack compilation runs ten generic covariant/ADM/Dirac/principal controls.
"""

from __future__ import annotations

import json

import pytest
import sympy as sp

from sigma_theory_compiler import v3_family_formal_ladder as ladder
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = ladder.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def receipt() -> dict:
    return ladder.run_formal_ladder(ROOT)


@pytest.fixture(scope="session")
def sealed() -> dict:
    return json.loads((ROOT / ladder.LADDER_RECEIPT_PATH).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Inputs and binding
# ---------------------------------------------------------------------------


def test_the_v3_screen_receipt_self_seal_replays() -> None:
    screen = ladder.load_screen_receipt(ROOT)
    body = {key: item for key, item in screen.items() if key != "content_sha256"}
    assert screen["content_sha256"] == canonical_sha256(body)
    assert screen["counts"]["passer_families"] == 71


def test_representative_set_carries_all_71_and_binds_to_the_screen() -> None:
    screen = ladder.load_screen_receipt(ROOT)
    families = ladder.load_representatives(ROOT, screen)
    assert len(families) == 71
    assert sum(item["size"] for item in families) == screen["counts"]["all_gate_passers"]
    artifact = json.loads(
        (ROOT / ladder.REPRESENTATIVES_PATH).read_text(encoding="utf-8")
    )
    assert artifact["source"]["content_sha256"] == screen["content_sha256"]
    assert artifact["counts"]["reported_in_sealed_receipt"] == 16
    assert artifact["counts"]["recovered_by_replay"] == 55


def test_representative_build_refuses_a_set_that_does_not_reproduce_the_sealed_sixteen() -> None:
    screen = ladder.load_screen_receipt(ROOT)
    families = ladder.load_representatives(ROOT, screen)
    tampered = [dict(item) for item in families]
    tampered[0] = {**tampered[0], "representative_ordinal": tampered[0]["representative_ordinal"] + 1}
    with pytest.raises(ladder.V3FormalLadderError, match="reproduce the sealed"):
        ladder.build_representatives(screen, tampered)
    with pytest.raises(ladder.V3FormalLadderError, match="count does not match"):
        ladder.build_representatives(screen, families[:-1])


def test_representative_artifact_bound_to_a_different_receipt_is_refused(tmp_path) -> None:
    screen = ladder.load_screen_receipt(ROOT)
    forged = dict(screen)
    forged["content_sha256"] = "0" * 64
    with pytest.raises(ladder.V3FormalLadderError, match="different v3 receipt"):
        ladder.load_representatives(ROOT, forged)


# ---------------------------------------------------------------------------
# Known-answer controls
# ---------------------------------------------------------------------------


def test_ladder_controls_fire_with_their_declared_verdicts(receipt: dict) -> None:
    controls = receipt["controls"]["ladder_controls"]
    assert set(controls) == set(ladder.CONTROL_ACTIONS)
    observed = {name: entry["observed_verdict"] for name, entry in controls.items()}
    assert observed == {
        "canonical_scalar": "FORMAL_PASS",
        "wrong_sign_ghost": "FORMAL_REJECT:ghost_freedom",
        "tensor_ghost": "FORMAL_REJECT:tensor_sector",
        "superluminal_kessence": "FORMAL_REJECT:principal_symbol_hyperbolicity",
    }


def test_the_canonical_scalar_control_clears_every_rung(receipt: dict) -> None:
    statuses = receipt["controls"]["ladder_controls"]["canonical_scalar"]["rung_statuses"]
    assert set(statuses) == {name for name, _ in ladder.LADDER_RUNGS}
    assert set(statuses.values()) == {"pass"}


def test_the_wrong_sign_control_rejects_on_the_first_rung(receipt: dict) -> None:
    statuses = receipt["controls"]["ladder_controls"]["wrong_sign_ghost"]["rung_statuses"]
    assert statuses["ghost_freedom"] == "reject"


def test_imported_negative_controls_are_replayed(receipt: dict) -> None:
    imported = receipt["controls"]["imported_principal_symbol_controls"]
    assert imported["passed"]
    assert imported["negative_fired"] == [
        "negative_kinetic_ghost",
        "negative_gradient",
        "superluminal_cone",
    ]
    legendre = receipt["controls"]["imported_kessence_legendre_control"]
    assert "wrong_sign_scalar" in legendre["negatives_rejected"]


def test_a_control_whose_verdict_moves_aborts_the_run(monkeypatch) -> None:
    monkeypatch.setitem(
        ladder.CONTROL_ACTIONS,
        "canonical_scalar",
        {**ladder.CONTROL_ACTIONS["canonical_scalar"], "expect": "FORMAL_REJECT:ghost_freedom"},
    )
    with pytest.raises(ladder.V3FormalLadderError, match="control canonical_scalar expected"):
        ladder.run_controls(ROOT)


def test_a_broken_imported_control_aborts_the_run(monkeypatch) -> None:
    monkeypatch.setattr(
        ladder, "run_principal_symbol_controls", lambda: {"passed": False, "controls": {}}
    )
    with pytest.raises(ladder.V3FormalLadderError, match="imported principal-symbol controls"):
        ladder.run_controls(ROOT)


# ---------------------------------------------------------------------------
# Materialization: the nonlocal arm must block, never pass
# ---------------------------------------------------------------------------


def test_every_family_carries_the_nonlocal_arm_and_is_blocked_on_it(receipt: dict) -> None:
    families = receipt["families"]
    assert len(families) == 71
    for entry in families:
        blockers = entry["materialization"]["full_lift_blockers"]
        assert "missing_adapter:nonlocal_fractional_operator" in blockers
        assert entry["materialization"]["full_lift_expressible"] is False
        assert entry["verdict"].startswith("BLOCKED:")
    assert receipt["counts"]["blocked_by_adapter"][
        "missing_adapter:nonlocal_fractional_operator"
    ] == 71


def test_the_nonlocal_arm_is_never_reported_as_a_formal_pass(receipt: dict) -> None:
    assert receipt["counts"]["formal_pass"] == 0
    for entry in receipt["families"]:
        assert entry["verdict"] != "FORMAL_PASS"


def test_classify_lift_names_the_declared_mechanisms() -> None:
    screen = ladder.load_screen_receipt(ROOT)
    families = ladder.load_representatives(ROOT, screen)
    seen = {
        mechanism
        for family in families
        for mechanism in ladder.classify_lift(family)["declared_mechanisms"]
    }
    assert "nonlocal_propagator_correction" in seen
    assert {"vainshtein_kinetic_braiding", "kmouflage_gradient_screening"} <= seen


def test_a_lift_with_only_an_expressible_sector_materializes() -> None:
    family = {
        "screening_family": "curvature",
        "covariant_lift_candidate": {
            "components": [{"mechanism": "kmouflage_gradient_screening"}]
        },
    }
    materialization = ladder.classify_lift(family)
    assert materialization["full_lift_expressible"] is True
    assert materialization["full_lift_blockers"] == []


def test_an_unknown_screening_family_is_refused() -> None:
    with pytest.raises(ladder.V3FormalLadderError, match="no declared covariant sector"):
        ladder.classify_lift(
            {"screening_family": "density", "covariant_lift_candidate": {"components": []}}
        )


# ---------------------------------------------------------------------------
# The declared sector ansaetze
# ---------------------------------------------------------------------------


def test_the_cubic_galileon_normalization_is_forced_by_the_declared_lagrangian() -> None:
    """``-(dphi)^2/2 - (Box phi)(dphi)^2/Lambda^3`` is ``g2 = x``, ``g3 = -2x`` exactly."""

    lam, box = sp.symbols("Lambda_phi box_phi", positive=True)
    x = sp.Symbol("x", positive=True)
    nabla_squared = -2 * x * lam**4  # the pack's x definition, inverted
    declared = -nabla_squared / 2 - box * nabla_squared / lam**3
    functions = ladder.SECTOR_ANSATZ["acceleration"]["functions"]
    g2 = sp.sympify(functions["g2"], locals={"x": x})
    g3 = sp.sympify(functions["g3"], locals={"x": x})
    pack_form = lam**4 * (g2 - g3 * box / lam**3)
    assert sp.simplify(declared - pack_form) == 0


def test_the_kmouflage_sector_is_the_repository_convex_g2_cell() -> None:
    ansatz = ladder.SECTOR_ANSATZ["curvature"]
    assert ansatz["functions"] == {"g2": "x + c_K*x**2", "g3": "0", "g4": "1/2"}
    assert ansatz["coefficients"] == ["c_K"]
    assert [cell["c_K"] for cell in ansatz["parameter_cells"]] == ["1/8", "1/4"]


def test_both_sectors_compile_into_the_repository_typed_action_ir(receipt: dict) -> None:
    for entry in receipt["families"]:
        action = entry["action_ir"]
        assert action["schema_version"] == "sigma-scalar-tensor-pack-ir-1.0"
        assert len(action["content_sha256"]) == 64
        assert action["status"] == "compiled_formal_adapters_unresolved"


# ---------------------------------------------------------------------------
# The rungs
# ---------------------------------------------------------------------------


def test_every_family_walks_the_declared_rung_sequence(receipt: dict) -> None:
    names = [name for name, _ in ladder.LADDER_RUNGS]
    for entry in receipt["families"]:
        assert [rung["rung"] for rung in entry["rungs"]] == names
        for rung in entry["rungs"]:
            assert rung["status"] in {"pass", "reject", "blocked"}


def test_the_acceleration_sector_blocks_on_the_cubic_g3_cone_adapter(receipt: dict) -> None:
    block = receipt["counts"]["by_screening_family"]["acceleration"]
    assert block["families"] == 46
    assert block["sector_verdicts"] == {
        "SECTOR_BLOCKED:principal_symbol_hyperbolicity:"
        "missing_adapter:cubic_g3_uniform_weak_field_cone": 46
    }
    entry = next(
        item for item in receipt["families"] if item["screening_family"] == "acceleration"
    )
    hyperbolic = next(
        rung for rung in entry["rungs"] if rung["rung"] == "principal_symbol_hyperbolicity"
    )
    assert hyperbolic["status"] == "blocked"
    assert hyperbolic["evidence"]["blocker"] == (
        "missing_adapter:cubic_g3_uniform_weak_field_cone"
    )
    assert "modified_harmonic_uniform_bound_required" in hyperbolic["evidence"][
        "formulation_routes"
    ]


def test_the_curvature_sector_clears_every_implemented_rung(receipt: dict) -> None:
    block = receipt["counts"]["by_screening_family"]["curvature"]
    assert block["families"] == 25
    assert block["sector_verdicts"] == {"SECTOR_PASS": 25}
    entry = next(
        item for item in receipt["families"] if item["screening_family"] == "curvature"
    )
    assert {rung["status"] for rung in entry["rungs"]} == {"pass"}
    assert entry["highest_rung_reached"] == "positive_energy_hamiltonian"


def test_every_rung_pass_rests_on_a_closed_on_shell_certificate(receipt: dict) -> None:
    for entry in receipt["families"]:
        passing = [rung for rung in entry["rungs"] if rung["status"] == "pass"]
        if not passing:
            continue
        for certificate in entry["background_certificates"]:
            assert certificate["certificate"]["status"] == "pass_interval_certified"
            assert certificate["certificate"]["errors"] == []


def test_a_rung_cannot_pass_without_a_certificate() -> None:
    symbolic = {"symbolic_status": "pass", "condition": "1 > 0", "domain": "x > 0"}
    assert ladder._combine(symbolic, None, None) == "blocked"
    assert ladder._combine(symbolic, "reject", None) == "blocked"
    assert ladder._combine(symbolic, "pass_interval_certified", None) == "pass"
    assert ladder._combine({**symbolic, "symbolic_status": "reject"}, None, None) == "reject"
    assert ladder._combine(symbolic, None, "violated") == "reject"


def test_an_illegal_rung_status_is_refused() -> None:
    with pytest.raises(ladder.V3FormalLadderError, match="illegal rung status"):
        ladder._rung("ghost_freedom", "unresolved", "why", {})


def test_the_sign_verdict_proves_positivity_and_negativity_and_admits_neither() -> None:
    assert ladder._sign_verdict("1", [])["symbolic_status"] == "pass"
    assert ladder._sign_verdict("-1", [])["symbolic_status"] == "reject"
    conditional = ladder._sign_verdict("1 - 48*x", [])
    assert conditional["symbolic_status"] == "conditional"
    assert conditional["condition"].endswith("> 0")


def test_free_coefficients_derive_the_parameter_inequality() -> None:
    verdict = ladder._sign_verdict("6*c_K*x + 1", ["c_K"])
    assert verdict["symbolic_status"] == "pass"
    assert verdict["unconstrained_status"] == "conditional"
    assert "c_K real" in verdict["unconstrained_domain"]


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def test_every_verdict_kind_is_reachable(receipt: dict) -> None:
    kinds = {entry["verdict"].split(":", 1)[0] for entry in receipt["families"]}
    control_kinds = {
        entry["observed_verdict"].split(":", 1)[0]
        for entry in receipt["controls"]["ladder_controls"].values()
    }
    assert kinds == {"BLOCKED"}
    assert control_kinds == {"FORMAL_PASS", "FORMAL_REJECT"}


def test_a_rejection_outranks_a_blocker() -> None:
    materialization = {"full_lift_blockers": ["missing_adapter:nonlocal_fractional_operator"]}
    rungs = [
        {"rung": "ghost_freedom", "status": "blocked", "evidence": {"blocker": None}},
        {"rung": "tensor_sector", "status": "reject", "evidence": {}},
    ]
    assert ladder.ladder_verdict(materialization, rungs) == "FORMAL_REJECT:tensor_sector"
    assert ladder.sector_verdict(rungs) == "SECTOR_BLOCKED:ghost_freedom:missing_adapter:unnamed"


def test_a_fully_expressible_and_fully_passing_family_is_a_formal_pass() -> None:
    materialization = {"full_lift_blockers": []}
    rungs = [
        {"rung": name, "status": "pass", "evidence": {}} for name, _ in ladder.LADDER_RUNGS
    ]
    assert ladder.ladder_verdict(materialization, rungs) == "FORMAL_PASS"
    assert ladder.sector_verdict(rungs) == "SECTOR_PASS"


def test_survivor_arithmetic_closes(receipt: dict) -> None:
    counts = receipt["counts"]
    assert counts["families_in"] == 71
    assert counts["not_eliminated"] + counts["eliminated"] == 71
    assert counts["sector_pass"] == 25
    assert sum(counts["per_rung_first_block"].values()) == 46
    assert sum(counts["per_rung_first_elimination"].values()) == 0


def test_no_screening_family_was_eliminated_entirely(receipt: dict) -> None:
    assert receipt["counts"]["screening_families_eliminated_entirely"] == []
    assert receipt["counts"]["screening_families_with_any_elimination"] == []


def test_survival_correlates_with_the_screening_axis_and_nothing_else(receipt: dict) -> None:
    correlation = receipt["counts"]["sector_verdict_by_kernel_axis"]
    screen_axis = correlation["screen"]
    assert set(screen_axis["curvature"]) == {"SECTOR_PASS"}
    assert all(key.startswith("SECTOR_BLOCKED") for key in screen_axis["acceleration"])
    for axis in ("L1", "L2", "p", "t", "w_yukawa", "w_power", "local"):
        mixed = [row for row in correlation[axis].values() if len(row) > 1]
        assert mixed, f"axis {axis} unexpectedly separates the verdicts"


def test_the_surviving_lagrangians_are_listed_with_their_conditions(receipt: dict) -> None:
    survivors = {block["sector_id"]: block for block in receipt["surviving_lagrangians"]}
    assert set(survivors) == {"cubic_galileon_kinetic_braiding", "kmouflage_convex_kessence"}
    galileon = survivors["cubic_galileon_kinetic_braiding"]
    assert galileon["families"] == 46
    assert galileon["normalized_functions"] == {"g2": "x", "g3": "-2*x", "g4": "1/2"}
    assert any("24*x**3" in text for text in galileon["background_conditions"])
    kmouflage = survivors["kmouflage_convex_kessence"]
    assert kmouflage["families"] == 25
    assert any("6*c_K*x + 1 > 0" in text for text in kmouflage["parameter_conditions"])
    assert any("2*c_K*x + 1 > 0" in text for text in kmouflage["parameter_conditions"])


def test_outstanding_repo_blockers_travel_with_every_verdict(receipt: dict) -> None:
    for entry in receipt["families"]:
        assert entry["outstanding_repo_blockers"] == list(ladder.OUTSTANDING_REPO_BLOCKERS)
        assert "global_positive_energy_on_general_nonmaximal_data_unresolved" in (
            entry["outstanding_repo_blockers"]
        )


def test_the_adapter_gap_report_names_every_adapter(receipt: dict) -> None:
    report = receipt["adapter_gap_report"]
    codes = {item["code"] for item in report["adapters_to_build"]}
    assert codes == set(ladder.BLOCKERS)
    assert report["build_order"][0] == "missing_adapter:nonlocal_fractional_operator"
    for item in report["adapters_to_build"]:
        assert item["adapter_to_build"]
        assert item["why"]


def test_the_adapter_gap_report_counts_both_block_sites(receipt: dict) -> None:
    counted = {item["code"]: item for item in receipt["adapter_gap_report"]["adapters_to_build"]}
    universal = counted["missing_adapter:nonlocal_fractional_operator"]
    assert universal["families_blocked_at_materialization"] == 71
    assert universal["families_blocked_at_a_ladder_rung"] == 0
    cubic = counted["missing_adapter:cubic_g3_uniform_weak_field_cone"]
    assert cubic["families_blocked_at_materialization"] == 0
    assert cubic["families_blocked_at_a_ladder_rung"] == 46
    assert receipt["counts"]["sector_blocked_by_adapter"] == {
        "missing_adapter:cubic_g3_uniform_weak_field_cone": 46
    }


# ---------------------------------------------------------------------------
# Receipt: determinism, tamper, validation, claims
# ---------------------------------------------------------------------------


def test_the_claims_block_is_the_declared_one(receipt: dict) -> None:
    assert receipt["claims"] == {
        "corpus_absence_establishes_novelty": False,
        "first_principles_derivation_pending": True,
        "formal_pass_is_not_physical_validation": True,
        "real_data_used": False,
        "synthetic_controls_only": True,
    }


def test_the_receipt_carries_no_floats(receipt: dict, sealed: dict) -> None:
    ladder._no_floats(receipt)
    ladder._no_floats(sealed)
    with pytest.raises(ladder.V3FormalLadderError, match="float in receipt"):
        ladder._no_floats({"a": [{"b": 1.5}]})


def test_the_receipt_is_deterministic(receipt: dict, sealed: dict) -> None:
    assert receipt["content_sha256"] == sealed["content_sha256"]
    body = {key: item for key, item in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == canonical_sha256(body)


def test_the_sealed_receipt_validates(sealed: dict) -> None:
    ladder.validate_receipt(sealed)


def test_a_reseal_after_tamper_is_still_refused(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    tampered["counts"]["formal_pass"] = 71
    with pytest.raises(ladder.V3FormalLadderError, match="receipt seal changed"):
        ladder.validate_receipt(tampered)
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(ladder.V3FormalLadderError, match="aggregate counts do not replay"):
        ladder.validate_receipt(tampered)


def test_a_doctored_surviving_lagrangian_block_is_refused(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    for block in tampered["surviving_lagrangians"]:
        block["parameter_conditions"] = []
        block["background_conditions"] = []
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(ladder.V3FormalLadderError, match="surviving Lagrangians do not replay"):
        ladder.validate_receipt(tampered)


def test_promoting_a_blocked_family_to_a_pass_is_refused(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    tampered["families"][0]["verdict"] = "FORMAL_PASS"
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(ladder.V3FormalLadderError, match="blocked family cannot be a FORMAL_PASS"):
        ladder.validate_receipt(tampered)


def test_an_unknown_blocker_or_rung_is_refused(sealed: dict) -> None:
    for verdict, message in (
        ("BLOCKED:missing_adapter:invented", "unknown adapter"),
        ("FORMAL_REJECT:invented_rung", "unknown rung"),
        ("MAYBE", "unknown verdict"),
    ):
        tampered = json.loads(json.dumps(sealed))
        tampered["families"][0]["verdict"] = verdict
        body = {key: item for key, item in tampered.items() if key != "content_sha256"}
        tampered["content_sha256"] = canonical_sha256(body)
        with pytest.raises(ladder.V3FormalLadderError, match=message):
            ladder.validate_receipt(tampered)


def test_a_silenced_control_is_refused(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    tampered["controls"]["ladder_controls"]["wrong_sign_ghost"]["observed_verdict"] = "FORMAL_PASS"
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(ladder.V3FormalLadderError, match="did not fire in the receipt"):
        ladder.validate_receipt(tampered)


def test_a_changed_claim_is_refused(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    tampered["claims"]["formal_pass_is_not_physical_validation"] = False
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(ladder.V3FormalLadderError, match="claims block changed"):
        ladder.validate_receipt(tampered)


def test_the_receipt_binds_the_v3_content_hash_and_the_representative_set(sealed: dict) -> None:
    screen = ladder.load_screen_receipt(ROOT)
    assert sealed["inputs"]["screen_receipt"]["content_sha256"] == screen["content_sha256"]
    artifact = json.loads((ROOT / ladder.REPRESENTATIVES_PATH).read_text(encoding="utf-8"))
    assert sealed["inputs"]["family_representatives"]["content_sha256"] == (
        artifact["content_sha256"]
    )
    assert sealed["inputs"]["polynomial_scalar_tensor_ir"]["file_sha256"] == (
        ladder.POLYNOMIAL_IR_FILE_SHA256
    )


def test_a_moved_polynomial_ir_binding_is_refused(sealed: dict) -> None:
    tampered = json.loads(json.dumps(sealed))
    tampered["inputs"]["polynomial_scalar_tensor_ir"]["file_sha256"] = "0" * 64
    body = {key: item for key, item in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(ladder.V3FormalLadderError, match="polynomial IR binding changed"):
        ladder.validate_receipt(tampered)


def test_the_cli_validates_the_sealed_receipt() -> None:
    assert ladder.main(["--root", str(ROOT), "--validate-checked"]) == 0


def test_the_cli_refuses_to_overwrite_an_immutable_receipt(tmp_path, sealed: dict) -> None:
    target = tmp_path / "receipt.json"
    ladder._write(sealed, target)
    ladder._write(sealed, target)
    with pytest.raises(ladder.V3FormalLadderError, match="immutable receipt"):
        ladder._write({**sealed, "decision": "changed"}, target)
