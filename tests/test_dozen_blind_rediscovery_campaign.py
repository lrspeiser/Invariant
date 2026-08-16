from __future__ import annotations

import copy
import io
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.dozen_blind_rediscovery_campaign import (
    CLAIMS,
    CONFIG_PATH,
    NON_CLAIM_NOTE,
    OUTPUT_PATH,
    TARGETS_PATH,
    DozenBlindError,
    _SealedTargetsGuard,
    _unseal_targets,
    build_artifacts,
    validate_artifacts,
    validate_campaign,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))

EXPECTED_VERDICTS = {
    "catalan_ratio": "REDISCOVERED_EXACT",
    "fibonacci": "REDISCOVERED_EXACT",
    "gauss_triangular": "REDISCOVERED_WITH_PROOF",
    "geometric_sum3": "REDISCOVERED_EXACT",
    "handshake": "REDISCOVERED_WITH_PROOF",
    "hanoi_mersenne": "REDISCOVERED_EXACT",
    "hockey_stick_c3": "REDISCOVERED_WITH_PROOF",
    "lucas": "REDISCOVERED_EXACT",
    "nicomachus_cubes": "REDISCOVERED_WITH_PROOF",
    "pell": "REDISCOVERED_EXACT",
    "square_pyramidal": "REDISCOVERED_WITH_PROOF",
    "weighted_geometric": "REDISCOVERED_EXACT",
}
EXPECTED_DISCOVERIES = {
    "catalan_ratio": "a(n+1)/a(n) = ((4)*n + (2))/(n + (2))",
    "fibonacci": "a(n) = (1)*a(n-1) + (1)*a(n-2)",
    "gauss_triangular": "a(n) = 1/2*n^1 + 1/2*n^2",
    "geometric_sum3": "a(n) = -1/2 + 1/2*3^n",
    "handshake": "a(n) = C(n,2)",
    "hanoi_mersenne": "a(n) = -1 + 2^n",
    "hockey_stick_c3": "a(n) = C(n,3)",
    "lucas": "a(n) = (1)*a(n-1) + (1)*a(n-2)",
    "nicomachus_cubes": "a(n) = 1/4*n^2 + 1/2*n^3 + 1/4*n^4",
    "pell": "a(n) = (2)*a(n-1) + (1)*a(n-2)",
    "square_pyramidal": "a(n) = 1/6*n^1 + 1/2*n^2 + 1/3*n^3",
    "weighted_geometric": "a(n) = 2 - 2*2^n + 2*n*2^n",
}
EXPECTED_HOLDOUT = {
    "catalan_ratio": 12,
    "fibonacci": 8,
    "gauss_triangular": 13,
    "geometric_sum3": 12,
    "handshake": 14,
    "hanoi_mersenne": 12,
    "hockey_stick_c3": 14,
    "lucas": 8,
    "nicomachus_cubes": 11,
    "pell": 8,
    "square_pyramidal": 12,
    "weighted_geometric": 11,
}
POLYNOMIAL_WORLDS = (
    "gauss_triangular",
    "square_pyramidal",
    "nicomachus_cubes",
    "handshake",
    "hockey_stick_c3",
)
EXPECTED_B6_ROUTES = {
    "gauss_triangular": 2,
    "handshake": 1,
    "hockey_stick_c3": 0,
    "nicomachus_cubes": 2,
    "square_pyramidal": 2,
}


@pytest.fixture(scope="module")
def artifacts() -> dict:
    return build_artifacts(ROOT)


@pytest.fixture(scope="module")
def checked(artifacts: dict) -> dict:
    campaign = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    assert campaign == artifacts["campaign"]
    worlds = {}
    for row in campaign["world_results"]:
        receipt = json.loads((ROOT / row["world_receipt_path"]).read_text(encoding="utf-8"))
        assert receipt == artifacts["worlds"][row["classical_id"]]
        assert receipt["content_sha256"] == row["world_receipt_sha256"]
        worlds[row["classical_id"]] = receipt
    validate_artifacts(campaign, worlds, root=ROOT)
    return {"campaign": campaign, "worlds": worlds}


def _row(campaign: dict, classical_id: str) -> dict:
    return next(
        row for row in campaign["world_results"] if row["classical_id"] == classical_id
    )


def _reseal(value: dict) -> None:
    value["content_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


def test_twelve_worlds_rediscovered_with_measured_verdict_table(checked: dict) -> None:
    campaign = checked["campaign"]
    assert campaign["decision"] == "PASS"
    verdicts = {row["classical_id"]: row["verdict"] for row in campaign["world_results"]}
    assert verdicts == EXPECTED_VERDICTS
    rediscovered = sum(
        1 for verdict in verdicts.values() if verdict.startswith("REDISCOVERED")
    )
    assert rediscovered == 12
    assert rediscovered >= CONFIG["policies"]["minimum_rediscoveries_for_pass"] == 8
    assert campaign["counts"] == {
        "holdout_confirmations_total": 135,
        "lean_sources_emitted": 12,
        "missed": 0,
        "partial": 0,
        "post_unseal_generation_events": 0,
        "prover_receipts": 12,
        "rediscovered_exact": 7,
        "rediscovered_total": 12,
        "rediscovered_with_proof": 5,
        "stage_receipts": 29,
        "target_fixture_reads": 1,
        "target_fixture_reads_denied_before_unseal": 1,
        "worlds": 12,
    }


def test_discovered_statements_and_holdout_confirmations(checked: dict) -> None:
    campaign = checked["campaign"]
    for classical_id, statement in EXPECTED_DISCOVERIES.items():
        row = _row(campaign, classical_id)
        assert row["discovered_statement"] == statement
        assert row["holdout_confirmations"] == EXPECTED_HOLDOUT[classical_id]
        # Every world keeps at least six untouched confirmations beyond its parameters.
        assert row["holdout_confirmations"] >= 6


def test_chronology_denied_probe_and_zero_prereads(checked: dict) -> None:
    chronology = checked["campaign"]["chronology"]
    assert chronology["unseal_batches"] == 1
    probe = chronology["denied_probe"]
    assert probe["attempted_target_reads"] == 1
    assert probe["denied_target_reads"] == 1
    assert probe["denied_content_bytes_exposed"] == 0
    assert probe["denied_paths"] == [TARGETS_PATH]
    assert probe["denied_surfaces"] == ["pathlib.Path.open"]
    assert probe["enforcement_surfaces"] == ["builtins.open", "io.open", "pathlib.Path.open"]
    assert len(chronology["phase_a_root"]) == 64
    assert [event["target_reads"] for event in chronology["events"]] == [0, 0, 0, 0, 0, 1, 1]
    unseal_event = chronology["events"][5]
    assert unseal_event["event"] == "atomic_target_unseal"
    for world in checked["worlds"].values():
        assert world["phase_a_root"] == chronology["phase_a_root"]


def test_sealed_guard_denies_every_read_surface_in_process() -> None:
    target = ROOT / TARGETS_PATH
    guard = _SealedTargetsGuard(ROOT)
    with guard:
        with pytest.raises(PermissionError):
            target.read_bytes()
        # Each patched surface is exercised on purpose; the guard raises before any
        # file object exists, so no context manager can apply.
        with pytest.raises(PermissionError):
            open(target, "rb")  # noqa: SIM115
        with pytest.raises(PermissionError):
            io.open(target, "rb")  # noqa: SIM115, UP020
        # Non-target files stay readable while the guard is active.
        assert (ROOT / CONFIG_PATH).read_bytes()
    certificate = guard.certificate()
    assert certificate["attempted_target_reads"] == 3
    assert certificate["denied_target_reads"] == 3
    assert certificate["denied_content_bytes_exposed"] == 0
    assert certificate["denied_surfaces"] == ["pathlib.Path.open", "builtins.open", "io.open"]
    # The guard restores the originals on exit.
    assert target.read_bytes()


def test_commitment_mismatch_fails_closed() -> None:
    config = copy.deepcopy(CONFIG)
    config["worlds"][0]["sealed_target_sha256"] = "0" * 64
    with pytest.raises(DozenBlindError, match="target commitment did not open"):
        _unseal_targets(ROOT, config, "a" * 64)


def test_target_fixture_binding_tamper_fails_closed() -> None:
    config = copy.deepcopy(CONFIG)
    config["target_fixture"]["content_sha256"] = "1" * 64
    with pytest.raises(DozenBlindError, match="target fixture content changed"):
        _unseal_targets(ROOT, config, "a" * 64)


def test_unseal_before_phase_a_seal_fails_closed() -> None:
    with pytest.raises(DozenBlindError, match="before candidate freeze"):
        _unseal_targets(ROOT, CONFIG, "not-a-sealed-phase-a-root")


def test_config_is_blind_to_classical_identities() -> None:
    config_text = (ROOT / CONFIG_PATH).read_text(encoding="utf-8")
    for classical_id in EXPECTED_VERDICTS:
        assert classical_id not in config_text
    for fragment in ("gauss", "fibonacci", "lucas", "pell", "catalan", "binomial", "mersenne"):
        assert fragment not in config_text.lower()
    fixture = json.loads((ROOT / TARGETS_PATH).read_text(encoding="utf-8"))
    for world, target in zip(CONFIG["worlds"], fixture["targets"], strict=True):
        assert world["world_id"] == target["world_id"]
        assert canonical_sha256(target) == world["sealed_target_sha256"]
        assert len(target["salt"]) == 32


def test_determinism_exact_replay(artifacts: dict) -> None:
    replayed = build_artifacts(ROOT)
    assert replayed["campaign"] == artifacts["campaign"]
    assert replayed["worlds"] == artifacts["worlds"]


def test_receipt_tamper_fails_closed(artifacts: dict) -> None:
    broken_seal = copy.deepcopy(artifacts["campaign"])
    broken_seal["content_sha256"] = "0" * 64
    with pytest.raises(DozenBlindError, match="seal changed"):
        validate_campaign(broken_seal, root=ROOT)
    resealed_lie = copy.deepcopy(artifacts["campaign"])
    row = next(
        entry
        for entry in resealed_lie["world_results"]
        if entry["classical_id"] == "hanoi_mersenne"
    )
    row["verdict"] = "REDISCOVERED_WITH_PROOF"
    _reseal(resealed_lie)
    with pytest.raises(DozenBlindError, match="exact replay changed"):
        validate_campaign(resealed_lie, root=ROOT)


def test_fibonacci_and_lucas_closed_form_non_claim_honesty(checked: dict) -> None:
    for classical_id, seeds in (("fibonacci", [0, 1]), ("lucas", [2, 1]), ("pell", [0, 1])):
        world = checked["worlds"][classical_id]
        phase_a = world["phase_a"]
        assert phase_a["closed_form_claimed"] is False
        stages = {stage["stage_id"]: stage for stage in phase_a["stages"]}
        assert stages["b1_basis_synthesis"]["decision"] == "BLOCK"
        assert stages["b7_structural_repair"]["decision"] == "BLOCK"
        conjectures = {
            row["kind"]: row
            for row in stages["b3_conjecture_generation"]["receipt"]["conjectures"]
        }
        assert conjectures["closed_form"]["status"] == "NOT_PROPOSED"
        assert conjectures["linear_recurrence"]["status"] == "SURVIVED"
        candidate = phase_a["candidate"]
        assert candidate["kind"] == "linear_recurrence"
        assert candidate["order"] == 2
        assert candidate["seeds"] == seeds
        assert world["unseal"]["verdict"] == "REDISCOVERED_EXACT"
        assert world["unseal"]["note"] == NON_CLAIM_NOTE
    fibonacci = checked["worlds"]["fibonacci"]["phase_a"]["candidate"]
    assert fibonacci["coefficients"] == [
        {"denominator": 1, "numerator": 1},
        {"denominator": 1, "numerator": 1},
    ]
    pell = checked["worlds"]["pell"]["phase_a"]["candidate"]
    assert pell["coefficients"] == [
        {"denominator": 1, "numerator": 2},
        {"denominator": 1, "numerator": 1},
    ]


def test_catalan_declared_transformation_logged_and_ratio_law_found(checked: dict) -> None:
    world = checked["worlds"]["catalan_ratio"]
    declaration = world["declared_transformations"][0]
    assert declaration["transformation_id"] == "adjacent_term_ratio"
    record = world["phase_a"]["transformation_records"][0]
    assert record["transformation_id"] == "adjacent_term_ratio"
    assert record["rows_consumed"] == 16
    assert record["rows_produced"] == 15
    assert record["input_rows_sha256"] == canonical_sha256(world["public_rows"])
    assert record["output_rows_sha256"] == canonical_sha256(record["output_rows"])
    stages = {stage["stage_id"]: stage for stage in world["phase_a"]["stages"]}
    assert stages["b1_basis_synthesis"]["decision"] == "BLOCK"
    assert stages["b2_nonlinear_coefficient_search"]["decision"] == "PASS"
    candidate = world["phase_a"]["candidate"]
    assert candidate["kind"] == "ratio_law"
    assert candidate["model_id"] == "linear_fractional"
    assert candidate["parameters"] == {
        "a": {"denominator": 1, "numerator": 4},
        "b": {"denominator": 1, "numerator": 2},
        "d": {"denominator": 1, "numerator": 2},
    }
    assert world["unseal"]["verdict"] == "REDISCOVERED_EXACT"
    assert world["unseal"]["comparison"]["equivalent"] is True


def test_lean_emitted_for_every_polynomial_world(checked: dict) -> None:
    for classical_id in POLYNOMIAL_WORLDS:
        world = checked["worlds"][classical_id]
        phase_a = world["phase_a"]
        assert phase_a["lean_emitted"] is True
        routes = {route["route"]: [] for route in phase_a["prover_routes"]}
        for route in phase_a["prover_routes"]:
            routes[route["route"]].append(route)
        (b5,) = routes["b5_lemma_decomposition"]
        assert b5["decision"] == "DECOMPOSED"
        assert b5["receipt"]["lean_source"].startswith("import Std.Tactic")
        assert b5["receipt"]["kernel_verified"] is False
        assert b5["lean_source_emitted"] is True
        b6_routes = routes.get("b6_quantified_inequality", [])
        assert len(b6_routes) == EXPECTED_B6_ROUTES[classical_id]
        for route in b6_routes:
            assert route["decision"] == "PROVED_LOCALLY"
            assert route["receipt"]["lean_source"].startswith("import Std.Tactic")
        assert world["unseal"]["verdict"] == "REDISCOVERED_WITH_PROOF"
    for classical_id in sorted(set(EXPECTED_VERDICTS) - set(POLYNOMIAL_WORLDS)):
        assert checked["worlds"][classical_id]["phase_a"]["lean_emitted"] is False


def test_companion_transforms_are_declared_for_shifted_binomials(checked: dict) -> None:
    handshake = checked["worlds"]["handshake"]["phase_a"]["prover_routes"][0]
    assert handshake["companion"] == {
        "base_value": 0,
        "closed_form_coefficients_ascending": [0, 1, 1],
        "scale": 2,
        "shift": 1,
        "step_coefficients_ascending": [2, 2],
    }
    hockey = checked["worlds"]["hockey_stick_c3"]["phase_a"]["prover_routes"][0]
    assert hockey["companion"] == {
        "base_value": 0,
        "closed_form_coefficients_ascending": [0, 2, 3, 1],
        "scale": 6,
        "shift": 2,
        "step_coefficients_ascending": [6, 9, 3],
    }


def test_claims_stay_inside_the_rediscovery_boundary(checked: dict) -> None:
    claims = checked["campaign"]["claims"]
    assert claims == CLAIMS
    assert claims["novelty_claimed"] is False
    assert claims["post_unseal_generation"] is False
    assert claims["machine_found_targets_unaided"] is True
    assert claims["rediscovery_of_classical_results"] is True
    assert claims["kernel_verified_lean"] is False
    assert claims["target_records_read_before_candidate_freeze"] == 0
    assert checked["campaign"]["policies"] == CONFIG["policies"]
    for world in checked["worlds"].values():
        assert world["unseal"]["commitment_opened"] is True
        assert world["sealed_target_sha256"] == canonical_sha256(
            world["unseal"]["target_record"]
        )
