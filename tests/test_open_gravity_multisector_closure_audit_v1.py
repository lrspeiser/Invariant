from __future__ import annotations

import copy
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_multisector_closure_audit_v1 as closure


@pytest.fixture(scope="module")
def packet() -> tuple[dict, dict]:
    config = closure.load_config()
    return config, closure.run_suite(config)


def test_config_and_exact_evidence_chain(packet: tuple[dict, dict]) -> None:
    config, _suite = packet
    closure.validate_config(config)
    assert len(config["evidence_packages"]) == 7
    assert len(config["new_bindings"]) == 3
    assert all(len(row["commit"]) == 40 for row in config["evidence_packages"])
    assert all(len(row["commit"]) == 40 for row in config["new_bindings"])


def test_exact_sector_inventory_and_status_counts(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    assert tuple(row["sector"] for row in config["sector_ledger"]) == closure._SECTORS
    assert suite["sectors"] == 11
    assert suite["status_counts"] == {"PARTIAL": 6, "BLOCKED": 5}


def test_all_twelve_multisector_gates_pass(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    assert list(suite["gates"]) == config["required_gates"]
    assert suite["passed"] == 12
    assert suite["failed"] == 0
    assert all(row["passed"] is True for row in suite["gates"].values())
    assert suite["observational_authority"] is False


def test_matter_identity_is_not_an_on_shell_source_solution(packet: tuple[dict, dict]) -> None:
    _config, suite = packet
    metrics = suite["gates"]["UNIVERSAL_MATTER_SOURCE_IDENTITY_ONLY"]["metrics"]
    assert metrics == {
        "source_identity": True,
        "physical_profile": False,
        "on_shell_background": False,
        "sector_status": "PARTIAL_UNIVERSAL_CONFORMAL_SOURCE_IDENTITY",
    }


def test_photon_clock_and_redshift_claims_remain_narrow(packet: tuple[dict, dict]) -> None:
    _config, suite = packet
    photon = suite["gates"]["PHOTON_CONFORMAL_CONE_WITH_LENSING_BLOCK"]["metrics"]
    clock = suite["gates"]["CLOCK_ENDPOINT_RULE_WITH_DYNAMIC_REDSHIFT_BLOCK"]["metrics"]
    assert photon["local_conformal_null_cone"] is True
    assert photon["lensing_prediction"] is False
    assert clock["static_endpoint_rule"] is True
    assert clock["extra_static_path_accumulation"] is False
    assert clock["dynamic_redshift_completion"] is False


def test_gw_solar_pulsar_and_cosmology_are_not_promoted(packet: tuple[dict, dict]) -> None:
    _config, suite = packet
    gw = suite["gates"]["GW_NECESSARY_CONE_BOUND_WITHOUT_OBSERVATIONAL_PASS"]["metrics"]
    solar = suite["gates"]["SOLAR_AND_PULSAR_PHYSICAL_GATES_BLOCKED"]["metrics"]
    flrw = suite["gates"]["FLRW_EQUATIONS_WITH_HISTORY_OBSTRUCTION"]["metrics"]
    assert gw["restricted_conditions"] is True and gw["gw_observational_pass"] is False
    assert solar["solar_viability"] is False and solar["binary_radiation_prediction"] is False
    assert flrw["background_equations"] is True and flrw["healthy_history"] is False


def test_capture_distinguishes_binding_from_irreversibility(packet: tuple[dict, dict]) -> None:
    _config, suite = packet
    metrics = suite["gates"]["CAPTURE_REQUIRES_ENERGY_ENTROPY_RECEIVER"]["metrics"]
    assert metrics["deeper_conservative_binding_possible"] is True
    assert metrics["irreversible_capture_derived"] is False
    assert metrics["receiver_defined"] is False


def test_hamiltonian_and_constraint_results_preserve_health_obstructions(
    packet: tuple[dict, dict],
) -> None:
    _config, suite = packet
    metrics = suite["gates"]["HAMILTONIAN_PRINCIPAL_AND_CONSTRAINT_PARTIALS"]["metrics"]
    assert metrics["external_metric_principal_symbol_partial"] is True
    assert metrics["restricted_hamiltonian"] is True
    assert metrics["conditional_adm_constraint_identity"] is True
    assert metrics["full_hyperbolicity"] is False
    assert metrics["physical_hamiltonian_positivity"] is False


def test_gp01_and_quantum_claims_remain_blocked(packet: tuple[dict, dict]) -> None:
    _config, suite = packet
    gp01 = suite["gates"]["GP01_HAS_NO_COMMON_MULTI_SECTOR_ACTION"]["metrics"]
    quantum = suite["gates"]["QUANTUM_CLAIMS_BLOCKED"]["metrics"]
    assert gp01["static_field_mechanics"] is True
    assert gp01["common_multisector_action"] is False
    assert quantum["typed_quantum_cards"] == 13
    assert quantum["quantized_gp01"] is False
    assert quantum["nonclassical_witness"] is False


def test_evidence_receipts_are_read_only_after_hash_verification(
    packet: tuple[dict, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _suite = packet
    original = closure.file_sha256
    verified: set[Path] = set()
    reads: list[Path] = []

    def tracking_hash(path: Path) -> str:
        verified.add(path.resolve())
        return original(path)

    original_read = closure._read_json

    def tracking_read(path: Path, label: str) -> dict:
        assert path.resolve() in verified
        reads.append(path.resolve())
        return original_read(path, label)

    monkeypatch.setattr(closure, "file_sha256", tracking_hash)
    monkeypatch.setattr(closure, "_read_json", tracking_read)
    evidence = closure.load_evidence(config)
    assert len(evidence) == 10
    assert len(reads) == 10


@pytest.mark.parametrize(
    "section",
    (
        "purpose",
        "evidence_packages",
        "new_bindings",
        "sector_ledger",
        "required_gates",
        "access_contract",
        "claim_boundary",
    ),
)
def test_every_semantic_section_is_hard_pinned(packet: tuple[dict, dict], section: str) -> None:
    config, _suite = packet
    changed = copy.deepcopy(config)
    changed[section] = None
    with pytest.raises(closure.ClosureAuditError, match="config semantics changed"):
        closure.validate_config(changed)


def test_noncanonical_receipt_path_rejected_before_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reads = 0

    def forbidden(*_args: object, **_kwargs: object) -> dict:
        nonlocal reads
        reads += 1
        return {}

    monkeypatch.setattr(closure, "OUTPUT_PATH", tmp_path / "private-response.json")
    monkeypatch.setattr(closure, "_read_json", forbidden)
    with pytest.raises(closure.ClosureAuditError, match="output path changed"):
        closure.validate_receipt()
    assert reads == 0


def test_receipt_rebuild_and_coherent_forgery_rejection(packet: tuple[dict, dict]) -> None:
    _config, _suite = packet
    receipt = closure.build_receipt()
    closure.validate_receipt_payload(receipt)
    forged = copy.deepcopy(receipt)
    forged["suite"]["observational_authority"] = True
    forged["content_sha256"] = closure.content_sha256(
        {key: value for key, value in forged.items() if key != "content_sha256"}
    )
    with pytest.raises(closure.ClosureAuditError, match="not reproducible"):
        closure.validate_receipt_payload(forged)


def test_zero_access_and_plain_language_claim_ceiling(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    receipt = closure.build_receipt()
    assert all(value == 0 for value in receipt["access_accounting"].values())
    assert all(row["plain_result"] for row in suite["sector_ledger"])
    assert "new cumulative redshift" in config["claim_boundary"]["does_not_establish"]
    assert "irreversible capture" in config["claim_boundary"]["does_not_establish"]
    assert "quantum gravity" in config["claim_boundary"]["does_not_establish"]
