from __future__ import annotations

import copy

import pytest

from sigma_theory_compiler import (
    open_gravity_differential_propagation_gw170817_source_preflight_v1 as preflight,
)


def test_exact_three_official_products_are_frozen() -> None:
    config = preflight.load_config()
    rows = config["products"]
    assert [row["detector"] for row in rows] == ["H1", "L1", "V1"]
    assert [row["content_length_bytes"] for row in rows] == [
        125217658,
        124266501,
        129470892,
    ]
    assert [row["published_md5"] for row in rows] == [
        "1a1cca3fb28686d5798539468a99dbae",
        "dbbde824db6df6a9f653db374fc5c88c",
        "8ea80f93257a292d82f0af497e2a4cff",
    ]
    assert all(row["sha256"] is None and row["payload_opened"] is False for row in rows)


def test_predecessor_and_package_are_hash_bound() -> None:
    assert set(preflight.validate_predecessor(preflight.load_config())) == {
        "config",
        "module",
        "test",
        "receipt",
    }
    assert set(preflight.validate_package_bindings()) == {
        "config_raw_sha256",
        "config_content_sha256",
        "module_semantic_sha256",
        "test_raw_sha256",
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("products", 0, "payload_opened"), True),
        (("products", 1, "sha256"), "0" * 64),
        (("products", 2, "content_length_bytes"), 1),
        (("execution_gate", "scoring_authority"), True),
        (("claim_boundary", "observational_fit_tested"), True),
        (("access", "builder_payload_files_opened"), 1),
    ],
)
def test_material_config_mutations_fail_closed(path: tuple[object, ...], value: object) -> None:
    mutated = copy.deepcopy(preflight.load_config())
    target = mutated
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(preflight.SourcePreflightError):
        preflight.validate_config(mutated)


def test_receipt_is_exact_and_claims_no_data_fit() -> None:
    receipt = preflight.build_receipt()
    preflight.validate_receipt(receipt)
    assert receipt["decision"] == "PASS_SOURCE_METADATA_ONLY__BLOCK_PAYLOAD_ACCESS_AND_SCORING"
    assert receipt["counts"] == {
        "detectors": 3,
        "products": 3,
        "declared_bytes": 378955051,
        "payload_files_opened": 0,
        "payload_rows_opened": 0,
        "scores_computed": 0,
    }
    assert receipt["claim_boundary"]["observational_fit_tested"] is False
    assert receipt["execution_gate"]["real_data_eligible"] is False


def test_receipt_forgery_is_rejected() -> None:
    forged = preflight.build_receipt()
    forged["claim_boundary"]["publication_ready"] = True
    with pytest.raises(preflight.SourcePreflightError):
        preflight.validate_receipt(forged)


def test_atomic_write_and_replay(tmp_path) -> None:
    output = tmp_path / "receipt.json"
    payload = preflight._receipt_bytes(preflight.build_receipt())
    assert preflight._atomic_no_clobber(output, payload) == "CREATED"
    assert output.read_bytes() == payload
    assert preflight._atomic_no_clobber(output, payload) == "EXISTING_IDENTICAL"
