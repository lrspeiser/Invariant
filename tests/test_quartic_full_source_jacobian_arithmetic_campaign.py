import json
from pathlib import Path

from sigma_theory_compiler.quartic_full_source_jacobian_arithmetic_campaign import (
    run_quartic_full_source_jacobian_arithmetic_campaign,
)
from sigma_theory_compiler.quartic_row0_arithmetic_expansion_campaign import (
    _content_hash,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PRINCIPAL = RUNS / "quartic-unspecialized-source-jacobian-campaign" / "campaign.json"
ROWS = tuple(
    RUNS / f"quartic-row{row}-arithmetic-expansion-campaign" / "campaign.json"
    for row in range(5)
) + (
    RUNS / "quartic-rows5-10-arithmetic-expansion-campaign" / "campaign.json",
)
CONFIG = ROOT / "configs" / "backgrounds" / "quartic_full_source_jacobian_arithmetic_campaign.json"
ARTIFACT = RUNS / "quartic-full-source-jacobian-arithmetic-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, ...]:
    return (_load(PRINCIPAL), *map(_load, ROWS), _load(CONFIG))


def test_all_1683_entries_have_exact_arithmetic_roots() -> None:
    result = run_quartic_full_source_jacobian_arithmetic_campaign(*_inputs())
    assert result["status"] == (
        "pass_all_12_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed"
    )
    packet = result["common_principal_arithmetic_packet"]
    assert packet["counts"]["chunks"] == 9
    assert packet["counts"]["entries"] == 1089
    assert packet["counts"]["semantic_or_tensor_operations"] == 0
    assert {node["op"] for node in packet["arithmetic_dag"]["nodes"]} <= set(
        packet["arithmetic_dag"]["allowed_operations"]
    )
    assert all(item["normalized_residual"] == "0" for item in packet["entries"])
    manifest = result["common_full_entry_manifest"]
    assert manifest["lower_entry_count"] == 594
    assert manifest["principal_entry_count"] == 1089
    assert manifest["total_entry_count"] == 1683
    assert {
        (item["source_row"], item["coordinate_column"])
        for item in manifest["entries"]
    } == {(row, column) for row in range(11) for column in range(153)}
    assert result["physical_pencil_J_identity"]["proved"]
    for certificate in result["certificates"]:
        assert certificate["full_11x153_source_Jacobian_entrywise_materialized"]
        assert certificate["physical_pencil_J_identity_proved"]
        assert not certificate["full_component_Frechet_tensors_orders_2_to_4_complete"]
        assert not certificate["paralinearization_remainder_bound_proved"]
        assert not certificate["full_H7_commutator_closed"]
    assert result == _load(ARTIFACT)


def test_reproducible_and_false_claim_or_rebound_provenance_rejects() -> None:
    inputs = _inputs()
    assert run_quartic_full_source_jacobian_arithmetic_campaign(
        *inputs
    ) == run_quartic_full_source_jacobian_arithmetic_campaign(*inputs)
    false_claim = dict(inputs[-1])
    false_claim["declare_component_remainder_proved"] = True
    result = run_quartic_full_source_jacobian_arithmetic_campaign(
        *inputs[:-1], false_claim
    )
    assert result["status"] == "reject"
    assert "cannot be declared" in result["errors"][0]

    row2 = json.loads(json.dumps(inputs[3]))
    row2["upstream_sha256"]["row1_arithmetic"] = "0" * 64
    row2_body = {key: value for key, value in row2.items() if key != "content_sha256"}
    row2["content_sha256"] = _content_hash(row2_body)
    result = run_quartic_full_source_jacobian_arithmetic_campaign(
        inputs[0], inputs[1], inputs[2], row2, *inputs[4:]
    )
    assert result["status"] == "reject"
    assert "provenance chain mismatch" in result["errors"][0]

    principal = json.loads(json.dumps(inputs[0]))
    principal["certificates"][0]["basis_and_injection_provenance"][
        "principal_jet_injection_sha256"
    ] = "1" * 64
    principal_body = {
        key: value for key, value in principal.items() if key != "content_sha256"
    }
    principal["content_sha256"] = _content_hash(principal_body)
    rebound_rows = [json.loads(json.dumps(item)) for item in inputs[1:7]]
    rebound_rows[0]["upstream_sha256"]["principal_source"] = principal[
        "content_sha256"
    ]
    body = {
        key: value
        for key, value in rebound_rows[0].items()
        if key != "content_sha256"
    }
    rebound_rows[0]["content_sha256"] = _content_hash(body)
    for index in range(1, 5):
        rebound_rows[index]["upstream_sha256"][
            f"row{index - 1}_arithmetic"
        ] = rebound_rows[index - 1]["content_sha256"]
        body = {
            key: value
            for key, value in rebound_rows[index].items()
            if key != "content_sha256"
        }
        rebound_rows[index]["content_sha256"] = _content_hash(body)
    rebound_rows[5]["upstream_sha256"]["row4_arithmetic"] = rebound_rows[4][
        "content_sha256"
    ]
    body = {
        key: value
        for key, value in rebound_rows[5].items()
        if key != "content_sha256"
    }
    rebound_rows[5]["content_sha256"] = _content_hash(body)
    result = run_quartic_full_source_jacobian_arithmetic_campaign(
        principal, *rebound_rows, inputs[-1]
    )
    assert result["status"] == "reject"
    assert "basis/J provenance mismatch" in result["errors"][0]
