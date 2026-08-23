from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import external_dataset_challenges as E
from sigma_theory_compiler.claim_specific_prior_art import HTTPResponse
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-08-23T16:00:00Z"


def _seal(value: dict[str, object]) -> None:
    value["content_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


def _nsw_csv() -> bytes:
    lines = ["rownames,data_id,treat,age,educ,black,hisp,marr,nodegree,re74,re75,re78"]
    for index in range(120):
        lines.append(f"{index + 1},fixture,0,30,12,0,0,0,0,0,0,{100 + index}")
    for index in range(120):
        lines.append(f"{index + 121},fixture,1,30,12,0,0,0,0,0,0,{103 + index}")
    return ("\n".join(lines) + "\n").encode()


def _filip_ascii() -> bytes:
    lines = [
        "NIST/ITL StRD",
        "Dataset Name: Filip",
        "Procedure: Linear Least Squares Regression",
    ]
    for index in range(11):
        estimate = "1" if index == 0 else "2" if index == 1 else "0"
        lines.append(f"B{index} {estimate} 0.01")
    lines.extend(
        [
            "Residual Standard Deviation 0.1",
            "R-Squared 0.9",
            "Data: y x",
        ]
    )
    for index in range(82):
        offset = "0.1" if index % 2 == 0 else "-0.1"
        response = 1 + 2 * index
        lines.append(f"{response}{offset[1:]} {index}" if offset.startswith("+") else f"{response + float(offset):.1f} {index}")
    return ("\n".join(lines) + "\n").encode()


WINE_HEADER = (
    '"fixed acidity";"volatile acidity";"citric acid";"residual sugar";'
    '"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";'
    '"pH";"sulphates";"alcohol";"quality"'
)


def _wine_csv(rows: int, *, white: bool) -> bytes:
    values = []
    for index in range(rows):
        fixed = 8 if white else 7
        sulfur = (20 if white else 10) + index % 3
        values.append(f"{fixed};0.3;0.2;2;0.05;12;{sulfur};0.99;3.2;0.5;10;6")
    return (WINE_HEADER + "\n" + "\n".join(values) + "\n").encode()


def _nhefs_csv() -> bytes:
    lines = ["rownames,seqn,qsmk,wt82_71,age,sex"]
    for index in range(1100):
        exposure = index % 2
        outcome = (index % 17) - 5 + exposure * 2
        lines.append(f"{index + 1},{1000 + index},{exposure},{outcome},{30 + index % 40},{index % 2}")
    return ("\n".join(lines) + "\n").encode()


def _responses() -> dict[str, bytes]:
    config = E.load_config(ROOT)
    responses: dict[str, bytes] = {}
    for challenge in config["challenges"]:
        documentation = challenge["documentation"]
        responses[documentation["source_uri"]] = (
            " ".join(documentation["required_markers"]) + " documented external source"
        ).encode()
        for source in challenge["datasets"]:
            if source["source_id"] == "causaldata-nsw-csv":
                body = _nsw_csv()
            elif source["source_id"] == "nist-filip-data":
                body = _filip_ascii()
            elif source["source_id"] == "uci-wine-red":
                body = _wine_csv(1599, white=False)
            elif source["source_id"] == "uci-wine-white":
                body = _wine_csv(4001, white=True)
            elif source["source_id"] == "causaldata-nhefs-csv":
                body = _nhefs_csv()
            else:  # pragma: no cover - the strict config makes this unreachable
                raise AssertionError(source["source_id"])
            responses[source["source_uri"]] = body
    return responses


class FakeTransport:
    def __init__(self, responses: dict[str, bytes]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def __call__(
        self, uri: str, headers: object, timeout_seconds: int, maximum_bytes: int
    ) -> HTTPResponse:
        del headers
        assert timeout_seconds == 30
        self.calls.append(uri)
        body = self.responses.get(uri)
        if body is None:
            return HTTPResponse(status=404, headers={}, body=b"missing")
        assert len(body) <= maximum_bytes
        return HTTPResponse(status=200, headers={"content-type": "text/plain"}, body=body)


def test_fake_external_pack_executes_four_challenges_and_refetches_identically() -> None:
    transport = FakeTransport(_responses())
    receipt = E.build_external_dataset_challenges(
        ROOT, transport=transport, retrieved_utc=NOW
    )
    E.validate_external_dataset_challenges(receipt, ROOT)
    reproduced = E.reproduce_external_dataset_challenges(
        ROOT, receipt, transport=FakeTransport(_responses())
    )
    assert reproduced == receipt
    assert receipt["summary"] == {
        "challenge_kinds": ["intervention", "noisy", "shifted", "unidentifiable"],
        "external_challenges_passed": 4,
        "external_principals": 3,
        "external_source_responses": 9,
        "mutation_controls_rejected": 4,
        "status": "PASS_EXTERNAL_DATASET_CHALLENGES",
        "unique_external_response_hashes": 9,
    }
    assert len(transport.calls) == 9
    by_kind = {item["kind"]: item["evidence"] for item in receipt["results"]}
    assert by_kind["intervention"]["observed_difference_in_means"] == "3"
    assert by_kind["noisy"]["observations"] == 82
    assert by_kind["shifted"]["train_rows"] == 1599
    assert by_kind["shifted"]["deployment_rows"] == 4001
    assert by_kind["unidentifiable"]["distinguishing_intervention_observed"] is False


def test_missing_external_documentation_marker_fails_closed() -> None:
    responses = _responses()
    uri = E.load_config(ROOT)["challenges"][0]["documentation"]["source_uri"]
    responses[uri] = b"generic dataset page without the documented experiment"
    with pytest.raises(E.ExternalDatasetError, match="documentation markers"):
        E.build_external_dataset_challenges(
            ROOT, transport=FakeTransport(responses), retrieved_utc=NOW
        )


def test_unavailable_external_source_fails_closed() -> None:
    responses = _responses()
    uri = E.load_config(ROOT)["challenges"][2]["datasets"][1]["source_uri"]
    del responses[uri]
    with pytest.raises(E.ExternalDatasetError, match="source unavailable"):
        E.build_external_dataset_challenges(
            ROOT, transport=FakeTransport(responses), retrieved_utc=NOW
        )


def test_small_shift_domain_fails_closed() -> None:
    responses = _responses()
    uri = E.load_config(ROOT)["challenges"][2]["datasets"][1]["source_uri"]
    responses[uri] = _wine_csv(20, white=True)
    with pytest.raises(E.ExternalDatasetError, match="domain split is too small"):
        E.build_external_dataset_challenges(
            ROOT, transport=FakeTransport(responses), retrieved_utc=NOW
        )


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("claims", "causal_effect_established", True, "claim boundary"),
        ("release_gate", "level5_eligible", True, "release boundary"),
        (
            "source_signature",
            "cryptographic_signature_verified",
            True,
            "signature or release boundary",
        ),
    ],
)
def test_receipt_cannot_promote_claims_or_unsigned_sources(
    section: str, key: str, value: object, message: str
) -> None:
    receipt = E.build_external_dataset_challenges(
        ROOT, transport=FakeTransport(_responses()), retrieved_utc=NOW
    )
    changed = copy.deepcopy(receipt)
    changed[section][key] = value
    _seal(changed)
    with pytest.raises(E.ExternalDatasetError, match=message):
        E.validate_external_dataset_challenges(changed, ROOT)


def test_refetch_rejects_a_resealed_source_hash_substitution() -> None:
    receipt = E.build_external_dataset_challenges(
        ROOT, transport=FakeTransport(_responses()), retrieved_utc=NOW
    )
    changed = copy.deepcopy(receipt)
    changed["results"][0]["source_evidence"][0]["response_sha256"] = "a" * 64
    hashes = {
        source["response_sha256"]
        for result in changed["results"]
        for source in result["source_evidence"]
    }
    changed["summary"]["unique_external_response_hashes"] = len(hashes)
    _seal(changed)
    E.validate_external_dataset_challenges(changed, ROOT)
    with pytest.raises(E.ExternalDatasetError, match="did not reproduce"):
        E.reproduce_external_dataset_challenges(
            ROOT, changed, transport=FakeTransport(_responses())
        )


def test_stored_live_external_dataset_receipt_validates() -> None:
    path = ROOT / E.OUTPUT_PATH
    if not path.is_file():
        pytest.skip("live external dataset receipt has not been built yet")
    E.validate_external_dataset_challenges(json.loads(path.read_text(encoding="utf-8")), ROOT)
