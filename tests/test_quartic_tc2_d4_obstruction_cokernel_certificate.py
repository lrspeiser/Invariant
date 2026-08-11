from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_obstruction_cokernel_certificate import (
    QuarticTC2D4ObstructionCokernelCertificateError,
    build_certificate,
    validate_certificate,
)
from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import _content_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/backgrounds/quartic_tc2_d4_obstruction_cokernel_certificate.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return build_certificate(ROOT, CONFIG)


def _rehash(document: dict) -> dict:
    document.pop("content_sha256", None)
    document["content_sha256"] = _content_hash(document)
    return document


def test_exact_rank_two_alpha_fifth_certificate(artifact: dict) -> None:
    validate_certificate(artifact)
    symbolic = artifact["exact_symbolic_certificate"]
    compression = symbolic["equal_eigenspace_compressions"]["zero_eigenspace"]
    assert compression["factorization"] == "(34816/15)*alpha^5*W"
    assert compression["generic_rank"] == 2
    assert compression["nonzero_entries"] == 4
    assert symbolic["range_certificate"]["compatibility_iff_over_Q_or_R"] == "alpha=0"
    assert symbolic["exact_candidate_gap"]["interval"] == "[1088/15,34816/15]"
    assert artifact["counts"]["candidate_obstructions_certified"] == 12
    assert {row["a10"] for row in symbolic["candidate_classification"]} == {
        "-1",
        "-1/2",
        "1/2",
        "1",
    }


def test_claims_remain_fail_closed(artifact: dict) -> None:
    claims = artifact["claims"]
    assert claims["canonical_D4_obligation_244_classified"] is True
    assert claims["canonical_D4_obligation_244_compatible"] is False
    assert claims["alternative_lower_jet_homogeneous_completion_ruled_out"] is False
    for key in (
        "full_fourth_jet_range_closed",
        "full_tube_Sylvester_identity",
        "CK1_closed",
        "CK3_closed",
        "TC2_closed",
        "B7_closed",
        "global_H7_closed",
        "lifespan_proved",
    ):
        assert claims[key] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("claims", "alternative_lower_jet_homogeneous_completion_ruled_out"), True),
        (
            (
                "exact_symbolic_certificate",
                "equal_eigenspace_compressions",
                "zero_eigenspace",
                "factorization",
            ),
            "0",
        ),
        (
            ("exact_symbolic_certificate", "candidate_classification", 0, "compatible"),
            True,
        ),
        (
            ("exact_symbolic_certificate", "candidate_classification", 0, "a10"),
            "0",
        ),
        (
            (
                "exact_symbolic_certificate",
                "candidate_classification",
                0,
                "compression_sha256",
            ),
            "0" * 64,
        ),
        (("counts", "candidate_obstructions_certified"), 11),
        (("negative_controls", "cancel_with_c20", "rejected"), False),
    ],
)
def test_validator_rejects_rehashed_tampering(
    artifact: dict, path: tuple[str | int, ...], value: object
) -> None:
    mutated = copy.deepcopy(artifact)
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(QuarticTC2D4ObstructionCokernelCertificateError):
        validate_certificate(_rehash(mutated))


def test_source_binding_tamper_fails_before_symbolic_replay(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["obstruction_chunk"]["file_sha256"] = "0" * 64
    config.pop("content_sha256")
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "tampered-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticTC2D4ObstructionCokernelCertificateError):
        build_certificate(ROOT, path)
