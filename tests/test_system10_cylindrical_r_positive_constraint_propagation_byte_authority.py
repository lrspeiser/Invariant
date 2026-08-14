from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.materialize_hash_bound_worktree import materialize

ROOT = Path(__file__).resolve().parents[1]
CONFIG_REL = Path("configs/system10_cylindrical_r_positive_constraint_propagation_attempt.json")
SOURCE_REL = Path(
    "src/sigma_theory_compiler/system10_cylindrical_r_positive_constraint_propagation_attempt.py"
)
TEST_REL = Path("tests/test_system10_cylindrical_r_positive_constraint_propagation_attempt.py")
RECEIPT_REL = Path(
    "runs/math/system10-cylindrical-r-positive-constraint-propagation-attempt/receipt.json"
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _as_crlf(raw: bytes) -> bytes:
    lf = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return lf.replace(b"\n", b"\r\n")


def test_exact_git_attributes_pin_portable_checked_bytes() -> None:
    lines = set((ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines())
    assert {
        f"{CONFIG_REL.as_posix()} text eol=lf",
        f"{SOURCE_REL.as_posix()} text eol=lf",
        f"{TEST_REL.as_posix()} text eol=lf",
        f"{RECEIPT_REL.as_posix()} -text -diff",
    } <= lines


def test_materializer_restores_crlf_checkout_to_sealed_lf_source_and_test(
    tmp_path: Path,
) -> None:
    config = json.loads((ROOT / CONFIG_REL).read_text(encoding="utf-8"))
    expected = {
        SOURCE_REL: config["source_evidence"]["source"]["file_sha256"],
        TEST_REL: config["source_evidence"]["test"]["file_sha256"],
    }

    unified = tmp_path / "configs/unified_engine_status.json"
    unified.parent.mkdir(parents=True)
    unified.write_text(json.dumps({"sources": []}), encoding="utf-8")
    copied_config = tmp_path / CONFIG_REL
    copied_config.parent.mkdir(parents=True, exist_ok=True)
    copied_config.write_bytes((ROOT / CONFIG_REL).read_bytes())

    for relative, expected_sha in expected.items():
        authoritative = (ROOT / relative).read_bytes()
        assert _sha(authoritative) == expected_sha
        crlf = _as_crlf(authoritative)
        assert crlf != authoritative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(crlf)
        assert _sha(target.read_bytes()) != expected_sha

    result = materialize(tmp_path, unified)

    assert result["files_rewritten"] == 2
    for relative, expected_sha in expected.items():
        assert (tmp_path / relative).read_bytes() == (ROOT / relative).read_bytes()
        assert _sha((tmp_path / relative).read_bytes()) == expected_sha
