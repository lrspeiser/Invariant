from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.materialize_hash_bound_worktree import materialize

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_REL = Path("configs/authorities/maxwell_proca_action_ir_byte_authority.json")
CONFLICT_REL = Path("configs/covariant_grammar_v3_seed_campaign.json")
SOURCE_CONFLICT_RELS = (
    Path("configs/grammar_v3_formal_preflight_service.json"),
    Path("configs/reviewed_future_parameter_formal_preflight_campaign.json"),
)


def _load(relative: Path) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _content_sha(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    content_sha = document.get("content_sha256")
    assert isinstance(content_sha, str)
    return content_sha


def _binding(document: dict[str, object], *keys: str) -> dict[str, object]:
    current: object = document
    for key in keys:
        assert isinstance(current, dict)
        current = current[key]
    assert isinstance(current, dict)
    return current


def _flat_source_bindings(value: object, path: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, dict):
        if value.get("source_path") == path and isinstance(value.get("source_file_sha256"), str):
            found.append(
                {
                    "path": path,
                    "file_sha256": value["source_file_sha256"],
                }
            )
        for child in value.values():
            found.extend(_flat_source_bindings(child, path))
    elif isinstance(value, list):
        for child in value:
            found.extend(_flat_source_bindings(child, path))
    return found


def test_authority_matches_every_checked_maxwell_consumer() -> None:
    authority = _load(AUTHORITY_REL)
    checked = authority["checked_representation"]
    expected = {
        "path": "runs/formal-controls-v1/action-health/proca_control/action-ir.json",
        "file_sha256": "951044da278d8a0f3aa53c2e1550a9bfc16618989d209e770b08e44b2b9cacf3",
        "content_sha256": "0ec3fd6c233855e1cabf767d35ea12729b1b75dbfb29d409066f4266004c7c0b",
    }
    assert checked == expected
    assert _sha((ROOT / checked["path"]).read_bytes()) == checked["file_sha256"]
    assert _content_sha(ROOT / checked["path"]) == checked["content_sha256"]

    consumers = {
        "configs/maxwell_arbitrary_background_hilbert_stress_divergence_gate.json": (
            "evidence_bindings",
            "proca_action_ir",
        ),
        "configs/maxwell_hilbert_noether_interface_gate.json": (
            "evidence_bindings",
            "proca_action_ir",
        ),
        "configs/universal_matter_coupled_pde_control_pack.json": (
            "evidence_bindings",
            "proca_action_ir",
        ),
        "runs/math/maxwell-arbitrary-background-stress-divergence-gate/receipt.json": (
            "source_bindings",
            "proca_action_ir",
        ),
        "runs/math/maxwell-hilbert-noether-interface-gate/receipt.json": (
            "source_bindings",
            "action_ir",
        ),
        "runs/math/universal-matter-coupled-pde-control-pack/receipt.json": (
            "source_bindings",
            "registered_evidence",
            "proca_action_ir",
        ),
    }
    assert authority["checked_consumers"] == sorted(consumers)
    for relative, keys in consumers.items():
        assert _binding(_load(Path(relative)), *keys) == expected


def test_covariant_crlf_registration_materializes_back_to_checked_lf_bytes(
    tmp_path: Path,
) -> None:
    authority = _load(AUTHORITY_REL)
    checked = authority["checked_representation"]
    covariant_seed = _load(CONFLICT_REL)
    conflicting = next(
        control["action_artifact"]
        for control in covariant_seed["known_answer_controls"]
        if control["control_id"] == "PROCA-GR"
    )

    assert AUTHORITY_REL.as_posix() < CONFLICT_REL.as_posix()
    assert conflicting["path"] == checked["path"]
    assert conflicting["content_sha256"] == checked["content_sha256"]
    assert conflicting["file_sha256"] != checked["file_sha256"]

    source = (ROOT / checked["path"]).read_bytes()
    lf = source.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    assert _sha(lf) == checked["file_sha256"]
    assert _sha(crlf) == conflicting["file_sha256"]

    unified = tmp_path / "configs/unified_engine_status.json"
    unified.parent.mkdir(parents=True)
    unified.write_text(json.dumps({"sources": []}), encoding="utf-8")
    authority_path = tmp_path / AUTHORITY_REL
    authority_path.parent.mkdir(parents=True)
    authority_path.write_bytes((ROOT / AUTHORITY_REL).read_bytes())
    conflict_path = tmp_path / CONFLICT_REL
    conflict_path.write_text(
        json.dumps({"controls": [{"action_artifact": conflicting}]}), encoding="utf-8"
    )
    target = tmp_path / checked["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(crlf)

    result = materialize(tmp_path, unified)

    assert result["files_rewritten"] == 1
    assert target.read_bytes() == lf
    assert _sha(target.read_bytes()) == checked["file_sha256"]
    assert _content_sha(target) == checked["content_sha256"]


def test_authority_matches_every_checked_maxwell_covariant_source_consumer() -> None:
    authority = _load(AUTHORITY_REL)
    checked = authority["checked_covariant_identity_source"]
    expected = {
        "path": "src/sigma_theory_compiler/covariant_identities.py",
        "file_sha256": "ed05fe41d43ec606fa406d236e08c72c8702fa01f2e812a302d5edadba0d86b6",
    }
    assert checked == expected
    assert _sha((ROOT / checked["path"]).read_bytes()) == checked["file_sha256"]

    consumers = {
        "configs/maxwell_arbitrary_background_hilbert_stress_divergence_gate.json": (
            "evidence_bindings",
            "registered_covariant_identity_source",
        ),
        "configs/maxwell_hilbert_noether_interface_gate.json": (
            "evidence_bindings",
            "covariant_identity_source",
        ),
        "runs/math/maxwell-arbitrary-background-stress-divergence-gate/receipt.json": (
            "source_bindings",
            "registered_covariant_identity_source",
        ),
        "runs/math/maxwell-hilbert-noether-interface-gate/receipt.json": (
            "source_bindings",
            "covariant_identity_source",
        ),
    }
    assert authority["checked_covariant_identity_consumers"] == sorted(consumers)
    for relative, keys in consumers.items():
        assert _binding(_load(Path(relative)), *keys) == expected


def test_preflight_lf_registration_materializes_back_to_checked_crlf_source_bytes(
    tmp_path: Path,
) -> None:
    authority = _load(AUTHORITY_REL)
    checked = authority["checked_covariant_identity_source"]
    conflicts = []
    for relative in SOURCE_CONFLICT_RELS:
        assert AUTHORITY_REL.as_posix() < relative.as_posix()
        bindings = _flat_source_bindings(_load(relative), checked["path"])
        assert len(bindings) == 1
        conflicts.extend(bindings)
    assert {binding["path"] for binding in conflicts} == {checked["path"]}
    assert len({binding["file_sha256"] for binding in conflicts}) == 1
    conflicting_sha = conflicts[0]["file_sha256"]
    assert conflicting_sha != checked["file_sha256"]

    source = (ROOT / checked["path"]).read_bytes()
    lf = source.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    assert _sha(lf) == conflicting_sha
    assert _sha(crlf) == checked["file_sha256"]

    unified = tmp_path / "configs/unified_engine_status.json"
    unified.parent.mkdir(parents=True)
    unified.write_text(json.dumps({"sources": []}), encoding="utf-8")
    authority_path = tmp_path / AUTHORITY_REL
    authority_path.parent.mkdir(parents=True)
    authority_path.write_bytes((ROOT / AUTHORITY_REL).read_bytes())
    for relative, binding in zip(SOURCE_CONFLICT_RELS, conflicts, strict=True):
        conflict_path = tmp_path / relative
        conflict_path.parent.mkdir(parents=True, exist_ok=True)
        conflict_path.write_text(
            json.dumps(
                {
                    "source_path": binding["path"],
                    "source_file_sha256": binding["file_sha256"],
                }
            ),
            encoding="utf-8",
        )
    target = tmp_path / checked["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(lf)

    result = materialize(tmp_path, unified)

    assert result["files_rewritten"] == 1
    assert target.read_bytes() == crlf
    assert _sha(target.read_bytes()) == checked["file_sha256"]
    assert target.read_text(encoding="utf-8").splitlines() == source.decode("utf-8").splitlines()
