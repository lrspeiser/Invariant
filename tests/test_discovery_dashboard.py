"""A3 discovery-dashboard gates.

The dashboard is a view over sealed evidence, so the load-bearing tests are the
prohibitions: no scalar score vocabulary may appear in any output, every row must
bind the exact evidence bytes it summarized, absent or broken sources must become
explicit rows instead of crashes, and any tamper — of the dashboard or of a source —
must be detected.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.discovery_dashboard import (
    CLAIMS,
    DASHBOARD_SCHEMA,
    FAILURE_STATUS,
    FORBIDDEN_SCORE_TOKENS,
    KINDS,
    ROW_KEYS,
    SOURCES,
    DiscoveryDashboardError,
    build_dashboard,
    main,
    render_html,
    validate_dashboard,
)
from sigma_theory_compiler.problem_queue import seal_queue
from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]

QUEUE_PATH = "configs/problem_queue_v1.json"
BILLION_PATH = "runs/gpu-baryonic-screen/billion-v1.json"
LENSING_PATH = "runs/gpu-baryonic-screen/lensing-cluster-v1.json"
SWEEP_PATH = "runs/math/counterexample-sweeps/collatz-halving-1e8.json"


def _seal(body: dict) -> dict:
    return {**body, "content_sha256": canonical_sha256(body)}


def _entry(entry_id: str, *, control: bool = False, synthetic: bool = False) -> dict:
    return {
        "id": entry_id,
        "domain": "math/dynamics",
        "statement": f"Synthetic statement for {entry_id}, used only by the dashboard tests.",
        "source_citation": "Dashboard test fixture registry, entry 1 of 2, 2026.",
        "believed_open_because": (
            "This is a synthetic fixture entry; it is not open mathematics and says so."
        ),
        "machine_form": {"kind": "sequence_rows", "generator": "fixture_gen", "max_point": 8},
        "progress_definition": "Progress is defined only for the fixture harness.",
        "control_rediscovery": control,
        "synthetic": synthetic,
    }


def _screen_receipt(decision: str = "SCREENED") -> dict:
    return _seal(
        {
            "counts": {"processed": 100, "survivors": 4},
            "decision": decision,
            "schema_version": "invariant-gpu-baryonic-interpolation-screen-result-1.0",
            "scope": "Synthetic screen fixture for dashboard tests.",
        }
    )


def _sweep_receipt() -> dict:
    return _seal(
        {
            "counts": {"checked": 999, "witnesses": 0},
            "decision": "NO_COUNTEREXAMPLE_IN_RANGE",
            "schema_version": "invariant-gpu-counterexample-sweep-result-1.0",
            "scope": "Synthetic sweep fixture for dashboard tests.",
        }
    )


def _write_json(root: Path, relpath: str, value: dict) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _make_root(
    tmp_path: Path,
    *,
    queue: bool = True,
    billion: bool = True,
    lensing: bool = True,
    sweep: bool = True,
) -> Path:
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    if queue:
        sealed = seal_queue([_entry("fixture_plain"), _entry("fixture_ctrl", control=True,
                                                             synthetic=True)])
        _write_json(root, QUEUE_PATH, sealed)
    if billion:
        _write_json(root, BILLION_PATH, _screen_receipt())
    if lensing:
        _write_json(root, LENSING_PATH, _screen_receipt("SEALED-NEGATIVE"))
    if sweep:
        _write_json(root, SWEEP_PATH, _sweep_receipt())
    return root


def _row(dashboard: dict, row_id: str) -> dict:
    matches = [row for row in dashboard["rows"] if row["row_id"] == row_id]
    assert len(matches) == 1, row_id
    return matches[0]


# ---------------------------------------------------------------------------
# The shipped sources
# ---------------------------------------------------------------------------


def test_shipped_sources_build_verbatim_status_rows():
    dashboard = build_dashboard(REPO_ROOT)
    validate_dashboard(dashboard)
    assert dashboard["schema_version"] == DASHBOARD_SCHEMA
    assert dashboard["counts"]["rows"] == len(dashboard["rows"])
    problems = [row for row in dashboard["rows"] if row["kind"] == "problem"]
    assert len(problems) == 10  # pinned by the A2 queue tests

    billion = _row(dashboard, "gpu-baryonic-screen-billion-v1")
    assert billion["status_text"].startswith("SCREENED")
    assert "processed 1129900996" in billion["status_text"]
    assert billion["lineage"]["receipt_decision"] == "SCREENED"
    raw = (REPO_ROOT / BILLION_PATH).read_bytes()
    assert billion["evidence_content_sha256"] == hashlib.sha256(raw).hexdigest()

    sweep = _row(dashboard, "collatz-halving-1e8")
    assert sweep["status_text"].startswith("NO_COUNTEREXAMPLE_IN_RANGE")
    assert "checked 99999999" in sweep["status_text"]

    # The lensing receipt may not exist yet; either way it must be an explicit row.
    lensing = _row(dashboard, "gpu-baryonic-screen-lensing-cluster-v1")
    assert lensing["kind"] == "screen_campaign"


def test_declared_sources_cover_the_required_paths():
    assert [source["path"] for source in SOURCES] == [
        QUEUE_PATH, BILLION_PATH, LENSING_PATH, SWEEP_PATH
    ]


# ---------------------------------------------------------------------------
# The hard rule: no scalar score vocabulary anywhere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", FORBIDDEN_SCORE_TOKENS)
def test_forbidden_score_tokens_absent_from_shipped_outputs(token):
    dashboard = build_dashboard(REPO_ROOT)
    json_blob = canonical_json_bytes(dashboard).decode("utf-8").lower()
    html_blob = render_html(dashboard).lower()
    assert token not in json_blob
    assert token not in html_blob


@pytest.mark.parametrize("token", FORBIDDEN_SCORE_TOKENS)
def test_forbidden_score_tokens_absent_from_synthetic_outputs(tmp_path, token):
    dashboard = build_dashboard(_make_root(tmp_path))
    assert token not in canonical_json_bytes(dashboard).decode("utf-8").lower()
    assert token not in render_html(dashboard).lower()


def test_source_smuggling_a_score_token_refuses_to_build(tmp_path):
    root = _make_root(tmp_path)
    _write_json(root, BILLION_PATH, _screen_receipt("PASS with probability 1"))
    with pytest.raises(DiscoveryDashboardError, match="forbidden score token"):
        build_dashboard(root)


# ---------------------------------------------------------------------------
# Evidence binding
# ---------------------------------------------------------------------------


def test_every_row_carries_evidence_path_and_content_hash(tmp_path):
    root = _make_root(tmp_path)
    dashboard = build_dashboard(root)
    assert dashboard["rows"]
    for row in dashboard["rows"]:
        assert set(row) == ROW_KEYS
        assert row["kind"] in KINDS
        assert row["evidence_path"]
        digest = row["evidence_content_sha256"]
        assert digest is not None
        raw = (root / row["evidence_path"]).read_bytes()
        assert digest == hashlib.sha256(raw).hexdigest()


def test_missing_sources_fail_soft_as_explicit_missing_rows(tmp_path):
    root = _make_root(tmp_path, queue=False, lensing=False)
    dashboard = build_dashboard(root)
    validate_dashboard(dashboard)
    for row_id in ("problem-queue-v1", "gpu-baryonic-screen-lensing-cluster-v1"):
        row = _row(dashboard, row_id)
        assert row["status_text"] == FAILURE_STATUS["missing"]
        assert row["evidence_content_sha256"] is None
        assert row["lineage"] == {"failure": "missing"}
    assert dashboard["counts"]["missing"] == 2
    assert dashboard["counts"]["present"] == 2


def test_unreadable_source_row_still_binds_the_found_bytes(tmp_path):
    root = _make_root(tmp_path)
    garbage = b"this is not a receipt"
    (root / SWEEP_PATH).write_bytes(garbage)
    dashboard = build_dashboard(root)
    row = _row(dashboard, "collatz-halving-1e8")
    assert row["status_text"] == FAILURE_STATUS["unreadable"]
    assert row["evidence_content_sha256"] == hashlib.sha256(garbage).hexdigest()
    assert dashboard["counts"]["unreadable"] == 1


def test_tampered_source_receipt_is_flagged_not_trusted(tmp_path):
    root = _make_root(tmp_path)
    tampered = _screen_receipt()
    tampered["decision"] = "EVERYTHING-PASSED"  # edited without resealing
    _write_json(root, BILLION_PATH, tampered)
    dashboard = build_dashboard(root)
    row = _row(dashboard, "gpu-baryonic-screen-billion-v1")
    assert row["status_text"] == FAILURE_STATUS["tampered"]
    assert "EVERYTHING-PASSED" not in json.dumps(dashboard)
    assert dashboard["counts"]["tampered"] == 1


def test_queue_schema_labels_surface_verbatim(tmp_path):
    dashboard = build_dashboard(_make_root(tmp_path))
    plain = _row(dashboard, "problem-queue-v1.fixture_plain")
    labeled = _row(dashboard, "problem-queue-v1.fixture_ctrl")
    assert plain["status_text"] == "QUEUED"
    assert labeled["status_text"] == "QUEUED (control_rediscovery, synthetic)"
    assert labeled["lineage"]["control_rediscovery"] is True
    assert labeled["lineage"]["synthetic"] is True
    assert plain["lineage"]["source_citation"].startswith("Dashboard test fixture")


# ---------------------------------------------------------------------------
# Determinism and tamper detection
# ---------------------------------------------------------------------------


def test_dashboard_build_is_deterministic(tmp_path):
    root = _make_root(tmp_path)
    first = build_dashboard(root)
    second = build_dashboard(root)
    assert first == second
    assert render_html(first) == render_html(second)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


def test_dashboard_tamper_without_reseal_fails(tmp_path):
    dashboard = build_dashboard(_make_root(tmp_path))
    tampered = json.loads(canonical_json_bytes(dashboard))
    tampered["rows"][0]["status_text"] = "PASS"
    with pytest.raises(DiscoveryDashboardError, match="seal"):
        validate_dashboard(tampered)


def test_dashboard_tamper_with_reseal_fails_on_counts(tmp_path):
    dashboard = build_dashboard(_make_root(tmp_path))
    body = {key: value for key, value in dashboard.items() if key != "content_sha256"}
    body = json.loads(canonical_json_bytes(body))
    body["counts"]["present"] += 1
    resealed = {**body, "content_sha256": canonical_sha256(body)}
    with pytest.raises(DiscoveryDashboardError, match="counts"):
        validate_dashboard(resealed)


def test_claims_are_pinned(tmp_path):
    dashboard = build_dashboard(_make_root(tmp_path))
    assert dashboard["claims"] == CLAIMS
    assert CLAIMS["scalar_score_of_any_kind_present"] is False
    assert CLAIMS["statuses_are_verbatim_source_decisions"] is True
    body = {key: value for key, value in dashboard.items() if key != "content_sha256"}
    body["claims"] = {**CLAIMS, "scalar_score_of_any_kind_present": True}
    resealed = {**body, "content_sha256": canonical_sha256(body)}
    with pytest.raises(DiscoveryDashboardError, match="claims"):
        validate_dashboard(resealed)


def test_validate_with_root_detects_source_drift(tmp_path):
    root = _make_root(tmp_path)
    dashboard = build_dashboard(root)
    validate_dashboard(dashboard, root=root)
    _write_json(root, BILLION_PATH, _screen_receipt("BLOCK"))  # properly resealed source edit
    with pytest.raises(DiscoveryDashboardError, match="replay"):
        validate_dashboard(dashboard, root=root)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def test_html_is_static_grouped_and_self_contained(tmp_path):
    dashboard = build_dashboard(_make_root(tmp_path))
    html_text = render_html(dashboard)
    assert "<script" not in html_text.lower()
    assert "href=" not in html_text.lower()
    assert "src=" not in html_text.lower()
    problems = html_text.index("Problems (A2 intake queue)")
    screens = html_text.index("Screen campaigns")
    sweeps = html_text.index("Sweep campaigns")
    assert problems < screens < sweeps
    for row in dashboard["rows"]:
        assert row["row_id"] in html_text
    assert dashboard["content_sha256"] in html_text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_writes_sealed_outputs_and_validates(tmp_path, capsys):
    root = _make_root(tmp_path)
    json_path = tmp_path / "out" / "status.json"
    html_path = tmp_path / "out" / "status.html"
    assert main([
        "--root", str(root), "--output-json", str(json_path), "--output-html", str(html_path)
    ]) == 0
    assert "BUILT rows=" in capsys.readouterr().out
    stored = json.loads(json_path.read_text(encoding="utf-8"))
    validate_dashboard(stored, root=root)
    assert html_path.read_text(encoding="utf-8") == render_html(stored)

    assert main([
        "--root", str(root), "--output-json", str(json_path), "--validate-checked"
    ]) == 0
    assert capsys.readouterr().out.startswith("VALID rows=")

    stored["rows"][0]["status_text"] = "PASS"
    json_path.write_bytes(canonical_json_bytes(stored) + b"\n")
    assert main([
        "--root", str(root), "--output-json", str(json_path), "--validate-checked"
    ]) == 1
    assert capsys.readouterr().out.startswith("INVALID")
