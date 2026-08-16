"""A3 — deterministic discovery dashboard over sealed evidence.

Discovery output is only trustworthy while it stays attached to its evidence.  This
module aggregates the declared discovery sources — the A2 problem queue, the GPU
baryonic screen receipts, and the counterexample-sweep receipts — into one sealed
status receipt plus a static HTML rendering.  It is a *view*, never a judge: every
row's status is the verbatim decision string carried by its source receipt, and every
row binds the exact bytes it was rendered from.

Three rules keep the dashboard honest.

**No scalar scores, ever.**  The dashboard never computes, copies, or renders a
scalar that ranks how "true", "likely", or "settled" anything is.  Statuses are
verbatim receipt decisions and integer counts.  Building the dashboard scans its own
outputs for the forbidden score vocabulary and refuses to emit them.

**Evidence is byte-bound.**  Every row carries the source path and the SHA-256 of the
exact bytes it summarized.  A source edited after the dashboard was built is
detectable without trusting the dashboard.

**Absence is reported, not repaired.**  A missing source becomes an explicit row with
a null byte hash; an unreadable or tampered source becomes a row saying exactly that,
still binding whatever bytes were found.  The dashboard never invents a status for
evidence it could not read.

Claim boundary: a dashboard row asserts that a source file said something, and which
bytes said it.  It establishes no novelty, significance, ranking, or validity beyond
what the underlying sealed receipt itself claims.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .problem_queue import ProblemQueueError, load_queue
from .sigma_core import SigmaCoreError, canonical_json_bytes, canonical_sha256

DASHBOARD_SCHEMA = "invariant-discovery-dashboard-1.0"

_ROW_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,159}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

KINDS = ("problem", "screen_campaign", "sweep_campaign")
KIND_HEADINGS = {
    "problem": "Problems (A2 intake queue)",
    "screen_campaign": "Screen campaigns",
    "sweep_campaign": "Sweep campaigns",
}

TOP_LEVEL_KEYS = {"schema_version", "rows", "counts", "claims", "scope", "content_sha256"}
ROW_KEYS = {
    "row_id",
    "kind",
    "title",
    "status_text",
    "evidence_path",
    "evidence_content_sha256",
    "lineage",
}

#: Score vocabulary that must never appear in any dashboard output, matched
#: case-insensitively over the full serialized JSON and HTML.
FORBIDDEN_SCORE_TOKENS = ("truth_score", "probability", "confidence", "% true", "likelihood")

#: Failure statuses a row may carry instead of a verbatim source decision.
FAILURE_STATUS = {
    "missing": "MISSING (no file at declared evidence path)",
    "unreadable": "UNREADABLE (evidence bytes are not a valid receipt)",
    "tampered": "TAMPERED (receipt seal does not match receipt body)",
}

CLAIMS = {
    "corpus_absence_establishes_novelty": False,
    "dashboard_row_establishes_significance_or_validity": False,
    "evidence_bytes_are_hash_bound_per_row": True,
    "missing_sources_become_explicit_rows": True,
    "scalar_score_of_any_kind_present": False,
    "statuses_are_verbatim_source_decisions": True,
}

SCOPE = (
    "Deterministic aggregation of declared discovery sources into sealed status rows. "
    "Each row restates a source's own decision string verbatim and binds the SHA-256 "
    "of the exact source bytes; missing, unreadable, or seal-broken sources are "
    "reported as such. No row carries any scalar rating of how settled, ranked, or "
    "credible its subject is, and row membership asserts nothing beyond what the "
    "cited sealed source itself claims."
)

#: Declared sources, in row order.  `path` is repo-root-relative POSIX.
SOURCES = (
    {
        "source_id": "problem-queue-v1",
        "loader": "queue",
        "kind": "problem",
        "path": "configs/problem_queue_v1.json",
        "title": "A2 open-problem intake queue",
    },
    {
        "source_id": "gpu-baryonic-screen-billion-v1",
        "loader": "receipt",
        "kind": "screen_campaign",
        "path": "runs/gpu-baryonic-screen/billion-v1.json",
        "title": "GPU baryonic screen, billion-law campaign (billion-v1)",
    },
    {
        "source_id": "gpu-baryonic-screen-lensing-cluster-v1",
        "loader": "receipt",
        "kind": "screen_campaign",
        "path": "runs/gpu-baryonic-screen/lensing-cluster-v1.json",
        "title": "GPU baryonic screen, lensing + cluster gates (lensing-cluster-v1)",
    },
    {
        "source_id": "collatz-halving-1e8",
        "loader": "receipt",
        "kind": "sweep_campaign",
        "path": "runs/math/counterexample-sweeps/collatz-halving-1e8.json",
        "title": "Collatz halving-relation counterexample sweep to 1e8",
    },
)


class DiscoveryDashboardError(ValueError):
    """Raised when dashboard construction or validation fails closed."""


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def _file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _failure_row(source: Mapping[str, Any], failure: str, digest: str | None) -> dict[str, Any]:
    return {
        "row_id": source["source_id"],
        "kind": source["kind"],
        "title": source["title"],
        "status_text": FAILURE_STATUS[failure],
        "evidence_path": source["path"],
        "evidence_content_sha256": digest,
        "lineage": {"failure": failure},
    }


def _status_with_counts(decision: str, counts: Any) -> str:
    """Verbatim decision, optionally followed by the receipt's own integer counts."""

    if not isinstance(counts, Mapping):
        return decision
    parts: list[str] = []
    for key in sorted(counts):
        value = counts[key]
        if isinstance(value, int) and not isinstance(value, bool) and isinstance(key, str):
            parts.append(f"{key} {value}")
    return f"{decision} ({', '.join(parts)})" if parts else decision


def _queue_rows(source: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    path = root / source["path"]
    if not path.is_file():
        return [_failure_row(source, "missing", None)]
    data = path.read_bytes()
    digest = _file_sha256(data)
    try:
        queue = load_queue(path)
    except ProblemQueueError:
        return [_failure_row(source, "unreadable", digest)]
    rows: list[dict[str, Any]] = []
    for entry in queue["entries"]:
        labels = [
            name for name in ("control_rediscovery", "synthetic") if entry[name] is True
        ]
        status = "QUEUED" if not labels else f"QUEUED ({', '.join(labels)})"
        rows.append(
            {
                "row_id": f"{source['source_id']}.{entry['id']}",
                "kind": source["kind"],
                "title": entry["statement"],
                "status_text": status,
                "evidence_path": source["path"],
                "evidence_content_sha256": digest,
                "lineage": {
                    "control_rediscovery": entry["control_rediscovery"],
                    "queue_content_sha256": queue["content_sha256"],
                    "source_citation": entry["source_citation"],
                    "synthetic": entry["synthetic"],
                },
            }
        )
    return rows


def _receipt_row(source: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    path = root / source["path"]
    if not path.is_file():
        return [_failure_row(source, "missing", None)]
    data = path.read_bytes()
    digest = _file_sha256(data)
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return [_failure_row(source, "unreadable", digest)]
    if not isinstance(value, Mapping):
        return [_failure_row(source, "unreadable", digest)]
    decision = value.get("decision")
    schema_version = value.get("schema_version")
    seal = value.get("content_sha256")
    if (
        not isinstance(decision, str)
        or not decision.strip()
        or not isinstance(schema_version, str)
        or not isinstance(seal, str)
    ):
        return [_failure_row(source, "unreadable", digest)]
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    try:
        expected = canonical_sha256(body)
    except SigmaCoreError:
        return [_failure_row(source, "unreadable", digest)]
    if seal != expected:
        return [_failure_row(source, "tampered", digest)]
    return [
        {
            "row_id": source["source_id"],
            "kind": source["kind"],
            "title": source["title"],
            "status_text": _status_with_counts(decision, value.get("counts")),
            "evidence_path": source["path"],
            "evidence_content_sha256": digest,
            "lineage": {
                "receipt_content_sha256": seal,
                "receipt_decision": decision,
                "receipt_schema_version": schema_version,
            },
        }
    ]


# ---------------------------------------------------------------------------
# Dashboard construction
# ---------------------------------------------------------------------------


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result = {
        "missing": 0,
        "present": 0,
        "rows": len(rows),
        "tampered": 0,
        "unreadable": 0,
    }
    for kind in KINDS:
        result[kind] = 0
    for row in rows:
        result[row["kind"]] += 1
        failure = row["lineage"].get("failure") if isinstance(row["lineage"], Mapping) else None
        if failure is None:
            result["present"] += 1
        else:
            result[failure] += 1
    return result


def _scan_forbidden(blob: str, label: str) -> None:
    lowered = blob.lower()
    for token in FORBIDDEN_SCORE_TOKENS:
        if token in lowered:
            raise DiscoveryDashboardError(f"forbidden score token {token!r} in {label}")


def build_dashboard(root: Path | str = ".") -> dict[str, Any]:
    """Build, seal, and self-validate the dashboard receipt from declared sources."""

    base = Path(root)
    rows: list[dict[str, Any]] = []
    for source in SOURCES:
        if source["loader"] == "queue":
            rows.extend(_queue_rows(source, base))
        else:
            rows.extend(_receipt_row(source, base))
    body = {
        "claims": CLAIMS,
        "counts": _counts(rows),
        "rows": rows,
        "schema_version": DASHBOARD_SCHEMA,
        "scope": SCOPE,
    }
    dashboard = {**body, "content_sha256": canonical_sha256(body)}
    validate_dashboard(dashboard)
    _scan_forbidden(render_html(dashboard), "html rendering")
    return dashboard


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _reject_floats(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        raise DiscoveryDashboardError(f"floating value forbidden at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_floats(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_floats(child, f"{path}[{index}]")


def _validate_row(row: Any, label: str) -> None:
    if not isinstance(row, Mapping) or set(row) != ROW_KEYS:
        raise DiscoveryDashboardError(f"{label} keys changed")
    row_id = row["row_id"]
    if not isinstance(row_id, str) or _ROW_ID.fullmatch(row_id) is None:
        raise DiscoveryDashboardError(f"{label}.row_id must match {_ROW_ID.pattern}")
    if row["kind"] not in KINDS:
        raise DiscoveryDashboardError(f"{label}.kind must be one of {KINDS}")
    for name in ("title", "status_text", "evidence_path"):
        item = row[name]
        if not isinstance(item, str) or not item.strip():
            raise DiscoveryDashboardError(f"{label}.{name} must be a nonempty string")
    digest = row["evidence_content_sha256"]
    lineage = row["lineage"]
    if not isinstance(lineage, Mapping) or not lineage:
        raise DiscoveryDashboardError(f"{label}.lineage must be a non-empty object")
    failure = lineage.get("failure")
    if failure is not None:
        if set(lineage) != {"failure"} or failure not in FAILURE_STATUS:
            raise DiscoveryDashboardError(f"{label}.lineage failure form changed")
        if row["status_text"] != FAILURE_STATUS[failure]:
            raise DiscoveryDashboardError(f"{label} failure status text changed")
        if failure == "missing":
            if digest is not None:
                raise DiscoveryDashboardError(f"{label} missing row cannot bind bytes")
            return
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise DiscoveryDashboardError(
            f"{label}.evidence_content_sha256 must be a lowercase SHA-256 digest"
        )


def validate_dashboard(value: Any, *, root: Path | str | None = None) -> None:
    """Reject structural, claims, counts, score-token, or seal violations.

    With `root`, additionally require an exact rebuild replay against the current
    source files, so a stale or source-divergent dashboard fails validation.
    """

    if not isinstance(value, Mapping) or set(value) != TOP_LEVEL_KEYS:
        raise DiscoveryDashboardError("dashboard top-level keys changed")
    _reject_floats(value)
    if value["schema_version"] != DASHBOARD_SCHEMA:
        raise DiscoveryDashboardError("dashboard schema changed")
    if value["claims"] != CLAIMS:
        raise DiscoveryDashboardError("dashboard claims changed")
    if not isinstance(value["scope"], str) or not value["scope"].strip():
        raise DiscoveryDashboardError("dashboard scope must be a nonempty string")
    rows = value["rows"]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise DiscoveryDashboardError("rows must be a non-empty list")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        _validate_row(row, f"rows[{index}]")
        if row["row_id"] in seen:
            raise DiscoveryDashboardError(f"duplicate row_id: {row['row_id']}")
        seen.add(row["row_id"])
    if value["counts"] != _counts(rows):
        raise DiscoveryDashboardError("counts do not match rows")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    try:
        expected = canonical_sha256(body)
    except SigmaCoreError as error:
        raise DiscoveryDashboardError(f"dashboard is not canonically encodable: {error}") from error
    if value["content_sha256"] != expected:
        raise DiscoveryDashboardError("dashboard seal changed")
    _scan_forbidden(canonical_json_bytes(value).decode("utf-8"), "dashboard receipt")
    if root is not None and dict(value) != build_dashboard(root):
        raise DiscoveryDashboardError("dashboard does not replay from current sources")


# ---------------------------------------------------------------------------
# HTML rendering (static, inline CSS, no scripts, no external assets)
# ---------------------------------------------------------------------------

_CSS = (
    "body{font-family:Georgia,serif;margin:2rem auto;max-width:70rem;color:#1a1a1a;"
    "background:#fdfdfa;padding:0 1rem}"
    "h1{font-size:1.6rem;border-bottom:3px double #1a1a1a;padding-bottom:.4rem}"
    "h2{font-size:1.2rem;margin-top:2rem}"
    "table{border-collapse:collapse;width:100%;font-size:.85rem}"
    "th,td{border:1px solid #999;padding:.4rem .5rem;text-align:left;vertical-align:top}"
    "th{background:#eee9df}"
    "code{font-family:Consolas,monospace;font-size:.78rem;word-break:break-all}"
    ".status{font-weight:bold;white-space:pre-wrap}"
    ".failure{color:#7a1f1f}"
    "footer{margin-top:2rem;font-size:.8rem;color:#444;border-top:1px solid #999;"
    "padding-top:.6rem}"
)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _row_html(row: Mapping[str, Any]) -> str:
    lineage = row["lineage"]
    failure = lineage.get("failure")
    status_class = "status failure" if failure is not None else "status"
    digest = row["evidence_content_sha256"]
    digest_html = f"<code>{_escape(digest)}</code>" if digest is not None else "(no bytes found)"
    lineage_parts = [
        f"{_escape(key)}: {_escape(lineage[key])}" for key in sorted(lineage)
    ]
    return (
        "<tr>"
        f"<td><code>{_escape(row['row_id'])}</code></td>"
        f"<td>{_escape(row['title'])}</td>"
        f"<td class=\"{status_class}\">{_escape(row['status_text'])}</td>"
        f"<td><code>{_escape(row['evidence_path'])}</code><br>{digest_html}</td>"
        f"<td>{'<br>'.join(lineage_parts)}</td>"
        "</tr>"
    )


def render_html(dashboard: Mapping[str, Any]) -> str:
    """Deterministic static HTML for a built dashboard.  No scripts of any kind."""

    counts = dashboard["counts"]
    sections: list[str] = []
    for kind in KINDS:
        rows = [row for row in dashboard["rows"] if row["kind"] == kind]
        if not rows:
            continue
        body = "".join(_row_html(row) for row in rows)
        sections.append(
            f"<h2>{_escape(KIND_HEADINGS[kind])}</h2>"
            "<table><thead><tr><th>Row</th><th>Title</th><th>Status</th>"
            "<th>Evidence (path + SHA-256 of bytes)</th><th>Lineage</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )
    count_text = ", ".join(f"{key} {counts[key]}" for key in sorted(counts))
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        "<title>Invariant discovery dashboard</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>Invariant discovery dashboard</h1>"
        f"<p>Statuses are verbatim decision strings from sealed source receipts; every "
        f"row binds the SHA-256 of the exact evidence bytes. Counts: {_escape(count_text)}.</p>"
        + "".join(sections)
        + "<footer>"
        f"<p>{_escape(dashboard['scope'])}</p>"
        f"<p>schema {_escape(dashboard['schema_version'])} &middot; receipt seal "
        f"<code>{_escape(dashboard['content_sha256'])}</code></p>"
        "</footer></body></html>\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discovery dashboard (A3).")
    parser.add_argument("--root", default=".", help="repository root holding the declared sources")
    parser.add_argument("--output-json", help="path for the sealed dashboard JSON receipt")
    parser.add_argument("--output-html", help="path for the static HTML rendering")
    parser.add_argument(
        "--validate-checked",
        action="store_true",
        help="validate the existing --output-json against structure, seal, and current sources",
    )
    args = parser.parse_args(argv)
    if args.validate_checked:
        if not args.output_json:
            parser.error("--validate-checked requires --output-json")
        try:
            stored = json.loads(Path(args.output_json).read_text(encoding="utf-8"))
            validate_dashboard(stored, root=args.root)
        except (OSError, ValueError) as error:
            print(f"INVALID {args.output_json}: {error}")
            return 1
        print(f"VALID rows={len(stored['rows'])} content_sha256={stored['content_sha256']}")
        return 0
    dashboard = build_dashboard(args.root)
    if args.output_json:
        _write_bytes(Path(args.output_json), canonical_json_bytes(dashboard) + b"\n")
    if args.output_html:
        _write_bytes(Path(args.output_html), render_html(dashboard).encode("utf-8"))
    if not args.output_json and not args.output_html:
        print(json.dumps(dashboard, indent=2, sort_keys=True))
    else:
        counts = dashboard["counts"]
        print(
            f"BUILT rows={counts['rows']} present={counts['present']} "
            f"missing={counts['missing']} content_sha256={dashboard['content_sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
