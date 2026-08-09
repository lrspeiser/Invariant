from __future__ import annotations

import ast
import hashlib
import json
import re
import sqlite3
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FORMULA_REGISTRY = Path("results/formula_prior_art_registry/formula_prior_art_registry.json")
FORMULA_SCORECARD = Path("results/formula_scorecard/formula_scorecard.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:20]}"


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return completed.stdout.strip()


def _walk_scalars(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _walk_scalars(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_scalars(child, f"{path}[{index}]")
    elif isinstance(value, (str, int, float, bool)) or value is None:
        yield path, value


class GateOntology:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.gates = payload["gates"]
        self.prohibited = [
            pattern.casefold() for pattern in payload["prohibited_evidence_patterns"]
        ]

    @classmethod
    def from_path(cls, path: str | Path) -> GateOntology:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def match_gates(self, text: str) -> list[str]:
        lowered = text.casefold()
        return [
            gate["id"]
            for gate in self.gates
            if any(keyword.casefold() in lowered for keyword in gate["keywords"])
        ]

    def evidence_admissibility(self, text: str) -> tuple[str, str]:
        lowered = text.casefold()
        matches = [pattern for pattern in self.prohibited if pattern in lowered]
        if matches:
            return "historical_only", f"prohibited evidence pattern: {', '.join(matches)}"
        if any(
            token in lowered
            for token in (
                "hamiltonian",
                "hessian",
                "characteristic",
                "covariant",
                "kinetic",
                "constraint",
                "newtonian limit",
                "noether",
                "bianchi",
                "theory-only",
            )
        ):
            return "theory_priority_allowed", "theory-side evidence"
        if any(token in lowered for token in ("raw", "detector", "spectrum", "angular", "doppler")):
            return (
                "measurement_requires_audit",
                "potential measurement; raw provenance audit required",
            )
        return "context_only", "not automatically admitted to priority scoring"


def _infer_outcome(*values: object) -> str:
    text = " ".join(str(value) for value in values if value is not None).casefold()
    failure = any(
        token in text
        for token in (
            "fail",
            "reject",
            "retire",
            "falsif",
            "no-go",
            "no_go",
            "stop_",
            "failed_closed",
            "unstable",
            "negative energy",
            "worsen",
            "no improvement",
            "null result",
        )
    )
    success = any(
        token in text
        for token in (
            " pass",
            "passed",
            "survive",
            "authorize",
            "proceed",
            "healthy",
            "complete",
        )
    )
    if failure and not success:
        return "reject"
    if success and not failure:
        return "pass"
    if failure and success:
        return "mixed"
    return "unknown"


def _mechanism_tags(text: str) -> list[str]:
    lowered = text.casefold()
    vocabulary = {
        "flux": ("flux", "displacement field", "d^i"),
        "gradient_state": ("gradient", "nabla d", "partial d", "spatial state"),
        "measured_state": ("z_b", "baryonic state", "stress", "coherence"),
        "scalar": ("scalar field", "scalar-tensor", "scalar sector"),
        "vector_aether": ("aether", "vector field", "aest"),
        "tensor_polarization": ("tensor", "polarization", "spin-2"),
        "dhost_horndeski": ("dhost", "horndeski", "galileon"),
        "nonlocal": ("nonlocal", "kernel", "memory"),
        "elastic_material": ("elastic", "strain", "material"),
        "pressure_thermal": ("pressure", "thermal", "temperature"),
        "lensing": ("lensing", "critical curve", "image topology"),
        "saturation": ("saturat", "sqrt(1+", "bounded"),
        "screening": ("screen", "symmetron", "high-field"),
        "first_derivative": ("first derivative", "partial_i", "nabla_i"),
    }
    return sorted(
        tag for tag, needles in vocabulary.items() if any(needle in lowered for needle in needles)
    )


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE gate_catalog (
  gate_id TEXT PRIMARY KEY, stage INTEGER NOT NULL, hard INTEGER NOT NULL,
  keywords_json TEXT NOT NULL
);
CREATE TABLE sources (
  source_id TEXT PRIMARY KEY, relative_path TEXT UNIQUE NOT NULL, kind TEXT NOT NULL,
  sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL, admissibility TEXT NOT NULL,
  admissibility_reason TEXT NOT NULL
);
CREATE TABLE tests (
  test_id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(source_id),
  name TEXT NOT NULL, line INTEGER NOT NULL, assertion_count INTEGER NOT NULL,
  call_names_json TEXT NOT NULL, tags_json TEXT NOT NULL, gate_ids_json TEXT NOT NULL
);
CREATE TABLE assertions (
  assertion_id TEXT PRIMARY KEY, test_id TEXT NOT NULL REFERENCES tests(test_id),
  ordinal INTEGER NOT NULL, expression TEXT NOT NULL, gate_ids_json TEXT NOT NULL
);
CREATE TABLE protocols (
  protocol_id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(source_id),
  protocol_version TEXT, status TEXT, purpose TEXT, inferred_outcome TEXT NOT NULL,
  tags_json TEXT NOT NULL, gate_ids_json TEXT NOT NULL, authorization_json TEXT NOT NULL
);
CREATE TABLE results (
  result_id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(source_id),
  status TEXT, decision TEXT, verdict TEXT, outcome_text TEXT,
  inferred_outcome TEXT NOT NULL, tags_json TEXT NOT NULL, gate_ids_json TEXT NOT NULL,
  gate_evidence_json TEXT NOT NULL
);
CREATE TABLE formulas (
  formula_id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(source_id),
  name TEXT NOT NULL, family TEXT, equation TEXT, eligibility TEXT,
  hard_exclusion INTEGER NOT NULL, classification_json TEXT NOT NULL,
  published_overlap_json TEXT NOT NULL, tags_json TEXT NOT NULL
);
CREATE TABLE edges (
  edge_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, relation TEXT NOT NULL,
  target TEXT NOT NULL, detail_json TEXT NOT NULL
);
CREATE TABLE lessons (
  lesson_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, subject TEXT NOT NULL,
  inferred_outcome TEXT NOT NULL, gate_ids_json TEXT NOT NULL, tags_json TEXT NOT NULL,
  admissible_for_priority INTEGER NOT NULL, evidence_class TEXT NOT NULL
);
CREATE TABLE mixed_signals (
  signal_id TEXT PRIMARY KEY, signal_type TEXT NOT NULL, signal_value TEXT NOT NULL,
  pass_count INTEGER NOT NULL, reject_count INTEGER NOT NULL,
  source_ids_json TEXT NOT NULL,
  interpretation TEXT NOT NULL
);
CREATE INDEX idx_tests_source ON tests(source_id);
CREATE INDEX idx_protocol_outcome ON protocols(inferred_outcome);
CREATE INDEX idx_result_outcome ON results(inferred_outcome);
CREATE INDEX idx_lessons_outcome ON lessons(inferred_outcome, admissible_for_priority);
"""


class KnowledgeBuilder:
    def __init__(self, repo: str | Path, ontology: GateOntology):
        self.repo = Path(repo).resolve()
        self.root = self.repo / "research" / "galaxy-cluster-unification"
        if not self.root.is_dir():
            raise FileNotFoundError(f"Missing galaxy-cluster-unification root: {self.root}")
        self.ontology = ontology
        self.source_ids: dict[str, str] = {}
        self.counts: Counter[str] = Counter()
        self.invalid_json: list[dict[str, str]] = []

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def _add_source(self, connection: sqlite3.Connection, path: Path, kind: str, text: str) -> str:
        relative = self._relative(path)
        if relative in self.source_ids:
            return self.source_ids[relative]
        source_id = _stable_id("SRC", relative, _sha256(path))
        admissibility, reason = self.ontology.evidence_admissibility(text)
        connection.execute(
            "INSERT INTO sources VALUES (?,?,?,?,?,?,?)",
            (source_id, relative, kind, _sha256(path), path.stat().st_size, admissibility, reason),
        )
        self.source_ids[relative] = source_id
        self.counts[f"source_{kind}"] += 1
        return source_id

    def _ingest_tests(self, connection: sqlite3.Connection) -> None:
        for path in sorted((self.root / "tests").glob("test_*.py")):
            text = path.read_text(encoding="utf-8")
            source_id = self._add_source(connection, path, "test_module", text)
            tree = ast.parse(text, filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not node.name.startswith("test_"):
                    continue
                assertions = [child for child in ast.walk(node) if isinstance(child, ast.Assert)]
                calls = sorted(
                    {
                        ast.unparse(child.func)
                        for child in ast.walk(node)
                        if isinstance(child, ast.Call)
                    }
                )
                combined = " ".join([path.name, node.name, ast.get_docstring(node) or ""])
                tags = _mechanism_tags(combined)
                gates = self.ontology.match_gates(combined)
                test_id = _stable_id("TST", self._relative(path), node.name, node.lineno)
                connection.execute(
                    "INSERT INTO tests VALUES (?,?,?,?,?,?,?,?)",
                    (
                        test_id,
                        source_id,
                        node.name,
                        node.lineno,
                        len(assertions),
                        json.dumps(calls),
                        json.dumps(tags),
                        json.dumps(gates),
                    ),
                )
                self.counts["test_functions"] += 1
                for ordinal, assertion in enumerate(assertions):
                    expression = ast.unparse(assertion.test)
                    assertion_gates = self.ontology.match_gates(f"{combined} {expression}")
                    connection.execute(
                        "INSERT INTO assertions VALUES (?,?,?,?,?)",
                        (
                            _stable_id("AST", test_id, ordinal, expression),
                            test_id,
                            ordinal,
                            expression,
                            json.dumps(assertion_gates),
                        ),
                    )
                    self.counts["assertions"] += 1

    def _gate_evidence(self, payload: Any) -> dict[str, Any]:
        evidence: dict[str, Any] = {}
        for path, value in _walk_scalars(payload):
            lowered = path.casefold()
            if any(token in lowered for token in ("gate", "pass", "fail", "decision", "verdict")):
                evidence[path] = value
            if len(evidence) >= 200:
                break
        return evidence

    def _add_edges_from_json(
        self, connection: sqlite3.Connection, source_id: str, payload: dict[str, Any]
    ) -> None:
        for json_path, value in _walk_scalars(payload):
            if not isinstance(value, str):
                continue
            if json_path.endswith(".path") or json_path == "path":
                connection.execute(
                    "INSERT OR IGNORE INTO edges VALUES (?,?,?,?,?)",
                    (
                        _stable_id("EDG", source_id, json_path, value),
                        source_id,
                        "references_path",
                        value.replace("\\", "/"),
                        json.dumps({"json_path": json_path}),
                    ),
                )
                self.counts["edges"] += 1
            elif "sha256" in json_path.casefold() and re.fullmatch(r"[0-9a-fA-F]{64}", value):
                connection.execute(
                    "INSERT OR IGNORE INTO edges VALUES (?,?,?,?,?)",
                    (
                        _stable_id("EDG", source_id, json_path, value),
                        source_id,
                        "commits_hash",
                        value.casefold(),
                        json.dumps({"json_path": json_path}),
                    ),
                )
                self.counts["edges"] += 1

    def _ingest_json(self, connection: sqlite3.Connection) -> None:
        paths = sorted((self.root / "configs").glob("*.json")) + sorted(
            (self.root / "results").rglob("*.json")
        )
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8-sig")
                payload = json.loads(text)
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                self.invalid_json.append(
                    {"path": self._relative(path), "error": f"{type(error).__name__}: {error}"}
                )
                continue
            if not isinstance(payload, dict):
                continue
            kind = "protocol_json" if path.parent.name == "configs" else "result_json"
            source_id = self._add_source(connection, path, kind, text[:200_000])
            self._add_edges_from_json(connection, source_id, payload)
            status = payload.get("status")
            decision = payload.get("decision")
            verdict = payload.get("verdict")
            outcome_text = payload.get("outcome")
            summary_text = " ".join(
                str(value)
                for value in (
                    path.name,
                    payload.get("protocol_version"),
                    status,
                    payload.get("purpose"),
                    decision,
                    verdict,
                    outcome_text,
                )
                if value is not None
            )
            tags = _mechanism_tags(summary_text)
            gates = self.ontology.match_gates(summary_text)
            inferred = _infer_outcome(status, decision, verdict, outcome_text)
            if kind == "protocol_json":
                protocol_id = _stable_id(
                    "PRO", self._relative(path), payload.get("protocol_version")
                )
                connection.execute(
                    "INSERT INTO protocols VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        protocol_id,
                        source_id,
                        payload.get("protocol_version"),
                        str(status) if status is not None else None,
                        str(payload.get("purpose")) if payload.get("purpose") is not None else None,
                        inferred,
                        json.dumps(tags),
                        json.dumps(gates),
                        json.dumps(payload.get("authorization", {}), sort_keys=True),
                    ),
                )
                self.counts["protocols"] += 1
                subject = str(payload.get("protocol_version") or path.stem)
            else:
                result_id = _stable_id("RES", self._relative(path), source_id)
                connection.execute(
                    "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        result_id,
                        source_id,
                        str(status) if status is not None else None,
                        str(decision) if decision is not None else None,
                        str(verdict) if verdict is not None else None,
                        str(outcome_text) if outcome_text is not None else None,
                        inferred,
                        json.dumps(tags),
                        json.dumps(gates),
                        json.dumps(self._gate_evidence(payload), sort_keys=True),
                    ),
                )
                self.counts["results"] += 1
                subject = path.parent.name if path.name == "report.json" else path.stem

            source_row = connection.execute(
                "SELECT admissibility FROM sources WHERE source_id=?", (source_id,)
            ).fetchone()
            admissible = source_row[0] == "theory_priority_allowed"
            if inferred != "unknown" and (tags or gates):
                connection.execute(
                    "INSERT INTO lessons VALUES (?,?,?,?,?,?,?,?)",
                    (
                        _stable_id("LES", source_id, subject, inferred),
                        source_id,
                        subject,
                        inferred,
                        json.dumps(gates),
                        json.dumps(tags),
                        int(admissible),
                        source_row[0],
                    ),
                )
                self.counts["lessons"] += 1

    def _ingest_docs(self, connection: sqlite3.Connection) -> None:
        for path in sorted((self.root / "docs").glob("*.md")):
            text = path.read_text(encoding="utf-8", errors="replace")
            source_id = self._add_source(connection, path, "research_note", text[:200_000])
            title = next(
                (line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")),
                path.stem,
            )
            outcome = _infer_outcome(path.name, title, " ".join(text.splitlines()[:30]))
            tags = _mechanism_tags(f"{path.name} {title}")
            gates = self.ontology.match_gates(f"{path.name} {title}")
            source_row = connection.execute(
                "SELECT admissibility FROM sources WHERE source_id=?", (source_id,)
            ).fetchone()
            if outcome != "unknown" and (tags or gates):
                connection.execute(
                    "INSERT OR IGNORE INTO lessons VALUES (?,?,?,?,?,?,?,?)",
                    (
                        _stable_id("LES", source_id, title, outcome),
                        source_id,
                        title,
                        outcome,
                        json.dumps(gates),
                        json.dumps(tags),
                        int(source_row[0] == "theory_priority_allowed"),
                        source_row[0],
                    ),
                )
                self.counts["lessons"] += 1

    def _ingest_formulas(self, connection: sqlite3.Connection) -> None:
        path = self.root / FORMULA_REGISTRY
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_id = self._add_source(
            connection, path, "formula_registry", json.dumps(payload)[:200_000]
        )
        forbidden_eligibility = {
            "prohibited_convenience_switch",
            "comparator_or_per_object_fit_not_final_theory",
            "diagnostic_not_final_theory",
            "lensing_only_closure_not_final_theory",
            "tested_and_rejected_or_failed",
            "published_control_not_novel",
        }
        for row in payload["tested_formulas"]:
            classification = row["classification"]
            eligibility = classification.get("final_theory_eligibility")
            hard_exclusion = bool(
                eligibility in forbidden_eligibility
                or classification.get("convenience_switch")
                or classification.get("per_object_gravity_fit")
                or classification.get("lensing_only_closure")
                or classification.get("empirical_composite")
                or classification.get("reported_failure_or_rejection")
                or _infer_outcome(row.get("verdict")) == "reject"
            )
            text = f"{row.get('family')} {row.get('formula')} {row.get('schematic_equation')}"
            connection.execute(
                "INSERT INTO formulas VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    row["registry_id"],
                    source_id,
                    row["formula"],
                    row.get("family"),
                    row.get("schematic_equation"),
                    eligibility,
                    int(hard_exclusion),
                    json.dumps(classification, sort_keys=True),
                    json.dumps(row.get("published_overlap_ids", [])),
                    json.dumps(_mechanism_tags(text)),
                ),
            )
            self.counts["formulas"] += 1

    def _build_mixed_signals(self, connection: sqlite3.Connection) -> None:
        aggregates: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {"pass": 0, "reject": 0, "sources": set()}
        )
        for source_id, outcome, gates_json, tags_json in connection.execute(
            "SELECT source_id, inferred_outcome, gate_ids_json, tags_json FROM lessons "
            "WHERE admissible_for_priority=1 AND inferred_outcome IN ('pass','reject')"
        ):
            signals = [("gate", value) for value in json.loads(gates_json)]
            signals.extend(("mechanism", value) for value in json.loads(tags_json))
            for signal in signals:
                aggregates[signal][outcome] += 1
                aggregates[signal]["sources"].add(source_id)
        for (signal_type, signal_value), aggregate in sorted(aggregates.items()):
            if not aggregate["pass"] or not aggregate["reject"]:
                continue
            connection.execute(
                "INSERT INTO mixed_signals VALUES (?,?,?,?,?,?,?)",
                (
                    _stable_id("MIX", signal_type, signal_value),
                    signal_type,
                    signal_value,
                    aggregate["pass"],
                    aggregate["reject"],
                    json.dumps(sorted(aggregate["sources"])),
                    "Mixed historical outcomes flag a review target; they are not automatically a contradiction.",
                ),
            )
            self.counts["mixed_signals"] += 1

    def build(self, database: str | Path, summary_path: str | Path) -> dict[str, Any]:
        database = Path(database)
        database.parent.mkdir(parents=True, exist_ok=True)
        if database.exists():
            database.unlink()
        connection = sqlite3.connect(database)
        try:
            connection.executescript(SCHEMA)
            for gate in self.ontology.gates:
                connection.execute(
                    "INSERT INTO gate_catalog VALUES (?,?,?,?)",
                    (gate["id"], gate["stage"], int(gate["hard"]), json.dumps(gate["keywords"])),
                )
            metadata = {
                "schema_version": "sigma-knowledge-graph-1.0",
                "repo_root": str(self.repo),
                "research_root": str(self.root),
                "repo_commit": _git(self.repo, "rev-parse", "HEAD"),
                "repo_dirty_paths": _git(self.repo, "status", "--short").splitlines(),
                "ontology_version": self.ontology.payload["schema_version"],
            }
            for key, value in metadata.items():
                connection.execute("INSERT INTO metadata VALUES (?,?)", (key, json.dumps(value)))
            self._ingest_tests(connection)
            self._ingest_json(connection)
            self._ingest_docs(connection)
            self._ingest_formulas(connection)
            self._build_mixed_signals(connection)
            connection.commit()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            admissibility = dict(
                connection.execute(
                    "SELECT admissibility, COUNT(*) FROM sources GROUP BY admissibility"
                ).fetchall()
            )
            outcomes = dict(
                connection.execute(
                    "SELECT inferred_outcome, COUNT(*) FROM lessons GROUP BY inferred_outcome"
                ).fetchall()
            )
            priority_lessons = connection.execute(
                "SELECT COUNT(*) FROM lessons WHERE admissible_for_priority=1"
            ).fetchone()[0]
        finally:
            connection.close()
        summary = {
            **metadata,
            "built_utc": datetime.now(UTC).isoformat(),
            "database": str(database),
            "database_sha256": _sha256(database),
            "database_size_bytes": database.stat().st_size,
            "integrity_check": integrity,
            "counts": dict(sorted(self.counts.items())),
            "source_admissibility": admissibility,
            "lesson_outcomes": outcomes,
            "priority_admissible_lessons": priority_lessons,
            "invalid_json": self.invalid_json,
            "scientific_rule": (
                "This graph prioritizes work only. Hard-gate failures remain terminal, and historical-only "
                "or unaudited observational evidence contributes zero priority weight."
            ),
        }
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary


def _dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    return all(left[key] >= right[key] for key in left) and any(
        left[key] > right[key] for key in left
    )


def pareto_fronts(rows: list[dict[str, Any]], axes: list[str]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[float, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(float(row[axis]) for axis in axes)].append(row)
    remaining = list(grouped)
    fronts: list[list[dict[str, Any]]] = []
    while remaining:
        front_points = []
        for point in remaining:
            if not any(
                other != point
                and all(left >= right for left, right in zip(other, point))
                and any(left > right for left, right in zip(other, point))
                for other in remaining
            ):
                front_points.append(point)
        front = [row for point in front_points for row in grouped[point]]
        fronts.append(
            sorted(front, key=lambda row: row.get("formula_id", row.get("family_id", "")))
        )
        front_set = set(front_points)
        remaining = [point for point in remaining if point not in front_set]
    return fronts


def prioritize_registered_formulas(database: str | Path, output: str | Path) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        formula_rows = connection.execute("SELECT * FROM formulas ORDER BY formula_id").fetchall()
        lesson_rows = connection.execute(
            "SELECT inferred_outcome, tags_json FROM lessons WHERE admissible_for_priority=1"
        ).fetchall()
    finally:
        connection.close()

    tag_history: dict[str, Counter[str]] = defaultdict(Counter)
    for lesson in lesson_rows:
        for tag in json.loads(lesson["tags_json"]):
            tag_history[tag][lesson["inferred_outcome"]] += 1

    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in formula_rows:
        classification = json.loads(row["classification_json"])
        tags = json.loads(row["tags_json"])
        equation = row["equation"] or ""
        integrity_penalties = sum(
            bool(classification.get(key))
            for key in (
                "convenience_switch",
                "per_object_gravity_fit",
                "lensing_only_closure",
                "empirical_composite",
            )
        )
        history = Counter()
        for tag in tags:
            history.update(tag_history[tag])
        pass_count = history["pass"]
        reject_count = history["reject"]
        item = {
            "formula_id": row["formula_id"],
            "name": row["name"],
            "family": row["family"],
            "equation": equation,
            "eligibility": row["eligibility"],
            "mechanism_tags": tags,
            "theoretical_integrity": max(0.0, 1.0 - 0.25 * integrity_penalties),
            "action_readiness": {
                "requires_single_action_derivation": 0.7,
                "experimental_incomplete_candidate": 0.45,
            }.get(row["eligibility"], 0.0),
            "parsimony": 1.0 / (1.0 + len(equation) / 80.0),
            "theory_history_signal": (pass_count + 1.0) / (pass_count + reject_count + 2.0),
            "theory_history_passes": pass_count,
            "theory_history_rejections": reject_count,
            "priority_semantics": "work ordering only; not probability of truth",
        }
        if row["hard_exclusion"]:
            item["exclusion_reason"] = "curated registry hard exclusion or reported failure"
            excluded.append(item)
        else:
            candidates.append(item)

    axes = ["theoretical_integrity", "action_readiness", "parsimony", "theory_history_signal"]
    fronts = pareto_fronts(candidates, axes)
    ranked: list[dict[str, Any]] = []
    for front_index, front in enumerate(fronts, start=1):
        for item in front:
            item["pareto_front"] = front_index
            ranked.append(item)
    report = {
        "schema_version": "sigma-formula-priority-1.0",
        "created_utc": datetime.now(UTC).isoformat(),
        "database": str(database),
        "axes": axes,
        "eligible_count": len(ranked),
        "excluded_count": len(excluded),
        "front_count": len(fronts),
        "front_one": fronts[0] if fronts else [],
        "eligible": ranked,
        "excluded": excluded,
        "forbidden_inputs": [
            "dark or invisible halo targets",
            "redshift-derived distances",
            "supernova distance moduli",
            "derived GR/NFW lensing accelerations",
            "observational fit quality as a substitute for action health",
        ],
        "interpretation": (
            "Pareto position orders expensive follow-up work among registry-eligible rows. It is not a "
            "truth score, and no axis can rescue a hard-gate failure."
        ),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
