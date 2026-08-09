from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import re
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-equation-universe-1.0"
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_FUNCTIONS: dict[str, Any] = {
    "sqrt": sp.sqrt,
    "exp": sp.exp,
    "log": sp.log,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "Abs": sp.Abs,
}
_ALLOWED_SOURCE_MODES = {"full", "formula_only", "metadata_only", "blocked"}
_ALLOWED_REPRESENTATIONS = {"scalar_sympy", "tensor_dsl", "latex_only"}


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE sources (
  source_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  url TEXT NOT NULL,
  authors_json TEXT NOT NULL,
  year INTEGER,
  source_kind TEXT NOT NULL,
  license_id TEXT,
  license_url TEXT,
  ingestion_mode TEXT NOT NULL,
  policy_reason TEXT NOT NULL,
  accessed_utc TEXT,
  UNIQUE(url)
);
CREATE TABLE equations (
  equation_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  domain TEXT NOT NULL,
  representation TEXT NOT NULL,
  expression TEXT NOT NULL,
  latex TEXT,
  normalized_expression TEXT,
  denominator_guard TEXT,
  semantic_hash TEXT,
  structural_hash TEXT,
  feature_json TEXT NOT NULL,
  dimension_status TEXT NOT NULL,
  dimension_detail_json TEXT NOT NULL,
  assumptions_json TEXT NOT NULL,
  validity_json TEXT NOT NULL,
  tags_json TEXT NOT NULL,
  source_id TEXT NOT NULL REFERENCES sources(source_id),
  source_locator TEXT,
  independently_encoded INTEGER NOT NULL,
  record_sha256 TEXT NOT NULL,
  created_utc TEXT NOT NULL
);
CREATE TABLE variables (
  equation_id TEXT NOT NULL REFERENCES equations(equation_id) ON DELETE CASCADE,
  symbol TEXT NOT NULL,
  canonical_name TEXT NOT NULL,
  meaning TEXT NOT NULL,
  dimension_json TEXT,
  field_kind TEXT NOT NULL,
  tensor_rank INTEGER NOT NULL,
  PRIMARY KEY(equation_id, symbol)
);
CREATE TABLE derivations (
  derivation_id TEXT PRIMARY KEY,
  target_equation_id TEXT NOT NULL REFERENCES equations(equation_id),
  operation TEXT NOT NULL,
  assumptions_json TEXT NOT NULL,
  proof_method TEXT NOT NULL,
  proof_status TEXT NOT NULL,
  proof_detail_json TEXT NOT NULL,
  source_id TEXT REFERENCES sources(source_id)
);
CREATE TABLE derivation_inputs (
  derivation_id TEXT NOT NULL REFERENCES derivations(derivation_id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL,
  equation_id TEXT NOT NULL REFERENCES equations(equation_id),
  PRIMARY KEY(derivation_id, ordinal)
);
CREATE TABLE equivalence_edges (
  edge_id TEXT PRIMARY KEY,
  left_equation_id TEXT NOT NULL REFERENCES equations(equation_id),
  right_equation_id TEXT NOT NULL REFERENCES equations(equation_id),
  equivalence_type TEXT NOT NULL,
  proof_method TEXT NOT NULL,
  proof_detail_json TEXT NOT NULL,
  UNIQUE(left_equation_id, right_equation_id, equivalence_type)
);
CREATE TABLE import_runs (
  import_id TEXT PRIMARY KEY,
  input_path TEXT NOT NULL,
  input_sha256 TEXT NOT NULL,
  started_utc TEXT NOT NULL,
  completed_utc TEXT NOT NULL,
  source_count INTEGER NOT NULL,
  equation_count INTEGER NOT NULL,
  derivation_count INTEGER NOT NULL,
  rejected_count INTEGER NOT NULL,
  detail_json TEXT NOT NULL
);
CREATE TABLE formula_spaces (
  space_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  protocol_version TEXT NOT NULL,
  generator_version TEXT NOT NULL,
  manifest_path TEXT NOT NULL,
  manifest_sha256 TEXT NOT NULL,
  basis_path TEXT NOT NULL,
  basis_sha256 TEXT NOT NULL,
  survivor_directory TEXT NOT NULL,
  basis_count INTEGER NOT NULL,
  max_action_terms INTEGER NOT NULL,
  total_declared_actions INTEGER NOT NULL,
  processed_actions INTEGER NOT NULL,
  survivor_count INTEGER NOT NULL,
  complete_declared_space INTEGER NOT NULL,
  query_semantics TEXT NOT NULL,
  registered_utc TEXT NOT NULL,
  UNIQUE(protocol_version, manifest_sha256)
);
CREATE INDEX idx_equations_semantic ON equations(semantic_hash);
CREATE INDEX idx_equations_structural ON equations(structural_hash);
CREATE INDEX idx_equations_domain ON equations(domain);
CREATE INDEX idx_derivations_target ON derivations(target_equation_id);
"""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\0".join(str(part) for part in parts)
    return f"{prefix}-{_sha256_text(payload)[:20]}"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validate_ast(text: str, symbol_names: set[str]) -> None:
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid scalar expression: {exc.msg}") from exc
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Name,
        ast.Constant,
        ast.Call,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Load,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise TypeError(f"unsupported scalar syntax: {type(node).__name__}")
        if (
            isinstance(node, ast.Name)
            and node.id not in symbol_names | set(_ALLOWED_FUNCTIONS) | {"pi", "E"}
        ):
            raise ValueError(f"undeclared symbol or function: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
                raise ValueError("only allowlisted scalar functions may be called")
            if node.keywords:
                raise ValueError("keyword arguments are not supported")


def _parse_scalar(text: str, variables: dict[str, dict[str, Any]]) -> sp.Expr:
    normalized = text.strip().replace("^", "**")
    symbols = {name: sp.Symbol(name, real=True) for name in variables}
    _validate_ast(normalized, set(symbols))
    locals_dict = {**symbols, **_ALLOWED_FUNCTIONS, "pi": sp.pi, "E": sp.E}
    return sp.sympify(normalized, locals=locals_dict, evaluate=True)


def parse_relation(
    expression: str, variables: dict[str, dict[str, Any]]
) -> tuple[sp.Expr, sp.Expr]:
    if expression.count("=") != 1:
        raise ValueError("scalar_sympy relations require exactly one '='")
    left, right = expression.split("=", 1)
    return _parse_scalar(left, variables), _parse_scalar(right, variables)


def _canonical_residual(residual: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    rational = sp.cancel(sp.together(residual))
    numerator, denominator = sp.fraction(rational)
    numerator = sp.expand(numerator)
    content, primitive = numerator.as_content_primitive(clear=True)
    if content != 0:
        numerator = sp.expand(primitive)
    positive = sp.srepr(numerator)
    negative = sp.srepr(-numerator)
    if negative < positive:
        numerator = -numerator
    denominator = sp.factor(denominator)
    return numerator, denominator


def _dimension_key(value: dict[str, Any] | None) -> str:
    if value is None:
        return "unknown"
    cleaned = {key: str(Fraction(item)) for key, item in sorted(value.items()) if item != 0}
    return _json(cleaned)


def _dimension_add(left: dict[str, Fraction], right: dict[str, Fraction]) -> dict[str, Fraction]:
    keys = set(left) | set(right)
    return {key: left.get(key, Fraction()) + right.get(key, Fraction()) for key in keys}


def _dimension_scale(value: dict[str, Fraction], scale: Fraction) -> dict[str, Fraction]:
    return {key: item * scale for key, item in value.items() if item * scale}


def _dimension_of(
    expression: sp.Expr, dimensions: dict[sp.Symbol, dict[str, Fraction] | None]
) -> dict[str, Fraction] | None:
    if expression.is_Number or expression in {sp.pi, sp.E}:
        return {}
    if isinstance(expression, sp.Symbol):
        return dimensions.get(expression)
    if isinstance(expression, sp.Add):
        items = [_dimension_of(arg, dimensions) for arg in expression.args]
        if any(item is None for item in items):
            return None
        first = items[0]
        if any(item != first for item in items[1:]):
            raise ValueError("additive terms have incompatible dimensions")
        return first
    if isinstance(expression, sp.Mul):
        total: dict[str, Fraction] = {}
        for arg in expression.args:
            item = _dimension_of(arg, dimensions)
            if item is None:
                return None
            total = _dimension_add(total, item)
        return {key: value for key, value in total.items() if value}
    if isinstance(expression, sp.Pow):
        base, exponent = expression.args
        if not exponent.is_Rational:
            return None
        base_dimension = _dimension_of(base, dimensions)
        if base_dimension is None:
            return None
        return _dimension_scale(base_dimension, Fraction(int(exponent.p), int(exponent.q)))
    if expression.func in {sp.Abs}:
        return _dimension_of(expression.args[0], dimensions)
    if expression.func in {sp.exp, sp.log, sp.sin, sp.cos, sp.tan}:
        argument = _dimension_of(expression.args[0], dimensions)
        if argument is None:
            return None
        if argument:
            raise ValueError(f"{expression.func.__name__} requires a dimensionless argument")
        return {}
    return None


def dimension_audit(
    left: sp.Expr, right: sp.Expr, variables: dict[str, dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    symbol_dimensions: dict[sp.Symbol, dict[str, Fraction] | None] = {}
    for name, variable in variables.items():
        raw = variable.get("dimension")
        symbol_dimensions[sp.Symbol(name, real=True)] = (
            None
            if raw is None
            else {key: Fraction(str(value)) for key, value in raw.items() if value != 0}
        )
    try:
        left_dimension = _dimension_of(left, symbol_dimensions)
        right_dimension = _dimension_of(right, symbol_dimensions)
    except ValueError as exc:
        return "fail", {"reason": str(exc)}
    detail = {
        "left": None if left_dimension is None else _dimension_key(left_dimension),
        "right": None if right_dimension is None else _dimension_key(right_dimension),
    }
    if left_dimension is None or right_dimension is None:
        return "unknown", detail
    if left_dimension != right_dimension:
        return "fail", {**detail, "reason": "left and right dimensions differ"}
    return "pass", detail


def _variable_map(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for variable in record.get("variables", []):
        symbol = str(variable["symbol"])
        if not _NAME.fullmatch(symbol):
            raise ValueError(f"invalid variable symbol: {symbol}")
        if symbol in result:
            raise ValueError(f"duplicate variable symbol: {symbol}")
        canonical = str(variable.get("canonical_name", symbol))
        if not _NAME.fullmatch(canonical):
            raise ValueError(f"invalid canonical variable name: {canonical}")
        result[symbol] = {
            "canonical_name": canonical,
            "meaning": str(variable.get("meaning", "")),
            "dimension": variable.get("dimension"),
            "field_kind": str(variable.get("field_kind", "scalar")),
            "tensor_rank": int(variable.get("tensor_rank", 0)),
        }
    return result


def _feature_counter(expression: sp.Expr) -> Counter[str]:
    features: Counter[str] = Counter()
    for node in sp.preorder_traversal(expression):
        if isinstance(node, sp.Symbol):
            features["Symbol"] += 1
        elif node.is_Number:
            features["Number"] += 1
        else:
            features[node.func.__name__] += 1
    return features


def _structural_form(
    expression: sp.Expr, canonical_variables: dict[str, dict[str, Any]]
) -> tuple[str, bool]:
    symbols = sorted(expression.free_symbols, key=lambda item: item.name)
    groups: dict[tuple[str, str, int], list[sp.Symbol]] = {}
    for symbol in symbols:
        variable = canonical_variables.get(symbol.name, {})
        key = (
            _dimension_key(variable.get("dimension")),
            str(variable.get("field_kind", "scalar")),
            int(variable.get("tensor_rank", 0)),
        )
        groups.setdefault(key, []).append(symbol)
    permutation_count = math.prod(math.factorial(len(group)) for group in groups.values())
    if permutation_count > 40320:
        fallback = {
            symbol: sp.Symbol(f"X{index}") for index, symbol in enumerate(symbols)
        }
        return sp.srepr(expression.xreplace(fallback)), False
    group_options: list[list[dict[sp.Symbol, sp.Symbol]]] = []
    offset = 0
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda item: item.name)
        targets = [sp.Symbol(f"X{index}") for index in range(offset, offset + len(group))]
        offset += len(group)
        group_options.append(
            [dict(zip(group, permutation, strict=True)) for permutation in itertools.permutations(targets)]
        )
    candidates: list[str] = []
    for choices in itertools.product(*group_options):
        mapping: dict[sp.Symbol, sp.Symbol] = {}
        for choice in choices:
            mapping.update(choice)
        candidate, _ = _canonical_residual(expression.xreplace(mapping))
        candidates.append(sp.srepr(candidate))
    return min(candidates, default=sp.srepr(expression)), True


def canonicalize_record(record: dict[str, Any]) -> dict[str, Any]:
    representation = str(record.get("representation", "scalar_sympy"))
    if representation not in _ALLOWED_REPRESENTATIONS:
        raise ValueError(f"unsupported representation: {representation}")
    variables = _variable_map(record)
    if representation != "scalar_sympy":
        normalized = " ".join(str(record["expression"]).split())
        features = {"representation": representation, "token_count": len(normalized.split())}
        return {
            "variables": variables,
            "normalized_expression": normalized,
            "denominator_guard": None,
            "semantic_hash": _sha256_text(f"{representation}\0{normalized}"),
            "structural_hash": None,
            "structural_hash_complete": False,
            "features": features,
            "dimension_status": "unknown",
            "dimension_detail": {"reason": "non-scalar representation"},
        }
    left, right = parse_relation(str(record["expression"]), variables)
    aliases = {
        sp.Symbol(symbol, real=True): sp.Symbol(variable["canonical_name"], real=True)
        for symbol, variable in variables.items()
    }
    canonical_variables = {
        variable["canonical_name"]: variable for variable in variables.values()
    }
    residual, denominator = _canonical_residual((left - right).xreplace(aliases))
    normalized = sp.srepr(residual)
    semantic_hash = _sha256_text(normalized)
    structural_form, structural_complete = _structural_form(residual, canonical_variables)
    dimension_status, dimension_detail = dimension_audit(left, right, variables)
    return {
        "variables": variables,
        "normalized_expression": str(residual),
        "denominator_guard": None if denominator == 1 else str(denominator),
        "semantic_hash": semantic_hash,
        "structural_hash": _sha256_text(structural_form),
        "structural_hash_complete": structural_complete,
        "features": dict(sorted(_feature_counter(residual).items())),
        "dimension_status": dimension_status,
        "dimension_detail": dimension_detail,
    }


def _cosine_features(left: dict[str, int], right: dict[str, int]) -> float:
    left_numeric = {key: value for key, value in left.items() if isinstance(value, int | float)}
    right_numeric = {key: value for key, value in right.items() if isinstance(value, int | float)}
    keys = set(left_numeric) | set(right_numeric)
    numerator = sum(left_numeric.get(key, 0) * right_numeric.get(key, 0) for key in keys)
    left_norm = math.sqrt(sum(value * value for value in left_numeric.values()))
    right_norm = math.sqrt(sum(value * value for value in right_numeric.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


class EquationUniverse:
    def __init__(self, database: str | Path):
        self.database = Path(database)

    def initialize(self, *, replace: bool = False) -> None:
        if self.database.exists():
            if not replace:
                raise FileExistsError(f"database already exists: {self.database}")
            self.database.unlink()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database) as connection:
            connection.executescript(SCHEMA)
            connection.executemany(
                "INSERT INTO metadata(key,value) VALUES (?,?)",
                [
                    ("schema_version", SCHEMA_VERSION),
                    ("created_utc", _utc_now()),
                    (
                        "novelty_policy",
                        "unmatched means not found in this corpus; it never proves novelty",
                    ),
                ],
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def register_generator_history(
        self,
        manifest: str | Path,
        basis: str | Path,
        survivor_directory: str | Path | None = None,
        *,
        name: str = "Sigma Generator v2 compact formula history",
    ) -> dict[str, Any]:
        from .formula_history import GeneratorFormulaHistory

        history = GeneratorFormulaHistory(manifest, basis, survivor_directory)
        description = history.describe()
        space_id = _stable_id(
            "SPACE",
            description["protocol_version"],
            description["manifest_sha256"],
        )
        with self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO formula_spaces(
                space_id,name,protocol_version,generator_version,manifest_path,manifest_sha256,
                basis_path,basis_sha256,survivor_directory,basis_count,max_action_terms,
                total_declared_actions,processed_actions,survivor_count,complete_declared_space,
                query_semantics,registered_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    space_id,
                    name,
                    description["protocol_version"],
                    description["generator_version"],
                    str(history.manifest_path),
                    description["manifest_sha256"],
                    str(history.basis_path),
                    description["basis_sha256"],
                    str(history.survivor_directory),
                    description["basis_count"],
                    description["max_action_terms"],
                    description["total_declared_actions"],
                    description["processed_actions"],
                    description["survivor_count"],
                    int(description["complete_declared_space"]),
                    (
                        "exact grammar decomposition plus deterministic ordinal lookup; compact "
                        "survivor binary search; no scientific-validity inference"
                    ),
                    _utc_now(),
                ),
            )
        return {"space_id": space_id, **description}

    def import_file(self, path: str | Path) -> dict[str, Any]:
        input_path = Path(path).resolve()
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported equation-universe import schema")
        started = _utc_now()
        rejected: list[dict[str, str]] = []
        source_count = equation_count = derivation_count = 0
        with self._connect() as connection:
            for source in payload.get("sources", []):
                mode = str(source["ingestion_mode"])
                if mode not in _ALLOWED_SOURCE_MODES:
                    raise ValueError(f"unsupported source ingestion mode: {mode}")
                connection.execute(
                    """INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        source["source_id"],
                        source["title"],
                        source["url"],
                        _json(source.get("authors", [])),
                        source.get("year"),
                        source.get("source_kind", "unknown"),
                        source.get("license_id"),
                        source.get("license_url"),
                        mode,
                        source.get("policy_reason", ""),
                        source.get("accessed_utc"),
                    ),
                )
                source_count += 1
            source_modes = {
                row["source_id"]: row["ingestion_mode"]
                for row in connection.execute("SELECT source_id,ingestion_mode FROM sources")
            }
            for record in payload.get("equations", []):
                try:
                    mode = source_modes[record["source_id"]]
                    independent = bool(record.get("independently_encoded", False))
                    if mode == "blocked":
                        raise ValueError("source policy blocks equation ingestion")
                    if mode == "metadata_only" and not independent:
                        raise ValueError(
                            "metadata-only sources require an independently encoded formula"
                        )
                    canonical = canonicalize_record(record)
                    if canonical["dimension_status"] == "fail":
                        raise ValueError(
                            f"dimension audit failed: {canonical['dimension_detail']}"
                        )
                    record_hash = _sha256_text(_json(record))
                    connection.execute(
                        """INSERT INTO equations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            record["equation_id"],
                            record["name"],
                            record.get("domain", "unknown"),
                            record.get("representation", "scalar_sympy"),
                            record["expression"],
                            record.get("latex"),
                            canonical["normalized_expression"],
                            canonical["denominator_guard"],
                            canonical["semantic_hash"],
                            canonical["structural_hash"],
                            _json(canonical["features"]),
                            canonical["dimension_status"],
                            _json(canonical["dimension_detail"]),
                            _json(record.get("assumptions", [])),
                            _json(record.get("validity", [])),
                            _json(record.get("tags", [])),
                            record["source_id"],
                            record.get("source_locator"),
                            int(independent),
                            record_hash,
                            _utc_now(),
                        ),
                    )
                    connection.executemany(
                        "INSERT INTO variables VALUES (?,?,?,?,?,?,?)",
                        [
                            (
                                record["equation_id"],
                                symbol,
                                value["canonical_name"],
                                value["meaning"],
                                None
                                if value["dimension"] is None
                                else _json(value["dimension"]),
                                value["field_kind"],
                                value["tensor_rank"],
                            )
                            for symbol, value in canonical["variables"].items()
                        ],
                    )
                    equation_count += 1
                except (KeyError, TypeError, ValueError, sqlite3.IntegrityError) as exc:
                    rejected.append(
                        {"equation_id": str(record.get("equation_id", "missing")), "reason": str(exc)}
                    )
            self._build_equivalence_edges(connection)
            for derivation in payload.get("derivations", []):
                try:
                    proof = self._verify_derivation(connection, derivation)
                    connection.execute(
                        "INSERT INTO derivations VALUES (?,?,?,?,?,?,?,?)",
                        (
                            derivation["derivation_id"],
                            derivation["target_equation_id"],
                            derivation["operation"],
                            _json(derivation.get("assumptions", {})),
                            proof["method"],
                            proof["status"],
                            _json(proof),
                            derivation.get("source_id"),
                        ),
                    )
                    connection.executemany(
                        "INSERT INTO derivation_inputs VALUES (?,?,?)",
                        [
                            (derivation["derivation_id"], index, equation_id)
                            for index, equation_id in enumerate(derivation["inputs"])
                        ],
                    )
                    derivation_count += 1
                except (KeyError, TypeError, ValueError, sqlite3.IntegrityError) as exc:
                    rejected.append(
                        {
                            "derivation_id": str(
                                derivation.get("derivation_id", "missing")
                            ),
                            "reason": str(exc),
                        }
                    )
            completed = _utc_now()
            import_id = _stable_id("IMP", input_path, _sha256_path(input_path), completed)
            connection.execute(
                "INSERT INTO import_runs VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    import_id,
                    str(input_path),
                    _sha256_path(input_path),
                    started,
                    completed,
                    source_count,
                    equation_count,
                    derivation_count,
                    len(rejected),
                    _json({"rejected": rejected}),
                ),
            )
        return {
            "import_id": import_id,
            "sources": source_count,
            "equations": equation_count,
            "derivations": derivation_count,
            "rejected": rejected,
        }

    def _build_equivalence_edges(self, connection: sqlite3.Connection) -> None:
        rows = list(
            connection.execute(
                "SELECT equation_id,semantic_hash,structural_hash FROM equations "
                "WHERE semantic_hash IS NOT NULL ORDER BY equation_id"
            )
        )
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                equivalence_type = None
                if left["semantic_hash"] == right["semantic_hash"]:
                    equivalence_type = "semantic_algebraic"
                elif (
                    left["structural_hash"]
                    and left["structural_hash"] == right["structural_hash"]
                ):
                    equivalence_type = "alpha_structural"
                if equivalence_type:
                    edge_id = _stable_id(
                        "EQV", left["equation_id"], right["equation_id"], equivalence_type
                    )
                    connection.execute(
                        "INSERT OR IGNORE INTO equivalence_edges VALUES (?,?,?,?,?,?)",
                        (
                            edge_id,
                            left["equation_id"],
                            right["equation_id"],
                            equivalence_type,
                            "canonical_hash",
                            _json({"hash_equal": True}),
                        ),
                    )

    def _equation_parse_data(
        self, connection: sqlite3.Connection, equation_id: str
    ) -> tuple[sp.Expr, list[sp.Symbol]]:
        row = connection.execute(
            "SELECT representation,expression FROM equations WHERE equation_id=?",
            (equation_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown equation: {equation_id}")
        if row["representation"] != "scalar_sympy":
            raise ValueError(f"derivation input is not scalar_sympy: {equation_id}")
        variables = {
            item["symbol"]: {
                "canonical_name": item["canonical_name"],
                "dimension": None
                if item["dimension_json"] is None
                else json.loads(item["dimension_json"]),
                "field_kind": item["field_kind"],
                "tensor_rank": item["tensor_rank"],
            }
            for item in connection.execute(
                "SELECT * FROM variables WHERE equation_id=?", (equation_id,)
            )
        }
        left, right = parse_relation(row["expression"], variables)
        aliases = {
            sp.Symbol(name, real=True): sp.Symbol(value["canonical_name"], real=True)
            for name, value in variables.items()
        }
        residual, _ = _canonical_residual((left - right).xreplace(aliases))
        return residual, sorted(residual.free_symbols, key=lambda item: item.name)

    def _verify_derivation(
        self, connection: sqlite3.Connection, derivation: dict[str, Any]
    ) -> dict[str, Any]:
        inputs: list[sp.Expr] = []
        symbols: set[sp.Symbol] = set()
        for equation_id in derivation["inputs"]:
            expression, used = self._equation_parse_data(connection, equation_id)
            inputs.append(expression)
            symbols.update(used)
        target, used = self._equation_parse_data(
            connection, derivation["target_equation_id"]
        )
        symbols.update(used)
        nonzero = derivation.get("assumptions", {}).get("nonzero", [])
        for index, name in enumerate(nonzero):
            symbol = sp.Symbol(name, real=True)
            if symbol not in symbols:
                raise ValueError(f"nonzero assumption names unused symbol: {name}")
            auxiliary = sp.Symbol(f"_nz{index}")
            symbols.add(auxiliary)
            inputs.append(sp.expand(auxiliary * symbol - 1))
        ordered = sorted(symbols, key=lambda item: item.name)
        try:
            basis = sp.groebner(inputs, *ordered, order="grevlex")
            _, remainder = basis.reduce(target)
            remainder = sp.factor(remainder)
            status = "verified" if remainder == 0 else "unverified"
            return {
                "method": "groebner_ideal_membership_with_nonzero_saturation",
                "status": status,
                "remainder": str(remainder),
                "basis_size": len(basis.polys),
                "nonzero_assumptions": list(nonzero),
            }
        except (sp.PolynomialError, ValueError) as exc:
            return {
                "method": "groebner_ideal_membership_with_nonzero_saturation",
                "status": "unsupported",
                "reason": str(exc),
                "nonzero_assumptions": list(nonzero),
            }

    def classify(self, record: dict[str, Any], *, nearest_limit: int = 5) -> dict[str, Any]:
        canonical = canonicalize_record(record)
        with self._connect() as connection:
            semantic = [
                dict(row)
                for row in connection.execute(
                    "SELECT equation_id,name,domain,source_id FROM equations WHERE semantic_hash=?",
                    (canonical["semantic_hash"],),
                )
            ]
            structural = [
                dict(row)
                for row in connection.execute(
                    "SELECT equation_id,name,domain,source_id FROM equations "
                    "WHERE structural_hash=? AND semantic_hash<>?",
                    (canonical["structural_hash"], canonical["semantic_hash"]),
                )
            ]
            nearest: list[dict[str, Any]] = []
            for row in connection.execute(
                "SELECT equation_id,name,domain,feature_json FROM equations "
                "WHERE representation='scalar_sympy'"
            ):
                score = _cosine_features(
                    canonical["features"], json.loads(row["feature_json"])
                )
                nearest.append(
                    {
                        "equation_id": row["equation_id"],
                        "name": row["name"],
                        "domain": row["domain"],
                        "structural_feature_similarity": score,
                    }
                )
            formula_spaces = [
                dict(row)
                for row in connection.execute(
                    "SELECT space_id,name,protocol_version,manifest_path,basis_path,"
                    "survivor_directory FROM formula_spaces ORDER BY space_id"
                )
            ]
        generator_history: list[dict[str, Any]] = []
        formula_space_expression = record.get("formula_space_expression")
        if formula_space_expression is not None:
            from .formula_history import GeneratorFormulaHistory

            for space in formula_spaces:
                try:
                    history = GeneratorFormulaHistory(
                        space["manifest_path"],
                        space["basis_path"],
                        space["survivor_directory"],
                    )
                    result = history.query(str(formula_space_expression))
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                    result = {
                        "status": "history_query_error",
                        "exact_project_history_match": False,
                        "error": f"{type(error).__name__}: {error}",
                        "scientific_validity_claimed": False,
                    }
                generator_history.append(
                    {
                        "space_id": space["space_id"],
                        "name": space["name"],
                        **result,
                    }
                )
        nearest.sort(
            key=lambda item: (-item["structural_feature_similarity"], item["equation_id"])
        )
        if semantic:
            classification = "known_semantic_equivalent"
        elif structural:
            classification = "known_structural_analogue"
        elif any(item.get("exact_project_history_match") for item in generator_history):
            classification = "known_project_history_exact"
        else:
            classification = "not_found_in_corpus"
        return {
            "classification": classification,
            "novelty_claim_allowed": False,
            "novelty_warning": (
                "absence from this finite corpus does not establish scientific or legal novelty"
            ),
            "canonical": canonical,
            "semantic_matches": semantic,
            "structural_matches": structural,
            "generator_history_matches": generator_history,
            "nearest": nearest[:nearest_limit],
        }

    def audit(self) -> dict[str, Any]:
        with self._connect() as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            counts = {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "sources",
                    "equations",
                    "variables",
                    "derivations",
                    "equivalence_edges",
                    "import_runs",
                    "formula_spaces",
                )
            }
            dimension_counts = {
                row["dimension_status"]: row["count"]
                for row in connection.execute(
                    "SELECT dimension_status,COUNT(*) AS count FROM equations "
                    "GROUP BY dimension_status"
                )
            }
            proof_counts = {
                row["proof_status"]: row["count"]
                for row in connection.execute(
                    "SELECT proof_status,COUNT(*) AS count FROM derivations GROUP BY proof_status"
                )
            }
            source_modes = {
                row["ingestion_mode"]: row["count"]
                for row in connection.execute(
                    "SELECT ingestion_mode,COUNT(*) AS count FROM sources GROUP BY ingestion_mode"
                )
            }
            formula_space_coverage = [
                dict(row)
                for row in connection.execute(
                    "SELECT space_id,protocol_version,total_declared_actions,processed_actions,"
                    "survivor_count,complete_declared_space FROM formula_spaces ORDER BY space_id"
                )
            ]
            unproven = [
                dict(row)
                for row in connection.execute(
                    "SELECT derivation_id,target_equation_id,proof_status FROM derivations "
                    "WHERE proof_status<>'verified'"
                )
            ]
        return {
            "schema_version": SCHEMA_VERSION,
            "database": str(self.database.resolve()),
            "integrity_check": integrity,
            "counts": counts,
            "dimension_status": dimension_counts,
            "derivation_proofs": proof_counts,
            "source_ingestion_modes": source_modes,
            "formula_space_coverage": formula_space_coverage,
            "unproven_derivations": unproven,
            "novelty_policy": (
                "canonical matches identify known overlap; unmatched records are only absent "
                "from this corpus and may not be labeled novel"
            ),
            "passed": integrity == "ok" and not unproven,
        }


def build_equation_universe(
    seed: str | Path, database: str | Path, report: str | Path, *, replace: bool = False
) -> dict[str, Any]:
    universe = EquationUniverse(database)
    universe.initialize(replace=replace)
    imported = universe.import_file(seed)
    audit = universe.audit()
    output = {"import": imported, "audit": audit}
    report_path = Path(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
