"""Render a sealed seven-galaxy Refracted Gravity manuscript evidence packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_rg_seven_galaxy_2d_manuscript_evidence_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_seven_galaxy_2d_manuscript_evidence_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_seven_galaxy_2d_manuscript_evidence_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-rg-seven-galaxy-2d-manuscript-evidence-v1/receipt.json"
)
ARTIFACT_DIRECTORY = Path(
    "runs/gravity/open-gravity-rg-seven-galaxy-2d-manuscript-evidence-v1/artifacts"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-seven-galaxy-2d-manuscript-evidence-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-rg-seven-galaxy-2d-manuscript-evidence-receipt-1.0"
_CONFIG_RAW_SHA256 = "8e3b196df428fa81d1596d17da895955dada60ccb68ecdc01c9e3f07a0fa79e0"
_CONFIG_CONTENT_SHA256 = "e1b4b1d2955abbb82696f3bea7ac9f3929463d8a6896fb4706cf4fa6e127b9fd"
_MODULE_SEMANTIC_SHA256 = "c8e79f7a406752bec27b21b764108a5842885cbd4ffafceeb424c87f29bf7fc8"
_TEST_RAW_SHA256 = "467b8c62713f45c2932c3e690ee3fa318822d3e0961d90ea6c50aed016f1fc9b"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')

_MODEL_KEYS = {
    "Newton": "NEWTON_3D_DST",
    "RAR": "RAR_2016_ON_NEWTON_3D",
    "MOND": "MOND_STANDARD_MU_ON_NEWTON_3D",
    "RG": "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG",
}
_MODEL_COLORS = {"Newton": "#4C78A8", "RAR": "#F58518", "MOND": "#54A24B", "RG": "#B279A2"}
_OBJECT_ORDER = ("UGC04305", "NGC2841", "IC2574", "DDO154", "NGC5055", "NGC6946", "NGC7331")


class ManuscriptEvidenceError(RuntimeError):
    """Raised when a manuscript evidence gate fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManuscriptEvidenceError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path) -> str:
    normalized, count = _MODULE_PIN_PATTERN.subn(
        rb"\g<1>" + b"0" * 64 + rb"\g<2>", path.read_bytes()
    )
    _require(count == 1, "module semantic pin pattern changed")
    return hashlib.sha256(normalized).hexdigest()


def _repo_path(relative: Path | str) -> Path:
    path = (_ROOT / relative).resolve()
    _require(path == _ROOT or _ROOT in path.parents, "path escaped repository")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManuscriptEvidenceError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def _package_bindings() -> dict[str, str]:
    return {
        "config_raw_sha256": _CONFIG_RAW_SHA256,
        "config_content_sha256": _CONFIG_CONTENT_SHA256,
        "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
        "test_raw_sha256": _TEST_RAW_SHA256,
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if _CONFIG_CONTENT_SHA256 != "0" * 64:
        _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "FROZEN_MANUSCRIPT_EVIDENCE_BUILD_FOR_BOUNDED_CORPUS_2D_RG_STRESS_TEST",
        "status changed",
    )
    _require(
        [row["role"] for row in config["input_bindings"]]
        == [
            "SIX_EXTERNAL_FIXED_2D_SCORE",
            "HOLMBERG_II_FIXED_2D_SCORE",
            "SEVEN_GALAXY_ROBUSTNESS",
            "BOUNDED_PRIMARY_LITERATURE_SYNTHESIS",
        ],
        "input inventory changed",
    )
    for binding in config["input_bindings"]:
        _require(len(binding["artifacts"]) == 4, "input artifact count changed")
        _require(
            all(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) for row in binding["artifacts"]),
            "invalid input artifact seal",
        )
        _require(
            re.fullmatch(r"[0-9a-f]{64}", binding["receipt_content_sha256"]) is not None,
            "invalid receipt content seal",
        )
    artifacts = config["artifact_contract"]
    _require(artifacts["directory"] == ARTIFACT_DIRECTORY.as_posix(), "artifact directory changed")
    _require(
        [(row["role"], row["path"], row["rows"]) for row in artifacts["artifacts"]]
        == [
            ("PRIMARY_RMSE_TABLE", "table-1-seven-primary-cells.csv", 7),
            ("ALL_SENSITIVITY_CELLS_TABLE", "table-2-all-48-sensitivity-cells.csv", 48),
            ("LEAVE_ONE_OUT_TABLE", "table-3-six-leave-one-out-reaggregations.csv", 6),
            ("PRIMARY_RMSE_FIGURE", "figure-1-seven-primary-rmse.svg", 7),
            ("RG_CELL_IMPROVEMENT_FIGURE", "figure-2-rg-vs-best-comparator-all-cells.svg", 48),
            ("MANUSCRIPT_SUMMARY", "manuscript-summary.md", 1),
        ],
        "artifact contract changed",
    )
    _require(artifacts["raw_response_maps_reopened"] is False, "raw response reopening enabled")
    _require(artifacts["raw_response_pixels_rendered"] is False, "raw response rendering enabled")
    manuscript = config["manuscript_contract"]
    _require(len(manuscript["required_caveats"]) == 5, "caveat inventory changed")
    _require(len(manuscript["forbidden_claims"]) == 6, "forbidden claims changed")
    claims = config["claim_boundary"]
    _require(claims["bounded_corpus_method_novelty_supported"] is True, "method boundary changed")
    _require(claims["ic2574_conditional_signal_retained"] is True, "IC2574 boundary changed")
    for key in (
        "deterministic_manuscript_artifacts_built",
        "universal_rg_replication",
        "unique_theory_established",
        "publication_ready",
    ):
        _require(claims[key] is False, f"claim promoted in frozen config: {key}")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output changed")


def _validate_package() -> None:
    if _MODULE_SEMANTIC_SHA256 != "0" * 64:
        _require(
            module_semantic_sha256(_repo_path(MODULE_PATH)) == _MODULE_SEMANTIC_SHA256,
            "module changed",
        )
    if _TEST_RAW_SHA256 != "0" * 64:
        _require(file_sha256(_repo_path(TEST_PATH)) == _TEST_RAW_SHA256, "tests changed")


def load_config(*, verify_package: bool = True) -> dict[str, Any]:
    path = _repo_path(CONFIG_PATH)
    if _CONFIG_RAW_SHA256 != "0" * 64:
        _require(file_sha256(path) == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "config")
    validate_config(config)
    if verify_package:
        _validate_package()
    return config


def _load_inputs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    inputs: dict[str, dict[str, Any]] = {}
    for binding in config["input_bindings"]:
        receipt_path: Path | None = None
        for artifact in binding["artifacts"]:
            path = _repo_path(artifact["path"])
            _require(path.is_file(), f"input missing: {artifact['path']}")
            _require(file_sha256(path) == artifact["sha256"], f"input changed: {artifact['path']}")
            if artifact["path"].endswith("/receipt.json"):
                receipt_path = path
        _require(receipt_path is not None, "input receipt missing")
        receipt = _read_json(receipt_path, f"{binding['role']} receipt")
        _require(
            receipt["content_sha256"] == binding["receipt_content_sha256"],
            f"receipt content seal changed: {binding['role']}",
        )
        expected_content = content_sha256({**receipt, "content_sha256": ""})
        _require(
            expected_content == receipt["content_sha256"],
            f"receipt self-hash failed: {binding['role']}",
        )
        inputs[binding["role"]] = receipt
    return inputs


def _object_id(score: Mapping[str, Any]) -> str:
    if "object_id" in score:
        return str(score["object_id"])
    cell = str(score["cell_score_id"])
    return cell.split("__", 1)[0]


def _score_row(score: Mapping[str, Any], sample_role: str, *, primary: bool) -> dict[str, Any]:
    rmses = {
        label: float(score["models"][key]["rmse_m_s"]) / 1000.0
        for label, key in _MODEL_KEYS.items()
    }
    best_comparator = min(("Newton", "RAR", "MOND"), key=lambda label: rmses[label])
    improvement = (rmses[best_comparator] - rmses["RG"]) / rmses[best_comparator]
    return {
        "sample_role": sample_role,
        "object_id": _object_id(score),
        "cell_score_id": score["cell_score_id"],
        "primary_cell": primary,
        "inclination_deg": float(score["inclination_deg"]),
        "conversion_cell_id": score["conversion_cell_id"],
        "resolution": score["resolution"],
        "common_pixel_count": int(score["common_pixel_count"]),
        "beam_equivalent_count": float(score["beam_equivalent_count"]),
        "newton_rmse_km_s": rmses["Newton"],
        "rar_rmse_km_s": rmses["RAR"],
        "mond_rmse_km_s": rmses["MOND"],
        "rg_rmse_km_s": rmses["RG"],
        "winner": score["winner"],
        "best_comparator": best_comparator,
        "rg_fractional_improvement_over_best_comparator": improvement,
        "rg_wins_cell": bool(score["rg_beats_all_three_comparators"]),
    }


def _rows(
    inputs: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    external = inputs["SIX_EXTERNAL_FIXED_2D_SCORE"]
    holmberg = inputs["HOLMBERG_II_FIXED_2D_SCORE"]
    robustness = inputs["SEVEN_GALAXY_ROBUSTNESS"]
    external_primary_ids = {
        row["cell_score_id"] for row in external["primary_adjudication"]["primary_cells"]
    }
    holmberg_primary_id = holmberg["primary_cell"]["cell_score_id"]
    all_rows = [
        _score_row(
            row, "HOLMBERG_II_DEVELOPMENT", primary=row["cell_score_id"] == holmberg_primary_id
        )
        for row in holmberg["scores"]
    ] + [
        _score_row(row, "SIX_GALAXY_EXTERNAL", primary=row["cell_score_id"] in external_primary_ids)
        for row in external["scores"]
    ]
    _require(len(all_rows) == 48, "combined score row count changed")
    _require(len({row["cell_score_id"] for row in all_rows}) == 48, "duplicate score cell")
    primary = sorted(
        (row for row in all_rows if row["primary_cell"]),
        key=lambda row: _OBJECT_ORDER.index(row["object_id"]),
    )
    _require(len(primary) == 7, "primary row count changed")
    loo = []
    for row in robustness["leave_one_object_out"]:
        values = row["equal_object_mean_rmse_m_s"]
        loo.append(
            {
                "omitted_object_id": row["omitted_object_id"],
                "retained_object_count": int(row["retained_object_count"]),
                "newton_mean_rmse_km_s": float(values[_MODEL_KEYS["Newton"]]) / 1000.0,
                "rar_mean_rmse_km_s": float(values[_MODEL_KEYS["RAR"]]) / 1000.0,
                "mond_mean_rmse_km_s": float(values[_MODEL_KEYS["MOND"]]) / 1000.0,
                "rg_mean_rmse_km_s": float(values[_MODEL_KEYS["RG"]]) / 1000.0,
                "rg_rank": int(row["rg_rank"]),
                "rg_fractional_improvement_over_newton": float(
                    row["rg_fractional_improvement_over_newton"]
                ),
            }
        )
    _require(len(loo) == 6, "leave-one-out row count changed")
    return primary, all_rows, loo


def _format_csv(rows: list[dict[str, Any]], columns: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        rendered: dict[str, Any] = {}
        for key in columns:
            value = row[key]
            if type(value) is float:
                rendered[key] = format(value, ".12g")
            elif type(value) is bool:
                rendered[key] = "true" if value else "false"
            else:
                rendered[key] = value
        writer.writerow(rendered)
    return stream.getvalue().encode("utf-8")


def _primary_svg(rows: list[dict[str, Any]]) -> bytes:
    width, height = 1120, 620
    left, right, top, bottom = 85, 30, 55, 120
    plot_w, plot_h = width - left - right, height - top - bottom
    ymax = max(row[f"{label.lower()}_rmse_km_s"] for row in rows for label in _MODEL_KEYS)
    ymax = max(10.0, ymax * 1.08)
    xs = [left + index * plot_w / (len(rows) - 1) for index in range(len(rows))]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.series{fill:none;stroke-width:2.5}.point{stroke:white;stroke-width:1}</style>",
        '<text x="560" y="28" text-anchor="middle" font-size="19">Seven preregistered primary cells: velocity-map RMSE</text>',
    ]
    for tick in range(6):
        value = ymax * tick / 5
        y = top + plot_h * (1 - tick / 5)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{y:.3f}" x2="{width - right}" y2="{y:.3f}"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.3f}" text-anchor="end" font-size="12">{value:.1f}</text>'
        )
    parts += [
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}"/>',
        f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" text-anchor="middle" font-size="14">RMSE (km/s)</text>',
    ]
    for x, row in zip(xs, rows, strict=True):
        label = "Ho II" if row["object_id"] == "UGC04305" else row["object_id"]
        parts.append(
            f'<text x="{x:.3f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="12">{html.escape(label)}</text>'
        )
        parts.append(
            f'<text x="{x:.3f}" y="{top + plot_h + 42}" text-anchor="middle" font-size="11">i={row["inclination_deg"]:.1f}°</text>'
        )
    for series_index, label in enumerate(_MODEL_KEYS):
        values = [row[f"{label.lower()}_rmse_km_s"] for row in rows]
        ys = [top + plot_h * (1 - value / ymax) for value in values]
        points = " ".join(f"{x:.3f},{y:.3f}" for x, y in zip(xs, ys, strict=True))
        color = _MODEL_COLORS[label]
        parts.append(f'<polyline class="series" stroke="{color}" points="{points}"/>')
        for x, y in zip(xs, ys, strict=True):
            parts.append(f'<circle class="point" cx="{x:.3f}" cy="{y:.3f}" r="4" fill="{color}"/>')
        legend_x = left + series_index * 145
        parts.append(
            f'<line x1="{legend_x}" y1="{height - 30}" x2="{legend_x + 24}" y2="{height - 30}" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{legend_x + 30}" y="{height - 26}" font-size="12">{html.escape(label)}</text>'
        )
    parts.append("</svg>")
    return ("\n".join(parts) + "\n").encode("utf-8")


def _improvement_svg(rows: list[dict[str, Any]]) -> bytes:
    ordered = sorted(
        rows, key=lambda row: (_OBJECT_ORDER.index(row["object_id"]), row["cell_score_id"])
    )
    width, height = 1200, 620
    left, right, top, bottom = 85, 25, 55, 120
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [row["rg_fractional_improvement_over_best_comparator"] for row in ordered]
    ymin, ymax = min(-1.05, min(values) * 1.08), max(0.5, max(values) * 1.08)

    def y_of(value: float) -> float:
        return top + plot_h * (ymax - value) / (ymax - ymin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1}.zero{stroke:#111;stroke-width:1.5}.separator{stroke:#ddd;stroke-width:1}</style>",
        '<text x="600" y="28" text-anchor="middle" font-size="19">RG fractional RMSE improvement over the best comparator in each cell</text>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
        f'<line class="zero" x1="{left}" y1="{y_of(0):.3f}" x2="{width - right}" y2="{y_of(0):.3f}"/>',
        f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" text-anchor="middle" font-size="14">(best comparator − RG) / best comparator</text>',
    ]
    for tick in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5):
        if ymin <= tick <= ymax:
            y = y_of(tick)
            parts.append(
                f'<text x="{left - 10}" y="{y + 4:.3f}" text-anchor="end" font-size="11">{tick:.2f}</text>'
            )
    counts = Counter(row["object_id"] for row in ordered)
    cursor = 0
    point_index = 0
    for object_id in _OBJECT_ORDER:
        count = counts[object_id]
        start = cursor
        for _ in range(count):
            row = ordered[point_index]
            x = left + (point_index + 0.5) * plot_w / len(ordered)
            value = row["rg_fractional_improvement_over_best_comparator"]
            color = "#2E8B57" if value > 0 else "#C44E52"
            parts.append(f'<circle cx="{x:.3f}" cy="{y_of(value):.3f}" r="4.5" fill="{color}"/>')
            point_index += 1
        cursor += count
        center = left + (start + count / 2) * plot_w / len(ordered)
        label = "Ho II" if object_id == "UGC04305" else object_id
        parts.append(
            f'<text x="{center:.3f}" y="{top + plot_h + 27}" text-anchor="middle" font-size="12">{html.escape(label)}</text>'
        )
        if cursor < len(ordered):
            xsep = left + cursor * plot_w / len(ordered)
            parts.append(
                f'<line class="separator" x1="{xsep:.3f}" y1="{top}" x2="{xsep:.3f}" y2="{top + plot_h}"/>'
            )
    parts += [
        f'<circle cx="{left}" cy="{height - 31}" r="4.5" fill="#2E8B57"/><text x="{left + 12}" y="{height - 27}" font-size="12">RG wins cell</text>',
        f'<circle cx="{left + 135}" cy="{height - 31}" r="4.5" fill="#C44E52"/><text x="{left + 147}" y="{height - 27}" font-size="12">Comparator wins cell</text>',
        "</svg>",
    ]
    return ("\n".join(parts) + "\n").encode("utf-8")


def _markdown(
    primary: list[dict[str, Any]], all_rows: list[dict[str, Any]], loo: list[dict[str, Any]]
) -> bytes:
    external_primary = [row for row in primary if row["sample_role"] == "SIX_GALAXY_EXTERNAL"]
    winner_counts = Counter(row["winner"] for row in all_rows)
    rg_cells = [row for row in all_rows if row["rg_wins_cell"]]
    ic2574_rg = [row for row in rg_cells if row["object_id"] == "IC2574"]
    rg_rank_one = sum(row["rg_rank"] == 1 for row in loo)
    text = f"""# A two-dimensional H I velocity-field stress test of Refracted Gravity across seven nearby galaxies

## Plain-language result

The same published Refracted Gravity parameter set was used without retuning. It did not win any of the six preregistered external primary galaxy tests. Across all source and map-resolution sensitivity cells, a localized IC 2574 pattern remains worth a targeted follow-up, but it is not evidence that Refracted Gravity works generally.

## Method

We formed response-blind, source-derived two-dimensional line-of-sight velocity predictions for Newtonian baryons, the empirical radial-acceleration relation (RAR), standard algebraic MOND, and Refracted Gravity (RG). We then opened the already sealed THINGS moment-1 response maps and applied one fixed pixel-level RMSE score with a rotation sign and systemic velocity shared by every model within each object-resolution cell. The source construction is **MODEL_LIFTED_2P5D**, not a full three-dimensional reconstruction.

The evidence set contains one Holmberg II development object and six externally selected galaxies. The external primary cell used the frozen fixed 3.6-micron mass-to-light conversion, published inclination, and natural-weighted THINGS map for each galaxy. Source conversions and resolutions are sensitivity cells, not independent galaxies.

## Results

- RG external primary wins: **0/{len(external_primary)}**.
- Winners over all 48 sensitivity cells: Newtonian baryons **{winner_counts[_MODEL_KEYS["Newton"]]}**, RAR **{winner_counts[_MODEL_KEYS["RAR"]]}**, MOND **{winner_counts[_MODEL_KEYS["MOND"]]}**, RG **{winner_counts[_MODEL_KEYS["RG"]]}**.
- External RG-winning sensitivity cells: **{len([row for row in rg_cells if row["sample_role"] == "SIX_GALAXY_EXTERNAL"])}/30**, all belonging to IC 2574.
- IC 2574 RG-winning cells: **{len(ic2574_rg)}/4**; RG loses its preregistered fixed-conversion natural-resolution primary cell.
- Leave-one-object-out external reaggregations with RG ranked first: **{rg_rank_one}/{len(loo)}**.

The six-galaxy equal-object primary mean gives RG only a 0.126% advantage over Newtonian baryons, while MOND has the lowest mean RMSE. This is below the preregistered two-percent diagnostic materiality threshold. NGC 6946, the low-inclination external object, provides a direct counterexample to a simple inclination-only explanation of the earlier Holmberg II sensitivity.

## What may be publishable

The defensible contribution is a reproducible, bounded-corpus two-dimensional RG velocity-field stress-test method and its negative/constraint result. No prior two-dimensional velocity-map RG application was identified in the frozen six-paper primary RG corpus. This is a bounded-corpus finding, not a global priority claim.

The result does **not** establish RG, a new theory of gravity, a general three-dimensional solution, an inclination mechanism, or elimination of dark matter. It is not publication-ready without independent scientific review.

## Required caveats

1. The source model is MODEL_LIFTED_2P5D.
2. Rotation sign and systemic velocity are response-derived but shared across models.
3. There is no gas-dynamical forward model.
4. No p-values or independent confirmation are claimed.
5. Source-conversion and angular-resolution cells are sensitivity analyses, not additional galaxies.

## Decisive next test

The next targeted experiment is an IC 2574 source-completeness and cube-level kinematic test using an independent stellar tracer and a gas-dynamical forward model. A true full-3D extension must bind an observed three-dimensional source or explicitly retain its vertical-density assumptions as model-lifted uncertainty.
"""
    return text.encode("utf-8")


def build_artifacts(config: Mapping[str, Any]) -> tuple[dict[str, bytes], dict[str, Any]]:
    inputs = _load_inputs(config)
    primary, all_rows, loo = _rows(inputs)
    score_columns = (
        "sample_role",
        "object_id",
        "cell_score_id",
        "primary_cell",
        "inclination_deg",
        "conversion_cell_id",
        "resolution",
        "common_pixel_count",
        "beam_equivalent_count",
        "newton_rmse_km_s",
        "rar_rmse_km_s",
        "mond_rmse_km_s",
        "rg_rmse_km_s",
        "winner",
        "best_comparator",
        "rg_fractional_improvement_over_best_comparator",
        "rg_wins_cell",
    )
    loo_columns = (
        "omitted_object_id",
        "retained_object_count",
        "newton_mean_rmse_km_s",
        "rar_mean_rmse_km_s",
        "mond_mean_rmse_km_s",
        "rg_mean_rmse_km_s",
        "rg_rank",
        "rg_fractional_improvement_over_newton",
    )
    payloads = {
        "table-1-seven-primary-cells.csv": _format_csv(primary, score_columns),
        "table-2-all-48-sensitivity-cells.csv": _format_csv(all_rows, score_columns),
        "table-3-six-leave-one-out-reaggregations.csv": _format_csv(loo, loo_columns),
        "figure-1-seven-primary-rmse.svg": _primary_svg(primary),
        "figure-2-rg-vs-best-comparator-all-cells.svg": _improvement_svg(all_rows),
        "manuscript-summary.md": _markdown(primary, all_rows, loo),
    }
    contract = {row["path"]: row for row in config["artifact_contract"]["artifacts"]}
    _require(set(payloads) == set(contract), "artifact payload inventory changed")
    index = []
    for name in sorted(payloads):
        payload = payloads[name]
        index.append(
            {
                "role": contract[name]["role"],
                "path": f"{ARTIFACT_DIRECTORY.as_posix()}/{name}",
                "format": contract[name]["format"],
                "rows": contract[name]["rows"],
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    summary = {
        "primary_objects": len(primary),
        "all_score_cells": len(all_rows),
        "leave_one_out_reaggregations": len(loo),
        "external_primary_rg_wins": sum(
            row["sample_role"] == "SIX_GALAXY_EXTERNAL" and row["rg_wins_cell"] for row in primary
        ),
        "all_rg_winning_cells": sum(row["rg_wins_cell"] for row in all_rows),
        "external_rg_winning_cells": sum(
            row["sample_role"] == "SIX_GALAXY_EXTERNAL" and row["rg_wins_cell"] for row in all_rows
        ),
        "ic2574_rg_winning_cells": sum(
            row["object_id"] == "IC2574" and row["rg_wins_cell"] for row in all_rows
        ),
        "loo_rg_rank_one": sum(row["rg_rank"] == 1 for row in loo),
        "winner_counts": dict(sorted(Counter(row["winner"] for row in all_rows).items())),
        "artifact_index": index,
        "artifact_index_sha256": content_sha256(index),
    }
    return payloads, summary


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    _, summary = build_artifacts(config)
    claims = dict(config["claim_boundary"])
    claims["deterministic_manuscript_artifacts_built"] = True
    receipt = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_DETERMINISTIC_SEVEN_GALAXY_2D_RG_MANUSCRIPT_EVIDENCE_BUILT",
        "decision": "PUBLICATION_CANDIDATE_BOUNDED_CORPUS_2D_RG_METHOD_AND_UNIVERSAL_REPLICATION_FAILURE",
        "package_bindings": _package_bindings(),
        "input_receipt_content_sha256": {
            row["role"]: row["receipt_content_sha256"] for row in config["input_bindings"]
        },
        "evidence_summary": summary,
        "scientific_boundary": {
            "raw_response_maps_reopened": False,
            "raw_response_pixels_rendered": False,
            "new_scores_computed": 0,
            "p_values_computed": 0,
            "model_parameters_retuned": 0,
            "independent_confirmation": False,
            "source_model_class": "MODEL_LIFTED_2P5D",
        },
        "claim_boundary": claims,
        "content_sha256": "",
    }
    receipt["content_sha256"] = content_sha256({**receipt, "content_sha256": ""})
    return receipt


def validate_receipt(config: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    _require(dict(receipt) == build_receipt(config), "receipt differs from deterministic rebuild")


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "refusing nonidentical overwrite")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == payload, "concurrent nonidentical output")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_packet() -> str:
    config = load_config()
    payloads, _ = build_artifacts(config)
    states = []
    for name in sorted(payloads):
        states.append(_atomic_no_clobber(_repo_path(ARTIFACT_DIRECTORY / name), payloads[name]))
    receipt = build_receipt(config)
    validate_receipt(config, receipt)
    states.append(_atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt) + b"\n"))
    return (
        "EXISTING_IDENTICAL"
        if all(state == "EXISTING_IDENTICAL" for state in states)
        else "CREATED"
    )


def check_packet() -> str:
    config = load_config()
    payloads, _ = build_artifacts(config)
    for name, payload in payloads.items():
        path = _repo_path(ARTIFACT_DIRECTORY / name)
        _require(path.is_file(), f"artifact missing: {name}")
        _require(path.read_bytes() == payload, f"artifact differs from rebuild: {name}")
    path = _repo_path(OUTPUT_PATH)
    _require(path.is_file(), "receipt missing")
    validate_receipt(config, _read_json(path, "receipt"))
    return "VALID"


def status() -> dict[str, Any]:
    config = load_config()
    if not _repo_path(OUTPUT_PATH).is_file():
        return {"package_id": config["package_id"], "status": "FROZEN_UNBUILT"}
    receipt = _read_json(_repo_path(OUTPUT_PATH), "receipt")
    validate_receipt(config, receipt)
    return {
        "package_id": config["package_id"],
        "status": receipt["status"],
        "decision": receipt["decision"],
        "artifacts": len(receipt["evidence_summary"]["artifact_index"]),
        "publication_ready": receipt["claim_boundary"]["publication_ready"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("check")
    sub.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        print(write_packet())
    elif arguments.command == "check":
        print(check_packet())
    else:
        print(json.dumps(status(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
