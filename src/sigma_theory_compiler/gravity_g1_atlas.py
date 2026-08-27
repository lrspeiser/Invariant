"""G1 production atlas: 100 million target-blind candidates for each of 139 galaxies."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .gravity_g0_experiment import load_config as load_g0_config
from .gravity_g1_pilot import _binding, _file_sha256, _load_json
from .gravity_g1_pilot_v3 import (
    load_config as load_pilot_config,
)
from .gravity_g1_pilot_v3 import (
    search_arm_galaxy,
)
from .gravity_g1_pilot_v3 import (
    validate_receipt as validate_pilot_receipt,
)
from .sigma_core import canonical_json_bytes, canonical_sha256
from .sparc_full_sample import assemble

SCHEMA = "invariant-gravity-g1-atlas-1.0"
CHECKPOINT_SCHEMA = "invariant-gravity-g1-atlas-checkpoint-1.0"
CONFIG_SCHEMA = "invariant-gravity-g1-atlas-config-1.0"
CONFIG_PATH = "configs/gravity_g1_atlas.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_g1_atlas.py"
TEST_PATH = "tests/test_gravity_g1_atlas.py"
OUTPUT_PATH = "runs/gravity/g1-atlas/galaxy-formula-atlas-v1.json"


class GravityG1AtlasError(ValueError):
    """The production atlas contract, checkpoint, or result is inconsistent."""


def load_config(root: Path) -> Mapping[str, Any]:
    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityG1AtlasError("G1 atlas config schema changed")
    binding = config.get("pilot_binding")
    if not isinstance(binding, Mapping):
        raise GravityG1AtlasError("G1 atlas has no pilot binding")
    path = root / str(binding.get("path"))
    if _file_sha256(path) != binding.get("file_sha256"):
        raise GravityG1AtlasError("G1 atlas pilot file binding changed")
    receipt = _load_json(path)
    validate_pilot_receipt(receipt, root=root)
    if receipt.get("content_sha256") != binding.get("content_sha256"):
        raise GravityG1AtlasError("G1 atlas pilot content binding changed")
    if receipt.get("decision") != binding.get("required_decision"):
        raise GravityG1AtlasError("G1 atlas is not pilot-authorized")
    segments = config.get("segments")
    if not isinstance(segments, list) or sum(int(row["candidate_count"]) for row in segments) != int(
        config["candidate_budget_per_galaxy"]
    ):
        raise GravityG1AtlasError("G1 atlas candidate budget does not sum exactly")
    population = assemble(root)
    if len(population.exploration) != int(config["population"]["galaxies"]):
        raise GravityG1AtlasError("G1 atlas exploration population changed")
    if sum(galaxy.count for galaxy in population.exploration) != int(
        config["population"]["points"]
    ):
        raise GravityG1AtlasError("G1 atlas exploration point count changed")
    return config


def _checkpoint_bindings(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "config_content_sha256": canonical_sha256(config),
        "source": _binding(root, SOURCE_PATH),
        "test": _binding(root, TEST_PATH),
    }


def _checkpoint_path(root: Path, config: Mapping[str, Any], galaxy: str) -> Path:
    return root / str(config["checkpoint"]["directory"]) / f"{galaxy}.json"


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    root: Path,
    config: Mapping[str, Any],
    expected_galaxy: str,
) -> None:
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise GravityG1AtlasError("G1 atlas checkpoint schema changed")
    body = dict(checkpoint)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityG1AtlasError("G1 atlas checkpoint content seal changed")
    if checkpoint.get("galaxy") != expected_galaxy:
        raise GravityG1AtlasError("G1 atlas checkpoint crossed galaxies")
    if checkpoint.get("source_bindings") != _checkpoint_bindings(root, config):
        raise GravityG1AtlasError("G1 atlas checkpoint source binding changed")
    if checkpoint.get("confirmation_evaluator_access_count") != 0:
        raise GravityG1AtlasError("G1 atlas checkpoint reports confirmation access")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if path.exists() and path.read_bytes() != payload:
        raise GravityG1AtlasError(f"refusing to overwrite a different atlas artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _runtime_pilot_config(root: Path, config: Mapping[str, Any]) -> Mapping[str, Any]:
    pilot = copy.deepcopy(load_pilot_config(root))
    pilot["gpu_prefilter"]["retained_candidates_per_arm_galaxy_for_cpu_replay"] = int(
        config["evaluator"]["retained_gpu_candidates_per_segment_galaxy"]
    )
    return pilot


def run_galaxy(
    root: Path,
    galaxy: Any,
    config: Mapping[str, Any],
    *,
    candidate_count_override: int | None = None,
    use_gpu: bool = True,
) -> dict[str, Any]:
    g0_config = load_g0_config(root)
    pilot_config = _runtime_pilot_config(root, config)
    arm_by_id = {row["id"]: row for row in pilot_config["arms"]}
    trials = []
    total = 0
    for segment in config["segments"]:
        count = int(
            segment["candidate_count"]
            if candidate_count_override is None
            else candidate_count_override
        )
        arm = arm_by_id[segment["v3_arm"]]
        trial = search_arm_galaxy(
            galaxy,
            arm,
            pilot_config,
            g0_config,
            candidate_count=count,
            use_gpu=use_gpu,
        )
        trial = {**trial, "segment_id": segment["id"]}
        trials.append(trial)
        total += count
    admitted = [
        {**candidate, "segment_id": trial["segment_id"]}
        for trial in trials
        for candidate in trial["cpu_fp64_admitted_pareto"]
    ]
    admitted.sort(
        key=lambda item: (
            float(item["aggregate_score"]["chi_square"]),
            int(item["description_length"]["total_bits"]),
            str(item["segment_id"]),
            int(item["ordinal"]),
        )
    )
    body: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "candidate_count": total,
        "confirmation_evaluator_access_count": 0,
        "covered": bool(admitted),
        "galaxy": galaxy.name,
        "point_count": galaxy.count,
        "retained_pareto": admitted[:64],
        "source_bindings": _checkpoint_bindings(root, config),
        "trials": trials,
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def build_atlas(
    root: Path,
    *,
    galaxy_limit: int | None = None,
    candidate_count_override: int | None = None,
    use_gpu: bool = True,
    persist_checkpoints: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    population = assemble(root)
    galaxies = list(population.exploration)
    if galaxy_limit is not None:
        galaxies = galaxies[:galaxy_limit]
    rows = []
    for index, galaxy in enumerate(galaxies, start=1):
        checkpoint_path = _checkpoint_path(root, config, galaxy.name)
        full_trial = candidate_count_override is None and use_gpu
        if full_trial and checkpoint_path.is_file():
            checkpoint = _load_json(checkpoint_path)
            validate_checkpoint(
                checkpoint, root=root, config=config, expected_galaxy=galaxy.name
            )
        else:
            checkpoint = run_galaxy(
                root,
                galaxy,
                config,
                candidate_count_override=candidate_count_override,
                use_gpu=use_gpu,
            )
            if full_trial and persist_checkpoints:
                _write_immutable(checkpoint_path, checkpoint)
        rows.append(checkpoint)
        print(
            json.dumps(
                {
                    "candidate_count": checkpoint["candidate_count"],
                    "covered": checkpoint["covered"],
                    "galaxy": galaxy.name,
                    "progress": f"{index}/{len(galaxies)}",
                    "retained": len(checkpoint["retained_pareto"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    full_run = (
        galaxy_limit is None
        and candidate_count_override is None
        and use_gpu
        and len(rows) == 139
        and all(row["candidate_count"] == 100_000_000 for row in rows)
    )
    covered = sum(row["covered"] for row in rows)
    passed = full_run and covered == 139
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "G1",
        "decision": "PASS_G1_ATLAS_139_OF_139" if passed else "BLOCK_G1_ATLAS_INCOMPLETE",
        "claims": {
            "alternative_to_gr_discovered": False,
            "confirmation_galaxy_evaluated": False,
            "formula_is_universal": False,
            "historical_novelty_established": False,
            "g2_equivalence_authorized": passed,
        },
        "config": {"content_sha256": canonical_sha256(config), "path": CONFIG_PATH},
        "counts": {
            "candidate_galaxy_trials": sum(int(row["candidate_count"]) for row in rows),
            "confirmation_evaluator_accesses": 0,
            "covered_galaxies": covered,
            "exploration_galaxies": len(rows),
            "exploration_points": sum(int(row["point_count"]) for row in rows),
        },
        "galaxies": rows,
        "limitations": [
            "This is a collection of per-galaxy formulas with two local acceleration coefficients, not one law.",
            "Cross-validation supports within-galaxy interpolation robustness but not prediction of a new galaxy.",
            "Discrete grammar choices were selected on exploration folds and are charged in description length.",
            "G2 must collapse algebraic and behavioral equivalences before population-level meta-law search.",
        ],
        "source_bindings": {
            "config": _binding(root, CONFIG_PATH),
            "source": _binding(root, SOURCE_PATH),
            "test": _binding(root, TEST_PATH),
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_atlas(atlas: Mapping[str, Any], *, root: Path) -> None:
    root = root.resolve()
    if atlas.get("schema_version") != SCHEMA:
        raise GravityG1AtlasError("G1 atlas schema changed")
    body = dict(atlas)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityG1AtlasError("G1 atlas content seal changed")
    config = load_config(root)
    if atlas.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityG1AtlasError("G1 atlas config binding changed")
    bindings = atlas.get("source_bindings", {})
    for key, relative in (("config", CONFIG_PATH), ("source", SOURCE_PATH), ("test", TEST_PATH)):
        if bindings.get(key) != _binding(root, relative):
            raise GravityG1AtlasError(f"G1 atlas {key} binding changed")
    if atlas.get("counts", {}).get("confirmation_evaluator_accesses") != 0:
        raise GravityG1AtlasError("G1 atlas reports confirmation access")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    parser.add_argument("--galaxy-limit", type=int)
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if args.validate_checked:
        validate_atlas(_load_json(output), root=root)
        return 0
    atlas = build_atlas(
        root,
        galaxy_limit=args.galaxy_limit,
        candidate_count_override=args.candidate_count,
        use_gpu=not args.cpu_only,
        persist_checkpoints=True,
    )
    _write_immutable(output, atlas)
    print(json.dumps({"content_sha256": atlas["content_sha256"], "decision": atlas["decision"]}))
    return 0 if atlas["decision"] == "PASS_G1_ATLAS_139_OF_139" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GravityG1AtlasError",
    "build_atlas",
    "load_config",
    "run_galaxy",
    "validate_atlas",
    "validate_checkpoint",
]
