from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .quartic_tc2_mixed_third_jet_continuation_service import (
    run_mixed_third_jet_continuation_service,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_second_atom_artifacts(project_root: Path) -> list[dict[str, Any]]:
    runs = project_root / "runs" / "physics-language"
    paths = [
        runs / "quartic-tc2-second-atom-chunk-campaign" / "campaign.json",
        *[
            runs / f"quartic-tc2-second-atom-chunk{offset}-campaign" / "campaign.json"
            for offset in range(64, 704, 64)
        ],
        *sorted((runs / "quartic-tc2-continuous-service" / "chunks").glob("offset-*.json")),
    ]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("canonical second-atom artifact sequence is incomplete")
    return [_load(path) for path in paths]


def run_parallel_epoch(
    project_root: Path,
    *,
    config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config = _load(config_path)
    runs = project_root / "runs" / "physics-language"
    diagonal = _load(runs / "quartic-tc2-diagonal-third-jet-campaign" / "campaign.json")
    quadratic = _load(runs / "quartic-tc2-quadratic-deltak-extension-campaign" / "campaign.json")
    return run_mixed_third_jet_continuation_service(
        project_root / str(config["initial_prior_path"]),
        project_root / str(config["initial_chunk_config_path"]),
        diagonal,
        quadratic,
        _canonical_second_atom_artifacts(project_root),
        config,
        output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one hash-bound parallel mixed-third-jet continuation epoch."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/backgrounds/quartic_tc2_mixed_third_jet_parallel_continuation_service.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/physics-language/quartic-tc2-mixed-third-jet-parallel-continuation-service"
        ),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    output_path = args.output if args.output.is_absolute() else project_root / args.output
    result = run_parallel_epoch(
        project_root,
        config_path=config_path,
        output_path=output_path,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
