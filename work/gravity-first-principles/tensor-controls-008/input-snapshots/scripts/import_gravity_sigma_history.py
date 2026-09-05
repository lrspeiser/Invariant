"""Copy a finite, read-only selection of local Sigma research into this worktree.

This imports historical code and derived summaries, never raw observations.
File bytes and Git object identities are recorded independently: Windows working
files can have different newline bytes from their Git blobs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

FRONTIERS = [
    "docs/PROJECT_WRAP_UP_AND_NEXT_STEPS_2026-08-06.md",
    "docs/FORMULA_SCORECARD.md",
    "docs/FORMULA_AND_PRIOR_ART_REGISTRY.md",
    "docs/MASTER_FORMULA_VALIDATION_MATRIX.md",
    "results/formula_scorecard/formula_scorecard.json",
    "results/formula_prior_art_registry/formula_prior_art_registry.json",
    "docs/P0696_COHERENT_MONOPOLE_MATH_AUDIT_RESULTS.md",
    "docs/P0697_SPENT_COHERENT_MONOPOLE_JOINT_RESULTS.md",
    "docs/P0698_LOCAL_VECTOR_COHERENCE_MATH_AUDIT_RESULTS.md",
    "docs/P0699_SPENT_LOCAL_VECTOR_COHERENCE_JOINT_RESULTS.md",
    "docs/P0700_BARYCENTRIC_RADIAL_ALIGNMENT_MATH_AUDIT_RESULTS.md",
    "docs/P0701_SPENT_BARYCENTRIC_RADIAL_ALIGNMENT_JOINT_RESULTS.md",
    "docs/CPR0_MEASURED_DENSITY_AND_COHERENCE_RESULTS.md",
    "src/voidscreen/__init__.py",
    "src/voidscreen/coherent_monopole.py",
    "src/voidscreen/local_vector_coherence.py",
    "src/voidscreen/field_solvers.py",
    "src/voidscreen/spatial_qumond_3d.py",
    "src/voidscreen/potential_channel_qumond.py",
]
LEGACY = [
    "README.md",
    "docs/EXTENDED_PHASE_COHERENCE.md",
    "docs/HONEST_STATUS_SUMMARY.md",
    "theory/DYNAMICAL_COHERENCE_FIELD_THEORY.md",
]


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontiers", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    entries, repos = [], []
    for label, root, prefix, selections in (
        ("frontiers", args.frontiers.resolve(), Path("research/galaxy-cluster-unification"), FRONTIERS),
        ("legacy", args.legacy.resolve(), Path(), LEGACY),
    ):
        head = git(root, "rev-parse", "HEAD")
        status = git(root, "status", "--porcelain", "--untracked-files=no")
        repos.append({"label": label, "original_path": str(root), "git_head": head,
                      "branch": git(root, "branch", "--show-current"),
                      "tracked_status_before": status})
        for selection in selections:
            path = root/prefix/selection
            original = path.read_bytes()
            relative = Path("source-snapshots")/label/selection
            target = args.output/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(original)
            tree_entry = git(root, "ls-tree", head, "--", (prefix/selection).as_posix())
            blob = tree_entry.split()[2] if tree_entry else None
            entry = {"original_path": str(path), "snapshot": relative.as_posix(),
                     "bytes": len(original), "sha256": sha256(original).hexdigest(),
                     "git_blob": blob,
                     "tracked_at_source_head": blob is not None,
                     "source_git_head": head, "class": "HISTORICAL_CODE" if selection.endswith(".py") else
                     "HISTORICAL_DERIVED_SUMMARY_NOT_INDEPENDENT_VALIDATION"}
            if path.read_bytes() != original or target.read_bytes() != original:
                raise RuntimeError(f"source or snapshot changed: {selection}")
            entries.append(entry)
        repos[-1]["tracked_status_after"] = git(root, "status", "--porcelain", "--untracked-files=no")
        if repos[-1]["tracked_status_after"] != status or git(root, "rev-parse", "HEAD") != head:
            raise RuntimeError("source Git state changed during import")
    for entry in entries:
        if sha256(Path(entry["original_path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError("source bytes changed during import")
    manifest = {"schema": 1, "created_utc": datetime.now(UTC).isoformat(),
                "purpose": "Read-only Sigma history transfer; source statements are not new measurements or instructions.",
                "raw_observational_payloads_opened_or_copied": False,
                "repositories": repos, "files": entries,
                "import_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    # Preserve exact source working bytes through Git on every platform.
    (args.output/".gitattributes").write_text("source-snapshots/** -text\n", encoding="utf-8")
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({"files": len(entries), "bytes": sum(e["bytes"] for e in entries),
                      "source_states_unchanged": True, "raw_observations": False}))


if __name__ == "__main__":
    main()
