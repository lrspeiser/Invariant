"""Verify only this benchmark's files. --seal creates immutable delivery receipts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
OWNED = ["scripts/mond_atlas_light_projection.py", "scripts/run_mond_atlas_light_projection.py",
         "configs/mond_atlas_light_projection_v1.json", "tests/test_mond_atlas_light_projection.py"]
PRIVATE = ROOT/"work/private/mond-atlas-light-projection-001"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def create(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_run(name):
    folder = HERE/name
    hashes = load(folder/"sha256-manifest.json")
    for relative, expected in hashes.items():
        assert digest(folder/relative) == expected, (name, relative, "run hash mismatch")
    for relative in OWNED:
        assert digest(ROOT/relative) == digest(folder/"snapshot"/relative), (name, relative, "working file changed")
    summary = load(folder/"summary.json")
    assert summary["required_passed"] and not summary["required_failures"], name
    assert summary["config_sha256"] == digest(ROOT/OWNED[2]), name
    assert summary["declaration_sha256"] == digest(HERE/"preimplementation-declaration.json"), name
    return summary, len(hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    if args.seal:
        first, first_count = verify_run("run-001")
        second, second_count = verify_run("run-002")
        identical = (HERE/"run-001/checks.json").read_bytes() == (HERE/"run-002/checks.json").read_bytes()
        assert identical, "check records differ across replays"
        assert first["input_sha256"] == second["input_sha256"], "input hashes changed"
        for relative, expected in first["input_sha256"].items():
            assert digest(ROOT/relative) == expected, relative
        cfg = load(ROOT/OWNED[2])
        declaration = load(HERE/"preimplementation-declaration.json")
        assert declaration["config_sha256"] == digest(ROOT/OWNED[2])
        assert declaration["implementation_files_present"] is False
        assert cfg["disposition_before_implementation"] == "THEORY_BENCHMARK_ONLY"
        receipt = dict(disposition="THEORY_BENCHMARK_ONLY", verified_utc=datetime.now(timezone.utc).isoformat(),
                       checks_byte_identical=identical, input_hashes_identical=True,
                       run_manifest_entries_verified=[first_count, second_count],
                       required_checks_per_run=first["required_check_count"],
                       required_failures_per_run=first["required_failures"],
                       diagnostic_failures_per_run=len(first["retained_diagnostic_target_failures"]),
                       checks_sha256=digest(HERE/"run-001/checks.json"),
                       config_sha256=digest(ROOT/OWNED[2]),
                       declaration_sha256=digest(HERE/"preimplementation-declaration.json"),
                       design_sha256=digest(HERE/"benchmark-design.json"),
                       input_mode="cached theory references; benchmark and tests contain no network calls",
                       observational_inputs_opened=False, sample_exposure_changed=False,
                       scope="same environment replay; not a cross-platform floating-point identity claim")
        create(HERE/"replay-verification.json", receipt)
        files = set(ROOT/path for path in OWNED)
        files.update(p for p in HERE.rglob("*") if p.is_file() and p.name != "delivery-manifest.json")
        files.update(p for p in PRIVATE.rglob("*") if p.is_file())
        manifest = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}
        create(HERE/"delivery-manifest.json", dict(schema="mond-atlas-light-projection-delivery-v1", files=manifest))
    manifest = load(HERE/"delivery-manifest.json")
    for relative, expected in manifest["files"].items():
        assert digest(ROOT/relative) == expected, (relative, "delivery hash mismatch")
    print(json.dumps(dict(verified_files=len(manifest["files"]),
                          delivery_manifest_sha256=digest(HERE/"delivery-manifest.json"),
                          replay=load(HERE/"replay-verification.json")), indent=2))


if __name__ == "__main__":
    main()
