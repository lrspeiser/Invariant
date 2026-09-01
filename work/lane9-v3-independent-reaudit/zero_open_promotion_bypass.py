from __future__ import annotations

import tempfile
from pathlib import Path

from sigma_theory_compiler import open_gravity_void_correlation_development_release_v3 as release


source_paths = {
    (release.REPO_ROOT / section["path"]).resolve()
    for section in release.load_config()["sources"].values()
}
original_open = Path.open
scientific_opens: list[str] = []


def guarded_open(path: Path, *args, **kwargs):
    resolved = path.resolve()
    if resolved in source_paths:
        scientific_opens.append(str(resolved))
        raise AssertionError(f"scientific source opened: {resolved}")
    return original_open(path, *args, **kwargs)


with tempfile.TemporaryDirectory(prefix="lane9-v3-zero-open-") as temporary:
    root = Path(temporary).resolve()
    release.REPO_ROOT = root
    release.FINAL_DIRECTORY = root / "runs/gravity/open-gravity-void-correlation-development-score-v3"
    release.STAGING_ROOT = root / "work/open-gravity-void-correlation-development-score-v3-staging"
    release.validate_package_payloads_v3 = lambda *args, **kwargs: {}
    Path.open = guarded_open
    try:
        payloads = {
            name: b"{}\n" for name in release._ARTIFACT_NAMES | {"receipt.json"}
        }
        first = release._promote_fixed_payloads(payloads, b"not-even-json", {})
        second = release._promote_fixed_payloads(payloads, b"not-even-json", {})
    finally:
        Path.open = original_open

assert first == "PROMOTED_COMPLETE"
assert second == "EXISTING_IDENTICAL"
assert scientific_opens == []
print("ZERO_OPEN_CALLER_PAYLOAD_PROMOTED_WITHOUT_MARKER_OR_VALID_AUTHORIZATION")
