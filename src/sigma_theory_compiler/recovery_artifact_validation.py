from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

BOUND_PREDECESSOR_KEYS = {"path", "file_sha256", "content_sha256"}
DATA_SEALS = {
    "observations_opened": False,
    "dark_matter_or_halo_inputs_opened": False,
    "redshift_or_cosmology_inputs_opened": False,
    "paid_llm_calls": False,
    "formal_or_theory_pass_promoted": False,
    "global_H7_closed": False,
    "lifespan_proved": False,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash_matches(value: dict[str, Any]) -> bool:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return value.get("content_sha256") == hashlib.sha256(canonical_json(body).encode()).hexdigest()


def validate_bound_inputs(
    config: dict[str, Any],
    inputs: dict[str, dict[str, Any]],
    expected_config_keys: set[str],
) -> None:
    if set(config) != expected_config_keys | {"predecessors"}:
        raise ValueError("campaign config key set mismatch")
    bindings = config.get("predecessors")
    if not isinstance(bindings, dict) or set(bindings) != set(inputs):
        raise ValueError("predecessor binding key set mismatch")
    for label, value in inputs.items():
        binding = bindings[label]
        if not isinstance(binding, dict) or set(binding) != BOUND_PREDECESSOR_KEYS:
            raise ValueError(f"{label} predecessor binding contract mismatch")
        if not content_hash_matches(value):
            raise ValueError(f"{label} predecessor content hash mismatch")
        if value.get("content_sha256") != binding.get("content_sha256"):
            raise ValueError(f"{label} predecessor is not the registered artifact")
        for hash_key in ("file_sha256", "content_sha256"):
            digest = binding.get(hash_key)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{label} predecessor {hash_key} is invalid")


def load_bound_inputs(
    root: Path,
    config: dict[str, Any],
    labels: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    resolved_root = root.resolve()
    loaded: dict[str, dict[str, Any]] = {}
    bindings = config.get("predecessors")
    if not isinstance(bindings, dict) or set(bindings) != set(labels):
        raise ValueError("predecessor binding key set mismatch")
    for label in labels:
        binding = bindings[label]
        if not isinstance(binding, dict) or set(binding) != BOUND_PREDECESSOR_KEYS:
            raise ValueError(f"{label} predecessor binding contract mismatch")
        path = (resolved_root / str(binding["path"])).resolve()
        if resolved_root != path and resolved_root not in path.parents:
            raise ValueError(f"{label} predecessor path escapes repository root")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != binding["file_sha256"]:
            raise ValueError(f"{label} predecessor file hash mismatch")
        value = json.loads(raw.decode("utf-8"))
        if (
            not isinstance(value, dict)
            or not content_hash_matches(value)
            or value.get("content_sha256") != binding["content_sha256"]
        ):
            raise ValueError(f"{label} predecessor content binding mismatch")
        loaded[label] = value
    return loaded


def validate_exact_rebuild(artifact: dict[str, Any], rebuilt: dict[str, Any]) -> None:
    if not content_hash_matches(artifact):
        raise ValueError("artifact content hash mismatch")
    if artifact.get("data_seals") != DATA_SEALS:
        raise ValueError("artifact data seals mismatch")
    if rebuilt.get("status") == "reject":
        raise ValueError("registered predecessor reconstruction rejected")
    if artifact != rebuilt:
        raise ValueError("artifact differs from deterministic reconstruction")
