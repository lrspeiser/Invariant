from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sigma_theory_compiler import external_creativity_multi_host as M
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / M.OUTPUT_PATH


def test_downloaded_multi_host_receipt_is_sealed_and_cross_platform() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    M.validate_receipt(value)
    assert value["acquisition"]["artifact_bytes_downloaded_and_hashed"]
    assert value["reproduction"]["received_machines"] == 4
    assert value["reproduction"]["distinct_runner_ids"] == 4
    assert value["reproduction"]["distinct_operating_systems"] == [
        "ubuntu-latest",
        "windows-latest",
    ]
    assert value["lean"]["kernel_checked"]
    assert not value["claim_boundary"]["physical_bare_metal_identity_claimed"]


def test_multi_host_receipt_rejects_collapsed_runner_identity() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    changed = deepcopy(value)
    for host in changed["hosts"]:
        host["runner_id"] = 1
    body = {key: item for key, item in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(M.MultiHostReproductionError, match="policy"):
        M.validate_receipt(changed)
