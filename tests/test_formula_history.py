from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from sigma_theory_compiler.equation_universe import EquationUniverse, build_equation_universe
from sigma_theory_compiler.formula_history import (
    GeneratorFormulaHistory,
    encode_ordinal,
)
from sigma_theory_compiler.high_throughput import decode_ordinal

ROOT = Path(__file__).resolve().parents[1]


def test_encode_ordinal_inverts_generator_decoder():
    for ordinal in (0, 1, 99, 1_000_000, 1_088_651_719):
        decoded = decode_ordinal(50, 6, ordinal)
        sign_mask = sum(
            1 << position
            for position, sign in enumerate(decoded["signs"])
            if sign > 0
        )
        assert encode_ordinal(50, 6, decoded["term_ids"], sign_mask) == ordinal


def test_compact_history_finds_survivor_and_rejected_formula(tmp_path):
    basis = [
        {
            "id": 0,
            "px": 0,
            "pq": 0,
            "pz": 1,
            "transform": "Identity",
            "dimension_l": 0,
            "dimension_t": 0,
            "derivative_order_in_h": 0,
            "has_measured_state": True,
            "high_field_growth_numerator": 0,
            "high_field_growth_denominator": 1,
            "expression": "z",
        }
    ]
    basis_path = tmp_path / "basis.json"
    basis_path.write_text(json.dumps(basis, indent=2), encoding="utf-8")
    canonical_basis = json.dumps(basis, separators=(",", ":"), ensure_ascii=False).encode()
    survivor_dir = tmp_path / "survivors"
    survivor_dir.mkdir()
    survivor_file = survivor_dir / "block.bin"
    header = struct.Struct("<8sHHQQQQ")
    record = struct.Struct("<QBBH6H")
    survivor_file.write_bytes(
        header.pack(b"SGSURV2\0", 1, record.size, 0, 0, 2, 1)
        + record.pack(1, 1, 1, 0, 0, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF)
    )
    manifest = {
        "protocol_version": "TEST-PROTOCOL",
        "generator_version": "test",
        "basis_count": 1,
        "basis_library_sha256": hashlib.sha256(canonical_basis).hexdigest(),
        "max_action_terms": 1,
        "total_declared_actions": 2,
        "start_ordinal": 0,
        "end_ordinal_exclusive": 2,
        "processed_actions": 2,
        "complete_declared_space": True,
        "survivor_count": 1,
        "survivor_export_directory": str(survivor_dir),
        "blocks": [
            {
                "block_index": 0,
                "start_ordinal": 0,
                "end_ordinal_exclusive": 2,
                "survivor_export": {
                    "file": survivor_file.name,
                    "record_count": 1,
                },
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    history = GeneratorFormulaHistory(manifest_path, basis_path, survivor_dir)
    survivor = history.query("+(z)")
    rejected = history.query("-(z)")
    outside = history.query("z**2")

    assert survivor["ordinal"] == 1
    assert survivor["recorded_outcome"] == "survived_sampled_static_export"
    assert rejected["ordinal"] == 0
    assert rejected["recorded_outcome"] == "rejected_before_sampled_static_survivor_export"
    assert outside["status"] == "outside_exact_generator_syntax"

    database = tmp_path / "equations.sqlite"
    build_equation_universe(
        ROOT / "configs" / "equation_universe" / "gravity_seed_v1.json",
        database,
        tmp_path / "build.json",
    )
    universe = EquationUniverse(database)
    universe.register_generator_history(manifest_path, basis_path, survivor_dir)
    classified = universe.classify(
        {
            "representation": "scalar_sympy",
            "expression": "F = z",
            "formula_space_expression": "+(z)",
            "variables": [
                {"symbol": "F", "dimension": {}},
                {"symbol": "z", "dimension": {}},
            ],
        }
    )
    assert classified["classification"] == "known_project_history_exact"
    assert classified["generator_history_matches"][0]["exact_project_history_match"] is True
    assert classified["novelty_claim_allowed"] is False
