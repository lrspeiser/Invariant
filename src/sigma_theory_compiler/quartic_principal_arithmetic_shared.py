from __future__ import annotations

from typing import Any

from .quartic_row0_arithmetic_expansion_campaign import (
    ArithmeticDag,
    _faddeev_leverrier_inverse,
    _matrix_vector_row,
)

DIMENSION = 11
PRINCIPAL_CHUNKS = (
    ("s01", "B_1", 1),
    ("s02", "B_2", 1),
    ("s03", "B_3", 1),
    ("s11", "C_11", 1),
    ("s12", "C_12", 2),
    ("s13", "C_13", 2),
    ("s22", "C_22", 1),
    ("s23", "C_23", 2),
    ("s33", "C_33", 1),
)


def build_principal_source_arithmetic_packet(
    physical_block_sha256: str,
    coordinate_atom_basis_sha256: str,
    principal_jet_injection_sha256: str,
) -> dict[str, Any]:
    """Lower the nine exact principal chunks to entrywise arithmetic roots."""

    if any(
        not isinstance(value, str) or len(value) != 64
        for value in (
            physical_block_sha256,
            coordinate_atom_basis_sha256,
            principal_jet_injection_sha256,
        )
    ):
        raise ValueError("principal arithmetic provenance hashes must be SHA-256")
    dag = ArithmeticDag()
    coefficient_a = [
        [
            dag.component(f"A[{row},{column}]", physical_block_sha256)
            for column in range(DIMENSION)
        ]
        for row in range(DIMENSION)
    ]
    inverse, determinant_root, inverse_evidence = _faddeev_leverrier_inverse(
        dag, coefficient_a
    )
    entries: list[dict[str, Any]] = []
    for chunk_index, (atom_family, block_label, multiplicity) in enumerate(
        PRINCIPAL_CHUNKS
    ):
        block = [
            [
                dag.component(
                    f"{block_label}[{row},{column}]", physical_block_sha256
                )
                for column in range(DIMENSION)
            ]
            for row in range(DIMENSION)
        ]
        for field in range(DIMENSION):
            block_column = [block[row][field] for row in range(DIMENSION)]
            for output_row in range(DIMENSION):
                root = dag.neg(
                    _matrix_vector_row(dag, inverse, block_column, output_row)
                )
                if multiplicity == 2:
                    root = dag.mul(dag.constant(2), root)
                entries.append(
                    {
                        "source_row": output_row,
                        "coordinate_column": 54 + chunk_index * DIMENSION + field,
                        "coordinate_atom": f"{atom_family}[{field}]",
                        "atom_family": atom_family,
                        "field": field,
                        "multiplicity": multiplicity,
                        "arithmetic_root": root,
                        "identity": (
                            f"D_{atom_family}F=-{multiplicity} A^-1 {block_label}"
                        ),
                        "normalized_residual": "0",
                    }
                )
    arithmetic_dag = dag.packet()
    return {
        "schema_version": "sigma-principal-source-entrywise-arithmetic-1.0",
        "physical_provenance": {
            "unspecialized_physical_block_sha256": physical_block_sha256,
            "coordinate_atom_basis_sha256": coordinate_atom_basis_sha256,
            "principal_jet_injection_sha256": principal_jet_injection_sha256,
            "identity": "D_Y E55 J_153x55(xi)=iP55(Y,xi)",
        },
        "chunk_order": [item[0] for item in PRINCIPAL_CHUNKS],
        "arithmetic_dag": arithmetic_dag,
        "determinant_coefficient_root": determinant_root,
        "inverse_evidence": inverse_evidence,
        "entries": entries,
        "counts": {
            "chunks": len(PRINCIPAL_CHUNKS),
            "entries": len(entries),
            "arithmetic_nodes": arithmetic_dag["node_count"],
            "semantic_or_tensor_operations": 0,
        },
    }
