from __future__ import annotations

import copy
import io
import json
import shutil
import struct
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_cluster_act_dr6_erass1_overlap_executor as executor


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _copy_package(tmp_path: Path) -> Path:
    root = _repo_root()
    config = json.loads((root / executor.CONFIG_PATH).read_text(encoding="utf-8"))
    paths = [
        executor.CONFIG_PATH,
        executor.MODULE_PATH,
        executor.TEST_PATH,
        executor.CURRENT_AUTH_PATH,
    ]
    parent = config["parent_binding"]
    paths.extend(
        Path(parent[key])
        for key in (
            "config_path",
            "module_path",
            "test_path",
            "authorization_path",
            "receipt_path",
        )
    )
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    return tmp_path


def _fits_value(value) -> str:
    if isinstance(value, str):
        return f"'{value}'"
    if value is True:
        return "T"
    if value is False:
        return "F"
    return str(value)


def _header(cards: list[tuple[str, object]]) -> bytes:
    rendered = [f"{key:<8}= {_fits_value(value):>20}".ljust(80) for key, value in cards]
    rendered.append("END".ljust(80))
    payload = "".join(rendered).encode("ascii")
    return payload + b" " * ((-len(payload)) % 2880)


def _synthetic_fits(path: Path, *, include_flags: bool = True) -> None:
    primary = _header([("SIMPLE", True), ("BITPIX", 8), ("NAXIS", 0), ("EXTEND", True)])
    columns = [("name", "16A"), ("RADeg", "1D"), ("M500", "1D")]
    if include_flags:
        columns.append(("flags", "1J"))
    row_bytes = 32 + (4 if include_flags else 0)
    cards: list[tuple[str, object]] = [
        ("XTENSION", "BINTABLE"),
        ("BITPIX", 8),
        ("NAXIS", 2),
        ("NAXIS1", row_bytes),
        ("NAXIS2", 2),
        ("PCOUNT", 0),
        ("GCOUNT", 1),
        ("TFIELDS", len(columns)),
    ]
    for index, (name, form) in enumerate(columns, 1):
        cards.extend([(f"TTYPE{index}", name), (f"TFORM{index}", form)])
    extension = _header(cards)
    rows = []
    for name, ra, forbidden_mass, flags in (
        ("OBJECT-1", 10.0, 9.9e14, 0),
        ("OBJECT-2", 11.0, 8.8e14, 0),
    ):
        payload = name.encode().ljust(16) + struct.pack(">dd", ra, forbidden_mass)
        if include_flags:
            payload += struct.pack(">i", flags)
        rows.append(payload)
    data = b"".join(rows)
    data += b"\0" * ((-len(data)) % 2880)
    path.write_bytes(primary + extension + data)


def _tiny_table_fits(
    path: Path,
    columns: list[tuple[str, str]],
    row_payload: bytes,
    *,
    extra_column_cards: dict[str, object] | list[tuple[str, object]] | None = None,
) -> None:
    primary = _header([("SIMPLE", True), ("BITPIX", 8), ("NAXIS", 0), ("EXTEND", True)])
    cards: list[tuple[str, object]] = [
        ("XTENSION", "BINTABLE"),
        ("BITPIX", 8),
        ("NAXIS", 2),
        ("NAXIS1", len(row_payload)),
        ("NAXIS2", 1),
        ("PCOUNT", 0),
        ("GCOUNT", 1),
        ("TFIELDS", len(columns)),
    ]
    for index, (name, form) in enumerate(columns, 1):
        cards.extend([(f"TTYPE{index}", name), (f"TFORM{index}", form)])
    extra_cards = extra_column_cards or {}
    items = extra_cards.items() if isinstance(extra_cards, dict) else extra_cards
    for key, value in items:
        cards.append((key, value))
    extension = _header(cards)
    padded = row_payload + b"\0" * ((-len(row_payload)) % 2880)
    path.write_bytes(primary + extension + padded)


def _authorized_manifest(root: Path) -> dict:
    config = executor.load_config(root)
    receipt = json.loads((root / executor.PREFLIGHT_PATH).read_text(encoding="utf-8"))
    auth = json.loads((root / executor.CURRENT_AUTH_PATH).read_text(encoding="utf-8"))
    auth["status"] = "AUTHORIZED_CATALOG_ONLY_EXECUTION"
    auth["authorization"] = True
    auth["run_id"] = executor.RUN_ID
    auth["authorized_by"] = "Henry"
    auth["approved_at_utc"] = "2026-08-29T23:59:59Z"
    auth["approval_phrase"] = executor.AUTHORIZATION_PHRASE
    auth["package_binding"] = executor._authorized_manifest_expected_package(root, receipt)
    for item in auth["catalog_authorizations"]:
        item["authorized"] = True
    auth["claim_boundary"]["authorized_successor_ready_to_execute"] = True
    executor.validate_authorized_manifest(auth, config, root)
    return auth


def _act_row(name: str, ra: float, *, warning: str = "") -> dict:
    return {
        "name": name,
        "RADeg": ra,
        "decDeg": 0.0,
        "fixed_SNR": 6.0,
        "flags": 0,
        "footprint_eROSITADe": True,
        "footprint_Legacy": True,
        "eRASS1CL": True,
        "redshift": 0.1,
        "redshiftErr": 0.001,
        "redshiftType": "spec",
        "opt_RADeg": 0.0,
        "opt_decDeg": 0.0,
        "opt_positionSource": "",
        "warnings": warning,
    }


def _erass_row(detuid: str, name: str, ra: float, *, match_name: str = "") -> dict:
    return {
        "DETUID": detuid,
        "NAME": name,
        "RA": ra,
        "DEC": 0.0,
        "RA_XFIT": ra,
        "DEC_XFIT": 0.0,
        "EXT_LIKE": 5.0,
        "DET_LIKE_0": 10.0,
        "BEST_Z": 0.1,
        "BEST_ZERR": 0.001,
        "BEST_Z_TYPE": "spec",
        "PCONT": 0.01,
        "MATCH_NAME": match_name,
    }


def test_current_package_and_preflight_receipt_match() -> None:
    root = _repo_root()
    config = executor.load_config(root)
    expected = executor.build_preflight_receipt(root)
    stored = json.loads((root / executor.PREFLIGHT_PATH).read_text(encoding="utf-8"))
    executor.validate_preflight_receipt(stored, root)
    assert stored == expected
    assert config["preflight_access_state"]["network_calls"] == 0
    assert config["claim_boundary"]["authorized_successor_ready_to_execute"] is False


def test_exact_future_command_and_ceiling_are_frozen() -> None:
    config = executor.load_config(_repo_root())
    contract = config["future_execution_contract"]
    assert contract["exact_command"].endswith(
        "--authorization runs/gravity/publication-readiness/act-dr6-erass1-overlap-executor-v2/authorization-approved.json "
        "--output-dir runs/gravity/publication-readiness/act-dr6-erass1-overlap-executor-v2-result"
    )
    assert contract["network_get_call_ceiling"] == 2
    assert contract["required_run_id"] == executor.RUN_ID
    assert contract["required_authorization_phrase"] == executor.AUTHORIZATION_PHRASE
    assert contract["network_head_call_ceiling"] == 0
    assert contract["network_redirect_call_ceiling"] == 0
    assert contract["network_retry_call_ceiling"] == 0
    assert contract["network_byte_ceiling"] == 9705600 + 23764095
    assert contract["catalog_row_ceiling"] == 3747 + 12247
    assert contract["scores_or_model_call_ceiling"] == 0
    assert (
        config["selection_match_and_exclusion_contract"]["pair_comparison_ceiling"]
        == executor.PAIR_COMPARISON_CEILING
        == 45_889_509
    )
    assert config["projection_contract"]["column_tform_contract"] == (
        executor.PROJECTION_TFORM_CONTRACT
    )
    assert [item["url"] for item in config["projection_contract"]["schema_metadata_sources"]] == [
        "https://lambda.gsfc.nasa.gov/cgi-bin/fitsheader.cgi?fitsfile=/data/suborbital/ACT/actadv_dr6_cluster_cat/DR6_cluster-catalog_v1.0.fits",
        "https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/Catalogues_dr1/BulbulE_DR1/erass1cl_primary_v3.2.html",
    ]


def test_current_unauthorized_manifest_fails_before_output_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = _copy_package(tmp_path)
    executor.write_preflight_receipt(copied)
    approved = copied / executor.APPROVED_AUTH_PATH
    approved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(copied / executor.CURRENT_AUTH_PATH, approved)
    calls = 0

    def bomb(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("network boundary crossed")

    monkeypatch.setattr(executor, "_download_exact", bomb)
    output = copied / executor.RESULT_DIR
    with pytest.raises(
        executor.GravityClusterActErassOverlapExecutorError, match="authorized identity"
    ):
        executor.execute_authorized(copied, approved, output)
    assert calls == 0
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.__setitem__("authorization", False), "authorized identity"),
        (lambda value: value.__setitem__("run_id", "WRONG_RUN"), "authorized identity"),
        (lambda value: value.__setitem__("approval_phrase", "yes"), "authorized identity"),
        (
            lambda value: value.__setitem__("unexpected", True),
            "authorization manifest keys changed",
        ),
        (
            lambda value: value["package_binding"].__setitem__("module_file_sha256", "0" * 64),
            "package binding",
        ),
        (
            lambda value: value["catalog_authorizations"][0].__setitem__(
                "url", "https://example.invalid"
            ),
            "catalog specification",
        ),
        (
            lambda value: value["catalog_authorizations"][1]["permitted_columns"].append("M500"),
            "catalog specification",
        ),
        (
            lambda value: value["network_and_output_scope"].__setitem__("maximum_get_calls", 3),
            "network/output scope",
        ),
        (
            lambda value: value["access_state"].pop("network_calls"),
            "authorization access state keys changed",
        ),
        (
            lambda value: value["access_state"].__setitem__("extra_false_state", False),
            "authorization access state keys changed",
        ),
        (lambda value: value["access_state"].__setitem__("catalog_rows_opened", 1), "chronology"),
        (
            lambda value: value["claim_boundary"].__setitem__("overlap_count_computed", True),
            "claim boundary",
        ),
        (
            lambda value: value.__setitem__("approved_at_utc", "2026-99-99T23:59:59Z"),
            "timestamp",
        ),
    ],
)
def test_authorized_manifest_mutations_fail_closed(tmp_path: Path, mutation, match: str) -> None:
    copied = _copy_package(tmp_path)
    executor.write_preflight_receipt(copied)
    config = executor.load_config(copied)
    auth = _authorized_manifest(copied)
    mutation(auth)
    with pytest.raises(executor.GravityClusterActErassOverlapExecutorError, match=match):
        executor.validate_authorized_manifest(auth, config, copied)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["future_execution_contract"].__setitem__("network_get_call_ceiling", 3),
        lambda value: value["catalog_assets"][0].__setitem__("expected_network_bytes", 1),
        lambda value: value["projection_contract"]["ACT_DR6_LEGACY_V1_0"].append("M500"),
        lambda value: value["selection_match_and_exclusion_contract"][
            "population_gate"
        ].__setitem__("confirmatory_target_clusters", 191),
        lambda value: value["selection_match_and_exclusion_contract"].__setitem__(
            "pair_comparison_ceiling", 45_889_508
        ),
        lambda value: value["selection_match_and_exclusion_contract"].__setitem__(
            "xcop_candidate_taint_rule", "allow later unique reuse"
        ),
        lambda value: value["sanitized_output_contract"]["ledger_fields"].append("M500"),
        lambda value: value["failure_and_publication_contract"].__setitem__(
            "download_failure_cleanup", "keep partial files"
        ),
        lambda value: value["preflight_access_state"].__setitem__("network_calls", 1),
        lambda value: value["claim_boundary"].__setitem__("overlap_count_computed", True),
        lambda value: value["parent_binding"].__setitem__("commit", "0" * 40),
    ],
)
def test_config_semantic_mutations_fail_closed(mutation) -> None:
    config = executor.load_config(_repo_root())
    changed = copy.deepcopy(config)
    mutation(changed)
    with pytest.raises(executor.GravityClusterActErassOverlapExecutorError):
        executor.validate_config(changed)


def test_fits_projection_seeks_only_allowlisted_cells(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.fits"
    _synthetic_fits(path)
    rows, schema = executor.read_fits_projection(
        path,
        ("name", "RADeg", "flags"),
        expected_tforms={"name": "16A", "RADeg": "1D", "flags": "1J"},
        expected_rows=2,
        maximum_rows=2,
    )
    assert rows == [
        {"name": "OBJECT-1", "RADeg": 10.0, "flags": 0.0},
        {"name": "OBJECT-2", "RADeg": 11.0, "flags": 0.0},
    ]
    assert all("M500" not in row for row in rows)
    assert schema == {
        "rows": 2,
        "row_bytes": 36,
        "fields_in_source_schema": 4,
        "fields_decoded": 3,
    }


def test_fits_missing_required_field_fails_without_values(tmp_path: Path) -> None:
    path = tmp_path / "missing.fits"
    _synthetic_fits(path, include_flags=False)
    with pytest.raises(
        executor.GravityClusterActErassOverlapExecutorError, match="column missing: flags"
    ):
        executor.read_fits_projection(
            path,
            ("name", "RADeg", "flags"),
            expected_tforms={"name": "16A", "RADeg": "1D", "flags": "1J"},
            expected_rows=2,
            maximum_rows=2,
        )


def test_fits_rejects_duplicate_casefolded_column_names(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.fits"
    _tiny_table_fits(path, [("Name", "1A"), ("NAME", "1A")], b"AB")
    with pytest.raises(
        executor.GravityClusterActErassOverlapExecutorError,
        match="duplicate case-folded FITS column name",
    ):
        executor.read_fits_projection(
            path,
            ("Name",),
            expected_tforms={"Name": "1A"},
            expected_rows=1,
            maximum_rows=1,
        )


def test_fits_rejects_invalid_logical_and_null_declarations(tmp_path: Path) -> None:
    logical = tmp_path / "invalid-logical.fits"
    _tiny_table_fits(logical, [("flag", "1L")], b"X")
    with pytest.raises(
        executor.GravityClusterActErassOverlapExecutorError, match="invalid FITS logical flag"
    ):
        executor.read_fits_projection(
            logical,
            ("flag",),
            expected_tforms={"flag": "1L"},
            expected_rows=1,
            maximum_rows=1,
        )

    invalid_null = tmp_path / "invalid-null.fits"
    _tiny_table_fits(
        invalid_null,
        [("value", "1D")],
        struct.pack(">d", 1.0),
        extra_column_cards={"TNULL1": 0},
    )
    with pytest.raises(
        executor.GravityClusterActErassOverlapExecutorError,
        match="invalid FITS integer-null declaration",
    ):
        executor.read_fits_projection(
            invalid_null,
            ("value",),
            expected_tforms={"value": "1D"},
            expected_rows=1,
            maximum_rows=1,
        )

    boolean_null = tmp_path / "boolean-null.fits"
    _tiny_table_fits(
        boolean_null,
        [("value", "1J")],
        struct.pack(">i", 1),
        extra_column_cards={"TNULL1": True},
    )
    with pytest.raises(
        executor.GravityClusterActErassOverlapExecutorError,
        match="invalid FITS integer-null declaration",
    ):
        executor.read_fits_projection(
            boolean_null,
            ("value",),
            expected_tforms={"value": "1J"},
            expected_rows=1,
            maximum_rows=1,
        )


@pytest.mark.parametrize(
    "duplicate_cards",
    [
        [("TTYPE1", "renamed")],
        [("TFORM1", "1K")],
        [("TNULL1", 0), ("TNULL1", 1)],
        [("TSCAL1", 1), ("TSCAL1", 2)],
        [("NAXIS1", 4)],
    ],
)
def test_fits_rejects_duplicate_structural_cards_before_overwrite(
    tmp_path: Path, duplicate_cards: list[tuple[str, object]]
) -> None:
    path = tmp_path / "duplicate-structural.fits"
    _tiny_table_fits(
        path,
        [("value", "1J")],
        struct.pack(">i", 1),
        extra_column_cards=duplicate_cards,
    )
    with pytest.raises(
        executor.GravityClusterActErassOverlapExecutorError,
        match="duplicate FITS structural card",
    ):
        executor.read_fits_projection(
            path,
            ("value",),
            expected_tforms={"value": "1J"},
            expected_rows=1,
            maximum_rows=1,
        )


@pytest.mark.parametrize("malformed", ["1JA", "1JD", "1JZ", "0J", "J(8)"])
def test_fits_rejects_malformed_or_unsupported_tform_suffixes(malformed: str) -> None:
    with pytest.raises(
        executor.GravityClusterActErassOverlapExecutorError,
        match="malformed or unsupported FITS column format",
    ):
        executor._tform_layout(malformed)


def test_a_valued_flags_are_rejected_by_tform_before_private_value_decode(
    tmp_path: Path,
) -> None:
    private_token = "PRIVATE_ROW_TOKEN"
    path = tmp_path / "a-valued-flags.fits"
    _tiny_table_fits(path, [("flags", "17A")], private_token.encode("ascii"))
    with pytest.raises(executor.GravityClusterActErassOverlapExecutorError) as caught:
        executor.read_fits_projection(
            path,
            ("flags",),
            expected_tforms={"flags": "K"},
            expected_rows=1,
            maximum_rows=1,
        )
    assert "TFORM changed: flags" in str(caught.value)
    assert private_token not in str(caught.value)


@pytest.mark.parametrize(
    "semantic_card",
    [
        {"TSCAL1": 2},
        {"TZERO1": 1},
        {"TNULL1": -1},
        {"TDIM1": "(1)"},
    ],
)
def test_allowlisted_scaling_null_and_shape_semantics_fail_closed(
    tmp_path: Path, semantic_card: dict[str, object]
) -> None:
    path = tmp_path / "semantic-mutation.fits"
    _tiny_table_fits(
        path,
        [("flags", "1J")],
        struct.pack(">i", 0),
        extra_column_cards=semantic_card,
    )
    with pytest.raises(
        executor.GravityClusterActErassOverlapExecutorError,
        match="scaling/null semantics changed: flags",
    ):
        executor.read_fits_projection(
            path,
            ("flags",),
            expected_tforms={"flags": "1J"},
            expected_rows=1,
            maximum_rows=1,
        )


def test_fits_row_padding_is_not_implicitly_accepted(tmp_path: Path) -> None:
    path = tmp_path / "padded-row.fits"
    _tiny_table_fits(path, [("flags", "1J")], struct.pack(">i", 0) + b"X")
    with pytest.raises(
        executor.GravityClusterActErassOverlapExecutorError,
        match="layout does not exactly equal row width",
    ):
        executor.read_fits_projection(
            path,
            ("flags",),
            expected_tforms={"flags": "1J"},
            expected_rows=1,
            maximum_rows=1,
        )


def test_private_row_coercion_failure_is_wrapped_without_value() -> None:
    private_token = "PRIVATE_ROW_TOKEN"
    act = [_act_row("ACT-PRIVATE", 10.0)]
    act[0]["flags"] = private_token
    with pytest.raises(executor.GravityClusterActErassOverlapExecutorError) as caught:
        executor.match_catalogs(act, [_erass_row("E1", "ERASS", 10.0)])
    assert "private catalog row failed frozen type/coercion validation" in str(caught.value)
    assert private_token not in str(caught.value)


def test_single_member_archive_guard_and_hash(tmp_path: Path) -> None:
    fits = tmp_path / "source.fits"
    _synthetic_fits(fits)
    archive = tmp_path / "source.tgz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(fits, arcname="nested/erass1cl_main_v3.2.fits")
    output = tmp_path / "extracted.fits"
    receipt = executor.extract_single_fits_member(
        archive,
        output,
        expected_basename="erass1cl_main_v3.2.fits",
        maximum_bytes=10000,
    )
    assert receipt["bytes"] == output.stat().st_size
    assert receipt["sha256"] == executor._file_sha(output)


@pytest.mark.parametrize(
    "member_type",
    [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE],
)
def test_archive_rejects_every_special_member_even_with_expected_fits(
    tmp_path: Path, member_type: bytes
) -> None:
    fits = tmp_path / "source.fits"
    _synthetic_fits(fits)
    archive = tmp_path / f"special-{member_type!r}.tgz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(fits, arcname="erass1cl_main_v3.2.fits")
        special = tarfile.TarInfo("unexpected-special")
        special.type = member_type
        special.linkname = "erass1cl_main_v3.2.fits"
        handle.addfile(special)
    with pytest.raises(
        executor.GravityClusterActErassOverlapExecutorError, match="prohibited special member"
    ):
        executor.extract_single_fits_member(
            archive,
            tmp_path / "must-not-exist.fits",
            expected_basename="erass1cl_main_v3.2.fits",
            maximum_bytes=10000,
        )


def test_archive_rejects_an_extra_regular_file(tmp_path: Path) -> None:
    fits = tmp_path / "source.fits"
    _synthetic_fits(fits)
    archive = tmp_path / "extra.tgz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(fits, arcname="erass1cl_main_v3.2.fits")
        payload = b"unexpected"
        extra = tarfile.TarInfo("unexpected.txt")
        extra.size = len(payload)
        handle.addfile(extra, io.BytesIO(payload))
    with pytest.raises(executor.GravityClusterActErassOverlapExecutorError, match="inventory"):
        executor.extract_single_fits_member(
            archive,
            tmp_path / "must-not-exist.fits",
            expected_basename="erass1cl_main_v3.2.fits",
            maximum_bytes=10000,
        )


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "/absolute/erass1cl_main_v3.2.fits",
        "../erass1cl_main_v3.2.fits",
        "nested/../../erass1cl_main_v3.2.fits",
        "C:\\absolute\\erass1cl_main_v3.2.fits",
        "C:/absolute/erass1cl_main_v3.2.fits",
        "C:drive-relative\\erass1cl_main_v3.2.fits",
        "\\\\server\\share\\erass1cl_main_v3.2.fits",
        "//server/share/erass1cl_main_v3.2.fits",
    ],
)
def test_archive_rejects_posix_windows_drive_and_unc_paths(
    tmp_path: Path, unsafe_name: str
) -> None:
    fits = tmp_path / "source.fits"
    _synthetic_fits(fits)
    archive = tmp_path / "unsafe-path.tgz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(fits, arcname="erass1cl_main_v3.2.fits")
        payload = b"unsafe"
        unsafe = tarfile.TarInfo(unsafe_name)
        unsafe.size = len(payload)
        handle.addfile(unsafe, io.BytesIO(payload))
    with pytest.raises(
        executor.GravityClusterActErassOverlapExecutorError,
        match="archive member path safety failed",
    ):
        executor.extract_single_fits_member(
            archive,
            tmp_path / "must-not-exist.fits",
            expected_basename="erass1cl_main_v3.2.fits",
            maximum_bytes=10000,
        )


class _FakeResponse:
    def __init__(self, url: str, payload: bytes, *, status: int = 200):
        self._url = url
        self._stream = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}
        self._status = status
        self.closed = False

    def getcode(self) -> int:
        return self._status

    def geturl(self) -> str:
        return self._url

    def read(self, size: int) -> bytes:
        return self._stream.read(size)

    def close(self) -> None:
        self.closed = True


def test_download_hashes_exact_bytes_and_cleans_failure(tmp_path: Path) -> None:
    url = "https://example.test/catalog"
    payload = b"synthetic-catalog"
    response = _FakeResponse(url, payload)
    destination = tmp_path / "raw.bin"
    receipt = executor._download_exact(
        {"url": url, "expected_network_bytes": len(payload)},
        destination,
        opener=lambda request, timeout: response,
    )
    assert receipt["bytes"] == len(payload)
    assert receipt["sha256"] == executor._file_sha(destination)
    assert response.closed is True

    bad_response = _FakeResponse(url, payload + b"overflow")
    failed = tmp_path / "failed.bin"
    with pytest.raises(executor.GravityClusterActErassOverlapExecutorError, match="Content-Length"):
        executor._download_exact(
            {"url": url, "expected_network_bytes": len(payload)},
            failed,
            opener=lambda request, timeout: bad_response,
        )
    assert not failed.exists()
    assert bad_response.closed is True


def test_redirect_is_refused_without_a_hidden_second_get(tmp_path: Path) -> None:
    url = "https://example.test/catalog"
    response = _FakeResponse("https://example.test/redirected", b"redirect", status=302)
    calls = 0
    budget = executor._GetBudget(2)

    def opener(request, timeout):
        nonlocal calls
        calls += 1
        return response

    destination = tmp_path / "redirect.bin"
    with pytest.raises(
        executor.GravityClusterActErassOverlapExecutorError, match="HTTP status changed"
    ):
        executor._download_exact(
            {"url": url, "expected_network_bytes": 8},
            destination,
            opener=opener,
            budget=budget,
        )
    assert calls == 1
    assert budget.calls == 1
    assert not destination.exists()
    assert response.closed is True


def test_no_redirect_handler_and_exact_two_get_budget() -> None:
    from urllib.error import HTTPError
    from urllib.request import Request

    handler = executor._make_no_redirect_handler()
    with pytest.raises(HTTPError):
        handler.redirect_request(
            Request("https://example.test/source"),
            None,
            302,
            "Found",
            {},
            "https://example.test/target",
        )
    budget = executor._GetBudget(2)
    budget.claim()
    with pytest.raises(
        executor.GravityClusterActErassOverlapExecutorError, match="exact GET call count"
    ):
        budget.verify_exact()
    budget.claim()
    budget.verify_exact()
    with pytest.raises(executor.GravityClusterActErassOverlapExecutorError, match="ceiling"):
        budget.claim()


def test_matching_quarantine_xcop_and_sanitized_schema() -> None:
    act = [
        _act_row("Abell 85", 10.0),
        _act_row("NEW-1", 20.0),
        _act_row("PROJECTED", 30.0, warning="Possible projected system"),
    ]
    erass = [
        _erass_row("E1", "A85", 10.001, match_name="Abell 85"),
        _erass_row("E2", "ERASS-NEW", 20.001),
    ]
    ledger, counts = executor.match_catalogs(act, erass)
    assert ledger[0]["xcop_identity"] == "A85"
    assert ledger[0]["xcop_excluded"] is True
    assert ledger[1]["eligible_catalog_overlap"] is True
    assert ledger[2]["quarantine_reason"] == "POSSIBLE_PROJECTED_SYSTEM"
    assert counts["xcop_excluded_rows"] == 1
    assert counts["xcop_excluded_distinct_erass_objects"] == 1
    assert counts["distinct_matched_erass_objects"] == 2
    assert counts["eligible_distinct_erass_objects"] == 1
    assert counts["spherical_pair_comparisons"] == 4
    assert counts["post_xcop_catalog_upper_bound"] == 1
    assert all(tuple(row) == executor.LEDGER_FIELDS for row in ledger)
    serialized = json.dumps(ledger)
    for forbidden in ("PCONT", "EXT_LIKE", "M500", "fixed_SNR", "warnings"):
        assert forbidden not in serialized


def test_multiple_candidates_are_not_resolved_by_name() -> None:
    act = [_act_row("OBJECT", 10.0)]
    erass = [
        _erass_row("E1", "OBJECT", 10.001, match_name="OBJECT"),
        _erass_row("E2", "OTHER", 10.002),
    ]
    ledger, counts = executor.match_catalogs(act, erass)
    assert ledger[0]["quarantine_reason"] == "MULTIPLE_IN_RADIUS_ERASS_CANDIDATES"
    assert ledger[0]["eligible_catalog_overlap"] is False
    assert counts["post_xcop_catalog_upper_bound"] == 0


@pytest.mark.parametrize(
    ("act_position", "erass_position", "redshift"),
    [
        ((359.99, 0.0), (0.01, 0.0), 0.1),
        ((0.0, 89.9), (180.0, 89.9), 0.01),
        ((0.0, 89.999), (179.0, 89.999), 0.1),
    ],
)
def test_exhaustive_spherical_scan_handles_ra_wrap_low_z_high_dec_and_poles(
    act_position: tuple[float, float],
    erass_position: tuple[float, float],
    redshift: float,
) -> None:
    act = _act_row("SPHERICAL-EDGE", act_position[0])
    act["decDeg"] = act_position[1]
    act["redshift"] = redshift
    erass = _erass_row("EDGE-E1", "SPHERICAL-ERASS", erass_position[0])
    erass["DEC"] = erass_position[1]
    erass["RA_XFIT"] = erass_position[0]
    erass["DEC_XFIT"] = erass_position[1]
    erass["BEST_Z"] = redshift
    ledger, counts = executor.match_catalogs([act], [erass])
    assert ledger[0]["eligible_catalog_overlap"] is True
    assert ledger[0]["match_state"] == "MATCHED"
    assert counts["spherical_pair_comparisons"] == 1
    assert counts["eligible_distinct_erass_objects"] == 1


def test_mixed_xcop_and_ordinary_reuse_is_grouped_before_eligibility() -> None:
    acts = [_act_row("Abell 85", 10.0), _act_row("ORDINARY", 10.0005)]
    erass = [_erass_row("SHARED-E1", "SHARED-ERASS", 10.00025)]
    ledger, counts = executor.match_catalogs(acts, erass)
    assert counts["unique_positional_match_rows"] == 2
    assert counts["distinct_matched_erass_objects"] == 1
    assert counts["reused_erass_groups"] == 1
    assert counts["reused_act_rows"] == 2
    assert counts["xcop_excluded_distinct_erass_objects"] == 1
    assert counts["eligible_distinct_erass_objects"] == 0
    assert counts["post_xcop_catalog_upper_bound"] == 0
    assert all(item["xcop_identity"] == "A85" for item in ledger)
    assert all(item["xcop_excluded"] is True for item in ledger)
    assert all(item["eligible_catalog_overlap"] is False for item in ledger)
    assert all(item["match_state"] == "QUARANTINED" for item in ledger)
    assert all(
        item["quarantine_reason"] == "ERASS_CANDIDATE_REUSED_BY_MULTIPLE_ACT_ROWS"
        for item in ledger
    )


def test_ambiguous_xcop_candidate_set_globally_taints_later_unique_match() -> None:
    acts = [_act_row("Abell 85", 10.0), _act_row("ORDINARY", 10.14)]
    erass = [
        _erass_row("E1", "NEUTRAL-E1", 10.07),
        _erass_row("E2", "NEUTRAL-E2", 9.93),
    ]
    ledger, counts = executor.match_catalogs(acts, erass)
    xcop_row, ordinary_row = ledger
    assert set(xcop_row["candidate_ids"]) == {"E1", "E2"}
    assert xcop_row["match_state"] == "QUARANTINED"
    assert xcop_row["quarantine_reason"] == "MULTIPLE_IN_RADIUS_ERASS_CANDIDATES"
    assert xcop_row["xcop_identity"] == "A85"
    assert xcop_row["xcop_excluded"] is True
    assert ordinary_row["candidate_ids"] == ["E1"]
    assert ordinary_row["match_state"] == "QUARANTINED"
    assert ordinary_row["quarantine_reason"] == "XCOP_CANDIDATE_TAINT_FROM_AMBIGUOUS_ACT_ROW"
    assert ordinary_row["xcop_identity"] == "A85"
    assert ordinary_row["xcop_excluded"] is True
    assert ordinary_row["eligible_catalog_overlap"] is False
    assert counts["xcop_candidate_taint_source_act_rows"] == 1
    assert counts["xcop_candidate_taint_edges"] == 2
    assert counts["xcop_candidate_tainted_distinct_erass_objects"] == 2
    assert counts["xcop_candidate_tainted_by_ambiguous_sets"] == 2
    assert counts["xcop_excluded_distinct_erass_objects"] == 2
    assert counts["eligible_distinct_erass_objects"] == 0
    assert counts["post_xcop_catalog_upper_bound"] == 0


@pytest.mark.parametrize("early_exit", ["projected", "failed_overlap"])
def test_xcop_candidate_taint_precedes_selection_and_projected_exits(
    early_exit: str,
) -> None:
    xcop = _act_row("Abell 85", 10.0)
    if early_exit == "projected":
        xcop["warnings"] = "Possible projected system"
    else:
        xcop["eRASS1CL"] = False
    acts = [xcop, _act_row("ORDINARY", 10.01)]
    erass = [_erass_row("E1", "NEUTRAL-E1", 10.005)]

    ledger, counts = executor.match_catalogs(acts, erass)

    xcop_row, ordinary_row = ledger
    assert xcop_row["candidate_ids"] == ["E1"]
    assert xcop_row["xcop_identity"] == "A85"
    assert xcop_row["xcop_excluded"] is True
    assert xcop_row["quarantine_reason"] == (
        "POSSIBLE_PROJECTED_SYSTEM" if early_exit == "projected" else "ACT_SELECTION_NOT_ELIGIBLE"
    )
    assert ordinary_row["candidate_ids"] == ["E1"]
    assert ordinary_row["match_state"] == "MATCHED_XCOP_EXCLUDED"
    assert ordinary_row["xcop_identity"] == "A85"
    assert ordinary_row["xcop_excluded"] is True
    assert ordinary_row["eligible_catalog_overlap"] is False
    assert counts["xcop_candidate_taint_source_act_rows"] == 1
    assert counts["xcop_candidate_taint_edges"] == 1
    assert counts["xcop_candidate_tainted_distinct_erass_objects"] == 1
    assert counts["xcop_excluded_distinct_erass_objects"] == 1
    assert counts["eligible_distinct_erass_objects"] == 0
    assert counts["post_xcop_catalog_upper_bound"] == 0


def test_spherical_pair_ceiling_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor, "PAIR_COMPARISON_CEILING", 0)
    with pytest.raises(
        executor.GravityClusterActErassOverlapExecutorError,
        match="spherical pair-comparison ceiling exceeded",
    ):
        executor.match_catalogs(
            [_act_row("PAIR-CEILING", 10.0)],
            [_erass_row("PAIR-E1", "PAIR-ERASS", 10.0)],
        )


def test_authorized_failure_removes_reserved_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = _copy_package(tmp_path)
    executor.write_preflight_receipt(copied)
    auth = _authorized_manifest(copied)
    approved = copied / executor.APPROVED_AUTH_PATH
    approved.parent.mkdir(parents=True, exist_ok=True)
    approved.write_text(json.dumps(auth, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    calls = 0

    def fail_download(asset, destination, *, budget):
        nonlocal calls
        del asset, budget
        calls += 1
        destination.write_bytes(b"partial opaque source bytes")
        raise RuntimeError("synthetic network failure")

    monkeypatch.setattr(executor, "_download_exact", fail_download)
    output = copied / executor.RESULT_DIR
    with pytest.raises(RuntimeError, match="synthetic network failure"):
        executor.execute_authorized(copied, approved, output)
    assert calls == 1
    assert not output.exists()


def test_cleanup_failure_is_loud_and_never_silently_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = _copy_package(tmp_path)
    executor.write_preflight_receipt(copied)
    auth = _authorized_manifest(copied)
    approved = copied / executor.APPROVED_AUTH_PATH
    approved.parent.mkdir(parents=True, exist_ok=True)
    approved.write_text(json.dumps(auth, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def fail_download(asset, destination, *, budget):
        del asset, budget
        destination.write_bytes(b"partial opaque source bytes")
        raise RuntimeError("synthetic download failure")

    real_rmtree = shutil.rmtree

    def fail_cleanup(path):
        raise OSError("synthetic cleanup failure")

    monkeypatch.setattr(executor, "_download_exact", fail_download)
    monkeypatch.setattr(executor.shutil, "rmtree", fail_cleanup)
    output = copied / executor.RESULT_DIR
    with pytest.raises(
        executor.GravityClusterActErassOverlapExecutorError,
        match="execution failed and cleanup failed loudly",
    ):
        executor.execute_authorized(copied, approved, output)
    assert output.exists()
    monkeypatch.setattr(executor.shutil, "rmtree", real_rmtree)
    executor._remove_tree_verified(output, label="test recovery")
    assert not output.exists()


def test_verified_cleanup_removes_private_staging(tmp_path: Path) -> None:
    staging = tmp_path / ".private-raw-staging"
    staging.mkdir()
    (staging / "opaque.bin").write_bytes(b"opaque")
    executor._remove_tree_verified(staging, label="synthetic private staging")
    assert not staging.exists()


def test_existing_result_directory_refuses_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = _copy_package(tmp_path)
    executor.write_preflight_receipt(copied)
    auth = _authorized_manifest(copied)
    approved = copied / executor.APPROVED_AUTH_PATH
    approved.parent.mkdir(parents=True, exist_ok=True)
    approved.write_text(json.dumps(auth, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output = copied / executor.RESULT_DIR
    output.mkdir(parents=True)
    calls = 0

    def bomb(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("network boundary crossed")

    monkeypatch.setattr(executor, "_download_exact", bomb)
    with pytest.raises(executor.GravityClusterActErassOverlapExecutorError, match="already exists"):
        executor.execute_authorized(copied, approved, output)
    assert calls == 0


def test_atomic_preflight_no_replace_and_race(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    expected = executor.build_preflight_receipt(copied)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: executor.write_preflight_receipt(copied), range(8)))
    assert paths == [copied / executor.PREFLIGHT_PATH] * 8
    assert json.loads((copied / executor.PREFLIGHT_PATH).read_text(encoding="utf-8")) == expected
    (copied / executor.PREFLIGHT_PATH).write_text('{"different":true}\n', encoding="utf-8")
    with pytest.raises(
        executor.GravityClusterActErassOverlapExecutorError,
        match="refusing to replace a different executor preflight receipt",
    ):
        executor.write_preflight_receipt(copied)


def test_no_authorized_manifest_or_result_exists_and_network_import_is_deferred() -> None:
    root = _repo_root()
    assert not (root / executor.APPROVED_AUTH_PATH).exists()
    assert not (root / executor.RESULT_DIR).exists()
    source = (root / executor.MODULE_PATH).read_text(encoding="utf-8")
    import_position = source.index("from urllib.request import Request, build_opener")
    authorization_position = source.index("def validate_authorized_manifest")
    assert import_position > authorization_position
    assert "ignore_errors=True" not in source
