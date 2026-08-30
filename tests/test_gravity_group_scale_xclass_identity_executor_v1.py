from __future__ import annotations

import copy
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_group_scale_xclass_identity_executor_v1 as executor


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _copy_package(tmp_path: Path) -> Path:
    root = _repo_root()
    config = executor.load_config(root)
    paths = [
        executor.CONFIG_PATH,
        executor.MODULE_PATH,
        executor.TEST_PATH,
        executor.PREFLIGHT_PATH,
        executor.UNAUTHORIZED_PATH,
    ]
    parent = config["parent_v3_binding"]
    paths.extend(Path(parent[key]) for key in executor.PARENT_PATH_KEYS)
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    return tmp_path


def _row(index: int, opaque: bytes = b"Q") -> bytes:
    allowed = (
        f"{index + 1:5d} {10.0 + index:8.4f} {(-5.0 + index / 20):7.4f} "
        f"{0.080 + index / 10000:5.3f}"
    ).encode("ascii")
    assert len(allowed) == 28
    suffix = opaque * 80 if len(opaque) == 1 else opaque.ljust(80, b"X")[:80]
    assert len(suffix) == 80
    return allowed + suffix + b"\n"


def _payload(opaque: bytes = b"Q") -> bytes:
    value = b"".join(_row(index, opaque) for index in range(executor.EXPECTED_ROWS))
    assert len(value) == executor.EXPECTED_BYTES
    return value


class _FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        status: int = 200,
        final_url: str = executor.SOURCE_URL,
        content_length: int | None = executor.EXPECTED_BYTES,
        content_encoding: str | None = None,
        transfer_encoding: str | None = None,
    ) -> None:
        self._payload = payload
        self.read_sizes: list[int] = []
        self.status = status
        self._final_url = final_url
        self.headers: dict[str, str] = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        if content_encoding is not None:
            self.headers["Content-Encoding"] = content_encoding
        if transfer_encoding is not None:
            self.headers["Transfer-Encoding"] = transfer_encoding

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self._final_url

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self._payload if size < 0 else self._payload[:size]


class _FakeOpener:
    def __init__(self, response: _FakeResponse | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = 0
        self.requests = []

    def open(self, request, timeout: int):
        self.calls += 1
        self.requests.append((request, timeout))
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def _approved_authorization(root: Path, **changes) -> tuple[Path, str]:
    body = {
        "schema_version": executor.AUTHORIZATION_SCHEMA,
        "status": executor.AUTHORIZED_STATUS,
        "authorized": True,
        "run_id": executor.RUN_ID,
        "authorization_phrase": executor.AUTHORIZATION_PHRASE,
        "approved_by": "independent_external_approver",
        "approved_at": "2026-08-29T23:00:00+00:00",
        "scientific_payload_exposure_acknowledged": True,
        "source_url": executor.SOURCE_URL,
        "maximum_get_calls": 1,
        "maximum_network_bytes": executor.EXPECTED_BYTES,
        "bindings": executor.expected_authorization_bindings(root),
    }
    body.update(changes)
    path = root / executor.APPROVED_AUTHORIZATION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_preflight_and_zero_run_state() -> None:
    root = _repo_root()
    config = executor.load_config(root)
    expected = executor.build_preflight_receipt(root)
    stored = json.loads((root / executor.PREFLIGHT_PATH).read_text(encoding="utf-8"))
    executor.validate_preflight_receipt(stored, root)
    assert stored == expected
    assert config["execution_accounting_at_freeze"]["executor_launches"] == 0
    assert config["claim_boundary"]["observational_authorization"] is False
    assert not (root / executor.APPROVED_AUTHORIZATION_PATH).exists()
    assert not (root / executor.ACCESS_INTENT_PATH).exists()
    assert not (root / executor.GET_ATTEMPT_PATH).exists()
    assert not (root / executor.RESULT_PATH).exists()


def test_exact_source_network_columns_and_blockers_are_frozen() -> None:
    config = executor.load_config(_repo_root())
    source = config["source_contract"]
    network = config["network_contract"]
    columns = config["column_contract"]
    assert source["url"] == executor.SOURCE_URL
    assert source["schema_source_url"] == (
        "https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/A%2BA/710/A77?format=html&tex=true"
    )
    assert source["expected_network_bytes"] == 16895
    assert source["expected_rows"] == 155
    assert source["wire_record_bytes_including_lf"] == 109
    assert network["get_calls"] == 1
    assert network["head_calls"] == network["redirect_calls"] == network["retry_calls"] == 0
    assert columns["decode_allowlist"] == ["XClass", "RAdeg", "DEdeg", "z"]
    assert columns["fixed_width_slices_1_based_inclusive"] == {
        "XClass": [1, 5],
        "RAdeg": [7, 14],
        "DEdeg": [16, 22],
        "z": [24, 28],
        "opaque_suffix": [29, 108],
    }
    assert columns["scientific_values_instantiated"] is False
    assert config["obsid_contract"]["obsid_mapping_executed"] is False
    assert config["obsid_contract"]["obsid_guessing_allowed"] is False
    assert config["xcop_overlap_contract"]["coordinate_ledger_bound"] is False
    assert config["xcop_overlap_contract"]["overlap_count"] is None


def test_current_unauthorized_manifest_refuses_before_intent_temp_or_network(
    tmp_path: Path,
) -> None:
    copied = _copy_package(tmp_path)
    opener = _FakeOpener(_FakeResponse(_payload()))
    path = copied / executor.UNAUTHORIZED_PATH
    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(
        executor.GravityGroupScaleXclassIdentityExecutorV1Error, match="not authorized"
    ):
        executor.execute(
            copied,
            path,
            expected_sha,
            executor.EXECUTE_SENTINEL,
            opener=opener,
        )
    assert opener.calls == 0
    assert not (copied / executor.ACCESS_INTENT_PATH).exists()
    assert not (copied / executor.GET_ATTEMPT_PATH).exists()
    assert not (copied / executor.TEMPORARY_PARENT).exists()
    assert not (copied / executor.RESULT_PATH).exists()


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"authorization_phrase": "forged"}, "phrase"),
        ({"scientific_payload_exposure_acknowledged": False}, "exposure"),
        ({"maximum_get_calls": 2}, "network bounds"),
        ({"source_url": "https://example.invalid/catalog"}, "source"),
        ({"bindings": {}}, "bindings"),
    ],
)
def test_forged_approved_authorization_refuses_before_network(
    tmp_path: Path, changes: dict, match: str
) -> None:
    copied = _copy_package(tmp_path)
    authorization, authorization_sha = _approved_authorization(copied, **changes)
    opener = _FakeOpener(_FakeResponse(_payload()))
    with pytest.raises(executor.GravityGroupScaleXclassIdentityExecutorV1Error, match=match):
        executor.execute(
            copied,
            authorization,
            authorization_sha,
            executor.EXECUTE_SENTINEL,
            opener=opener,
        )
    assert opener.calls == 0
    assert not (copied / executor.ACCESS_INTENT_PATH).exists()


def test_external_authorization_hash_and_sentinel_are_required_before_network(
    tmp_path: Path,
) -> None:
    copied = _copy_package(tmp_path)
    authorization, authorization_sha = _approved_authorization(copied)
    opener = _FakeOpener(_FakeResponse(_payload()))
    with pytest.raises(executor.GravityGroupScaleXclassIdentityExecutorV1Error, match="hash"):
        executor.execute(copied, authorization, "0" * 64, executor.EXECUTE_SENTINEL, opener=opener)
    with pytest.raises(executor.GravityGroupScaleXclassIdentityExecutorV1Error, match="sentinel"):
        executor.execute(copied, authorization, authorization_sha, "wrong", opener=opener)
    assert opener.calls == 0
    assert not (copied / executor.ACCESS_INTENT_PATH).exists()


def test_fake_single_get_publishes_only_allowlisted_identity_and_deletes_raw(
    tmp_path: Path,
) -> None:
    copied = _copy_package(tmp_path)
    authorization, authorization_sha = _approved_authorization(copied)
    private_token = b"PRIVATE_TX_L500_R500_CHI2_TOKEN"
    opener = _FakeOpener(_FakeResponse(_payload(private_token)))
    path = executor.execute(
        copied,
        authorization,
        authorization_sha,
        executor.EXECUTE_SENTINEL,
        opener=opener,
    )
    assert path == copied / executor.RESULT_PATH
    assert opener.calls == 1
    assert opener.response is not None
    assert opener.response.read_sizes == [executor.EXPECTED_BYTES]
    request, timeout = opener.requests[0]
    assert request.full_url == executor.SOURCE_URL
    assert request.get_method() == "GET"
    assert timeout == 60
    result_text = path.read_text(encoding="utf-8")
    result = json.loads(result_text)
    executor.validate_result(result, copied)
    assert len(result["records"]) == 155
    assert set(result["records"][0]) == {"source_object_id", "ra_deg", "dec_deg", "redshift"}
    assert result["records"][0]["source_object_id"] == "1"
    assert result["records"][0]["dec_deg"] == -5.0
    assert result["records"][-1]["dec_deg"] > 0.0
    assert (
        result["source_receipt"]["source_sha256"]
        == hashlib.sha256(_payload(private_token)).hexdigest()
    )
    assert result["obsid_mapping"]["executed"] is False
    assert result["xcop_overlap"]["count"] is None
    assert result["accounting"]["scientific_values_decoded"] == 0
    assert private_token.decode("ascii") not in result_text
    assert "\ufffd" not in result_text
    assert (copied / executor.ACCESS_INTENT_PATH).is_file()
    assert (copied / executor.GET_ATTEMPT_PATH).is_file()
    temporary_parent = copied / executor.TEMPORARY_PARENT
    assert temporary_parent.is_dir()
    assert list(temporary_parent.iterdir()) == []


@pytest.mark.parametrize("opaque", [b"\r", b"\n"])
def test_fixed_chunks_do_not_compare_opaque_suffix_contents(tmp_path: Path, opaque: bytes) -> None:
    copied = _copy_package(tmp_path)
    authorization, authorization_sha = _approved_authorization(copied)
    response = _FakeResponse(_payload(opaque))
    path = executor.execute(
        copied,
        authorization,
        authorization_sha,
        executor.EXECUTE_SENTINEL,
        opener=_FakeOpener(response),
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    executor.validate_result(result, copied)
    assert len(result["records"]) == executor.EXPECTED_ROWS
    assert response.read_sizes == [executor.EXPECTED_BYTES]


@pytest.mark.parametrize("replacement", [True, 1])
def test_result_rejects_bool_and_non_float_numeric_fields(
    tmp_path: Path, replacement: object
) -> None:
    copied = _copy_package(tmp_path)
    authorization, authorization_sha = _approved_authorization(copied)
    path = executor.execute(
        copied,
        authorization,
        authorization_sha,
        executor.EXECUTE_SENTINEL,
        opener=_FakeOpener(_FakeResponse(_payload())),
    )
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["records"][0]["ra_deg"] = replacement
    body = dict(changed)
    body.pop("content_sha256")
    changed["content_sha256"] = executor._sha(body)
    with pytest.raises(
        executor.GravityGroupScaleXclassIdentityExecutorV1Error,
        match="coordinate invalid",
    ):
        executor.validate_result(changed, copied)


def test_result_bounds_and_retained_intent_are_revalidated(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    authorization, authorization_sha = _approved_authorization(copied)
    path = executor.execute(
        copied,
        authorization,
        authorization_sha,
        executor.EXECUTE_SENTINEL,
        opener=_FakeOpener(_FakeResponse(_payload())),
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    changed = copy.deepcopy(result)
    changed["records"][0]["ra_deg"] = 999.0
    body = dict(changed)
    body.pop("content_sha256")
    changed["content_sha256"] = executor._sha(body)
    with pytest.raises(
        executor.GravityGroupScaleXclassIdentityExecutorV1Error, match="frozen bounds"
    ):
        executor.validate_result(changed, copied)

    intent_path = copied / executor.ACCESS_INTENT_PATH
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    intent["status"] = "forged"
    intent_body = dict(intent)
    intent_body.pop("content_sha256")
    intent["content_sha256"] = executor._sha(intent_body)
    intent_path.write_text(json.dumps(intent, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(executor.GravityGroupScaleXclassIdentityExecutorV1Error, match="intent"):
        executor.validate_result(result, copied)


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (_FakeResponse(_payload()[:-1]), "response byte count"),
        (_FakeResponse(_payload(), content_length=None), "Content-Length"),
        (_FakeResponse(_payload(), transfer_encoding="chunked"), "transfer encoding"),
        (_FakeResponse(_payload(), content_encoding="gzip"), "content encoding"),
        (_FakeResponse(_payload(), status=302), "status"),
        (_FakeResponse(_payload(), final_url="https://example.invalid/redirect"), "final URL"),
    ],
)
def test_response_boundary_failures_retain_intent_and_cleanup(
    tmp_path: Path, response: _FakeResponse, match: str
) -> None:
    copied = _copy_package(tmp_path)
    authorization, authorization_sha = _approved_authorization(copied)
    opener = _FakeOpener(response)
    with pytest.raises(executor.GravityGroupScaleXclassIdentityExecutorV1Error, match=match):
        executor.execute(
            copied,
            authorization,
            authorization_sha,
            executor.EXECUTE_SENTINEL,
            opener=opener,
        )
    assert opener.calls == 1
    assert (copied / executor.ACCESS_INTENT_PATH).is_file()
    assert (copied / executor.GET_ATTEMPT_PATH).is_file()
    assert not (copied / executor.RESULT_PATH).exists()
    assert list((copied / executor.TEMPORARY_PARENT).iterdir()) == []


def test_shifted_fixed_width_layout_is_rejected(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    authorization, authorization_sha = _approved_authorization(copied)
    rows = [_row(index) for index in range(executor.EXPECTED_ROWS)]
    rows[0] = b" " + rows[0][:-2] + b"\n"
    opener = _FakeOpener(_FakeResponse(b"".join(rows)))
    with pytest.raises(
        executor.GravityGroupScaleXclassIdentityExecutorV1Error,
        match="fixed-width",
    ):
        executor.execute(
            copied,
            authorization,
            authorization_sha,
            executor.EXECUTE_SENTINEL,
            opener=opener,
        )
    assert not (copied / executor.RESULT_PATH).exists()
    assert list((copied / executor.TEMPORARY_PARENT).iterdir()) == []


def test_three_fraction_digit_dec_is_rejected_against_f7_4(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    authorization, authorization_sha = _approved_authorization(copied)
    rows = [_row(index) for index in range(executor.EXPECTED_ROWS)]
    malformed = bytearray(rows[0])
    malformed[15:22] = b" -5.000"
    rows[0] = bytes(malformed)
    opener = _FakeOpener(_FakeResponse(b"".join(rows)))
    with pytest.raises(
        executor.GravityGroupScaleXclassIdentityExecutorV1Error,
        match="fixed-width field",
    ):
        executor.execute(
            copied,
            authorization,
            authorization_sha,
            executor.EXECUTE_SENTINEL,
            opener=opener,
        )
    assert not (copied / executor.RESULT_PATH).exists()
    assert list((copied / executor.TEMPORARY_PARENT).iterdir()) == []


def test_delimiter_error_never_discloses_private_suffix_token(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    authorization, authorization_sha = _approved_authorization(copied)
    private_token = b"PRIVATE_TX_L500_R500_CHI2_TOKEN"
    rows = [_row(index, private_token) for index in range(executor.EXPECTED_ROWS)]
    malformed = bytearray(rows[0])
    malformed[5] = ord("X")
    rows[0] = bytes(malformed)
    opener = _FakeOpener(_FakeResponse(b"".join(rows)))
    with pytest.raises(executor.GravityGroupScaleXclassIdentityExecutorV1Error) as captured:
        executor.execute(
            copied,
            authorization,
            authorization_sha,
            executor.EXECUTE_SENTINEL,
            opener=opener,
        )
    assert "delimiter" in str(captured.value)
    assert private_token.decode("ascii") not in str(captured.value)
    assert not (copied / executor.RESULT_PATH).exists()
    assert list((copied / executor.TEMPORARY_PARENT).iterdir()) == []


def test_parser_refuses_duplicate_identity_without_decoding_suffix(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    authorization, authorization_sha = _approved_authorization(copied)
    rows = [_row(index) for index in range(executor.EXPECTED_ROWS)]
    rows[1] = rows[0]
    opener = _FakeOpener(_FakeResponse(b"".join(rows)))
    with pytest.raises(executor.GravityGroupScaleXclassIdentityExecutorV1Error, match="duplicate"):
        executor.execute(
            copied,
            authorization,
            authorization_sha,
            executor.EXECUTE_SENTINEL,
            opener=opener,
        )
    assert not (copied / executor.RESULT_PATH).exists()
    assert list((copied / executor.TEMPORARY_PARENT).iterdir()) == []


def test_network_exception_cleans_raw_and_intent_blocks_relaunch(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    authorization, authorization_sha = _approved_authorization(copied)
    first = _FakeOpener(error=OSError("simulated transport failure"))
    with pytest.raises(executor.GravityGroupScaleXclassIdentityExecutorV1Error, match="GET failed"):
        executor.execute(
            copied,
            authorization,
            authorization_sha,
            executor.EXECUTE_SENTINEL,
            opener=first,
        )
    assert first.calls == 1
    assert list((copied / executor.TEMPORARY_PARENT).iterdir()) == []
    second = _FakeOpener(_FakeResponse(_payload()))
    with pytest.raises(executor.GravityGroupScaleXclassIdentityExecutorV1Error, match="intent"):
        executor.execute(
            copied,
            authorization,
            authorization_sha,
            executor.EXECUTE_SENTINEL,
            opener=second,
        )
    assert second.calls == 0


def test_concurrent_creators_allow_only_one_get(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    authorization, authorization_sha = _approved_authorization(copied)
    openers = [_FakeOpener(_FakeResponse(_payload())) for _ in range(2)]

    def run(index: int):
        try:
            return executor.execute(
                copied,
                authorization,
                authorization_sha,
                executor.EXECUTE_SENTINEL,
                opener=openers[index],
            )
        except executor.GravityGroupScaleXclassIdentityExecutorV1Error as error:
            return str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(run, range(2)))
    assert sum(opener.calls for opener in openers) == 1
    assert sum(isinstance(value, Path) for value in outcomes) == 1
    assert (copied / executor.RESULT_PATH).is_file()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["source_contract"].__setitem__("expected_network_bytes", 1), "source"),
        (lambda value: value["network_contract"].__setitem__("retry_calls", 1), "network"),
        (
            lambda value: value["column_contract"]["decode_allowlist"].append("TX"),
            "column",
        ),
        (
            lambda value: value["private_payload_contract"].__setitem__(
                "raw_payload_deleted_before_output_publish", False
            ),
            "private payload",
        ),
        (
            lambda value: value["authorization_contract"].__setitem__(
                "authorized_manifest_present_at_freeze", True
            ),
            "authorization",
        ),
        (
            lambda value: value["obsid_contract"].__setitem__("obsid_guessing_allowed", True),
            "ObsID",
        ),
        (
            lambda value: value["xcop_overlap_contract"].__setitem__("overlap_count", 0),
            "X-COP",
        ),
        (
            lambda value: value["execution_accounting_at_freeze"].__setitem__(
                "executor_launches", 1
            ),
            "freeze accounting",
        ),
        (
            lambda value: value["claim_boundary"].__setitem__("source_identity_acquired", True),
            "claim boundary",
        ),
        (
            lambda value: value["parent_v3_binding"].__setitem__("receipt_file_sha256", "0" * 64),
            "parent",
        ),
    ],
)
def test_nested_mutations_fail_closed(mutation, match: str) -> None:
    changed = copy.deepcopy(executor.load_config(_repo_root()))
    mutation(changed)
    with pytest.raises(executor.GravityGroupScaleXclassIdentityExecutorV1Error, match=match):
        executor.validate_config(changed)


def test_preflight_publication_is_atomic_no_replace(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    (copied / executor.PREFLIGHT_PATH).unlink()
    expected = executor.build_preflight_receipt(copied)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: executor.write_preflight_receipt(copied), range(8)))
    assert paths == [copied / executor.PREFLIGHT_PATH] * 8
    assert json.loads((copied / executor.PREFLIGHT_PATH).read_text(encoding="utf-8")) == expected
    (copied / executor.PREFLIGHT_PATH).write_text('{"different":true}\n', encoding="utf-8")
    with pytest.raises(executor.GravityGroupScaleXclassIdentityExecutorV1Error, match="replace"):
        executor.write_preflight_receipt(copied)
