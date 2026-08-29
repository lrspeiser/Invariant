from __future__ import annotations

from sigma_theory_compiler.gravity_item30_response_schema_correction import (
    parse_corrected_skyserver_csv,
)


def test_item30_schema_correction_removes_only_metadata_comments() -> None:
    payload = (
        b"#Table1\r\n"
        b"plateifu,stellar_sigma_1re,stellar_rchi2_1re,stellar_vel_lo_clip,stellar_vel_hi_clip\r\n"
        b"1000-1901,123.0,1.1,-40.0,42.0\r\n"
    )
    rows, comments = parse_corrected_skyserver_csv(payload)
    assert comments == ["#Table1"]
    assert rows == [
        {
            "plateifu": "1000-1901",
            "stellar_sigma_1re": "123.0",
            "stellar_rchi2_1re": "1.1",
            "stellar_vel_lo_clip": "-40.0",
            "stellar_vel_hi_clip": "42.0",
        }
    ]


def test_item30_schema_correction_rejects_comment_only_payload() -> None:
    try:
        parse_corrected_skyserver_csv(b"#Table1\n")
    except RuntimeError as error:
        assert "empty SkyServer CSV" in str(error)
    else:
        raise AssertionError("comment-only payload was accepted")
