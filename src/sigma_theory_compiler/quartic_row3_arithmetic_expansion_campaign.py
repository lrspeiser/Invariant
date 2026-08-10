from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .quartic_arithmetic_campaign_runner import run_next_row_arithmetic_campaign

SCHEMA_VERSION = "sigma-quartic-row3-arithmetic-expansion-campaign-1.0"
CERTIFICATE_SCHEMA = "sigma-quartic-row3-arithmetic-expansion-certificate-1.0"
PREVIOUS_STATUS = (
    "pass_all_12_rows0_2_arithmetic_materialized_other_rows_fail_closed"
)
SUCCESS_STATUS = (
    "pass_all_12_rows0_3_arithmetic_materialized_other_rows_fail_closed"
)


def run_quartic_row3_arithmetic_expansion_campaign(
    row2_campaign: dict[str, Any],
    metric_rows_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return run_next_row_arithmetic_campaign(
        row2_campaign,
        metric_rows_campaign,
        config,
        schema_version=SCHEMA_VERSION,
        certificate_schema=CERTIFICATE_SCHEMA,
        output_row=3,
        previous_status=PREVIOUS_STATUS,
        success_status=SUCCESS_STATUS,
        packet_key="common_row3_arithmetic_packet",
    )


def write_quartic_row3_arithmetic_expansion_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
