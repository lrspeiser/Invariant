from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.gravity_item38_emergent_evaluator import (
    ESD_TO_ACCELERATION,
    EXPLORATION_NAMES,
    PROFILE_MINIMA,
    _covariance_loss,
    _parse_covariance,
    _parse_profile,
    _robustness,
    _sign_permutation_p,
)
from sigma_theory_compiler.gravity_item38_emergent_gravity import load_config

ROOT = Path(__file__).resolve().parents[1]


def _profile_payload(scale: float = 1.0) -> bytes:
    lines = ["# header"]
    for index, g_bar in enumerate(np.logspace(-14, -12, 6)):
        esd = scale * (1.0 + index)
        lines.append(f"{g_bar} {esd} 0.0 0.1 0.98 0.08 1.0 1.0")
    return ("\n".join(lines) + "\n").encode()


def _covariance_payload(profiles: list[dict[str, object]]) -> bytes:
    lines = ["# header"]
    for left in profiles:
        for right in profiles:
            for radius_i in np.asarray(left["g_bar_m_s2"]):
                for radius_j in np.asarray(right["g_bar_m_s2"]):
                    value = 0.01 if radius_i == radius_j and left is right else 0.0
                    lines.append(
                        f"{left['profile_minimum']} {right['profile_minimum']} "
                        f"{radius_i} {radius_j} {value} 0.0 {0.98**2}"
                    )
    return ("\n".join(lines) + "\n").encode()


def test_item38_profile_parser_applies_published_ESD_calibration() -> None:
    profile = _parse_profile(_profile_payload(), EXPLORATION_NAMES[0])
    assert len(profile["g_bar_m_s2"]) == 6
    assert profile["profile_minimum"] == PROFILE_MINIMA[EXPLORATION_NAMES[0]]
    assert profile["esd_t_corrected"][0] == pytest.approx(1.0 / 0.98)
    assert profile["g_obs_m_s2"][0] == pytest.approx(ESD_TO_ACCELERATION / 0.98)


def test_item38_covariance_parser_builds_complete_log_space_matrix() -> None:
    profiles = [
        _parse_profile(_profile_payload(1.0 + index), name)
        for index, name in enumerate(EXPLORATION_NAMES)
    ]
    covariance = _parse_covariance(_covariance_payload(profiles), profiles)
    assert covariance.shape == (18, 18)
    assert np.allclose(covariance, covariance.T)
    assert np.all(np.diag(covariance) > 0.0)


def test_item38_covariance_loss_rewards_the_exact_prediction() -> None:
    covariance = np.eye(6) * 0.04
    assert _covariance_loss(np.zeros(6), covariance) == 0.0
    assert _covariance_loss(np.ones(6) * 0.2, covariance) == pytest.approx(1.0)


def test_item38_single_profile_influence_is_retained_not_hidden() -> None:
    rows = [
        {
            "profile": "a",
            "candidate_loss": 1.0,
            "reference_loss": 2.0,
            "comparative_difference": -1.0,
        },
        {
            "profile": "b",
            "candidate_loss": 1.0,
            "reference_loss": 2.0,
            "comparative_difference": -1.0,
        },
        {
            "profile": "c",
            "candidate_loss": 20.0,
            "reference_loss": 2.0,
            "comparative_difference": 18.0,
        },
    ]
    audit = _robustness(rows, load_config(ROOT))
    assert audit["most_influential_profile"] == "c"
    assert audit["leave_one_changes_sign"] is True
    assert audit["trim_changes_sign"] is True


def test_item38_three_profile_permutation_gate_cannot_fake_high_significance() -> None:
    rows = [
        {"reference_loss": 2.0, "candidate_loss": 1.0},
        {"reference_loss": 2.0, "candidate_loss": 1.0},
        {"reference_loss": 2.0, "candidate_loss": 1.0},
    ]
    assert _sign_permutation_p(rows) == pytest.approx(2.0 / 9.0)


def test_item38_confirmation_member_is_not_an_exploration_constant() -> None:
    from sigma_theory_compiler.gravity_item38_emergent_evaluator import (
        SEALED_CONFIRMATION,
        TRANSFER_NAMES,
    )

    assert SEALED_CONFIRMATION not in EXPLORATION_NAMES
    assert SEALED_CONFIRMATION not in TRANSFER_NAMES
