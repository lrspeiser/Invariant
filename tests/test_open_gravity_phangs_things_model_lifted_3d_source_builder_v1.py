from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import (
    open_gravity_phangs_things_model_lifted_3d_source_builder_v1 as builder,
)


@pytest.fixture(scope="module")
def config() -> dict:
    value = json.loads(Path(builder.CONFIG_PATH).read_text(encoding="utf-8"))
    builder.validate_config(value)
    return value


@pytest.fixture(scope="module")
def profiles(config: dict) -> dict:
    return builder.build_profiles(config)


def test_config_and_package_scope_are_exact(config: dict) -> None:
    assert config["cell_contract"]["total_cells"] == 225
    assert config["scientific_boundary"]["response_rows_opened"] == 0
    assert config["scientific_boundary"]["scores_computed"] == 0
    assert config["scientific_boundary"]["network_calls"] == 0
    assert config["claims"]["scientific_response_scored"] is False
    assert config["claims"]["publication_ready"] is False


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("status",), "PUBLICATION_READY"),
        (("cell_contract", "total_cells"), 1),
        (("cell_contract", "response_based_cell_selection"), True),
        (("scientific_boundary", "response_rows_opened"), 1),
        (("scientific_boundary", "network_calls"), 1),
        (("claims", "potential_depth_discriminator_established"), True),
        (("published_anchor_contract", "rule"), "trust the implementation"),
    ],
)
def test_config_mutations_fail_closed(
    config: dict, path: tuple[str, ...], replacement: object
) -> None:
    mutated = copy.deepcopy(config)
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises(builder.SourceBuilderError):
        builder.validate_config(mutated)


def test_every_physical_layer_has_a_public_anchor(config: dict) -> None:
    anchors = config["published_anchor_contract"]["datasets_and_methods"]
    assert [row["id"] for row in anchors] == [
        "S4G_P5_STELLAR_MASS",
        "S4G_ML_3P6",
        "THINGS_HI",
        "PHANGS_ALMA_CO21",
        "FREEMAN_EXPONENTIAL_DISK",
        "CASERTANO_FINITE_THICKNESS_DISK",
    ]
    assert all(row["url"].startswith("https://") and row["tests"] for row in anchors)
    assert "No source conversion" in config["published_anchor_contract"]["rule"]


def test_vertical_kernel_is_finite_symmetric_and_thickness_softens(config: dict) -> None:
    gravity = config["vertical_and_gravity_model"]
    thin = builder.vertical_kernel(
        33,
        100.0,
        1.0,
        nodes=48,
        g_constant=gravity["newton_g_pc_kms2_msun"],
    )
    thick = builder.vertical_kernel(
        33,
        100.0,
        400.0,
        nodes=48,
        g_constant=gravity["newton_g_pc_kms2_msun"],
    )
    assert np.isfinite(thin).all()
    assert np.allclose(thin, thin[::-1, ::-1], rtol=0.0, atol=1.0e-15)
    center = 16
    assert thin[center, center] < thick[center, center] < 0.0


def test_published_gravity_benchmarks_pass(profiles: dict) -> None:
    report = profiles["benchmarks"]
    assert all(report["passed"].values())
    assert report["point_mass_far_field_relative_error"] < 0.03
    assert report["freeman_max_relative_error"] < 0.08
    assert 0.0 < report["finite_thickness_force_ratio_at_2p2rd"] < 1.0


def test_profiles_have_exact_objects_cells_and_radial_grid(profiles: dict) -> None:
    assert [row["object_id"] for row in profiles["objects"]] == [
        "NGC2903",
        "NGC3351",
        "NGC3627",
    ]
    assert sum(len(row["cell_summaries"]) for row in profiles["objects"]) == 225
    for object_row in profiles["objects"]:
        assert len(object_row["cell_summaries"]) == 75
        assert len(object_row["cell_profiles"]) == 75
        assert len({row["cell_id"] for row in object_row["cell_summaries"]}) == 75
        assert all(len(row["radial_profile"]) == 60 for row in object_row["cell_profiles"])


def test_primary_source_masses_and_resolution_are_observationally_sane(profiles: dict) -> None:
    expected = {
        "NGC2903": (3.737168407e10, 4.726701593e9, 2.496841214e9),
        "NGC3351": (3.345557588e10, 1.748765908e9, 6.793724890e8),
        "NGC3627": (3.829479884e10, 9.495825964e8, 3.206086522e9),
    }
    for row in profiles["objects"]:
        primary = row["primary_summary"]
        star, hi, co = expected[row["object_id"]]
        assert primary["stellar_mass_msun"] == pytest.approx(star, rel=2.0e-9)
        assert primary["hi_helium_mass_msun"] == pytest.approx(hi, rel=2.0e-9)
        assert primary["co_helium_mass_msun"] == pytest.approx(co, rel=2.0e-9)
        assert 50.0 < primary["target_fwhm_pc"] < 2500.0
        assert 500.0 < row["rhalf_pc"] < 15_000.0


def test_matched_acceleration_preserves_the_new_discriminators(profiles: dict) -> None:
    target = 1.2e-10
    depths = []
    asymmetries = []
    for row in profiles["objects"]:
        matched = row["primary_summary"]["matched_acceleration"]
        assert abs(math.log10(matched["g_b_m_s2"] / target)) < 0.025
        assert matched["potential_depth_c2"] > 0.0
        assert matched["tidal_frobenius_s2"] > 0.0
        assert matched["rho_midplane_msun_pc3"] > 0.0
        assert matched["radial_force_rms_asymmetry"] > 0.0
        depths.append(matched["potential_depth_c2"])
        asymmetries.append(matched["radial_force_rms_asymmetry"])
    assert max(depths) / min(depths) < 1.06
    assert max(asymmetries) / min(asymmetries) > 2.0


def test_numerical_convergence_controls_pass_and_sip_is_not_hidden(profiles: dict) -> None:
    for row in profiles["objects"]:
        assert row["convergence"]["passed"] is True
        assert row["convergence"]["coarse_g_relative"] < 0.30
        assert row["convergence"]["padded_potential_relative"] < 0.30
        primary = next(
            value for value in row["cell_summaries"] if value["cell_id"] == row["primary_cell_id"]
        )
        sip = next(
            value
            for value in row["cell_summaries"]
            if value["cell_id"] == "S4G_SIP_HEADER_SENSITIVITY_PRIMARY_PHYSICS"
        )
        assert primary["profile_sha256"] != sip["profile_sha256"]


def test_co_omission_and_thickness_are_real_source_controls(profiles: dict) -> None:
    for row in profiles["objects"]:
        summaries = {value["cell_id"]: value for value in row["cell_summaries"]}
        with_co = summaries["ROBUST_PRIMARY:FIXED_0P6:WITH_CO:HS0.136986301369863:HG200"]
        without_co = summaries["ROBUST_PRIMARY:FIXED_0P6:WITHOUT_CO:HS0.136986301369863:HG200"]
        thin = summaries["ROBUST_PRIMARY:FIXED_0P6:WITH_CO:HS0.1:HG100"]
        thick = summaries["ROBUST_PRIMARY:FIXED_0P6:WITH_CO:HS0.2:HG400"]
        assert with_co["co_helium_mass_msun"] > 0.0
        assert without_co["co_helium_mass_msun"] == 0.0
        assert with_co["profile_sha256"] != without_co["profile_sha256"]
        assert thin["profile_sha256"] != thick["profile_sha256"]


def test_packet_retains_private_profiles_but_public_summary_only(
    config: dict, profiles: dict
) -> None:
    built_profiles, receipt = builder.build_packet(config)
    assert built_profiles == profiles
    assert receipt["cell_count"] == 225
    assert receipt["private_profile_raw_sha256"] == builder.content_sha256(profiles)
    assert "cell_profiles" not in receipt
    assert receipt["access_state"]["response_rows_opened"] == 0
    assert receipt["access_state"]["scores_computed"] == 0
    assert receipt["claims"]["publication_ready"] is False


def test_package_hash_pins_match_after_seal() -> None:
    if builder._MODULE_SEMANTIC_SHA256 == "0" * 64 or builder._TEST_RAW_SHA256 == "0" * 64:
        pytest.skip("package self-pins are installed only at the final mutation seal")
    assert (
        builder.module_semantic_sha256(builder._repo_path(builder.MODULE_PATH))
        == builder._MODULE_SEMANTIC_SHA256
    )
    assert builder.file_sha256(builder._repo_path(builder.TEST_PATH)) == builder._TEST_RAW_SHA256
