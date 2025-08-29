# tests/test_polarization_api.py
import numpy as np
import pytest
from unittest.mock import patch
from TypedUnit import ureg

from OptikTools.polarization import PolarizationState

def test_linear_construction_and_repr_smoke():
    pol = PolarizationState.from_linear_polarization(0.0 * ureg.degree)  # horizontal
    s = repr(pol)
    assert "PolarizationState" in s
    assert "Stokes" in s
    # Stokes for horizontal (normalized): S0=1, S1=+1, S2=0, S3=0
    s0, s1, s2, s3 = pol.to_stokes_parameters()
    assert pytest.approx(s0, rel=0, abs=1e-12) == 1.0
    assert pytest.approx(s1, rel=0, abs=1e-12) == 1.0
    assert pytest.approx(s2, rel=0, abs=1e-12) == 0.0
    assert pytest.approx(s3, rel=0, abs=1e-12) == 0.0

@pytest.mark.parametrize(
    "angle_deg,expected_s1,expected_s2",
    [
        (0.0,  +1.0, 0.0),  # H
        (90.0, -1.0, 0.0),  # V
        (45.0,  0.0, 1.0),  # D
        (-45.0, 0.0, -1.0), # A
    ],
)
def test_linear_stokes_map(angle_deg, expected_s1, expected_s2):
    pol = PolarizationState.from_linear_polarization(angle_deg * ureg.degree)
    s0, s1, s2, s3 = pol.to_stokes_parameters()
    assert pytest.approx(s0, abs=1e-12) == 1.0
    assert pytest.approx(s1, abs=1e-12) == expected_s1
    assert pytest.approx(s2, abs=1e-12) == expected_s2
    assert pytest.approx(s3, abs=1e-12) == 0.0

def test_circular_like_state():
    pol = PolarizationState.from_ellipse_parameters(
        ellipse_azimuth=0.0 * ureg.radian,
        ellipse_ellipticity=45.0 * ureg.degree
    )
    s0, s1, s2, s3 = pol.to_stokes_parameters()
    assert pytest.approx(s0, abs=1e-12) == 1.0
    assert pytest.approx(s1, abs=1e-6) == 0.0
    assert pytest.approx(s2, abs=1e-6) == 0.0
    assert pytest.approx(s3, abs=1e-6) == 1.0  # right-circular ideal

def test_basis_rotation_is_relabeling():
    pol = PolarizationState.from_linear_polarization(0.0 * ureg.degree)  # horizontal
    relabeled = pol.change_basis(30.0 * ureg.degree)
    # Stokes must be identical for a pure basis change
    s0, s1, s2, s3 = pol.to_stokes_parameters()
    s0r, s1r, s2r, s3r = relabeled.to_stokes_parameters()

    assert s0r == pytest.approx(s0, abs=1e-12)
    assert s1r == pytest.approx(s1, abs=1e-12)
    assert s2r == pytest.approx(s2, abs=1e-12)
    assert s3r == pytest.approx(s3, abs=1e-12)

def test_physical_rotation_changes_linear_stokes_as_expected():
    pol = PolarizationState.from_linear_polarization(0.0 * ureg.degree)  # horizontal: S1=+1, S2=0, S3=0
    rotated = pol.rotate_physical_state(30.0 * ureg.degree)
    s0, s1, s2, s3 = rotated.to_stokes_parameters()
    assert s0 == pytest.approx(1.0, abs=1e-12)
    assert s1 == pytest.approx(np.cos(np.deg2rad(60.0)), abs=1e-12)   # 0.5
    assert s2 == pytest.approx(np.sin(np.deg2rad(60.0)), abs=1e-12)   # +0.866...
    assert s3 == pytest.approx(0.0, abs=1e-12)


def test_half_wave_plate_action_on_linear():
    # Input: ψ = 10°, HWP fast axis α = 15° → ψ' = 2α − ψ = 20° (mod 180°)
    pol = PolarizationState.from_linear_polarization(10.0 * ureg.degree)
    out = pol.apply_half_wave_plate(fast_axis_angle=15.0 * ureg.degree)
    az, _ = out.to_polarization_ellipse_parameters()
    az_deg = (np.rad2deg(az) % 180.0)
    assert az_deg == pytest.approx(20.0, abs=1e-2)


def test_quarter_wave_plate_linear_to_circular():
    # Linear at 45° into a QWP at 0° → circular-like
    pol = PolarizationState.from_linear_polarization(45.0 * ureg.degree)
    out = pol.apply_quarter_wave_plate(fast_axis_angle=0.0 * ureg.radian)
    s0, s1, s2, s3 = out.to_stokes_parameters()
    assert pytest.approx(s0, abs=1e-6) == 1.0
    assert abs(s1) < 1e-6 and abs(s2) < 1e-6 and pytest.approx(abs(s3), abs=1e-6) == 1.0

def test_linear_polarizer_power_fraction_and_state():
    # Input at 30°, polarizer at 0°: transmitted fraction = cos^2(30°) = 0.75
    pol = PolarizationState.from_linear_polarization(30.0 * ureg.degree)
    out_state, frac = pol.apply_linear_polarizer(transmission_axis_angle=0.0 * ureg.radian)
    assert pytest.approx(frac, abs=1e-12) == (np.cos(np.deg2rad(30.0))**2)
    # Output state should be linear at 0°
    az, _ = out_state.to_polarization_ellipse_parameters()
    assert pytest.approx(np.rad2deg(az) % 180.0, abs=1e-6) == 0.0

@patch("matplotlib.pyplot.show")
def test_plot_methods_smoke(mock_plt):
    # Ensure plotting calls do not crash
    pol = PolarizationState.from_linear_polarization(0.0 * ureg.degree)
    pol.plot_polarization_ellipse()
    pol.plot()


if __name__ == "__main__":
    pytest.main(["-W error", "-s", __file__])
