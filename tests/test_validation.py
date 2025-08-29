# tests/test_validation.py
import numpy as np
import pytest
from TypedUnit import ureg

from OptikTools.polarization import PolarizationState
from OptikTools.beam import EllipticalGaussianBeam
from OptikTools.components import BeamExpander, CylindricalTelescope, ThinFocusingLens

def test_na_waist_round_trip():
    lam = 532 * ureg.nanometer
    na_x = 0.06
    na_y = 0.03
    pol = PolarizationState.from_linear_polarization(0.0 * ureg.degree)

    b = EllipticalGaussianBeam.from_numerical_apertures(
        vacuum_wavelength=lam,
        numerical_aperture_x=na_x,
        numerical_aperture_y=na_y,
        polarization_state=pol,
        optical_power=1.0 * ureg.milliwatt,
    )
    # Now recompute NA from w0
    na_x_rt = b.compute_numerical_aperture_x().magnitude
    na_y_rt = b.compute_numerical_aperture_y().magnitude
    assert pytest.approx(na_x_rt, rel=1e-12) == na_x
    assert pytest.approx(na_y_rt, rel=1e-12) == na_y

def test_composite_magnification_zoom_then_cylinder():
    lam = 405 * ureg.nanometer
    pol = PolarizationState.from_linear_polarization(0.0 * ureg.degree)

    b = EllipticalGaussianBeam.from_waists(
        vacuum_wavelength=lam,
        beam_waist_radius_x=1.0 * ureg.millimeter,
        beam_waist_radius_y=1.0 * ureg.millimeter,
        polarization_state=pol,
        optical_power=1.0 * ureg.milliwatt,
    )
    zoom = BeamExpander(linear_magnification=1.5)
    cyl = CylindricalTelescope(
        focal_length_element_1=75 * ureg.millimeter,
        focal_length_element_2=225 * ureg.millimeter,
        acted_axis="x",
    )
    out = cyl.apply_to_beam(zoom.apply_to_beam(b))
    # Expected: x scaled by 1.5 * (225/75)=1.5*3=4.5 ; y scaled by 1.5 only
    assert pytest.approx(out.beam_waist_radius_x.to("mm").magnitude, rel=1e-12) == 4.5
    assert pytest.approx(out.beam_waist_radius_y.to("mm").magnitude, rel=1e-12) == 1.5

def test_polarizer_transmitted_power_fraction_matches_cos2():
    pol = PolarizationState.from_linear_polarization(30.0 * ureg.degree)
    transmitted_state, transmitted_fraction = pol.apply_linear_polarizer(transmission_axis_angle=0.0 * ureg.degree)
    assert pytest.approx(transmitted_fraction, abs=1e-12) == (np.cos(np.deg2rad(30.0))**2)
    # State is linear at 0°
    az, _ = transmitted_state.to_polarization_ellipse_parameters()

    assert pytest.approx(np.rad2deg(az) % 180.0, abs=1e-6) == 0.0

def test_focus_rayleigh_range_consistency():
    lam = 633 * ureg.nanometer
    pol = PolarizationState.from_linear_polarization(0.0 * ureg.degree)
    pupil = EllipticalGaussianBeam.from_waists(
        vacuum_wavelength=lam,
        beam_waist_radius_x=2.0 * ureg.millimeter,
        beam_waist_radius_y=1.0 * ureg.millimeter,
        polarization_state=pol,
        optical_power=1.0 * ureg.milliwatt,
    )
    lens = ThinFocusingLens(focal_length=100 * ureg.millimeter)
    focus = lens.focus_collimated_beam(pupil)

    # zR = π w0^2 / λ; check x-axis
    zrx = focus.compute_rayleigh_range_x().to("um").magnitude
    w0x = focus.beam_waist_radius_x.to("um").magnitude
    lam_um = lam.to("um").magnitude
    assert pytest.approx(zrx, rel=1e-12) == np.pi * (w0x**2) / lam_um


if __name__ == "__main__":
    pytest.main(["-W error", "-s", __file__])
