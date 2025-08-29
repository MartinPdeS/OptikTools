# tests/test_components_api.py
import numpy as np
import pytest

from TypedUnit import ureg
from OptikTools.polarization import PolarizationState
from OptikTools.beam import EllipticalGaussianBeam
from OptikTools.components import BeamExpander, CylindricalTelescope, ThinFocusingLens

def test_zoom_expander_scales_both_axes():
    pol = PolarizationState.from_linear_polarization(0.0 * ureg.degree)
    beam = EllipticalGaussianBeam.from_waists(
        vacuum_wavelength=532 * ureg.nanometer,
        beam_waist_radius_x=0.5 * ureg.millimeter,
        beam_waist_radius_y=0.25 * ureg.millimeter,
        polarization_state=pol,
        optical_power=1.0 * ureg.milliwatt,
    )
    expander = BeamExpander(linear_magnification=3.0)
    out = expander.apply_to_beam(beam)
    assert pytest.approx(out.beam_waist_radius_x, rel=1e-12) == 1.5
    assert pytest.approx(out.beam_waist_radius_y, rel=1e-12) == 0.75

def test_cylindrical_telescope_scales_one_axis_only():
    pol = PolarizationState.from_linear_polarization(0.0 * ureg.degree)
    beam = EllipticalGaussianBeam.from_waists(
        vacuum_wavelength=405 * ureg.nanometer,
        beam_waist_radius_x=1.0 * ureg.millimeter,
        beam_waist_radius_y=1.0 * ureg.millimeter,
        polarization_state=pol,
        optical_power=1.0 * ureg.milliwatt,
    )
    cyl = CylindricalTelescope(
        focal_length_element_1=100 * ureg.millimeter,
        focal_length_element_2=200 * ureg.millimeter,
        acted_axis="x",
    )
    out = cyl.apply_to_beam(beam)
    assert pytest.approx(out.beam_waist_radius_x, rel=1e-12) == 2.0
    assert pytest.approx(out.beam_waist_radius_y, rel=1e-12) == 1.0

def test_focusing_lens_waist_formula_matches_lambda_f_over_pi_wp():
    pol = PolarizationState.from_linear_polarization(0.0 * ureg.degree)
    pupil_beam = EllipticalGaussianBeam.from_waists(
        vacuum_wavelength=405 * ureg.nanometer,
        beam_waist_radius_x=2.0 * ureg.millimeter,   # at pupil
        beam_waist_radius_y=1.0 * ureg.millimeter,
        polarization_state=pol,
        optical_power=1.0 * ureg.milliwatt,
    )
    lens = ThinFocusingLens(focal_length=80 * ureg.millimeter)
    focus = lens.focus_collimated_beam(pupil_beam)
    # w0 = λ f / (π w_pupil)
    w0x_expected = (405e-9 * 80e-3) / (np.pi * 2.0e-3) * ureg.meter
    w0y_expected = (405e-9 * 80e-3) / (np.pi * 1.0e-3) * ureg.meter
    assert pytest.approx(focus.beam_waist_radius_x, rel=1e-12) == w0x_expected
    assert pytest.approx(focus.beam_waist_radius_y, rel=1e-12) == w0y_expected


if __name__ == "__main__":
    pytest.main(["-W error", "-s", __file__])
