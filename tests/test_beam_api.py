# tests/test_beam_api.py
import numpy as np
import pytest
from unittest.mock import patch
from TypedUnit import ureg

from OptikTools.polarization import PolarizationState
from OptikTools.beam import EllipticalGaussianBeam

def test_constructor_from_numerical_apertures_and_repr_smoke():
    pol = PolarizationState.from_linear_polarization(0.0 * ureg.degree)
    beam = EllipticalGaussianBeam.from_numerical_apertures(
        vacuum_wavelength=405 * ureg.nanometer,
        numerical_aperture_x=0.05,
        numerical_aperture_y=0.05,
        polarization_state=pol,
        optical_power=2.0 * ureg.milliwatt,
    )
    s = repr(beam)
    assert "EllipticalGaussianBeam" in s

def test_constructor_from_waists_and_basic_properties():
    pol = PolarizationState.from_linear_polarization(0.0 * ureg.degree)
    beam = EllipticalGaussianBeam.from_waists(
        vacuum_wavelength=532 * ureg.nanometer,
        beam_waist_radius_x=1.0 * ureg.millimeter,
        beam_waist_radius_y=2.0 * ureg.millimeter,
        polarization_state=pol,
        optical_power=1.0 * ureg.milliwatt,
    )
    na_x = beam.compute_numerical_aperture_x()
    na_y = beam.compute_numerical_aperture_y()
    # NA ≈ λ/(π w0)
    assert pytest.approx(na_x.magnitude, rel=1e-12) == (532e-9) / (np.pi * 1.0e-3)
    assert pytest.approx(na_y.magnitude, rel=1e-12) == (532e-9) / (np.pi * 2.0e-3)

def test_rayleigh_ranges():
    pol = PolarizationState.from_linear_polarization(0.0 * ureg.degree)
    beam = EllipticalGaussianBeam.from_waists(
        vacuum_wavelength=1064 * ureg.nanometer,
        beam_waist_radius_x=100 * ureg.micrometer,
        beam_waist_radius_y=50 * ureg.micrometer,
        polarization_state=pol,
        optical_power=1.0 * ureg.milliwatt,
    )
    zrx = beam.compute_rayleigh_range_x()
    zry = beam.compute_rayleigh_range_y()
    # zR = π w0^2 / λ
    assert pytest.approx(zrx.to("mm").magnitude, rel=1e-12) == (np.pi * (100e-6)**2 / 1064e-9) * 1e3
    assert pytest.approx(zry.to("mm").magnitude, rel=1e-12) == (np.pi * (50e-6)**2 / 1064e-9) * 1e3

@patch("matplotlib.pyplot.show")
def test_plot_beam_radius_vs_axial_distance_smoke(mock_plt):
    pol = PolarizationState.from_linear_polarization(0.0 * ureg.degree)
    beam = EllipticalGaussianBeam.from_numerical_apertures(
        vacuum_wavelength=405 * ureg.nanometer,
        numerical_aperture_x=0.05,
        numerical_aperture_y=0.05,
        polarization_state=pol,
        optical_power=2.0 * ureg.milliwatt,
    )
    beam.plot_beam_radius_vs_axial_distance(2.0 * ureg.millimeter)


if __name__ == "__main__":
    pytest.main(["-W error", "-s", __file__])
