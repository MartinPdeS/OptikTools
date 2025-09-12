"""
Half-Wave Plate Effect
======================

This example shows how a half-wave plate rotates
the polarization direction of a linearly polarized beam.
"""

from TypedUnit import ureg

from OptikTools.beam import EllipticalGaussianBeam
from OptikTools.polarization import PolarizationState
from OptikTools.components import HalfWavePlate

# Initial linear polarization at 10°
polarization = PolarizationState.from_linear_polarization(10.0 * ureg.degree)

# Define a beam
beam = EllipticalGaussianBeam.from_waists(
    vacuum_wavelength=633 * ureg.nanometer,
    beam_waist_radius_x=1.0 * ureg.millimeter,
    beam_waist_radius_y=1.0 * ureg.millimeter,
    polarization_state=polarization,
    optical_power=1.0 * ureg.milliwatt,
)

# Apply half-wave plate with fast axis at 25°
hwp = HalfWavePlate(fast_axis_angle=25.0 * ureg.degree)
beam_rotated = hwp.apply_to_beam(beam)

print("Before:", beam.compute_polarization_ellipse_parameters())
print("After :", beam_rotated.compute_polarization_ellipse_parameters())

# Plot polarization ellipses before and after
beam.polarization_state.plot_polarization_ellipse()
beam_rotated.polarization_state.plot_polarization_ellipse()
