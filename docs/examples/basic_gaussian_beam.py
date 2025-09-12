"""
Basic Gaussian Beam
===================

This example shows how to construct an initial Gaussian beam,
inspect its properties, and plot the beam radius evolution.
"""

from TypedUnit import ureg

from OptikTools.beam import EllipticalGaussianBeam
from OptikTools.polarization import PolarizationState

# Define horizontal linear polarization
polarization = PolarizationState.from_linear_polarization(0.0 * ureg.degree)

# Create a circular Gaussian beam (NAx = NAy = 0.1)
beam = EllipticalGaussianBeam.from_numerical_apertures(
    vacuum_wavelength=532 * ureg.nanometer,
    numerical_aperture_x=0.1,
    numerical_aperture_y=0.1,
    polarization_state=polarization,
    optical_power=1.0 * ureg.milliwatt,
)

print(beam)

# Plot the beam radius vs axial distance
beam.plot_beam_radius_vs_axial_distance(1.0 * ureg.millimeter)
