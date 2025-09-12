"""
Optical Component Sequence
==========================

This example demonstrates how to apply a sequence of optical
components to a beam: polarizer, beam expander, wave plate,
cylindrical telescope, and focusing lens.
"""

from TypedUnit import ureg
from OptikTools.beam import EllipticalGaussianBeam
from OptikTools.polarization import PolarizationState
from OptikTools.components import (
    IdealLinearPolarizer,
    BeamExpander,
    HalfWavePlate,
    CylindricalTelescope,
    ThinFocusingLens,
    OpticalComponentSequence,
)

# Define initial beam (405 nm, NA = 0.12)
polarization = PolarizationState.from_linear_polarization(0.0 * ureg.degree)
beam = EllipticalGaussianBeam.from_numerical_apertures(
    vacuum_wavelength=405 * ureg.nanometer,
    numerical_aperture_x=0.12,
    numerical_aperture_y=0.12,
    polarization_state=polarization,
    optical_power=200.0 * ureg.milliwatt,
)

# Sequence of components
components = [
    IdealLinearPolarizer(transmission_axis_angle=0.0 * ureg.degree),
    BeamExpander(linear_magnification=2.0),
    HalfWavePlate(fast_axis_angle=40 * ureg.degree),
    CylindricalTelescope(
        focal_length_element_1=100 * ureg.millimeter,
        focal_length_element_2=200 * ureg.millimeter,
        acted_axis="x",
    ),
    ThinFocusingLens(focal_length=100 * ureg.millimeter),
]

optical_system = OpticalComponentSequence(components)
beam_out = optical_system.apply_to_beam(beam)

print(beam_out)
beam_out.plot_intensity_profile()
