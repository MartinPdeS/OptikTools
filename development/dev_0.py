
from TypedUnit import ureg
from OptikTools.beam import EllipticalGaussianBeam
from OptikTools.components import BeamExpander, BeamExpander, CylindricalTelescope, HalfWavePlate, ThinFocusingLens, OpticalComponentSequence, IdealLinearPolarizer
from OptikTools.polarization import PolarizationState


polarization_horizontal = PolarizationState.from_linear_polarization(0.0 * ureg.degree)

beam = EllipticalGaussianBeam.from_numerical_apertures(
    vacuum_wavelength=405 * ureg.nanometer,
    numerical_aperture_x=0.12,
    numerical_aperture_y=0.12,
    polarization_state=polarization_horizontal,
    optical_power=200.0 * ureg.milliwatt
)

component_list = [
    IdealLinearPolarizer(transmission_axis_angle=0 * ureg.degree),
    BeamExpander(linear_magnification=2.0),
    HalfWavePlate(fast_axis_angle=0 * ureg.degree),
    CylindricalTelescope(focal_length_element_1=30 * ureg.millimeter, focal_length_element_2=200 * ureg.millimeter, acted_axis="x"),
    ThinFocusingLens(focal_length=100 * ureg.millimeter)
]

optical_setup = OpticalComponentSequence(component_list)

beam = optical_setup.apply_to_beam(beam)

print(beam)

beam.plot_intensity_profile()
beam.polarization_state.plot()