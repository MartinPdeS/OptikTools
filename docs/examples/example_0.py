import numpy as np
from TypedUnit import ureg


from OptikTools.polarization import PolarizationState
from OptikTools.beam import EllipticalGaussianBeam
from OptikTools.components import SymmetricZoomBeamExpander, CylindricalKeplerianTelescope, ThinFocusingLens

polarization_horizontal = PolarizationState.from_linear_polarization(0.0)

polarization_circular_right = PolarizationState.from_ellipse_parameters(
    ellipse_azimuth_radians=0.0,
    ellipse_ellipticity_radians=np.deg2rad(45.0)
)


# Example A: Build from numerical apertures
beam_from_na = EllipticalGaussianBeam.from_numerical_apertures(
    vacuum_wavelength=405 * ureg.nanometer,
    numerical_aperture_x=0.05,
    numerical_aperture_y=0.05,
    polarization_state=polarization_horizontal,
    optical_power=2.0 * ureg.milliwatt
)
print(beam_from_na)
beam_from_na.plot_beam_radius_vs_axial_distance(2.0 * ureg.millimeter)

# Apply a symmetric zoom beam expander
zoom_expander = SymmetricZoomBeamExpander(linear_magnification=2.0)
expanded_beam = zoom_expander.apply_to_beam(beam_from_na)
print(expanded_beam)

# Apply a cylindrical Keplerian telescope acting on x with magnification 2 (200/100)
cylindrical_telescope = CylindricalKeplerianTelescope(
    focal_length_element_1=100 * ureg.millimeter,
    focal_length_element_2=200 * ureg.millimeter,
    acted_axis="x"
)

anisotropic_beam = cylindrical_telescope.apply_to_beam(expanded_beam)


# Focus with a thin focusing lens of focal length 80 mm
focusing_lens = ThinFocusingLens(focal_length=80 * ureg.millimeter)
focused_beam = focusing_lens.focus_collimated_beam(anisotropic_beam)
print(focused_beam)
focused_beam.plot_beam_radius_vs_axial_distance(0.5 * ureg.millimeter)

# Example B: Build from waist radii directly
beam_from_waists = EllipticalGaussianBeam.from_waists(
    vacuum_wavelength=405 * ureg.nanometer,
    beam_waist_radius_x=1.5 * ureg.millimeter,
    beam_waist_radius_y=1.5 * ureg.millimeter,
    polarization_state=polarization_circular_right,
    optical_power=1.0 * ureg.milliwatt
)
print(beam_from_waists)
