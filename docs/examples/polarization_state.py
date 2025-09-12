"""
Polarization States
===================

This example illustrates linear, circular, and elliptical
polarization states and plots their ellipses.
"""

import numpy as np
from TypedUnit import ureg

from OptikTools.polarization import PolarizationState

# Linear at 45°
linear_pol = PolarizationState.from_linear_polarization(45 * ureg.degree)
linear_pol.plot_polarization_ellipse()

# Right circular
circular_pol = PolarizationState.from_ellipse_parameters(
    ellipse_azimuth_radians=0.0, ellipse_ellipticity_radians=45 * ureg.degree
)
circular_pol.plot_polarization_ellipse()

# Elliptical
elliptical_pol = PolarizationState.from_ellipse_parameters(
    ellipse_azimuth_radians=30 * ureg.degree,
    ellipse_ellipticity_radians=20 * ureg.degree,
)
elliptical_pol.plot_polarization_ellipse()
