"""
optical_beams_explicit.py

Elliptical Gaussian beam modeling with units (pint), polarization (Jones vectors),
and common optical components. All names are explicit; no abbreviations.

Dependencies: pint, numpy, matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from pydantic.dataclasses import dataclass
from TypedUnit import Length, Power, Dimensionless, ureg

from OptikTools.polarization import PolarizationState
from OptikTools.utils import config_dict


# ------------------------------ Core beams -----------------------------------

@dataclass(config=config_dict)
class EllipticalGaussianBeam:
    vacuum_wavelength: Length                # [length]
    beam_waist_radius_x: Length               # [length] 1/e^2 radius
    beam_waist_radius_y: Length               # [length] 1/e^2 radius
    polarization_state: PolarizationState       # polarization state
    optical_power: Power = 1.0 * ureg.milliwatt  # [power]

    # ---- Constructors ----
    @classmethod
    def from_waists(cls, vacuum_wavelength: Length, beam_waist_radius_x: Length, beam_waist_radius_y: Length, polarization_state: PolarizationState, optical_power: Power = 1.0 * ureg.milliwatt) -> "EllipticalGaussianBeam":
        """
        Create an EllipticalGaussianBeam from beam waists.

        Parameters
        ----------
        vacuum_wavelength : Length
            The vacuum wavelength of the beam.
        beam_waist_radius_x : Length
            The beam waist radius in the x direction.
        beam_waist_radius_y : Length
            The beam waist radius in the y direction.
        polarization_state : PolarizationState
            The polarization state of the beam.
        optical_power : Power
            The optical power of the beam.
        """
        return cls(
            vacuum_wavelength=vacuum_wavelength,
            beam_waist_radius_x=beam_waist_radius_x,
            beam_waist_radius_y=beam_waist_radius_y,
            polarization_state=polarization_state,
            optical_power=optical_power
        )

    @classmethod
    def from_numerical_apertures(cls, vacuum_wavelength: Length, numerical_aperture_x: Dimensionless, numerical_aperture_y: Dimensionless, polarization_state: PolarizationState, optical_power: Power = 1.0 * ureg.milliwatt) -> "EllipticalGaussianBeam":
        """
        Create an EllipticalGaussianBeam from numerical apertures.

        Parameters
        ----------
        vacuum_wavelength : Length
            The vacuum wavelength of the beam.
        numerical_aperture_x : Dimensionless
            The numerical aperture in the x direction.
        numerical_aperture_y : Dimensionless
            The numerical aperture in the y direction.
        polarization_state : PolarizationState
            The polarization state of the beam.
        optical_power : Power
            The optical power of the beam.
        """
        # w0 = λ / (π NA)
        beam_waist_radius_x = (vacuum_wavelength / (np.pi * numerical_aperture_x)).to(ureg.meter)
        beam_waist_radius_y = (vacuum_wavelength / (np.pi * numerical_aperture_y)).to(ureg.meter)

        return cls(
            vacuum_wavelength=vacuum_wavelength,
            beam_waist_radius_x=beam_waist_radius_x,
            beam_waist_radius_y=beam_waist_radius_y,
            polarization_state=polarization_state,
            optical_power=optical_power
        )

    # ---- Derived quantities ----
    def compute_numerical_aperture_x(self) -> Dimensionless:
        r"""
        Compute the numerical aperture in the x direction (dimensionless).
        NA_x ≈ λ / (π * w0_x) in the scalar Gaussian/paraxial limit.
        """
        numerical_aperture_x = (self.vacuum_wavelength / (np.pi * self.beam_waist_radius_x))
        return numerical_aperture_x.to(ureg.dimensionless)

    def compute_numerical_aperture_y(self) -> Dimensionless:
        r"""
        Compute the numerical aperture in the y direction (dimensionless).
        NA_y ≈ λ / (π * w0_y) in the scalar Gaussian/paraxial limit.
        """
        numerical_aperture_y = (self.vacuum_wavelength / (np.pi * self.beam_waist_radius_y))
        return numerical_aperture_y.to(ureg.dimensionless)

    def compute_rayleigh_range_x(self) -> Length:
        r"""
        Compute the Rayleigh range (z_R) in the x direction (meters).
        z_R,x = π w0_x^2 / λ
        """
        rayleigh_range_x = (np.pi * self.beam_waist_radius_x**2 / self.vacuum_wavelength)
        return rayleigh_range_x.to(ureg.meter)

    def compute_rayleigh_range_y(self) -> Length:
        """
        Compute the Rayleigh range (z_R) in the y direction (meters).
        z_R,y = π w0_y^2 / λ
        """
        rayleigh_range_y = (np.pi * self.beam_waist_radius_y**2 / self.vacuum_wavelength)
        return rayleigh_range_y.to(ureg.meter)

    def compute_stokes_parameters(self) -> tuple[float, float, float, float]:
        """
        Compute the Stokes parameters (S0, S1, S2, S3) from the polarization state.
        """
        return self.polarization_state.to_stokes_parameters()

    def compute_polarization_ellipse_parameters(self) -> tuple[float, float]:
        """
        Compute the polarization ellipse (azimuth, ellipticity) in radians
        from the polarization state.
        """
        return self.polarization_state.to_polarization_ellipse_parameters()

    # ---- Representation ----
    def __repr__(self) -> str:
        """
        Pretty representation of the EllipticalGaussianBeam.
        """
        azimuth_radians, ellipticity_radians = self.compute_polarization_ellipse_parameters()
        return (
            f"\n<EllipticalGaussianBeam>\n"
            f"  Wavelength (vacuum)       : {self.vacuum_wavelength.to('nm'):.2f}\n"
            f"  Beam waist radius (x)     : {self.beam_waist_radius_x.to('um'):.2f}\n"
            f"  Beam waist radius (y)     : {self.beam_waist_radius_y.to('um'):.2f}\n"
            f"  Numerical aperture (x)    : {self.compute_numerical_aperture_x():.4f}\n"
            f"  Numerical aperture (y)    : {self.compute_numerical_aperture_y():.4f}\n"
            f"  Polarization azimuth      : {np.degrees(azimuth_radians):.2f} deg\n"
            f"  Polarization ellipticity  : {np.degrees(ellipticity_radians):.2f} deg\n"
            f"  Optical power             : {self.optical_power.to('mW'):.2f}\n"
            f"</EllipticalGaussianBeam>\n"
        )


    # ---- Plots ----
    def plot_beam_radius_vs_axial_distance(self, axial_distance_range: Length = 1.0 * ureg.millimeter, number_of_points: int = 300) -> None:
        """
        Plot the beam radius versus axial distance for the elliptical Gaussian beam.

        Parameters
        ----------
        axial_distance_range : Length
            The range of axial distances to consider (default: 1.0 mm).
        number_of_points : int
            The number of points to sample within the axial distance range (default: 300).
        """
        axial_distance_array = np.linspace(-axial_distance_range.m_as('m'), axial_distance_range.m_as('m'), number_of_points) * ureg.meter
        beam_radius_x = self.beam_waist_radius_x * np.sqrt(1.0 + (axial_distance_array / self.compute_rayleigh_range_x())**2)
        beam_radius_y = self.beam_waist_radius_y * np.sqrt(1.0 + (axial_distance_array / self.compute_rayleigh_range_y())**2)

        plt.figure()
        plt.plot(axial_distance_array.to('mm').magnitude, beam_radius_x.to('um').magnitude, label='beam_radius_x')
        plt.plot(axial_distance_array.to('mm').magnitude, beam_radius_y.to('um').magnitude, label='beam_radius_y')
        plt.xlabel("axial distance z [mm]")
        plt.ylabel("beam radius [µm]")
        plt.title("Elliptical Gaussian beam radius versus axial distance")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_polarization_ellipse(self, ellipse_scale: float = 1.0) -> None:
        """
        Plot the polarization ellipse of the beam.

        Parameters
        ----------
        ellipse_scale : float
            The scale factor for the ellipse (default: 1.0).
        """
        azimuth_radians, ellipticity_radians = self.compute_polarization_ellipse_parameters()
        major_axis = ellipse_scale
        minor_axis = ellipse_scale * np.tan(ellipticity_radians) if np.cos(ellipticity_radians) != 0 else ellipse_scale
        parametric_angle = np.linspace(0.0, 2.0 * np.pi, 400)
        coordinate_x = major_axis * np.cos(parametric_angle) * np.cos(azimuth_radians) - minor_axis * np.sin(parametric_angle) * np.sin(azimuth_radians)
        coordinate_y = major_axis * np.cos(parametric_angle) * np.sin(azimuth_radians) + minor_axis * np.sin(parametric_angle) * np.cos(azimuth_radians)

        plt.figure()
        plt.plot(coordinate_x, coordinate_y)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("electric field Ex (arb.)")
        plt.ylabel("electric field Ey (arb.)")
        plt.title("Polarization ellipse")
        plt.grid(True)
        plt.show()

    def plot_intensity_profile(
        self,
        axial_position: Length = 0.0 * ureg.meter,
        grid_half_width: Length | None = None,
        resolution: int = 400,
        log_scale: bool = False,
        normalize: bool = True,
    ) -> None:
        """
        Plot the 2D transverse intensity profile I(x, y; z) of the elliptical Gaussian beam.

        Parameters
        ----------
        axial_position : Length
            Axial position z at which to plot the profile (0 = waist plane). Default: 0 m.
        grid_half_width : Length | None
            Half-width of the square plotting window (in *transverse* dimensions).
            If None, it is set to 3 × max( w_x(z), w_y(z) ).
        resolution : int
            Number of samples per axis (resolution x resolution grid). Default: 400.
        log_scale : bool
            If True, display log10 of normalized intensity (with a floor). Default: False.
        normalize : bool
            If True, normalize peak intensity to 1 for display. Default: True.
        """
        # --- Radii at requested z ---
        z = axial_position.to("m")
        zrx = self.compute_rayleigh_range_x()
        zry = self.compute_rayleigh_range_y()
        w0x = self.beam_waist_radius_x
        w0y = self.beam_waist_radius_y

        wx = w0x * np.sqrt(1.0 + (z / zrx) ** 2)
        wy = w0y * np.sqrt(1.0 + (z / zry) ** 2)

        # --- Field of view (defaults to ±3×max radius at z) ---
        if grid_half_width is None:
            grid_half_width = 3.0 * max(wx, wy)

        # Build transverse coordinate grid in meters
        x = np.linspace(
            -grid_half_width.m_as("m"),
            +grid_half_width.m_as("m"),
            resolution,
        ) * ureg.meter
        y = np.linspace(
            -grid_half_width.m_as("m"),
            +grid_half_width.m_as("m"),
            resolution,
        ) * ureg.meter
        X, Y = np.meshgrid(x, y, indexing="xy")

        # --- Intensity model: I(x,y;z) ∝ exp(-2 x^2 / w_x(z)^2 - 2 y^2 / w_y(z)^2) ---
        # Strip units for numpy exponentials, keep them for axes labels
        Xm = X.to("m").magnitude
        Ym = Y.to("m").magnitude
        wxm = wx.to("m").magnitude
        wym = wy.to("m").magnitude

        I = np.exp(-2.0 * (Xm**2 / (wxm**2) + Ym**2 / (wym**2)))

        # Normalize peak to 1 if requested
        if normalize and I.max() > 0:
            I = I / I.max()

        # Optional log scaling
        if log_scale:
            floor = 1e-6  # prevents log(0); ~60 dB dynamic range
            I_plot = np.log10(np.maximum(I, floor))
            im_label = "log10(Intensity / peak)"
        else:
            I_plot = I
            im_label = "Normalized intensity" if normalize else "Intensity (arb.)"

        # --- Plot ---
        extent_um = [
            x[0].to("um").magnitude,
            x[-1].to("um").magnitude,
            y[0].to("um").magnitude,
            y[-1].to("um").magnitude,
        ]

        plt.figure()
        plt.imshow(
            I_plot,
            extent=extent_um,
            origin="lower",
            aspect="equal",
            interpolation="nearest",
        )
        cbar = plt.colorbar()
        cbar.set_label(im_label)

        # 1/e^2 contour overlay: I/I0 = exp(-2)  => level = exp(-2) in linear, or log10(exp(-2)) in log scale
        if log_scale:
            level = np.log10(np.exp(-2.0))
        else:
            level = np.exp(-2.0)

        # Contour expects X, Y in display units (µm)
        Xum = X.to("um").magnitude
        Yum = Y.to("um").magnitude
        plt.contour(Xum, Yum, I_plot, levels=[level])

        plt.xlabel("x [µm]")
        plt.ylabel("y [µm]")

        # Title with plane and radii
        plt.title(fr"Elliptical Gaussian intensity")
        plt.tight_layout()
        plt.show()
