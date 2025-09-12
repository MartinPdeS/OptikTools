"""
polarization_state.py

A self-contained, explicit PolarizationState class to model and visualize polarization.
- Uses Jones calculus internally.
- Provides constructors from linear angle, ellipse parameters, Jones vector, or Stokes parameters.
- Offers conversions to Stokes and ellipse parameters.
- Supports rotations, ideal linear polarizer, and general wave plates (including half-wave and quarter-wave plates).
- Includes __repr__ and plotting methods (polarization ellipse and Poincaré sphere point).

Dependencies: numpy, matplotlib
"""

from TypedUnit import ureg, Angle
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from MPSPlots import helper


def create_rotation_matrix_for_jones(rotation_angle: Angle) -> np.ndarray:
    """
    Standard CCW rotation matrix for Jones vectors.
    """
    c = np.cos(rotation_angle.to("radian").magnitude)
    s = np.sin(rotation_angle.to("radian").magnitude)
    # NOTE: +θ rotates a vector counterclockwise.
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def create_jones_vector_for_linear_polarization(linear_azimuth: Angle) -> np.ndarray:
    """
    Create a normalized Jones vector for linear polarization at a given azimuth.

    Parameters
    ----------
    linear_azimuth : Angle
        Azimuth angle (from x toward y) in radians.

    Returns
    -------
    np.ndarray
        Normalized Jones vector [Ex, Ey]^T.
    """
    cosine_value = np.cos(linear_azimuth.to("radian").magnitude)
    sine_value = np.sin(linear_azimuth.to("radian").magnitude)
    jones_vector = np.array([cosine_value, sine_value], dtype=np.complex128)
    norm = np.linalg.norm(jones_vector)
    return jones_vector / (norm if norm != 0 else 1.0)


def create_jones_vector_for_ellipse(
    ellipse_azimuth: Angle, ellipse_ellipticity: Angle
) -> np.ndarray:
    """
    Create a normalized Jones vector from ellipse azimuth and ellipticity.

    Parameters
    ----------
    ellipse_azimuth_radians : Angle
        Polarization ellipse azimuth (ψ), the major-axis angle relative to x, in radians.
    ellipse_ellipticity_radians : Angle
        Polarization ellipticity (χ), where tan(χ) = minor/major axis, in radians.

    Returns
    -------
    np.ndarray
        Normalized Jones vector.
    """
    cosine_azimuth = np.cos(ellipse_azimuth.to("radian").magnitude)
    sine_azimuth = np.sin(ellipse_azimuth.to("radian").magnitude)
    cosine_ellipticity = np.cos(ellipse_ellipticity.to("radian").magnitude)
    sine_ellipticity = np.sin(ellipse_ellipticity.to("radian").magnitude)

    electric_field_x = (
        cosine_ellipticity * cosine_azimuth - 1j * sine_ellipticity * sine_azimuth
    )
    electric_field_y = (
        cosine_ellipticity * sine_azimuth + 1j * sine_ellipticity * cosine_azimuth
    )
    jones_vector = np.array([electric_field_x, electric_field_y], dtype=np.complex128)
    norm = np.linalg.norm(jones_vector)
    return jones_vector / (norm if norm != 0 else 1.0)


def compute_stokes_parameters_from_jones(
    jones_vector: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Compute Stokes parameters from a Jones vector (up to overall power scale).

    Parameters
    ----------
    jones_vector : np.ndarray
        Jones vector [Ex, Ey]^T.

    Returns
    -------
    Tuple[float, float, float, float]
        Stokes parameters (S0, S1, S2, S3). If the Jones vector is normalized,
        S0 = 1.0.
    """
    electric_field_x, electric_field_y = jones_vector
    stokes_0 = (np.abs(electric_field_x) ** 2 + np.abs(electric_field_y) ** 2).real
    stokes_1 = (np.abs(electric_field_x) ** 2 - np.abs(electric_field_y) ** 2).real
    stokes_2 = (2.0 * np.real(electric_field_x * np.conj(electric_field_y))).real
    stokes_3 = (-2.0 * np.imag(electric_field_x * np.conj(electric_field_y))).real
    return float(stokes_0), float(stokes_1), float(stokes_2), float(stokes_3)


def compute_ellipse_parameters_from_jones(
    jones_vector: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute ellipse azimuth and ellipticity from a Jones vector.

    Parameters
    ----------
    jones_vector : np.ndarray
        Jones vector [Ex, Ey]^T.

    Returns
    -------
    Tuple[float, float]
        (ellipse_azimuth_radians, ellipse_ellipticity_radians).
    """
    stokes_0, stokes_1, stokes_2, stokes_3 = compute_stokes_parameters_from_jones(
        jones_vector
    )
    if stokes_0 == 0.0:
        return 0.0, 0.0
    normalized_s1 = stokes_1 / stokes_0
    normalized_s2 = stokes_2 / stokes_0
    normalized_s3 = stokes_3 / stokes_0
    ellipse_azimuth_radians = 0.5 * np.arctan2(normalized_s2, normalized_s1)
    ellipse_ellipticity_radians = 0.5 * np.arcsin(np.clip(normalized_s3, -1.0, 1.0))
    return float(ellipse_azimuth_radians), float(ellipse_ellipticity_radians)


@dataclass
class PolarizationState:
    """
    A complete polarization state based on a normalized Jones vector.

    The class stores a Jones vector and provides:
      - Construction from linear angle, ellipse parameters, Stokes parameters, or direct Jones.
      - Conversion to Stokes parameters and ellipse parameters.
      - Ideal rotations (basis rotations).
      - Ideal components: linear polarizer and general wave plate (including half-wave and quarter-wave).
      - Text representation and plots (polarization ellipse and Poincaré sphere position).

    Notes
    -----
    The Jones vector is assumed to be normalized internally (||E|| = 1). Power
    should be tracked externally and multiplied by the transmitted fraction when
    using a linear polarizer.
    """

    jones_vector: np.ndarray

    def copy(self) -> "PolarizationState":
        return PolarizationState(self.jones_vector.copy())

    # ---------------------- Constructors ----------------------

    @classmethod
    def from_linear_polarization(cls, linear_azimuth: Angle) -> "PolarizationState":
        return cls(create_jones_vector_for_linear_polarization(linear_azimuth))

    @classmethod
    def from_ellipse_parameters(
        cls, ellipse_azimuth: Angle, ellipse_ellipticity: Angle
    ) -> "PolarizationState":
        return cls(
            create_jones_vector_for_ellipse(ellipse_azimuth, ellipse_ellipticity)
        )

    @classmethod
    def from_jones_vector(cls, jones_vector: np.ndarray) -> "PolarizationState":
        norm = np.linalg.norm(jones_vector)
        normalized = jones_vector / (norm if norm != 0 else 1.0)
        return cls(normalized)

    @classmethod
    def from_stokes_parameters(
        cls, stokes_0: float, stokes_1: float, stokes_2: float, stokes_3: float
    ) -> "PolarizationState":
        """
        Construct a Jones-equivalent state from Stokes parameters.
        Note: For fully polarized light, S0^2 = S1^2 + S2^2 + S3^2.
              This method picks a consistent Jones vector up to a global phase.
        """
        if stokes_0 <= 0:
            # default to horizontal if degenerate
            return cls.from_linear_polarization(0.0)

        normalized_s1 = stokes_1 / stokes_0
        normalized_s2 = stokes_2 / stokes_0
        normalized_s3 = stokes_3 / stokes_0

        # Recover ellipse parameters and map to Jones
        ellipse_azimuth_radians = 0.5 * np.arctan2(normalized_s2, normalized_s1)
        ellipse_ellipticity_radians = 0.5 * np.arcsin(np.clip(normalized_s3, -1.0, 1.0))
        return cls.from_ellipse_parameters(
            ellipse_azimuth_radians, ellipse_ellipticity_radians
        )

    # ---------------------- Conversions ----------------------

    def to_stokes_parameters(self) -> Tuple[float, float, float, float]:
        return compute_stokes_parameters_from_jones(self.jones_vector)

    def to_polarization_ellipse_parameters(self) -> Tuple[float, float]:
        return compute_ellipse_parameters_from_jones(self.jones_vector)

    # ---------------------- Operations ----------------------

    def change_basis(self, rotation_angle: Angle) -> "PolarizationState":
        """
        Re-express the same physical polarization in a coordinate frame
        rotated by +rotation_angle_radians. This is a *basis relabeling*.
        Jones components transform with the inverse rotation: E' = R(-θ) E.
        Stokes parameters are invariant under this operation.
        """
        rotation_to_new_basis = create_rotation_matrix_for_jones(-rotation_angle)
        new_jones = rotation_to_new_basis @ self.jones_vector
        new_jones /= np.linalg.norm(new_jones)
        return PolarizationState(self.jones_vector.copy())

    def rotate_physical_state(self, rotation_angle: Angle) -> "PolarizationState":
        """
        Physically rotate the polarization by +rotation_angle_radians
        (e.g., what a rotator would do). Jones transforms as E' = R(+θ) E.
        Stokes parameters rotate on the Poincaré sphere; for linear states,
        S1' = cos(2θ), S2' = sin(2θ), S3' = S3.
        """
        rotation_matrix = create_rotation_matrix_for_jones(rotation_angle)
        new_jones = rotation_matrix @ self.jones_vector
        new_jones /= np.linalg.norm(new_jones)
        return PolarizationState(new_jones)

    def rotate_basis(self, rotation_angle: Angle) -> "PolarizationState":
        """
        Rotate the polarization basis by the given angle (e.g., to re-express
        the Jones vector in a different lab frame). This does not change the
        physical state, only its representation.
        """
        rotation_matrix = create_rotation_matrix_for_jones(rotation_angle)
        new_jones = rotation_matrix @ self.jones_vector
        # keep normalized
        new_jones /= np.linalg.norm(new_jones)
        return PolarizationState(new_jones)

    def apply_linear_polarizer(
        self, transmission_axis_angle: Angle
    ) -> Tuple["PolarizationState", float]:
        """
        Apply an ideal linear polarizer with a given transmission axis.

        Returns a new PolarizationState and the transmitted power fraction
        (to be multiplied externally by the optical power).
        """
        rotation_to_local = create_rotation_matrix_for_jones(-transmission_axis_angle)
        local_jones = rotation_to_local @ self.jones_vector

        # Projection onto the transmission axis (local x)
        transmitted_local = np.array([local_jones[0], 0.0 + 0.0j], dtype=np.complex128)
        transmitted_power_fraction = float(
            np.abs(local_jones[0]) ** 2
        )  # since ||jones||=1

        rotation_to_global = create_rotation_matrix_for_jones(transmission_axis_angle)
        transmitted_global = rotation_to_global @ transmitted_local
        norm = np.linalg.norm(transmitted_global)
        if norm == 0:
            transmitted_global = create_jones_vector_for_linear_polarization(
                transmission_axis_angle
            )
        else:
            transmitted_global = transmitted_global / norm

        return PolarizationState(transmitted_global), transmitted_power_fraction

    def apply_general_wave_plate(
        self, retardance: Angle, fast_axis_angle: Angle
    ) -> "PolarizationState":
        """
        Apply an ideal general wave plate to the polarization state.

        Parameters
        ----------
        retardance : Angle
            Phase retardance Δ (π for half-wave, π/2 for quarter-wave).
        fast_axis_angle : Angle
            Orientation of the fast axis relative to x.

        Returns
        -------
        PolarizationState
            New polarization after the retarder.
        """
        rotation_minus = create_rotation_matrix_for_jones(-fast_axis_angle)
        rotation_plus = create_rotation_matrix_for_jones(fast_axis_angle)
        jones_retarder = np.array(
            [
                [np.exp(-1j * retardance.to("radian").magnitude / 2.0), 0.0],
                [0.0, np.exp(+1j * retardance.to("radian").magnitude / 2.0)],
            ],
            dtype=np.complex128,
        )
        output_jones = (
            rotation_plus @ jones_retarder @ rotation_minus @ self.jones_vector
        )
        output_jones /= np.linalg.norm(output_jones)
        return PolarizationState(output_jones)

    def apply_half_wave_plate(self, fast_axis_angle: Angle) -> "PolarizationState":
        return self.apply_general_wave_plate(np.pi * ureg.radian, fast_axis_angle)

    def apply_quarter_wave_plate(self, fast_axis_angle: Angle) -> "PolarizationState":
        return self.apply_general_wave_plate(np.pi / 2.0 * ureg.radian, fast_axis_angle)

    # ---------------------- Representation ----------------------

    def __repr__(self) -> str:
        ellipse_azimuth_radians, ellipse_ellipticity_radians = (
            self.to_polarization_ellipse_parameters()
        )
        degrees_azimuth = float(np.degrees(ellipse_azimuth_radians))
        degrees_ellipticity = float(np.degrees(ellipse_ellipticity_radians))
        stokes_0, stokes_1, stokes_2, stokes_3 = self.to_stokes_parameters()
        return (
            f"<PolarizationState "
            f"Jones=[{self.jones_vector[0]:.3f}, {self.jones_vector[1]:.3f}], "
            f"Ellipse(azimuth_deg={degrees_azimuth:.2f}, ellipticity_deg={degrees_ellipticity:.2f}), "
            f"Stokes(S0={stokes_0:.3f}, S1={stokes_1:.3f}, S2={stokes_2:.3f}, S3={stokes_3:.3f})>"
        )

    # ---------------------- Plots ----------------------
    @helper.post_mpl_plot
    def plot_polarization_ellipse(self, ellipse_scale: float = 1.0) -> None:
        """
        Plot the polarization ellipse in the transverse plane (arbitrary units).
        """
        ellipse_azimuth_radians, ellipse_ellipticity_radians = (
            self.to_polarization_ellipse_parameters()
        )

        # Major and minor axes (up to an arbitrary global scale)
        major_axis = ellipse_scale
        minor_axis = (
            ellipse_scale * np.tan(ellipse_ellipticity_radians)
            if np.cos(ellipse_ellipticity_radians) != 0
            else ellipse_scale
        )

        parametric_angle = np.linspace(0.0, 2.0 * np.pi, 600)
        coordinate_x = major_axis * np.cos(parametric_angle) * np.cos(
            ellipse_azimuth_radians
        ) - minor_axis * np.sin(parametric_angle) * np.sin(ellipse_azimuth_radians)
        coordinate_y = major_axis * np.cos(parametric_angle) * np.sin(
            ellipse_azimuth_radians
        ) + minor_axis * np.sin(parametric_angle) * np.cos(ellipse_azimuth_radians)

        figure, ax = plt.subplots(1, 1)
        ax.plot(coordinate_x, coordinate_y, label="polarization ellipse")

        ax.set(
            xlabel="electric field Ex (arbitrary units)",
            ylabel="electric field Ey (arbitrary units)",
            title="Polarization ellipse",
            xlim=[-1, 1],
            ylim=[-1, 1],
        )
        plt.grid(True)
        plt.legend()
        ax.set_aspect("equal", adjustable="box")
        return figure

    @helper.post_mpl_plot
    def plot(self) -> None:
        """
        Plot the normalized Stokes vector (S1/S0, S2/S0, S3/S0) as a point on the Poincaré sphere.
        """
        stokes_0, stokes_1, stokes_2, stokes_3 = self.to_stokes_parameters()
        if stokes_0 == 0.0:
            stokes_0 = 1.0
        s1 = stokes_1 / stokes_0
        s2 = stokes_2 / stokes_0
        s3 = stokes_3 / stokes_0

        # Sphere wireframe for context (coarse)
        theta = np.linspace(0.0, np.pi, 32)  # polar
        phi = np.linspace(0.0, 2.0 * np.pi, 64)  # azimuth
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        x_grid = np.sin(theta_grid) * np.cos(phi_grid)
        y_grid = np.sin(theta_grid) * np.sin(phi_grid)
        z_grid = np.cos(theta_grid)

        figure, ax = plt.figure()
        ax = figure.add_subplot(111, projection="3d")
        ax.plot_wireframe(x_grid, y_grid, z_grid, linewidth=0.5, alpha=0.3)
        ax.scatter([s1], [s2], [s3], s=60, depthshade=True, label="Stokes vector")
        ax.set(
            xlabel="S1 / S0",
            ylabel="S2 / S0",
            zlabel="S3 / S0",
            title="Poincaré sphere position",
        )

        ax.legend()
        # Keep axes equal-ish
        lim = 1.1
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])
        return figure
