"""
optical_beams_explicit.py

Elliptical Gaussian beam modeling with units (pint), polarization (Jones vectors),
and common optical components. All names are explicit; no abbreviations.

Dependencies: pint, numpy, matplotlib
"""

from TypedUnit import ureg, Length, Angle
from dataclasses import replace
from typing import Literal, Tuple, List
from pydantic.dataclasses import dataclass
import numpy as np

from OptikTools import polarization
from OptikTools.beam import EllipticalGaussianBeam
from OptikTools.utils import config_dict

__all__ = [
    "BeamExpander",
    "CylindricalTelescope",
    "ThinFocusingLens",
    "IdealLinearPolarizer",
    "GeneralWavePlate",
    "HalfWavePlate",
    "QuarterWavePlate",
    "IdealFaradayPolarizationRotator",
    "IdealPolarizingBeamSplitterCube",
    "OpticalComponentSequence",
]

# --------------------------- Optical components ------------------------------

@dataclass(config=config_dict)
class BeamExpander:
    """
    A beam expander model.

    Parameters
    ----------
    linear_magnification : float
        The linear magnification factor of the beam expander.
    optical_throughput : float, optional
        The optical throughput of the beam expander (default is 1.0).
    """
    linear_magnification: float
    optical_throughput: float = 1.0

    def apply_to_beam(self, input_beam: EllipticalGaussianBeam) -> EllipticalGaussianBeam:
        """
        Apply the beam expander to the input beam.

        Parameters
        ----------
        input_beam : EllipticalGaussianBeam
            The input beam to be expanded.

        Returns
        -------
        EllipticalGaussianBeam
            The expanded output beam.
        """
        return replace(
            input_beam,
            beam_waist_radius_x=input_beam.beam_waist_radius_x * self.linear_magnification,
            beam_waist_radius_y=input_beam.beam_waist_radius_y * self.linear_magnification,
            optical_power=input_beam.optical_power * self.optical_throughput
        )

    def __repr__(self) -> str:
        return f"<BeamExpander linear magnification={self.linear_magnification:.3f} *, optical throughput={self.optical_throughput:.3f}>"

@dataclass(config=config_dict)
class CylindricalTelescope:
    """
    A cylindrical Keplerian telescope model.

    Parameters
    ----------
    focal_length_element_1 : Length
        Focal length of the first optical element.
    focal_length_element_2 : Length
        Focal length of the second optical element.
    acted_axis : Literal["x", "y"], optional
        The axis along which the telescope acts (default is "x").
    optical_throughput : float, optional
        The optical throughput of the telescope (default is 1.0).
    """
    focal_length_element_1: Length
    focal_length_element_2: Length
    acted_axis: Literal["x", "y"] = "x"
    optical_throughput: float = 1.0

    def __post_init__(self):
        self.linear_magnification = abs((self.focal_length_element_2 / self.focal_length_element_1).to_base_units().m)
        self.optical_throughput = float(self.optical_throughput)

    def apply_to_beam(self, input_beam: EllipticalGaussianBeam) -> EllipticalGaussianBeam:
        """
        Apply the cylindrical Keplerian telescope model to the input beam.

        Parameters
        ----------
        input_beam : EllipticalGaussianBeam
            The input beam to which the telescope model is applied.

        Returns
        -------
        EllipticalGaussianBeam
            The output beam after applying the telescope model.
        """
        if self.acted_axis == "x":
            return replace(
                input_beam,
                beam_waist_radius_x=input_beam.beam_waist_radius_x * self.linear_magnification,
                optical_power=input_beam.optical_power * self.optical_throughput
            )
        else:
            return replace(
                input_beam,
                beam_waist_radius_y=input_beam.beam_waist_radius_y * self.linear_magnification,
                optical_power=input_beam.optical_power * self.optical_throughput
            )

    def __repr__(self) -> str:
        return f"<CylindricalTelescope acted_axis={self.acted_axis}, linear magnification={self.linear_magnification:.3f} *, optical throughput={self.optical_throughput:.3f}>"


@dataclass(config=config_dict)
class ThinFocusingLens:
    """
    A thin focusing lens model.

    Parameters
    ----------
    focal_length : Length
        Focal length of the lens.
    optical_throughput : float, optional
        The optical throughput of the lens (default is 1.0).
    """
    focal_length: Length
    optical_throughput: float = 1.0

    def apply_to_beam(self, beam):
        return self.focus_collimated_beam(beam)

    def focus_collimated_beam(self, pupil_beam: EllipticalGaussianBeam) -> EllipticalGaussianBeam:
        """
        Focus a collimated beam using the lens.

        Parameters
        ----------
        pupil_beam : EllipticalGaussianBeam
            The collimated beam to be focused.

        Returns
        -------
        EllipticalGaussianBeam
            The focused output beam.
        """
        # Gaussian waist at focus: w0 = λ f / (π w_pupil)
        waist_x = (pupil_beam.vacuum_wavelength * self.focal_length / (np.pi * pupil_beam.beam_waist_radius_x)).to(ureg.meter)
        waist_y = (pupil_beam.vacuum_wavelength * self.focal_length / (np.pi * pupil_beam.beam_waist_radius_y)).to(ureg.meter)
        return EllipticalGaussianBeam.from_waists(
            vacuum_wavelength=pupil_beam.vacuum_wavelength,
            beam_waist_radius_x=waist_x,
            beam_waist_radius_y=waist_y,
            polarization_state=pupil_beam.polarization_state.copy(),
            optical_power=pupil_beam.optical_power * self.optical_throughput
        )

    def __repr__(self) -> str:
        return f"<ThinFocusingLens focal length={self.focal_length.to('mm'):.2f}, optical throughput={self.optical_throughput:.3f}>"

# ------------------------ Polarization components ----------------------------

@dataclass(config=config_dict)
class IdealLinearPolarizer:
    """
    An ideal linear polarizer.

    Parameters
    ----------
    transmission_axis_angle : Angle
        The angle of the transmission axis of the polarizer.
    """

    transmission_axis_angle: Angle

    def apply_to_beam(self, input_beam: EllipticalGaussianBeam) -> EllipticalGaussianBeam:
        """
        Apply the polarizer to the input beam.

        Parameters
        ----------
        input_beam : EllipticalGaussianBeam
            The input beam to be polarized.

        Returns
        -------
        EllipticalGaussianBeam
            The polarized output beam.
        """
        R_to_local = polarization.create_rotation_matrix_for_jones(-self.transmission_axis_angle)
        E_in_lab = input_beam.polarization_state.jones_vector
        E_local = R_to_local @ E_in_lab

        # 2) Ideal linear polarizer: keep only the local x component
        E_local_after = np.array([E_local[0], 0.0 + 0.0j], dtype=np.complex128)

        # 3) Return to lab basis
        R_to_lab = polarization.create_rotation_matrix_for_jones(self.transmission_axis_angle)
        E_out_lab = R_to_lab @ E_local_after

        # 4) Power transmission fraction (since ||E||=1, this is |Ex_local|^2)
        S0_in, *_ = polarization.compute_stokes_parameters_from_jones(E_in_lab)
        S0_out, *_ = polarization.compute_stokes_parameters_from_jones(E_out_lab)
        transmitted_fraction = 0.0 if S0_in == 0.0 else (S0_out / S0_in)

        # 5) Normalize Jones for the new polarization state (unless completely extinguished)
        if S0_out > 0.0:
            E_out_unit = E_out_lab / np.linalg.norm(E_out_lab)
        else:
            # If extinguished numerically, define state as linear along the transmission axis
            E_out_unit = polarization.create_jones_vector_for_linear_polarization(self.transmission_axis_angle)

        new_polarization_state = polarization.PolarizationState.from_jones_vector(E_out_unit)
        new_optical_power = input_beam.optical_power * transmitted_fraction

        # 6) Beam geometry unchanged; update polarization and power
        return replace(
            input_beam,
            polarization_state=new_polarization_state,
            optical_power=new_optical_power,
        )

    def __repr__(self) -> str:
        return f"<IdealLinearPolarizer\n transmission axis angle={self.transmission_axis_angle:~P}>"


@dataclass(config=config_dict)
class GeneralWavePlate:
    """
    A general wave plate model.

    Parameters
    ----------
    retardance : Angle
        The retardance of the wave plate.
    fast_axis_angle : Angle
        The angle of the fast axis of the wave plate.
    """
    retardance: Angle
    fast_axis_angle: Angle

    def apply_to_beam(self, input_beam: EllipticalGaussianBeam) -> EllipticalGaussianBeam:
        """
        Apply the wave plate to the input beam.

        Parameters
        ----------
        input_beam : EllipticalGaussianBeam
            The input beam to be affected by the wave plate.

        Returns
        -------
        EllipticalGaussianBeam
            The output beam after passing through the wave plate.
        """
        # 1) Rotate field into retarder eigenbasis (fast axis aligned to local x)
        R_to_local = polarization.create_rotation_matrix_for_jones(-self.fast_axis_angle)

        # 2) Apply the phase retardance between local axes (fast vs slow)
        J_delta = np.array(
            [[np.exp(-1j * self.retardance.to('radian').magnitude / 2.0), 0.0],
             [0.0, np.exp(+1j * self.retardance.to('radian').magnitude / 2.0)]],
            dtype=np.complex128
        )

        # 3) Rotate back to lab basis
        R_to_lab = polarization.create_rotation_matrix_for_jones(self.fast_axis_angle)

        # 4) Propagate the Jones vector (use the underlying vector, not the object)
        E_in = input_beam.polarization_state.jones_vector
        E_out = R_to_lab @ (J_delta @ (R_to_local @ E_in))

        # 5) Normalize Jones (retarder is lossless) and update polarization state
        E_out /= np.linalg.norm(E_out)
        new_pol = polarization.PolarizationState.from_jones_vector(E_out)

        # 6) Geometry unchanged; power unchanged (ideal retarder)
        return replace(input_beam, polarization_state=new_pol)

    def __repr__(self) -> str:
        return f"<GeneralWavePlate\n retardance={self.retardance:~P}\n fast axis angle={self.fast_axis_angle:~P}>"


class HalfWavePlate(GeneralWavePlate):
    """
    A half-wave plate model.

    Parameters
    ----------
    fast_axis_angle : Angle
        The angle of the fast axis of the half-wave plate.
    """
    def __init__(self, fast_axis_angle: Angle):
        super().__init__(retardance=np.pi * ureg.radian, fast_axis_angle=fast_axis_angle)

    def __repr__(self) -> str:
        return f"<HalfWavePlate\n fast axis angle={self.fast_axis_angle:~P}>"

class QuarterWavePlate(GeneralWavePlate):
    """
    A quarter-wave plate model.

    Parameters
    ----------
    fast_axis_angle : Angle
        The angle of the fast axis of the quarter-wave plate.
    """
    def __init__(self, fast_axis_angle: Angle):
        super().__init__(retardance=np.pi / 2 * ureg.radian, fast_axis_angle=fast_axis_angle)

    def __repr__(self) -> str:
        return f"<QuarterWavePlate\n fast axis angle={self.fast_axis_angle:~P}>"


@dataclass(config=config_dict)
class IdealFaradayPolarizationRotator:
    """
    An ideal Faraday polarization rotator.

    Parameters
    ----------
    rotation_angle : Angle
        The angle of rotation of the polarization.
    """
    rotation_angle: Angle

    def apply_to_beam(self, input_beam: EllipticalGaussianBeam) -> EllipticalGaussianBeam:
        output_jones = polarization.create_jones_vector_for_linear_polarization(self.rotation_angle) @ input_beam.polarization_state
        return replace(input_beam, polarization_state=output_jones / np.linalg.norm(output_jones))

    def __repr__(self) -> str:
        return f"<IdealFaradayPolarizationRotator\n rotation angle degree={self.rotation_angle:~P}>"


@dataclass(config=config_dict)
class IdealPolarizingBeamSplitterCube:
    """
    An ideal polarizing beam splitter (PBS) cube.

    Parameters
    ----------
    transmission_axis_angle : Angle
        The angle of the transmission axis of the PBS.
    """
    transmission_axis_angle: Angle

    def split_beam(self, input_beam: EllipticalGaussianBeam) -> Tuple[EllipticalGaussianBeam, EllipticalGaussianBeam]:
        """
        Split the input beam into two output beams: transmitted and reflected.

        Parameters
        ----------
        input_beam : EllipticalGaussianBeam
            The input beam to be split.

        Returns
        -------
        Tuple[EllipticalGaussianBeam, EllipticalGaussianBeam]
            The transmitted and reflected output beams.
        """
        rotation_to_local = polarization.create_jones_vector_for_linear_polarization(-self.transmission_axis_angle)
        local_jones = rotation_to_local @ input_beam.polarization_state
        local_component_x = local_jones[0]
        local_component_y = local_jones[1]

        if abs(local_component_x) > 0:
            transmitted_local = np.array([local_component_x, 0.0 + 0.0j], dtype=np.complex128)
            transmitted_global = polarization.create_jones_vector_for_linear_polarization(self.transmission_axis_angle) @ transmitted_local
            transmitted_global /= np.linalg.norm(transmitted_global)
        else:
            transmitted_global = polarization.create_jones_vector_for_linear_polarization(self.transmission_axis_angle) @ np.array([1.0 + 0.0j, 0.0 + 0.0j])

        if abs(local_component_y) > 0:
            reflected_local = np.array([0.0 + 0.0j, local_component_y], dtype=np.complex128)
            reflected_global = polarization.create_jones_vector_for_linear_polarization(self.transmission_axis_angle) @ reflected_local
            reflected_global /= np.linalg.norm(reflected_global)
        else:
            reflected_global = polarization.create_jones_vector_for_linear_polarization(self.transmission_axis_angle) @ np.array([0.0 + 0.0j, 1.0 + 0.0j])

        stokes_local_0, *_ = polarization.compute_stokes_parameters_from_jones(local_jones)
        transmitted_fraction = float((np.abs(local_component_x)**2).real / max(stokes_local_0, 1e-16))
        reflected_fraction = float((np.abs(local_component_y)**2).real / max(stokes_local_0, 1e-16))

        transmitted_beam = replace(
            input_beam,
            polarization_state=transmitted_global,
            optical_power=input_beam.optical_power * transmitted_fraction
        )

        reflected_beam = replace(
            input_beam,
            polarization_state=reflected_global,
            optical_power=input_beam.optical_power * reflected_fraction
        )

        return transmitted_beam, reflected_beam

    def __repr__(self) -> str:
        return f"<IdealPolarizingBeamSplitterCube\n transmission axis angle={self.transmission_axis_angle:~P}>"

# --------------------------- Optional: optical train -------------------------

class OpticalComponentSequence:
    """
    Simple sequencer for components that expose a method named apply_to_beam.
    """
    def __init__(self, component_list: List[object]):
        self.component_list = component_list

    def apply_to_beam(self, input_beam: EllipticalGaussianBeam) -> EllipticalGaussianBeam:
        current_beam = input_beam
        for component in self.component_list:
            if hasattr(component, "apply_to_beam"):
                current_beam = component.apply_to_beam(current_beam)
            else:
                raise TypeError(f"Component {component} does not implement apply_to_beam")
        return current_beam

    def __repr__(self) -> str:
        return f"<OpticalComponentSequence number of components={len(self.component_list)}>"


