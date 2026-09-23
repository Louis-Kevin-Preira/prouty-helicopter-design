"""Chapter 6 -- Airfoils for rotor blades, p. 426-434.

NACA 0012 lift and drag as functions of angle of attack and Mach number,
fitted to the equations of p. 426-434 rather than tabulated.

    AirfoilHoverGroup           small angles, one sign, p. 427-432
    AirfoilForwardFlightGroup   the full 0-360 deg range, p. 433-434
    StallAngleComp              sweep and dynamic overshoot on the stall
                                angles, for the Chapter 3 numerical rotor

AirfoilForwardFlightGroup takes stall_angles='external' to accept sec(Lambda)
and delta_alpha_stall from the caller, which is what the blade element
integration of Chapter 3 needs; the default reproduces the printed model.

Validation in docs/validation_airfoil.md. Note that the model returns a finite
number for any input, including Mach numbers above 1 and angles far past
stall: its arguments are solver states in Chapter 3, and a NaN in a state is
silent and fatal. See docs/validation_forward_flight.md section 3.
"""

from prouty.airfoil.airfoil_hover_group import AirfoilHoverGroup
from prouty.airfoil.airfoil_forward_flight_group import AirfoilForwardFlightGroup
from prouty.airfoil.stall_angle_comp import StallAngleComp

__all__ = ["AirfoilHoverGroup", "AirfoilForwardFlightGroup", "StallAngleComp"]
