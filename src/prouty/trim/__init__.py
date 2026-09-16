"""Chapter 8 -- The Helicopter in Trim, p. 481-539.

Three sections, three groups:

    TrimElementsGroup           p. 485-515   forces and moments of each
                                             component, given the unknowns
    LongitudinalTrimGroup       p. 516-530   X, Z, M -- Theta, T_M, a1s_M
    LateralDirectionalTrimGroup p. 531-538   Y, R, N -- b1s_M, Phi, T_T

The equilibrium and element components take ``linearized=True/False``.
``True`` reproduces Table 8.4 and Table 8.11 term for term, which is the
form Prouty solves; ``False`` keeps the trigonometry of p. 485-515.
"""

from prouty.trim.control_positions_comp import ControlPositionsComp
from prouty.trim.dyn_pressure_increase_comp import \
    DynPressureIncreaseComp
from prouty.trim.end_plate_factor_comp import EndPlateFactorComp
from prouty.trim.fuselage_alpha_comp import FuselageAlphaComp
from prouty.trim.fuselage_derivatives_comp import FuselageDerivativesComp
from prouty.trim.fuselage_forces_comp import FuselageForcesComp
from prouty.trim.fuselage_sidewash_comp import FuselageSidewashComp
from prouty.trim.fuselage_downwash_horiz_stab_comp import \
    FuselageDownwashAtHorizStabComp
from prouty.trim.horiz_stab_alpha_comp import HorizStabAlphaComp
from prouty.trim.hover_lateral_trim_comp import HoverLateralTrimComp
from prouty.trim.hover_longitudinal_trim_comp import \
    HoverLongitudinalTrimComp
from prouty.trim.horiz_stab_forces_comp import HorizStabForcesComp
from prouty.trim.horiz_stab_lift_drag_comp import HorizStabLiftDragComp
from prouty.trim.lat_cyclic_pitch_comp import LatCyclicPitchComp
from prouty.trim.lat_n_equilibrium_comp import LatNEquilibriumComp
from prouty.trim.lateral_directional_trim_group import \
    LateralDirectionalTrimGroup
from prouty.trim.lat_r_equilibrium_comp import LatREquilibriumComp
from prouty.trim.lat_y_equilibrium_comp import LatYEquilibriumComp
from prouty.trim.lift_curve_slope_comp import LiftCurveSlopeComp
from prouty.trim.long_cyclic_pitch_comp import LongCyclicPitchComp
from prouty.trim.long_m_equilibrium_comp import LongMEquilibriumComp
from prouty.trim.long_x_equilibrium_comp import LongXEquilibriumComp
from prouty.trim.long_z_equilibrium_comp import LongZEquilibriumComp
from prouty.trim.longitudinal_trim_group import LongitudinalTrimGroup
from prouty.trim.main_rotor_forces_comp import MainRotorForcesComp
from prouty.trim.maneuver_weight_comp import ManeuverWeightComp
from prouty.trim.rotor_downwash_comp import RotorDownwashComp
from prouty.trim.interference_factor_comp import InterferenceFactorComp
from prouty.trim.span_efficiency_comp import SpanEfficiencyComp
from prouty.trim.tail_rotor_forces_comp import TailRotorForcesComp
from prouty.trim.tail_rotor_sidewash_comp import TailRotorSidewashComp
from prouty.trim.tail_rotor_tpp_angle_comp import TailRotorTppAngleComp
from prouty.trim.vert_stab_effective_ar_comp import \
    VertStabEffectiveARComp
from prouty.trim.vert_stab_forces_comp import VertStabForcesComp
from prouty.trim.vert_stab_interference_drag_comp import \
    VertStabInterferenceDragComp
from prouty.trim.trim_elements_group import TrimElementsGroup
from prouty.trim.trim_gradient_comp import TrimGradientComp
from prouty.trim.vert_stab_lift_drag_comp import VertStabLiftDragComp
