"""Chapter 3 -- Aerodynamics of Forward Flight.

Prouty, "Helicopter Performance, Stability and Control", pp. 119-272.

Two independent rotor models and the trim problems built on them. Departures
from the printed text, and disagreements found inside it, are recorded in
docs/validation_forward_flight.md rather than in each module; the modules carry the
reasoning, the notes carry the measurements.

    G0   ForwardFlightConditionsGroup   flight conditions and coefficients
    G1   ClosedFormRotorGroup           closed-form thrust, torque, H-force
    G1b  TrimConditionsGroup            level flight, climb, autorotation
    G2   NumericalRotorGroup            blade element integration over the disc
    G3   RotorChartGenerator            isolated rotor charts from G2
    G4   FixedCollectiveTrimGroup       dive and auxiliary propulsion
         WindTunnelRotorGroup           isolated rotor with wall corrections

The two rotor models are interchangeable. TrimConditionsGroup,
FixedCollectiveTrimGroup and WindTunnelRotorGroup all take
rotor='closed_form' or 'numerical'; the first is instant and assumes a mean
drag coefficient, the second integrates the disc element by element and needs
the Chapter 6 airfoil. Swapping matters most where the induced torque is
negative -- a dive, an autorotation, a rotor at positive shaft angle in a
tunnel -- because there C_Q is a small difference of cancelling terms and a
mean c_d has no signal left to give.

G2 is differentiable through its Newton, including the trim and the inflow
loop, so it can drive a gradient-based optimiser. Accuracy of the totals is
about 1e-3 in the smooth interior and degrades to a few per cent near the
Chapter 6 clips; see the validation notes before trusting them at a new
operating point.
"""

# --- G0, conditions and non-dimensionalisation -------------------------
from .advance_ratio_comp import AdvanceRatioComp
from .avg_lift_coef_comp import AvgLiftCoefComp
from .blade_area_comp import BladeAreaComp
from .drag_rise_mach_comp import DragRiseMachComp, m_dr3_from_thickness
from .dynamic_pressure_comp import DynamicPressureComp
from .forward_flight_conditions_group import ForwardFlightConditionsGroup
from .induced_velocity_comp import InducedVelocityComp
from .inflow_comp import InflowComp
from .lock_number_comp import LockNumberComp
from .thrust_coef_comp import ThrustCoefComp
from .tip_mach_comp import TipMachComp
from .tpp_angle_comp import TppAngleComp

# --- G1, closed form ---------------------------------------------------
from .closed_form_rotor_group import ClosedFormRotorGroup
from .collective_pitch_comp import CollectivePitchComp
from .compressibility_torque_comp import CompressibilityTorqueComp
from .coning_comp import ConingComp
from .h_force_coef_comp import HForceCoefComp
from .lat_flapping_comp import LatFlappingComp
from .long_flapping_comp import LongFlappingComp
from .mean_drag_coef import MeanDragCoefGroup
from .torque_coef_comp import TorqueCoefComp

# --- G1b, the helicopter in trim ---------------------------------------
from .flight_path_comp import FlightPathComp
from .fuselage_aero_comp import FuselageAeroComp
from .fuselage_angle_comp import FuselageAngleComp
from .losses_torque_comp import LossesTorqueComp
from .tail_rotor_group import TailRotorGroup
from .tail_rotor_load_comp import RotorPowerComp, TailRotorLoadComp
from .tail_rotor_trim_comp import TailRotorTrimComp
from .trim_conditions_group import TrimConditionsGroup

# --- G2, numerical integration over the disc ---------------------------
from .alpha_comp import AlphaComp
from .chord_force_comp import ChordForceComp
from .disc_airfoil_group import DiscAirfoilGroup
from .disc_integral_comp import DiscIntegralComp
from .gyro_moment_comp import GyroMomentComp
from .lift_coef_bounds_comp import LiftCoefBoundsComp
from .loadings_comp import LoadingsComp
from .local_mach_comp import LocalMachComp
from .normal_force_comp import NormalForceComp
from .numerical_rotor_group import NumericalRotorGroup
from .pitch_dist_comp import PitchDistComp
from .resultant_vel_comp import ResultantVelComp
from .rotor_disc_grid_comp import RotorDiscGridComp
from .stall_delay_comp import StallDelayComp
from .stall_indicators_comp import StallIndicatorsComp
from .sweep_angle_comp import SweepAngleComp
from .unsteady_lift_comp import UnsteadyLiftComp
from .velocity_comps import PerpVelComp, RadialVelComp, TangentialVelComp

# --- G3, isolated rotor charts -----------------------------------------
from .chart_parameter_comp import ChartParameterComp
from .rotor_chart_generator import CHART_ROTOR, RotorChartGenerator, as_arrays

# --- G4, Table 3.5 cases 6 to 9 ----------------------------------------
from .climb_drag_area_comp import ClimbDragAreaComp
from .descent_angle_comp import DescentAngleComp
from .fixed_collective_trim_group import FixedCollectiveTrimGroup
from .propulsive_balance_comp import PropulsiveBalanceComp
from .rotor_drag_area_comp import RotorDragAreaComp
from .rotor_side_force_comp import RotorSideForceComp
from .tunnel_wall_correction_comp import TunnelWallCorrectionComp
from .wind_tunnel_rotor_group import WindTunnelRotorGroup

GROUPS = ('ForwardFlightConditionsGroup', 'ClosedFormRotorGroup',
          'TrimConditionsGroup', 'NumericalRotorGroup', 'DiscAirfoilGroup',
          'TailRotorGroup', 'FixedCollectiveTrimGroup',
          'WindTunnelRotorGroup')

__all__ = [
    'AdvanceRatioComp', 'AlphaComp', 'AvgLiftCoefComp', 'BladeAreaComp',
    'CHART_ROTOR', 'ChartParameterComp', 'ChordForceComp',
    'ClimbDragAreaComp', 'ClosedFormRotorGroup', 'CollectivePitchComp',
    'CompressibilityTorqueComp', 'ConingComp', 'DescentAngleComp',
    'DiscAirfoilGroup', 'DiscIntegralComp', 'DragRiseMachComp',
    'DynamicPressureComp', 'FixedCollectiveTrimGroup', 'FlightPathComp',
    'ForwardFlightConditionsGroup', 'FuselageAeroComp', 'FuselageAngleComp',
    'GROUPS', 'GyroMomentComp', 'HForceCoefComp', 'InducedVelocityComp',
    'InflowComp', 'LatFlappingComp', 'LiftCoefBoundsComp', 'LoadingsComp',
    'LocalMachComp', 'LockNumberComp', 'LongFlappingComp', 'LossesTorqueComp',
    'MeanDragCoefGroup', 'NormalForceComp', 'NumericalRotorGroup',
    'PerpVelComp', 'PitchDistComp', 'PropulsiveBalanceComp', 'RadialVelComp',
    'ResultantVelComp', 'RotorChartGenerator', 'RotorDiscGridComp',
    'RotorDragAreaComp', 'RotorPowerComp', 'RotorSideForceComp',
    'StallDelayComp', 'StallIndicatorsComp', 'SweepAngleComp',
    'TailRotorGroup', 'TailRotorLoadComp', 'TailRotorTrimComp',
    'TangentialVelComp', 'ThrustCoefComp', 'TipMachComp', 'TorqueCoefComp',
    'TppAngleComp', 'TrimConditionsGroup', 'TunnelWallCorrectionComp',
    'UnsteadyLiftComp', 'WindTunnelRotorGroup', 'as_arrays',
    'm_dr3_from_thickness',
]
