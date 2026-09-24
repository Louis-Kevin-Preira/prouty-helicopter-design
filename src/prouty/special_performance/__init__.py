"""Chapter 5 -- Special performance problems, p. 339-377.

    G1  TurnsPullupsGroup, turns and pullups                        p. 340-346
        LoadFactorComp, TurnKinematicsComp, TurnCyclicReliefComp,
        TurnEnergyPowerComp, ThrustCapabilityComp (Figure 5.2)
        SteadyTurnPowerGroup: EffectiveWeightComp
        (+ ForwardFlightPowerGroup of Chapter 4)
    G2a RotorSpeedDecayGroup, rotor speed decay                     p. 348-350
        DriveInertiaComp, KineticEnergyTimeComp, RotorSpeedDecayComp
    G2f AutorotativeIndicesGroup, autorotative indices              p. 363-364
        EquivalentHoverTimeComp, AutorotativeIndexComp
    G2b AutorotationDescentGroup, steady autorotative descent       p. 350-351
        BestAutorotationSpeedGroup (DescentSpeedStencilComp)
    G2c ZoomGlideGroup, zoom maneuver and glide distance            p. 351-352
        ZoomClimbAngleComp, ZoomAltitudeGainComp, GlideDistanceComp
        ZoomGlideChainGroup: G2c fed by G2b and Chapter 4 (SpeedNodesComp,
        ChainSplitComp)
    G2d HeightVelocityGroup, height-velocity diagram                p. 352-358
        LowHoverHeightComp, MinPowerSpeedComp, CriticalSpeedComp,
        MultiEngineCriticalSpeedComp, HighHoverHeightComp, HVBoundaryComp
    G2e MinTouchdownSpeedGroup, minimum touchdown speed             p. 358-363
        FlarePitchRateComp, FlareTimeComp, FlareAngleComp,
        FlareAutorotationGroup (FlareConditionsComp + Chapter 3 rotor),
        TouchdownSpeedComp
    G3  MaxAccelerationGroup, maximum acceleration                   p. 364-365
        HoverAccelerationComp, AvailableTorqueComp, SmoothMinComp,
        RotorForceLimitGroup (WeightCoefComp, FlareConditionsComp,
        Chapter 3 rotor, DecelerationForceComp)
    G4  MaxDecelerationGroup, maximum deceleration                   p. 365-366
        RotorForceLimitGroup(mode='decel'), SmoothMinComp
    G7  TowingGroup, towing                                         p. 371-372
        TowlineTensionComp
"""

from prouty.special_performance.load_factor_comp import LoadFactorComp
from prouty.special_performance.turn_kinematics_comp import TurnKinematicsComp
from prouty.special_performance.turn_cyclic_relief_comp import TurnCyclicReliefComp
from prouty.special_performance.turn_energy_power_comp import TurnEnergyPowerComp
from prouty.special_performance.thrust_capability_comp import FIG_5_2, ThrustCapabilityComp
from prouty.special_performance.turns_pullups_group import TurnsPullupsGroup
from prouty.special_performance.effective_weight_comp import EffectiveWeightComp
from prouty.special_performance.steady_turn_power_group import SteadyTurnPowerGroup
from prouty.special_performance.drive_inertia_comp import DriveInertiaComp
from prouty.special_performance.kinetic_energy_time_comp import KineticEnergyTimeComp
from prouty.special_performance.rotor_speed_decay_comp import RotorSpeedDecayComp
from prouty.special_performance.rotor_speed_decay_group import RotorSpeedDecayGroup
from prouty.special_performance.equivalent_hover_time_comp import EquivalentHoverTimeComp
from prouty.special_performance.autorotative_index_comp import AutorotativeIndexComp
from prouty.special_performance.autorotative_indices_group import AutorotativeIndicesGroup
from prouty.special_performance.towline_tension_comp import TowlineTensionComp
from prouty.special_performance.towing_group import TowingGroup
from prouty.special_performance.zoom_climb_angle_comp import ZoomClimbAngleComp
from prouty.special_performance.zoom_altitude_gain_comp import ZoomAltitudeGainComp
from prouty.special_performance.glide_distance_comp import GlideDistanceComp
from prouty.special_performance.zoom_glide_group import ZoomGlideGroup
from prouty.special_performance.low_hover_height_comp import LowHoverHeightComp
from prouty.special_performance.min_power_speed_comp import MinPowerSpeedComp
from prouty.special_performance.critical_speed_comp import CriticalSpeedComp
from prouty.special_performance.multi_engine_critical_speed_comp import \
    MultiEngineCriticalSpeedComp
from prouty.special_performance.high_hover_height_comp import HighHoverHeightComp
from prouty.special_performance.hv_boundary_comp import HVBoundaryComp
from prouty.special_performance.height_velocity_group import HeightVelocityGroup
from prouty.special_performance.flare_pitch_rate_comp import FlarePitchRateComp
from prouty.special_performance.flare_time_comp import FlareTimeComp
from prouty.special_performance.flare_angle_comp import FlareAngleComp
from prouty.special_performance.flare_conditions_comp import FlareConditionsComp
from prouty.special_performance.flare_autorotation_group import FlareAutorotationGroup
from prouty.special_performance.touchdown_speed_comp import TouchdownSpeedComp
from prouty.special_performance.min_touchdown_speed_group import MinTouchdownSpeedGroup
from prouty.special_performance.autorotation_descent_group import AutorotationDescentGroup
from prouty.special_performance.descent_speed_stencil_comp import DescentSpeedStencilComp
from prouty.special_performance.best_autorotation_speed_group import BestAutorotationSpeedGroup
from prouty.special_performance.speed_nodes_comp import SpeedNodesComp
from prouty.special_performance.chain_split_comp import ChainSplitComp
from prouty.special_performance.zoom_glide_chain_group import ZoomGlideChainGroup
from prouty.special_performance.hover_acceleration_comp import HoverAccelerationComp
from prouty.special_performance.available_torque_comp import AvailableTorqueComp
from prouty.special_performance.smooth_min_comp import SmoothMinComp
from prouty.special_performance.max_acceleration_group import MaxAccelerationGroup
from prouty.special_performance.weight_coef_comp import WeightCoefComp
from prouty.special_performance.deceleration_force_comp import DecelerationForceComp
from prouty.special_performance.rotor_force_limit_group import RotorForceLimitGroup
from prouty.special_performance.max_deceleration_group import MaxDecelerationGroup

__all__ = ['LoadFactorComp', 'TurnKinematicsComp', 'TurnCyclicReliefComp',
           'TurnEnergyPowerComp', 'FIG_5_2', 'ThrustCapabilityComp', 'TurnsPullupsGroup',
           'EffectiveWeightComp', 'SteadyTurnPowerGroup',
           'DriveInertiaComp', 'KineticEnergyTimeComp', 'RotorSpeedDecayComp',
           'RotorSpeedDecayGroup', 'EquivalentHoverTimeComp', 'AutorotativeIndexComp',
           'AutorotativeIndicesGroup', 'TowlineTensionComp', 'TowingGroup',
           'ZoomClimbAngleComp', 'ZoomAltitudeGainComp', 'GlideDistanceComp', 'ZoomGlideGroup',
           'LowHoverHeightComp', 'MinPowerSpeedComp', 'CriticalSpeedComp',
           'MultiEngineCriticalSpeedComp', 'HighHoverHeightComp', 'HVBoundaryComp',
           'HeightVelocityGroup',
           'FlarePitchRateComp', 'FlareTimeComp', 'FlareAngleComp', 'FlareConditionsComp',
           'FlareAutorotationGroup', 'TouchdownSpeedComp', 'MinTouchdownSpeedGroup',
           'AutorotationDescentGroup', 'DescentSpeedStencilComp', 'BestAutorotationSpeedGroup',
           'SpeedNodesComp', 'ChainSplitComp', 'ZoomGlideChainGroup',
           'HoverAccelerationComp', 'AvailableTorqueComp', 'SmoothMinComp',
           'MaxAccelerationGroup', 'WeightCoefComp', 'DecelerationForceComp',
           'RotorForceLimitGroup', 'MaxDecelerationGroup']
