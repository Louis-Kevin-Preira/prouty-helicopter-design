"""Chapter 1 -- Aerodynamics of hovering flight, p. 1-80.

Combined momentum and blade element theory with empirical corrections, the
21 step procedure of p. 69-72, as six groups:

    RotorPreprocessGroup        steps 1 to 3     geometry, atmosphere, grid
    BladeElementGroup           steps 4 to 7     inflow coupled to the airfoil
    ThrustGroup                 steps 8 to 11    loading, tip loss, thrust
    TorqueGroup                 steps 12 to 15   profile and induced torque
    EmpiricalCorrectionsGroup   steps 16 to 20   wake charts, total torque
    RotorPerformanceGroup       step 21          thrust, power, figure of merit

HoverRotorGroup assembles all six and offers an 'analysis' mode, where the
collective is given, and a 'trim' mode, where it is solved for a target thrust.
"""

from prouty.hover.hover_rotor_group import HoverRotorGroup
from prouty.hover.rotor_preprocess_group import RotorPreprocessGroup
from prouty.hover.inflow_group import InflowGroup
from prouty.hover.blade_element_group import BladeElementGroup
from prouty.hover.thrust_group import ThrustGroup
from prouty.hover.torque_group import TorqueGroup
from prouty.hover.empirical_corrections_group import EmpiricalCorrectionsGroup
from prouty.hover.rotor_performance_group import RotorPerformanceGroup

from prouty.hover.rotor_geometry_comp import RotorGeometryComp
from prouty.hover.atmosphere_comp import AtmosphereComp
from prouty.hover.blade_grid_comp import BladeGridComp
from prouty.hover.chord_dist_comp import ChordDistComp
from prouty.hover.twist_dist_comp import TwistDistComp
from prouty.hover.local_mach_comp import LocalMachComp
from prouty.hover.pitch_comp import PitchComp
from prouty.hover.inflow_ratio_comp import InflowRatioComp
from prouty.hover.angle_of_attack_comp import AngleOfAttackComp
from prouty.hover.thrust_loading_comp import ThrustLoadingComp
from prouty.hover.integration_weights_comp import IntegrationWeightsComp
from prouty.hover.tip_loss_comp import TipLossComp
from prouty.hover.integral_comp import IntegralComp
from prouty.hover.profile_torque_loading_comp import ProfileTorqueLoadingComp
from prouty.hover.induced_torque_loading_comp import InducedTorqueLoadingComp
from prouty.hover.wake_rotation_comp import WakeRotationComp
from prouty.hover.disc_loading_comp import DiscLoadingComp
from prouty.hover.thrust_solidity_comp import ThrustSolidityComp
from prouty.hover.wake_contraction_comp import WakeContractionComp
from prouty.hover.total_torque_comp import TotalTorqueComp
from prouty.hover.dimensional_perf_comp import DimensionalPerfComp
from prouty.hover.figure_of_merit_comp import FigureOfMeritComp

__all__ = [
    "HoverRotorGroup",
    "RotorPreprocessGroup", "InflowGroup", "BladeElementGroup",
    "ThrustGroup", "TorqueGroup", "EmpiricalCorrectionsGroup",
    "RotorPerformanceGroup",
    "RotorGeometryComp", "AtmosphereComp", "BladeGridComp", "ChordDistComp",
    "TwistDistComp", "LocalMachComp", "PitchComp", "InflowRatioComp",
    "AngleOfAttackComp", "ThrustLoadingComp", "IntegrationWeightsComp",
    "TipLossComp", "IntegralComp", "ProfileTorqueLoadingComp",
    "InducedTorqueLoadingComp", "WakeRotationComp", "DiscLoadingComp",
    "ThrustSolidityComp", "WakeContractionComp", "TotalTorqueComp",
    "DimensionalPerfComp", "FigureOfMeritComp",
]
