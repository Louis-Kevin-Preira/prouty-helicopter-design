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

__all__ = ['LoadFactorComp', 'TurnKinematicsComp', 'TurnCyclicReliefComp',
           'TurnEnergyPowerComp', 'FIG_5_2', 'ThrustCapabilityComp', 'TurnsPullupsGroup',
           'EffectiveWeightComp', 'SteadyTurnPowerGroup',
           'DriveInertiaComp', 'KineticEnergyTimeComp', 'RotorSpeedDecayComp',
           'RotorSpeedDecayGroup', 'EquivalentHoverTimeComp', 'AutorotativeIndexComp',
           'AutorotativeIndicesGroup', 'TowlineTensionComp', 'TowingGroup']
