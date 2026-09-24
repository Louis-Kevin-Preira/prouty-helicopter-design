"""Chapter 5 -- Special performance problems, p. 339-377.

    G1  TurnsPullupsGroup, turns and pullups                        p. 340-346
        LoadFactorComp, TurnKinematicsComp, TurnCyclicReliefComp,
        TurnEnergyPowerComp, ThrustCapabilityComp (Figure 5.2)
        SteadyTurnPowerGroup: EffectiveWeightComp
        (+ ForwardFlightPowerGroup of Chapter 4)
"""

from prouty.special_performance.load_factor_comp import LoadFactorComp
from prouty.special_performance.turn_kinematics_comp import TurnKinematicsComp
from prouty.special_performance.turn_cyclic_relief_comp import TurnCyclicReliefComp
from prouty.special_performance.turn_energy_power_comp import TurnEnergyPowerComp
from prouty.special_performance.thrust_capability_comp import FIG_5_2, ThrustCapabilityComp
from prouty.special_performance.turns_pullups_group import TurnsPullupsGroup
from prouty.special_performance.effective_weight_comp import EffectiveWeightComp
from prouty.special_performance.steady_turn_power_group import SteadyTurnPowerGroup

__all__ = ['LoadFactorComp', 'TurnKinematicsComp', 'TurnCyclicReliefComp',
           'TurnEnergyPowerComp', 'FIG_5_2', 'ThrustCapabilityComp', 'TurnsPullupsGroup',
           'EffectiveWeightComp', 'SteadyTurnPowerGroup']
