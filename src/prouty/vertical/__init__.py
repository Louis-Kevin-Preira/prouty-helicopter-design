"""Chapter 2 -- Aerodynamics of vertical flight, p. 93-117.

    G0  FlowStatesGroup, induced velocity in axial flight          p. 93-95, 112-113
        DescentRatioComp, AxialInducedVelocityComp
        (ClimbInducedVelocityComp of Chapter 4 for v_hov)
    G2  ClimbPowerGroup, power required in a vertical climb        p. 97-101
        ClimbPowerApproxComp
        (ClimbInducedVelocityComp, VerticalClimbPowerComp, TailRotorThrustComp of Chapter 4)
    G3  ClimbCollectiveComp, collective pitch required              p. 101
    G4  ThrustDampingGroup, rotor thrust damping                    p. 101-102
        AxialThrustIdealTwistComp, ThrustDampingComp
    G5  VortexRingBoundariesComp, vortex ring state boundaries      p. 95, 102-107
    G6  TailRotorVortexRingGroup, tail rotor and vortex ring state  p. 107-109
        TailRotorAxialVelocityComp (+ VortexRingBoundariesComp of G5)
    G7  VerticalAutorotationGroup, rate of descent in autorotation  p. 109-115
        MeanAngleOfAttackComp, AutorotationParameterComp, AutorotationDescentComp,
        AutorotationCollectiveComp, ParachuteAnalogyComp
        (Chapter 6 NACA 0012 drag, Chapter 3 ThrustCoefComp, Chapter 4 v_hov)
    G1  conditions at the blade element                             p. 95-97
        HoverRotorGroup(flight='climb') of Chapter 1, with ClimbInflowRatioComp
        (in prouty.hover, to keep hover free of any import from this package)
    G8  TailRotorDriveTorqueComp, effect of rapid pitch changes     p. 115-116
"""

from prouty.vertical.descent_ratio_comp import DescentRatioComp
from prouty.vertical.axial_induced_velocity_comp import (FIGURE_2_13, AxialInducedVelocityComp,
                                                         axial_inflow)
from prouty.vertical.flow_states_group import FlowStatesGroup
from prouty.vertical.climb_power_approx_comp import ClimbPowerApproxComp
from prouty.vertical.climb_power_group import ClimbPowerGroup
from prouty.vertical.climb_collective_comp import ClimbCollectiveComp
from prouty.vertical.axial_thrust_ideal_twist_comp import AxialThrustIdealTwistComp
from prouty.vertical.thrust_damping_comp import ThrustDampingComp
from prouty.vertical.thrust_damping_group import ThrustDampingGroup
from prouty.vertical.mean_angle_of_attack_comp import MeanAngleOfAttackComp
from prouty.vertical.autorotation_parameter_comp import AutorotationParameterComp
from prouty.vertical.autorotation_descent_comp import AutorotationDescentComp, solve_descent
from prouty.vertical.autorotation_collective_comp import AutorotationCollectiveComp
from prouty.vertical.parachute_analogy_comp import ParachuteAnalogyComp
from prouty.vertical.vertical_autorotation_group import VerticalAutorotationGroup
from prouty.vertical.vortex_ring_boundaries_comp import VortexRingBoundariesComp
from prouty.vertical.tail_rotor_axial_velocity_comp import TailRotorAxialVelocityComp
from prouty.vertical.tail_rotor_vortex_ring_group import TailRotorVortexRingGroup
from prouty.vertical.tail_rotor_drive_torque_comp import TailRotorDriveTorqueComp

__all__ = ['DescentRatioComp', 'AxialInducedVelocityComp', 'axial_inflow', 'FIGURE_2_13',
           'FlowStatesGroup', 'ClimbPowerApproxComp', 'ClimbPowerGroup', 'ClimbCollectiveComp',
           'AxialThrustIdealTwistComp', 'ThrustDampingComp', 'ThrustDampingGroup',
           'MeanAngleOfAttackComp', 'AutorotationParameterComp', 'AutorotationDescentComp',
           'solve_descent', 'AutorotationCollectiveComp', 'ParachuteAnalogyComp',
           'VerticalAutorotationGroup', 'VortexRingBoundariesComp',
           'TailRotorAxialVelocityComp', 'TailRotorVortexRingGroup', 'TailRotorDriveTorqueComp']
