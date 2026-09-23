"""Chapter 7 -- Rotor flapping characteristics, p. 443-480.

The five analytical sections of the chapter, as five groups:

    HoverFlappingGroup          p. 455-462   frequency ratio, damping, phase
                                             lag, cross-coupling, time constant
    ForwardFlightFlappingGroup  p. 463-469   coning and first harmonic flapping
    RateFlappingGroup           p. 469-476   flapping from pitch and roll rates,
                                             and the cyclic it costs in a turn
    FlappingMomentsGroup        p. 476-477   rotor stiffness, hub couple, and
                                             the moment about the c.g.
    HForceFlappingGroup         p. 478-479   how inflow cuts the in-plane force
                                             produced by flapping

Chapter7FlappingGroup assembles all five. Everything is feed-forward: the
induced velocity loop of p. 467-468 is linear and closed algebraically, and
the flapping system is linear and solved explicitly, so the chapter needs no
solver anywhere.

Four of the printed expressions with nested fractions -- the steady flapping
of p. 468-469 and the rate flapping of p. 473 -- turn out to be the solution
of one 2x2 system with different right hand sides. That system lives in
flapping_2x2 and is shared.

Eleven departures from the printed text are recorded in
docs/validation_flapping.md: eight are errors or approximations worth knowing
about, three are apparent inconsistencies that turned out to be sound and are
written down so they are not re-opened.
"""

from prouty.flapping.chapter7_flapping_group import Chapter7FlappingGroup
from prouty.flapping.hover_flapping_group import HoverFlappingGroup
from prouty.flapping.forward_flight_flapping_group import \
    ForwardFlightFlappingGroup
from prouty.flapping.rate_flapping_group import RateFlappingGroup
from prouty.flapping.flapping_moments_group import FlappingMomentsGroup
from prouty.flapping.h_force_flapping_group import HForceFlappingGroup

from prouty.flapping.blade_inertia_comp import BladeInertiaComp
from prouty.flapping.lock_number_comp import LockNumberComp
from prouty.flapping.flap_frequency_comp import FlapFrequencyComp
from prouty.flapping.flap_damping_comp import FlapDampingComp
from prouty.flapping.phase_angle_comp import PhaseAngleComp
from prouty.flapping.accel_coupling_comp import AccelCouplingComp
from prouty.flapping.blade_time_constant_comp import BladeTimeConstantComp

from prouty.flapping.thrust_inflow_comp import ThrustInflowComp
from prouty.flapping.flapping_matrix_comp import FlappingMatrixComp
from prouty.flapping.flapping_solve_comp import FlappingSolveComp
from prouty.flapping.closed_form_flapping_comp import ClosedFormFlappingComp

from prouty.flapping.rate_flapping_comp import RateFlappingComp
from prouty.flapping.flapping_superposition_comp import \
    FlappingSuperpositionComp
from prouty.flapping.maneuver_rate_comp import ManeuverRateComp
from prouty.flapping.lateral_cyclic_turn_comp import LateralCyclicTurnComp
from prouty.flapping.longitudinal_cyclic_turn_comp import \
    LongitudinalCyclicTurnComp

from prouty.flapping.rotor_stiffness_comp import RotorStiffnessComp
from prouty.flapping.hub_moment_comp import HubMomentComp
from prouty.flapping.cg_moment_comp import CGMomentComp

from prouty.flapping.h_force_flapping_deriv_comp import HForceFlappingDerivComp
from prouty.flapping.flapping_moment_buildup_comp import \
    FlappingMomentBuildupComp
