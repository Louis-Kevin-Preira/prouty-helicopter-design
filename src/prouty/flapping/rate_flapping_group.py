"""Flapping due to pitch and roll velocities, and the cyclic it costs.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 469-476.
"""

import openmdao.api as om

from prouty.flapping.flapping_superposition_comp import FlappingSuperpositionComp
from prouty.flapping.lateral_cyclic_turn_comp import LateralCyclicTurnComp
from prouty.flapping.longitudinal_cyclic_turn_comp import \
    LongitudinalCyclicTurnComp
from prouty.flapping.maneuver_rate_comp import ManeuverRateComp
from prouty.flapping.rate_flapping_comp import RateFlappingComp


class RateFlappingGroup(om.Group):
    """Flapping produced by angular rates, and the cyclic needed to trim it.

    ========================== ========= ====================================
    Component                  Pages     Produces
    ========================== ========= ====================================
    ManeuverRateComp           474       q
    RateFlappingComp           469-473   a_1s_rate, b_1s_rate
    FlappingSuperpositionComp  473       a_1s_total, b_1s_total
    LateralCyclicTurnComp      474-475   delta_A_1, A_1_maneuver
    LongitudinalCyclicTurnComp 475       delta_B_1, da1s_dB1, db1s_dB1
    ========================== ========= ====================================

    Options
    -------
    num_nodes : int
    rate_source : {'maneuver', 'external'}
        ``'maneuver'`` builds ``q`` from airspeed and load factor (p. 474).
        ``'external'`` takes ``p`` and ``q`` as group inputs, for a general
        manoeuvre or for stability derivative work.
    maneuver : {'level_turn', 'pull_up'}
        Passed to ``ManeuverRateComp``.
    coning_source : {'load_factor', 'direct'}
        Passed to ``LateralCyclicTurnComp``.
    exact_denominator : bool
        Passed to ``RateFlappingComp`` and ``LongitudinalCyclicTurnComp``.

    Coupling with steady flight
    ---------------------------
    ``a_1s`` and ``b_1s`` are group inputs, meant to come from
    ``ForwardFlightFlappingGroup``. Left unconnected they default to zero and
    the totals are just the rate contributions. The addition is exact, not an
    approximation: both problems reduce to the same 2x2 system, so their
    right-hand sides add.

    What this group does not model
    ------------------------------
    Two further sources of cross-coupling in a pitching manoeuvre are named
    on p. 474 but not derived. The change in coning as thrust rises is
    partly captured, through ``coning_source``. The sideslip that builds up
    with pedals fixed during a pull-up, which produces the rotor dihedral
    effect, is not: it belongs to the trim problem, not to flapping.

    Notes
    -----
    Feed-forward, no solver. Two corrections to the printed text are applied,
    C7-4 and C7-10 of ``docs/validation_flapping.md``.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('rate_source', values=('maneuver', 'external'),
                             default='maneuver')
        self.options.declare('maneuver', values=('level_turn', 'pull_up'),
                             default='level_turn')
        self.options.declare('coning_source',
                             values=('load_factor', 'direct'),
                             default='load_factor')
        self.options.declare('exact_denominator', types=bool, default=True)

    def setup(self):
        nn = self.options['num_nodes']
        exact = self.options['exact_denominator']

        if self.options['rate_source'] == 'maneuver':
            self.add_subsystem(
                'maneuver_rate',
                ManeuverRateComp(num_nodes=nn,
                                 maneuver=self.options['maneuver']),
                promotes=['*'])

        self.add_subsystem('rate_flapping',
                           RateFlappingComp(num_nodes=nn,
                                            exact_denominator=exact),
                           promotes=['*'])
        self.add_subsystem('superposition',
                           FlappingSuperpositionComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem(
            'lateral_cyclic',
            LateralCyclicTurnComp(num_nodes=nn,
                                  coning_source=self.options['coning_source']),
            promotes=['*'])
        self.add_subsystem('longitudinal_cyclic',
                           LongitudinalCyclicTurnComp(num_nodes=nn,
                                                      exact_denominator=exact),
                           promotes=['*'])
