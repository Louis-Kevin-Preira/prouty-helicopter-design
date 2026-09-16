"""Chapter 7 assembled: rotor flapping characteristics.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 455-479.
"""

import openmdao.api as om

from prouty.flapping.flapping_moments_group import FlappingMomentsGroup
from prouty.flapping.forward_flight_flapping_group import \
    ForwardFlightFlappingGroup
from prouty.flapping.h_force_flapping_group import HForceFlappingGroup
from prouty.flapping.hover_flapping_group import HoverFlappingGroup
from prouty.flapping.rate_flapping_group import RateFlappingGroup


class Chapter7FlappingGroup(om.Group):
    """The five sections of Chapter 7 wired together.

    ============================== ========= ==============================
    Group                          Pages     Section
    ============================== ========= ==============================
    HoverFlappingGroup             455-462   Flapping in hover with offset
    ForwardFlightFlappingGroup     463-469   Flapping in forward flight
    RateFlappingGroup              469-476   Flapping from pitch and roll
    FlappingMomentsGroup           476-477   Moments produced by flapping
    HForceFlappingGroup            478-479   H-force due to flapping
    ============================== ========= ==============================

    Who owns what
    -------------
    Three of the subgroups can build the blade inertia themselves, which
    would collide on the promoted names. Here ``HoverFlappingGroup`` owns it:
    it produces ``I_b``, ``M_b/g``, ``M_b``, ``e`` and, through the Lock
    number, ``gamma``. The other groups take them as inputs. That is why the
    hover section is always present, even if only forward flight is wanted.

    Which flapping the moments see
    ------------------------------
    When the rate section is included, the moments and the H-force are built
    from ``a_1s_total`` and ``b_1s_total`` rather than the steady values. The
    two coincide when the rates are zero, so this changes nothing in steady
    flight and is the physically consistent choice in a manoeuvre. Prouty
    never combines them, so it is an extension, not a printed result.

    Options
    -------
    Reaching HoverFlappingGroup: ``blade_input``, ``hinge_input``.
    Reaching ForwardFlightFlappingGroup: ``method``, ``inflow``,
    ``include_weight``.
    Reaching RateFlappingGroup: ``rate_source``, ``maneuver``.
    Reaching FlappingMomentsGroup: ``convention``, ``inplane_force``.
    Reaching HForceFlappingGroup: ``form``.
    Selecting sections: ``include_rates``, ``include_moments``,
    ``include_h_force``.

    For finer control, or for the inverse uses (sizing a hingeless rotor from
    its hub stiffness, comparing the numerical and closed-form flapping side
    by side), assemble the subgroups directly rather than through this one.

    Notes
    -----
    Feed-forward throughout, no solver anywhere in the chapter.

    Solidity is a plain input. ``sigma = b c / (pi R)`` would tie it to the
    geometry, but blade area conventions vary and Chapter 7 never defines it,
    so it is left to the caller.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

        self.options.declare('blade_input', values=('m', 'I_b'), default='I_b')
        self.options.declare('hinge_input',
                             values=('e_over_R', 'omega_n_ratio'),
                             default='e_over_R')

        self.options.declare('method', values=('numerical', 'closed_form'),
                             default='numerical')
        self.options.declare('inflow', values=('internal', 'external'),
                             default='internal')
        self.options.declare('include_weight', types=bool, default=True)

        self.options.declare('rate_source', values=('maneuver', 'external'),
                             default='maneuver')
        self.options.declare('maneuver', values=('level_turn', 'pull_up'),
                             default='level_turn')

        self.options.declare('convention', values=('hinge', 'center'),
                             default='hinge')
        self.options.declare('inplane_force', values=('simple', 'external'),
                             default='simple')
        self.options.declare('form', values=('lambda', 'theta75'),
                             default='lambda')

        self.options.declare('include_rates', types=bool, default=True)
        self.options.declare('include_moments', types=bool, default=True)
        self.options.declare('include_h_force', types=bool, default=True)

    def setup(self):
        nn = self.options['num_nodes']
        opts = self.options
        rates = opts['include_rates']

        # Owns the blade properties and the Lock number
        self.add_subsystem(
            'hover',
            HoverFlappingGroup(num_nodes=nn, blade_input=opts['blade_input'],
                               hinge_input=opts['hinge_input']),
            promotes=['*'])

        self.add_subsystem(
            'forward_flight',
            ForwardFlightFlappingGroup(
                num_nodes=nn, method=opts['method'], inflow=opts['inflow'],
                include_weight=opts['include_weight'], blade_input='external'),
            promotes=['*'])

        if rates:
            self.add_subsystem(
                'rate',
                RateFlappingGroup(num_nodes=nn,
                                  rate_source=opts['rate_source'],
                                  maneuver=opts['maneuver']),
                promotes=['*'])

        # Downstream sections read the total flapping when rates are modelled.
        # The H-force group only takes a_1s: b_1s reaches it as an output,
        # through d(C_Y/sigma)/db_1s.
        both = ([('a_1s', 'a_1s_total'), ('b_1s', 'b_1s_total'), '*'] if rates
                else ['*'])
        longitudinal = [('a_1s', 'a_1s_total'), '*'] if rates else ['*']

        if opts['include_moments']:
            self.add_subsystem(
                'moments',
                FlappingMomentsGroup(num_nodes=nn, blade_input='external',
                                     convention=opts['convention'],
                                     inplane_force=opts['inplane_force']),
                promotes_inputs=both, promotes_outputs=['*'])

        if opts['include_h_force']:
            self.add_subsystem(
                'h_force',
                HForceFlappingGroup(
                    num_nodes=nn, form=opts['form'],
                    include_buildup=opts['include_moments']),
                promotes_inputs=longitudinal, promotes_outputs=['*'])
