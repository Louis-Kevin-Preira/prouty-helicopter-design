"""The complete hover stability derivative set.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", pp. 563-573::

    Table 9.1  basic rotor derivatives, one instance per rotor   pp. 564-565
    Table 9.2  main rotor dimensional derivatives                pp. 566-569
    Table 9.3  tail rotor dimensional derivatives                pp. 569-570
    Table 9.4  the two summed                                    pp. 571-573

p. 563 states the scope: in hover only the aerodynamics of the two rotors
matter, so no airframe contribution enters until forward flight.
"""

import numpy as np
import openmdao.api as om

from prouty.stability.hover_chart_slopes_comp import HoverChartSlopesComp
from prouty.stability.basic_rotor_derivatives_hover_comp import (
    DEPENDENCIES,
    MIRRORS,
    BasicRotorDerivativesHoverComp,
)
from prouty.stability.main_rotor_derivatives_hover_comp import (
    ROWS as MAIN_ROWS,
)
from prouty.stability.monomial_rows_comp import factors_of
from prouty.stability.tail_rotor_derivatives_hover_comp import (
    TailRotorDerivativesHoverComp,
    build_rows,
)
from prouty.stability.main_rotor_derivatives_hover_comp import (
    MainRotorDerivativesHoverComp,
)
from prouty.stability.total_derivatives_hover_comp import (
    TotalDerivativesHoverComp,
)

#: Inputs that are the same physical quantity for both rotors.
SHARED = ('rho',)

#: Inputs that already say which rotor they belong to, or that only one rotor
#: has, so they take no suffix.
UNSUFFIXED = ('h_M', 'l_M', 'y_M', 'i_M', 'h_T', 'l_T',
              'a1s_bar', 'b1s_bar', 'CQ_sigma_bar')

#: Everything Table 9.1 produces.
BASIC_OUTPUTS = tuple(DEPENDENCIES) + tuple(MIRRORS)

#: Everything Table 9.1 consumes.
BASIC_INPUTS = ('e_over_R', 'a', 'sigma', 'R', 'A_b', 'rho', 'Omega',
                'Omega_R', 'gamma', 'CT_sigma', 'theta_0', 'theta_1',
                'v_1_over_Omega_R', 'a_0')

#: Inputs that reach both a rotor's Table 9.1 component and its dimensional
#: table, so the two declared defaults have to be reconciled once. Values are
#: the example helicopter's; the tail rotor pair is what Table 9.3 implies.
GEOMETRY_DEFAULTS = {
    'A_b_M': (240.0, 'ft**2'), 'Omega_R_M': (650.0, 'ft/s'),
    'Omega_M': (650.0 / 30.0, 'rad/s'), 'R_M': (30.0, 'ft'),
    'A_b_T': (19.397, 'ft**2'), 'Omega_R_T': (650.0, 'ft/s'),
    'R_T': (6.5, 'ft'),
}

#: The two entries p. 564 sends to the Chapter 1 hover charts.
CHART_ENTRIES = ('dCT_sigma_dtheta0', 'dCQ_sigma_dtheta0')

#: Table 9.1 thrust damping of the example helicopter, p. 564: the defaults
#: of the two inputs promoted when thrust_damping='external'.
DAMPING_DEFAULTS = {'dCT_sigma_dlambda_M': 0.49, 'dCT_sigma_dlambda_T': 0.44}

#: Inputs HoverChartSlopesComp shares with the Table 9.1 component.
SLOPE_INPUTS = ('CT_sigma', 'sigma', 'a')

#: Table 9.1 outputs the group exposes on top of the Table 9.4 rows, because
#: the Hohenemser period of p. 600 is written in terms of them.
EXPOSED_BASIC_OUTPUTS = ('d_a1s_d_mu', 'd_a1s_dq')

#: Table 9.1 values for the example helicopter's two chart entries, p. 564.
CHART_DEFAULTS = {
    'dCT_sigma_dtheta0_M': 0.61, 'dCQ_sigma_dtheta0_M': 0.078,
    'dCT_sigma_dtheta0_T': 0.50, 'dCQ_sigma_dtheta0_T': 0.089,
}


def suffixed(name, suffix):
    """Promoted name of ``name`` inside the subtree for one rotor."""
    if name in SHARED or name in UNSUFFIXED:
        return name
    return f'{name}{suffix}'


class HoverDerivativesGroup(om.Group):
    """Table 9.1 twice, then Tables 9.2 and 9.3, then Table 9.4.

    Five subsystems::

        basic_main --+--> main_table --+
                     |                 +--> total
        basic_tail --+--> tail_table --+

    Both ``basic_*`` are the same component, since Table 9.1 prints one
    equation column for two rotors. Which of its outputs each dimensional
    table consumes is what differs, and the connection lists are derived from
    the components' own declarations rather than retyped, so a row added to
    Table 9.2 wires itself.

    Naming
    ------
    Every input of a rotor's subtree is promoted with an ``_M`` or ``_T``
    suffix -- ``A_b_M``, ``gamma_T``, ``CT_sigma_M`` -- following the
    ``T_M`` / ``T_T`` convention of the trim package. Three exceptions: ``rho``
    is shared and promoted plain; the offsets ``h_M``, ``l_M``, ``y_M``,
    ``i_M``, ``h_T``, ``l_T`` already carry their rotor; and the trim values
    ``a1s_bar``, ``b1s_bar`` and ``CQ_sigma_bar`` are main rotor only and have
    no tail rotor counterpart in Table 9.2.

    Outputs are the 44 rows of Table 9.4, promoted from ``total``, plus
    ``d_a1s_d_mu_M``, ``d_a1s_dq_M`` and their tail rotor counterparts from
    Table 9.1, which ``HoverStabilityGroup`` needs for the Hohenemser period
    of p. 600.

    The two chart entries
    ---------------------
    p. 564 sends ``dCT/sigma/dtheta0`` and ``dCQ/sigma/dtheta0`` to the hover
    charts of Chapter 1 rather than giving an equation for them. They are
    ordinary inputs of this group -- ``dCT_sigma_dtheta0_M`` and the other
    three -- with the Table 9.1 values as defaults, so the group runs
    standalone and reproduces the book. Deriving them from the Chapter 1 model
    instead is a separate question; see the note in
    ``docs/validation_stability.md``.

    Options
    -------
    num_nodes : int
    blade_closest : {'up', 'down'}
        Passed to ``TailRotorDerivativesHoverComp``; see C9-9.
    rate_flapping : {'table', 'exact'}
        Passed to both ``BasicRotorDerivativesHoverComp``; see C9-2.
    derivative_source : {'table', 'model'}
        Where the two chart entries of p. 564 come from. ``'table'``, the
        default, makes them inputs with the Table 9.1 values, so the group
        reproduces the book. ``'model'`` inserts a ``HoverChartSlopesComp``
        per rotor, deriving them in closed form from Chapter 1's combined
        momentum and blade element theory so that an optimiser sees them move
        with the design. That costs accuracy against the printed table: 7 %
        high on thrust and 18 % low on torque. See that component's docstring
        for the numbers.
    thrust_damping : {'table', 'external'}
        Passed to both ``BasicRotorDerivativesHoverComp``. ``'external'``
        promotes their ``dCT_sigma_dlambda_ext`` as ``dCT_sigma_dlambda_M`` and
        ``dCT_sigma_dlambda_T``, to be fed by the Chapter 2 thrust damping
        (``prouty.vertical.ThrustDampingComp``); see C2-7.

    Two rows differ from the book
    -----------------------------
    ``dR/dydot`` and ``dM/dtheta0_T``, both from the tail rotor table
    (C9-8, C9-9). Nothing in this group introduces a third.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('blade_closest', values=('up', 'down'),
                             default='up')
        self.options.declare('rate_flapping', values=('table', 'exact'),
                             default='table')
        self.options.declare('derivative_source', values=('table', 'model'),
                             default='table')
        self.options.declare('thrust_damping', values=('table', 'external'),
                             default='table')

    def setup(self):
        nn = self.options['num_nodes']
        basic = dict(num_nodes=nn, rate_flapping=self.options['rate_flapping'],
                     thrust_damping=self.options['thrust_damping'])

        main_inputs = factors_of([t for terms in MAIN_ROWS.values()
                                  for t in terms])
        tail_rows = build_rows(1.0)
        tail_inputs = factors_of([t for terms in tail_rows.values()
                                  for t in terms])

        self._add_rotor('main', BasicRotorDerivativesHoverComp(**basic),
                        MainRotorDerivativesHoverComp(num_nodes=nn),
                        main_inputs, '_M')
        self._add_rotor('tail', BasicRotorDerivativesHoverComp(**basic),
                        TailRotorDerivativesHoverComp(
                            num_nodes=nn,
                            blade_closest=self.options['blade_closest']),
                        tail_inputs, '_T')

        self.add_subsystem('total', TotalDerivativesHoverComp(num_nodes=nn),
                           promotes_outputs=['*'])
        for rotor, suffix in (('main', 'main'), ('tail', 'tail')):
            rows = MAIN_ROWS if rotor == 'main' else tail_rows
            for name in rows:
                self.connect(f'{rotor}_table.{name}', f'total.{name}_{suffix}')

        self.set_input_defaults('rho', val=np.full(nn, 0.002378),
                                units='slug/ft**3')
        if self.options['derivative_source'] == 'table':
            for name, value in CHART_DEFAULTS.items():
                self.set_input_defaults(name, val=np.full(nn, value))
        else:
            # CT_sigma, sigma and a now reach two components each
            for suffix in ('_M', '_T'):
                self.set_input_defaults(f'CT_sigma{suffix}',
                                        val=np.full(nn, 0.085))
                self.set_input_defaults(f'sigma{suffix}', val=0.085)
                self.set_input_defaults(f'a{suffix}', val=6.0, units='1/rad')
        if self.options['thrust_damping'] == 'external':
            for name, value in DAMPING_DEFAULTS.items():
                self.set_input_defaults(name, val=np.full(nn, value))
        for name, (value, units) in GEOMETRY_DEFAULTS.items():
            scalar = name.startswith(('A_b', 'R_'))
            self.set_input_defaults(
                name, val=value if scalar else np.full(nn, value), units=units)

    def _add_rotor(self, rotor, basic_comp, table_comp, table_inputs, suffix):
        """Table 9.1 for one rotor, feeding its dimensional table."""
        modelled = (CHART_ENTRIES
                    if self.options['derivative_source'] == 'model' else ())
        # the exposed Table 9.1 outputs are promoted rather than connected, so
        # the dimensional table picks them up under the same promoted name
        from_upstream = [name for name in table_inputs
                         if (name in BASIC_OUTPUTS or name in modelled)
                         and name not in EXPOSED_BASIC_OUTPUTS]
        promoted = [name for name in table_inputs if name not in from_upstream]

        basic_inputs = BASIC_INPUTS
        if self.options['thrust_damping'] == 'external':
            basic_inputs += ('dCT_sigma_dlambda_ext',)
        self.add_subsystem(
            f'{rotor}_basic', basic_comp,
            promotes_inputs=[(name, suffixed(name, suffix))
                             if name != 'dCT_sigma_dlambda_ext'
                             else (name, f'dCT_sigma_dlambda{suffix}')
                             for name in basic_inputs],
            promotes_outputs=[(name, f'{name}{suffix}')
                              for name in EXPOSED_BASIC_OUTPUTS])

        if modelled:
            self.add_subsystem(
                f'{rotor}_slopes',
                HoverChartSlopesComp(num_nodes=self.options['num_nodes']),
                promotes_inputs=[(name, suffixed(name, suffix))
                                 for name in SLOPE_INPUTS + ('k_induced',
                                                             'dcd_dalpha')])
            self.connect(f'{rotor}_basic.dCT_sigma_dlambda',
                         f'{rotor}_slopes.dCT_sigma_dlambda')

        self.add_subsystem(
            f'{rotor}_table', table_comp,
            promotes_inputs=[(name, suffixed(name, suffix))
                             for name in promoted])

        for name in from_upstream:
            source = f'{rotor}_slopes' if name in modelled else f'{rotor}_basic'
            self.connect(f'{source}.{name}', f'{rotor}_table.{name}')
