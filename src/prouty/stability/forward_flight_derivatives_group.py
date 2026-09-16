"""The complete forward-flight stability derivative set.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", pp. 574-595::

    Table 9.5   chart derivatives, one column per rotor      p.  574
    Table 9.6   basic main rotor derivatives                 pp. 576-577
    Table 9.7   basic tail rotor derivatives                 p.  578
    Table 9.8   main rotor dimensional derivatives           pp. 578-582
    Table 9.9   tail rotor dimensional derivatives           pp. 582-583
    Table 9.10  nondimensional horizontal stabilizer         p.  584
    Table 9.11  horizontal stabilizer dimensional            pp. 585-586
    Table 9.12  nondimensional vertical stabilizer           p.  587
    Table 9.13  vertical stabilizer dimensional              pp. 587-589
    Table 9.14  nondimensional fuselage                      pp. 589-590
    Table 9.15  fuselage dimensional                         pp. 590-591
    Table 9.16  the five summed                              pp. 591-595
"""

from functools import lru_cache

import numpy as np
import openmdao.api as om

from prouty.stability.basic_main_rotor_derivatives_ff_comp import (
    BasicMainRotorDerivativesFFComp,
)
from prouty.stability.basic_tail_rotor_derivatives_ff_comp import (
    BasicTailRotorDerivativesFFComp,
)
from prouty.stability.fuselage_derivatives_ff_comp import (
    FuselageDerivativesFFComp,
)
from prouty.stability.fuselage_nondim_derivatives_comp import (
    FuselageNondimDerivativesComp,
)
from prouty.stability.horiz_stab_derivatives_ff_comp import (
    HorizStabDerivativesFFComp,
)
from prouty.stability.horiz_stab_nondim_derivatives_comp import (
    HorizStabNondimDerivativesComp,
)
from prouty.stability.main_rotor_derivatives_ff_comp import (
    MainRotorDerivativesFFComp,
)
from prouty.stability.rotor_chart_derivatives_comp import (
    RotorChartDerivativesComp,
)
from prouty.stability.tail_rotor_derivatives_ff_comp import (
    TailRotorDerivativesFFComp,
)
from prouty.stability.total_derivatives_ff_comp import (
    CONTRIBUTORS,
    TotalDerivativesFFComp,
)
from prouty.stability.vert_stab_derivatives_ff_comp import (
    VertStabDerivativesFFComp,
)
from prouty.stability.vert_stab_nondim_derivatives_comp import (
    VertStabNondimDerivativesComp,
)

#: Promotion tag for each of Table 9.16's five columns.
TAG = {'main': 'M', 'tail': 'T', 'horiz': 'H', 'vert': 'V', 'fuse': 'F'}
TAGS = tuple(f'_{letter}' for letter in TAG.values())

#: One physical quantity for the whole aircraft, promoted without a tag.
SHARED = ('rho', 'V', 'q', 'A_M', 'Z_M_bar', 'T_T')

#: Names that already say which component they belong to. Everything else
#: takes its subsystem's tag, because the same symbol means different things
#: on different components -- ``sigma`` and ``A_b`` above all.
UNAMBIGUOUS = frozenset({
    'qH_q', 'qV_q', 'A_T', 'D_int', 'detaF_dbeta', 'vH_v1', 'vF_v1',
    'deps_F_dalpha_F', 'alpha_TPP_bar',
    'X_H_bar', 'Z_H_bar', 'alpha_H_bar',
    'X_V_bar', 'Y_V_bar', 'alpha_V_bar',
    'X_F_bar', 'Z_F_bar', 'L_F_bar', 'D_F_bar', 'M_F_bar',
    'df_dalphaF', 'dLq_dalphaF', 'dSFq_dbeta', 'dMq_dalphaF', 'dNq_dbeta',
    'dRq_dbeta',
    'd_alphaH_d_xdot', 'd_alphaH_d_zdot', 'd_alphaH_d_zddot',
    'd_epsMH_d_xdot', 'd_epsMH_d_zdot', 'd_epsFH_d_zdot',
    'd_alphaV_d_xdot', 'd_alphaV_d_ydot',
    'd_etaTV_d_xdot', 'd_etaTV_d_ydot', 'd_etaFV_d_ydot',
    'd_alphaF_d_xdot', 'd_alphaF_d_zdot',
    'd_epsMF_d_xdot', 'd_epsMF_d_zdot',
})

#: Example helicopter at 115 knots, for every input two subsystems share.
#: Sources are named in the validation notes; nothing here was fitted.
EXAMPLE_DEFAULTS = {
    'rho': (0.002378, 'slug/ft**3'), 'V': (194.1, 'ft/s'),
    'q': (44.793, 'lbf/ft**2'), 'A_M': (2827.4, 'ft**2'),
    'Z_M_bar': (-20543.0, 'lbf'), 'T_T': (661.0, 'lbf'),
    # main rotor
    'A_b_M': (240.0, 'ft**2'), 'Omega_R_M': (650.0, 'ft/s'),
    'Omega_M': (650.0 / 30.0, 'rad/s'), 'R_M': (30.0, 'ft'),
    'sigma_M': (0.085, None), 'a_M': (6.0, '1/rad'),
    'e_over_R_M': (0.05, None), 'gamma_M': (8.1, None),
    'CT_sigma_bar_M': (0.086, None), 'CH_sigma_bar_M': (-0.000445, None),
    'CQ_sigma_bar_M': (0.0048004, None),
    'alpha_TPP_bar': (-0.036631, 'rad'),
    'a1s_bar_M': (-0.017398, 'rad'), 'b1s_bar_M': (-0.013609, 'rad'),
    'i_M': (0.0, 'rad'), 'A_1_M': (-0.049737, 'rad'),
    'B_1_M': (0.153608, 'rad'),
    'h_M': (7.5, 'ft'), 'l_M': (-0.5, 'ft'), 'y_M': (0.0, 'ft'),
    # tail rotor
    'A_b_T': (19.397, 'ft**2'), 'Omega_R_T': (650.0, 'ft/s'),
    'sigma_T': (0.14614, None), 'CT_sigma_bar_T': (0.033912, None),
    'a1s_bar_T': (0.065266, 'rad'),
    'h_T': (6.0, 'ft'), 'l_T': (37.0, 'ft'),
    # horizontal stabilizer
    'vH_v1': (1.4946, None), 'deps_F_dalpha_F': (0.23293, None),
    'l_H': (33.12, 'ft'), 'h_H': (-2.968, 'ft'),
    'qH_q': (0.6, None), 'A_H': (18.0, 'ft**2'), 'a_H': (4.0, '1/rad'),
    'A_R_H': (4.5, None), 'delta_H': (0.02, None), 'C_D0_H': (0.0064, None),
    'alpha_LO_H': (0.0, 'rad'), 'i_H': (-0.052, 'rad'),
    'alpha_H_bar': (-0.1401, 'rad'),
    'X_H_bar': (-14.06, 'lbf'), 'Z_H_bar': (271.0, 'lbf'),
    # vertical stabilizer
    'qV_q': (0.6, None), 'A_T': (132.73, 'ft**2'),
    'detaF_dbeta': (0.06, None),
    'A_V': (33.0, 'ft**2'), 'a_V': (3.0, '1/rad'), 'A_R_V': (3.2, None),
    'delta_V': (0.01, None), 'alpha_LO_V': (-0.1012, 'rad'),
    'alpha_V_bar': (0.0066660, 'rad'),
    'Y_V_bar': (287.0, 'lbf'), 'X_V_bar': (-58.0, 'lbf'),
    'D_int': (42.8, 'lbf'), 'h_V': (3.0, 'ft'), 'l_V': (35.0, 'ft'),
    # fuselage
    'vF_v1': (1.0, None),
    'X_F_bar': (-794.0, 'lbf'), 'Z_F_bar': (281.0, 'lbf'),
    'L_F_bar': (-281.0, 'lbf'), 'D_F_bar': (794.0, 'lbf'),
    'M_F_bar': (-11733.0, 'lbf*ft'),
    'df_dalphaF': (-2.0, 'ft**2/rad'), 'dLq_dalphaF': (74.5, 'ft**2/rad'),
    'dSFq_dbeta': (-220.0, 'ft**2/rad'), 'dMq_dalphaF': (1780.0, 'ft**3/rad'),
    'dNq_dbeta': (-820.0, 'ft**3/rad'), 'dRq_dbeta': (230.0, 'ft**3/rad'),
}

#: Which of those are one value rather than one per flight condition.
SCALAR_DEFAULTS = frozenset({
    'A_M', 'A_b_M', 'R_M', 'sigma_M', 'a_M', 'e_over_R_M', 'i_M',
    'h_M', 'l_M', 'y_M', 'A_b_T', 'sigma_T', 'h_T', 'l_T',
    'l_H', 'h_H', 'A_H', 'a_H', 'A_R_H', 'delta_H', 'C_D0_H', 'alpha_LO_H',
    'i_H', 'A_T', 'A_V', 'a_V', 'A_R_V', 'delta_V', 'alpha_LO_V',
    'h_V', 'l_V', 'df_dalphaF', 'dLq_dalphaF', 'dSFq_dbeta', 'dMq_dalphaF',
    'dNq_dbeta', 'dRq_dbeta',
})


#: The five dimensional tables promote their outputs with Table 9.16's own
#: column word rather than the letter tag, so they meet the totalling
#: component's inputs by promotion and need no connect statement.
COLUMN = {'main_table': 'main', 'tail_table': 'tail', 'horiz_table': 'horiz',
          'vert_table': 'vert', 'fuse_table': 'fuse'}

#: The three places an airframe table reads a rotor table. These are the only
#: explicit connections in the group.
CROSS_LINKS = (
    ('dZ_dxdot_main', 'dZ_dxdot_M'),
    ('dZ_dzdot_main', 'dZ_dzdot_M'),
    ('dY_dxdot_tail', 'dY_dxdot_T'),
    ('dY_dydot_tail', 'dY_dydot_T'),
)


def promoted(name, tag):
    """Promoted name of ``name`` inside the subtree tagged ``tag``."""
    if name in SHARED or name in UNAMBIGUOUS or name.endswith(TAGS):
        return name
    return f'{name}_{tag}'


class ForwardFlightDerivativesGroup(om.Group):
    """Tables 9.5 to 9.16, wired.

    Thirteen subsystems::

        main_chart --> main_basic --> main_table ----+--> total
        tail_chart --> tail_basic --> tail_table --+ |
                                                   | |
        main_table --> horiz_nondim --> horiz_table--+
        main_table --> fuse_nondim  --> fuse_table --+
        tail_table --> vert_nondim  --> vert_table --+

    The graph is genuinely coupled, not a parallel stack. The horizontal
    stabilizer and the fuselage both fly in the main rotor's downwash, so
    Table 9.10 and Table 9.14 read ``dZ/dxdot`` and ``dZ/dzdot`` out of
    Table 9.8. The vertical stabilizer flies in the tail rotor's sidewash, so
    Table 9.12 reads ``dY/dxdot`` and ``dY/dydot`` out of Table 9.9. Three of
    the five airframe inputs come from the rotors.

    Naming
    ------
    Every variable takes its subsystem's tag -- ``_M``, ``_T``, ``_H``, ``_V``
    or ``_F`` -- unless it is one physical quantity for the whole aircraft
    (``rho``, ``V``, ``q``, ``A_M``, ``Z_M_bar``, ``T_T``) or already says
    which component it belongs to (``A_H``, ``qV_q``, ``L_F_bar`` and the
    rest of ``UNAMBIGUOUS``).

    The tags are not decoration. ``sigma`` is .085 on the main rotor and .146
    on the tail; ``A_b`` is 240 ft^2 and 19.4; ``A_R``, ``delta`` and
    ``alpha_LO`` are different surfaces on the two stabilizers; and
    ``d_beta_d_ydot`` is computed four separate times, by Tables 9.6, 9.7,
    9.12 and 9.14, which all agree on ``1/V`` but are four outputs all the
    same.

    Outputs are the 50 rows of Table 9.16.

    Options
    -------
    num_nodes : int

    What it reproduces
    ------------------
    Fed the example helicopter at 115 knots -- and every default here is
    either printed in Chapter 9 or taken unchanged from the Chapter 8 work
    already in this repository -- it produces the Table 9.16 totals, and
    through them the ten coefficients of Table 9.20 that the whole
    forward-flight stability analysis of pp. 617-634 is built on.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        nodes = (('num_nodes', nn),)
        stack = (
            ('main_chart', RotorChartDerivativesComp,
             nodes + (('rotor', 'main'),), 'M'),
            ('tail_chart', RotorChartDerivativesComp,
             nodes + (('rotor', 'tail'),), 'T'),
            ('main_basic', BasicMainRotorDerivativesFFComp, nodes, 'M'),
            ('tail_basic', BasicTailRotorDerivativesFFComp, nodes, 'T'),
            ('main_table', MainRotorDerivativesFFComp, nodes, 'M'),
            ('tail_table', TailRotorDerivativesFFComp, nodes, 'T'),
            ('horiz_nondim', HorizStabNondimDerivativesComp, nodes, 'H'),
            ('horiz_table', HorizStabDerivativesFFComp, nodes, 'H'),
            ('vert_nondim', VertStabNondimDerivativesComp, nodes, 'V'),
            ('vert_table', VertStabDerivativesFFComp, nodes, 'V'),
            ('fuse_nondim', FuselageNondimDerivativesComp, nodes, 'F'),
            ('fuse_table', FuselageDerivativesFFComp, nodes, 'F'),
        )
        for name, comp_class, options, tag in stack:
            comp = comp_class(**dict(options))
            inputs, outputs = _variables(comp_class, options)
            column = COLUMN.get(name)
            self.add_subsystem(
                name, comp,
                promotes_inputs=[(v, promoted(v, tag)) for v in inputs],
                promotes_outputs=[(v, f'{v}_{column}' if column
                                   else promoted(v, tag)) for v in outputs])

        self.add_subsystem('total', TotalDerivativesFFComp(num_nodes=nn),
                           promotes=['*'])
        for source, target in CROSS_LINKS:
            self.connect(source, target)

        for name, (value, units) in EXAMPLE_DEFAULTS.items():
            scalar = name in SCALAR_DEFAULTS
            self.set_input_defaults(
                name, val=value if scalar else np.full(nn, value), units=units)


@lru_cache(maxsize=None)
def _variables(comp_class, options):
    """The input and output names a component declares, as two tuples.

    The components declare their variables inside ``setup``, so the only way
    to know the list before wiring is to set one up. Cached on the class and
    its options, so it happens once per configuration rather than once per
    group.
    """
    probe = om.Problem()
    probe.model.add_subsystem('c', comp_class(**dict(options)))
    probe.setup()
    probe.final_setup()

    inputs = tuple(meta['prom_name'].split('.')[-1] for _, meta
                   in probe.model.list_inputs(out_stream=None, prom_name=True,
                                              val=False))
    outputs = tuple(meta['prom_name'].split('.')[-1] for _, meta
                    in probe.model.list_outputs(out_stream=None,
                                                prom_name=True, val=False,
                                                residuals=False))
    return inputs, outputs
