"""Control response in hover, wired.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", pp. 598-613: the
two-degree-of-freedom system of p. 598, the transfer function of p. 606, and
the handling qualities requirements of Table 9.18 and Figures 9.13 and 9.14.
"""

import numpy as np
import openmdao.api as om

from prouty.stability.control_response_hover import HoverControlColumnComp
from prouty.stability.hover_long_two_dof_matrix_comp import (
    HoverLongTwoDofMatrixComp,
)
from prouty.stability.mil_response_requirements_comp import (
    MilResponseRequirementsComp,
)
from prouty.stability.transfer_function_group import TransferFunctionGroup

#: Degrees of longitudinal cyclic per inch of stick. Chapter 9 never prints
#: the control linkage gearing; 2.71 deg/in is what puts the example
#: helicopter on Figure 9.13's pitch abscissa at 0.32. See the class
#: docstring -- this is the one figure-derived number in the chain.
DEG_PER_INCH = 2.71

EXAMPLE_DEFAULTS = {
    'dX_dxdot': (-5.0, 'lbf*s/ft'), 'dX_dq': (1008.0, 'lbf*s/rad'),
    'dM_dxdot': (143.0, 'lbf*s'), 'dM_dq': (-28659.0, 'lbf*ft*s/rad'),
    'dX_dB1': (9516.0, 'lbf/rad'), 'dM_dB1': (-270578.0, 'lbf*ft/rad'),
    'G_W': (20000.0, 'lbf'), 'I_yy': (40000.0, 'slug*ft**2'),
    'g': (32.2, 'ft/s**2'),
    'deg_per_inch': (DEG_PER_INCH, 'deg'),
}


class ControlResponseHoverGroup(om.Group):
    """The longitudinal control response and what it is judged against.

    Four subsystems and two conversions::

        matrix  --> transfer_function --> numerator_coeffs
        column  -->                       denominator_coeffs
        damping --+
        power   --+-> requirements    --> Table 9.18 and Figures 9.13, 9.14

    ``matrix`` is the two-degree-of-freedom system of p. 598, ``column`` the
    right-hand side of p. 606, and ``transfer_function`` is the
    ``TransferFunctionGroup`` written for G0. ``requirements`` is Table 9.18.

    Two sign conversions
    --------------------
    Table 9.18 and Figure 9.13 are written in magnitudes -- damping in
    ft lb/rad/sec and control power in ft lb/in, both positive -- while the
    derivatives that feed them are negative: ``dM/dq = -28,659`` and
    ``dM/dB1 = -270,578``. Two ``ExecComp`` lines do the conversion rather
    than hiding a sign inside a component, so the negation is visible in the
    model and appears in ``list_inputs``.

    The gearing is the one figure-derived number
    --------------------------------------------
    Control power per inch needs degrees of cyclic per inch of stick, and
    Chapter 9 prints it nowhere. ``deg_per_inch = 2.71`` is what puts the
    example helicopter at 0.32 on Figure 9.13's pitch abscissa, so it is read
    back off a figure. Everything else in this group comes from an equation.

    The time history is post-processing
    -----------------------------------
    ``numerator_coeffs`` and ``denominator_coeffs`` go to
    ``heaviside_step_response`` for Figure 9.12, and ``figure_9_14_violations``
    reads the verdict off ``steady_rate`` and ``time_constant``. Neither is a
    component: a time history has no derivatives to contribute and a verdict is
    discrete.

    Options
    -------
    num_nodes : int

    Example helicopter in hover
    ---------------------------
    ``Theta/B1`` denominator ``s^3 + .7245 s^2 + .1151``, the p. 598 cubic;
    attitude gain -6.764 against p. 607's -6.78; one-second displacement
    7.32 deg/in against a required 2.65; damping 28,659 against a required
    24,977, a 15 % margin.

    On Figure 9.14 it lands at **25.6 deg/sec/in and a 1.40 second time
    constant**, outside the pitch box on both axes -- oversensitive and too
    slow. That is not a contradiction of Table 9.18: p. 613 offers those boxes
    as the tighter modern view, and the aircraft passing MIL-H-8501A while
    failing them is the point of printing both.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ones = np.ones(nn)

        self.add_subsystem('matrix', HoverLongTwoDofMatrixComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem('column', HoverControlColumnComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem(
            'transfer_function',
            TransferFunctionGroup(n=2, degree=2, column=1, num_nodes=nn),
            promotes=['*'])

        # Table 9.18 works in magnitudes; the derivatives are negative
        self.add_subsystem('damping', om.ExecComp(
            'damping = -dM_dq', shape=(nn,),
            damping={'units': 'lbf*ft*s/rad'},
            dM_dq={'units': 'lbf*ft*s/rad', 'val': -28659.0 * ones}),
            promotes=['*'])
        self.add_subsystem('power', om.ExecComp(
            'control_power = -dM_dB1 * deg_per_inch', shape=(nn,),
            control_power={'units': 'lbf*ft'},
            dM_dB1={'units': 'lbf*ft/rad', 'val': -270578.0 * ones},
            deg_per_inch={'units': 'rad', 'val': np.deg2rad(DEG_PER_INCH)
                          * ones}),
            promotes=['*'])

        self.add_subsystem(
            'requirements',
            MilResponseRequirementsComp(num_nodes=nn, axis='longitudinal',
                                        instrument=True),
            promotes_inputs=['G_W', 'damping', 'control_power',
                             ('inertia', 'I_yy')],
            promotes_outputs=['*'])

        for name, (value, units) in EXAMPLE_DEFAULTS.items():
            self.set_input_defaults(name, val=np.full(nn, value), units=units)
