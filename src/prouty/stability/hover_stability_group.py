"""The hover stability analyses of Chapter 9, on one derivative set.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", pp. 596-605::

    pp. 596-598  longitudinal, three degrees of freedom     quartic
    p.  598      longitudinal, two degrees of freedom       cubic
    pp. 600-601  Hohenemser, one degree of freedom          period only
    p.  602      Routh's discriminant, Table 9.17
    p.  604      lateral, two degrees of freedom            cubic
    p.  605      yaw, one degree of freedom                 one root
"""

import numpy as np
import openmdao.api as om

from prouty.stability.hohenemser_period_comp import HohenemserPeriodComp
from prouty.stability.hover_lateral_matrix_comp import HoverLateralMatrixComp
from prouty.stability.hover_long_two_dof_matrix_comp import (
    HoverLongTwoDofMatrixComp,
)
from prouty.stability.hover_longitudinal_matrix_comp import (
    HoverLongitudinalMatrixComp,
)
from prouty.stability.poly_determinant_comp import PolyDeterminantComp
from prouty.stability.routh_discriminant_comp import RouthDiscriminantComp
from prouty.stability.yaw_mode_hover_comp import YawModeHoverComp

#: Table 9.4 totals (pp. 571-573) and the three inertias Table 9.20 implies
#: (p. 614): -621 s^2 gives G.W./g, -40,000 s^2 gives I_yy, -5000 s^2 gives
#: I_xx and -35,000 s^2 gives I_zz.
EXAMPLE_DEFAULTS = {
    'dX_dxdot': (-5.0, 'lbf*s/ft'), 'dX_dzdot': (0.0, 'lbf*s/ft'),
    'dX_dq': (1008.0, 'lbf*s/rad'),
    'dZ_dxdot': (0.0, 'lbf*s/ft'), 'dZ_dzdot': (-182.0, 'lbf*s/ft'),
    'dZ_dzddot': (0.0, 'lbf*s**2/ft'), 'dZ_dq': (0.0, 'lbf*s/rad'),
    'dM_dxdot': (143.0, 'lbf*s'), 'dM_dzdot': (91.0, 'lbf*s'),
    'dM_dzddot': (0.0, 'lbf*s**2'), 'dM_dq': (-28659.0, 'lbf*ft*s/rad'),
    'dY_dydot': (-18.0, 'lbf*s/ft'), 'dY_dp': (-1086.0, 'lbf*s/rad'),
    'dR_dydot': (-65.0, 'lbf*s'), 'dR_dp': (-29127.0, 'lbf*ft*s/rad'),
    'dN_dr': (-13326.0, 'lbf*ft*s/rad'),
    'G_W': (20000.0, 'lbf'), 'g': (32.2, 'ft/s**2'),
    'I_xx': (5000.0, 'slug*ft**2'), 'I_yy': (40000.0, 'slug*ft**2'),
    'I_zz': (35000.0, 'slug*ft**2'),
    'd_a1s_d_mu_M': (0.34, None), 'd_a1s_dq_M': (-0.105, 's'),
    'Omega_R_M': (650.0, 'ft/s'), 'CT_sigma_bar_M': (0.086, None),
    'gamma_M': (8.1, None),
}

#: Scalar, not one value per flight condition.
SCALAR_DEFAULTS = {'a_M': (6.0, '1/rad'), 'R_M': (30.0, 'ft')}


class HoverStabilityGroup(om.Group):
    """Five analyses of the same hover, side by side.

    Takes the Table 9.4 derivative set plus the three inertias and produces
    every stability result of pp. 596-605::

        longitudinal      3 DOF  -> char_coeffs_long, routh_discriminant_long
        longitudinal_2dof 2 DOF  -> char_coeffs_long_2dof, routh_..._long_2dof
        lateral           2 DOF  -> char_coeffs_lateral, routh_..._lateral
        hohenemser        1 DOF  -> omega_N_squared, period, period_radius, ...
        yaw               1 DOF  -> root_yaw, time_to_half_yaw

    Roots and modal metrics are deliberately absent: they come from
    ``prouty.stability.modes`` in post-processing, since root ordering is
    discontinuous in the coefficients.

    Input naming
    ------------
    Derivatives, ``G_W``, the inertias and ``g`` keep the names
    ``TotalDerivativesHoverComp`` gives them, so ``HoverDerivativesGroup``
    connects straight through. The five Hohenemser inputs that belong to the
    main rotor rather than the whole aircraft take the ``_M`` suffix of
    ``HoverDerivativesGroup`` -- ``Omega_R_M``, ``gamma_M``, ``a_M``, ``R_M``,
    and the two flapping derivatives.

    ``CT_sigma_bar_M`` is deliberately *not* called ``CT_sigma_M``. p. 601 uses
    the trim thrust coefficient including the vertical drag penalty, .086
    (Chapter 1, p. 18), while the value that reproduces Table 9.1 is .0849.
    Keeping the names apart stops the two groups fighting over one variable.

    Sharing between branches
    ------------------------
    ``dX_dxdot``, ``dX_dq``, ``dM_dxdot``, ``dM_dq``, ``G_W``, ``I_yy`` and
    ``g`` reach both longitudinal branches, and ``dM_dxdot``, ``dM_dq`` and
    ``g`` reach the Hohenemser component as well. That is the point: the three
    longitudinal analyses are the same physics at three levels of
    simplification, and they must see the same numbers. ``set_input_defaults``
    fixes each at the example helicopter's value so the group runs standalone
    and reproduces the chapter.

    Options
    -------
    num_nodes : int

    What the group reproduces
    -------------------------
    ``s^4 + 1.0175 s^3 + .2123 s^2 + .1151 s + .0337`` (p. 597),
    ``s^3 + .7245 s^2 + .1151`` (p. 598), R.D.(3) = -.1151 (p. 602),
    period 15.68 s (p. 600), lateral ``s^3 + 5.854 s^2 + .1461 s + .4186``
    (p. 604), yaw root -.3807 (p. 605).
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self._add_branch('longitudinal', HoverLongitudinalMatrixComp(num_nodes=nn),
                         n=3, order=4, suffix='long', routh=True)
        self._add_branch('longitudinal_2dof',
                         HoverLongTwoDofMatrixComp(num_nodes=nn),
                         n=2, order=3, suffix='long_2dof', routh=True)
        self._add_branch('lateral', HoverLateralMatrixComp(num_nodes=nn),
                         n=2, order=3, suffix='lateral', routh=True)

        self.add_subsystem(
            'hohenemser', HohenemserPeriodComp(num_nodes=nn),
            promotes_inputs=['dM_dxdot', 'dM_dq', 'g',
                             ('d_a1s_d_mu', 'd_a1s_d_mu_M'),
                             ('d_a1s_dq', 'd_a1s_dq_M'),
                             ('Omega_R', 'Omega_R_M'),
                             ('CT_sigma_bar', 'CT_sigma_bar_M'),
                             ('gamma', 'gamma_M'), ('a', 'a_M'), ('R', 'R_M')],
            promotes_outputs=['*'])

        self.add_subsystem(
            'yaw', YawModeHoverComp(num_nodes=nn),
            promotes_inputs=['dN_dr', 'I_zz'],
            promotes_outputs=[('root', 'root_yaw'),
                              ('time_to_half', 'time_to_half_yaw')])

        for name, (value, units) in EXAMPLE_DEFAULTS.items():
            self.set_input_defaults(name, val=np.full(nn, value), units=units)
        for name, (value, units) in SCALAR_DEFAULTS.items():
            self.set_input_defaults(name, val=value, units=units)

    def _add_branch(self, name, matrix_comp, n, order, suffix, routh):
        """One set of equations: matrix, determinant, and Routh if it applies."""
        nn = self.options['num_nodes']
        coeffs = f'char_coeffs_{suffix}'

        self.add_subsystem(f'{name}_matrix', matrix_comp, promotes_inputs=['*'])
        self.add_subsystem(
            f'{name}_det',
            PolyDeterminantComp(n=n, degree=2, degree_out=order, num_nodes=nn),
            promotes_outputs=[('char_coeffs', coeffs)])
        self.connect(f'{name}_matrix.matrix_coeffs', f'{name}_det.matrix_coeffs')

        if routh:
            self.add_subsystem(
                f'{name}_routh',
                RouthDiscriminantComp(degree=order, num_nodes=nn),
                promotes_inputs=[('char_coeffs', coeffs)],
                promotes_outputs=[('routh_discriminant',
                                   f'routh_discriminant_{suffix}')])
