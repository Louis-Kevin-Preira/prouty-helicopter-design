"""Acceleration cross-coupling and the cyclic angle of attack it costs.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 459-461.
"""

import numpy as np
import openmdao.api as om


class AccelCouplingComp(om.ExplicitComponent):
    """Lateral flapping that accompanies longitudinal flapping, and its cost.

    A rotor with hinge offset lags its aerodynamic input by less than a
    quarter cycle, so tilting the tip path plane purely nose up also rolls it.
    Prouty gives the magnitude of that coupling as (p. 459)::

        b_1s / a_1s = -cot(phi)

    which the code evaluates from the frequency and damping ratios::

        b_1s / a_1s = -[(omega_n/Omega)^2 - 1] / [2 (c/c_crit) (omega_n/Omega)]

    For a uniform blade this collapses to the closed form of p. 460::

        b_1s / a_1s = -(12/gamma) (e/R)
                      / {[1 + (1/3)(e/R)] [1 - e/R]^4}

    Sustaining that flapping costs a cyclic change in blade angle of attack.
    With the inflow convention of p. 464 the blade sees ``Delta_alpha =
    -r' beta_dot / [Omega (r' + e)]``, which at ``psi = 180 deg`` gives
    (p. 461)::

        Delta_alpha / a_1s = -(b_1s / a_1s) (r' / (r' + e))
                           = -(b_1s / a_1s) [1 - (e/R) / (r/R)]

    Known book inconsistency
    ------------------------
    The ``b_1s`` written on p. 461 drops the ``(1 - e/R)^4`` factor that
    appears in the ratio on p. 460, and the numbers quoted on both pages
    (-0.07 and 0.07 for the example helicopter) follow that reduced form. The
    complete form is the one consistent with the p. 459 derivation and with
    the phase angle: it gives -0.0895 and 0.0835. This component implements
    the complete form; see ``docs/validation_flapping.md``.

    Notes
    -----
    ``b_1s/a_1s`` carries no direct dependence on ``e/R`` here: the offset
    enters through ``omega_n_ratio`` and ``zeta`` upstream. ``e/R`` is needed
    only for the blade station factor of ``Delta_alpha``.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('omega_n_ratio', val=1.0,
                       desc='Undamped flapping frequency ratio.')
        self.add_input('zeta', val=np.full(nn, 0.5),
                       desc='Damping ratio, c/c_crit.')
        self.add_input('e_over_R', val=0.0, desc='Hinge offset ratio, e/R.')
        self.add_input('r_over_R', val=0.75,
                       desc='Blade station for the angle of attack ratio.')

        self.add_output('b1s_over_a1s', val=np.zeros(nn),
                        desc='Lateral to longitudinal flapping ratio.')
        self.add_output('dalpha_over_a1s', val=np.zeros(nn),
                        desc='Cyclic angle of attack per unit flapping.')

        ar = np.arange(nn)
        self.declare_partials(['b1s_over_a1s', 'dalpha_over_a1s'], 'zeta',
                              rows=ar, cols=ar)
        self.declare_partials(['b1s_over_a1s', 'dalpha_over_a1s'],
                              'omega_n_ratio')
        self.declare_partials('dalpha_over_a1s', ['e_over_R', 'r_over_R'])

    def compute(self, inputs, outputs):
        nu, zeta = inputs['omega_n_ratio'], inputs['zeta']

        b_ratio = -(nu ** 2 - 1.0) / (2.0 * zeta * nu)
        station = 1.0 - inputs['e_over_R'] / inputs['r_over_R']

        outputs['b1s_over_a1s'] = b_ratio
        outputs['dalpha_over_a1s'] = -b_ratio * station

    def compute_partials(self, inputs, J):
        nu, zeta = inputs['omega_n_ratio'], inputs['zeta']
        x, rr = inputs['e_over_R'], inputs['r_over_R']

        b_ratio = -(nu ** 2 - 1.0) / (2.0 * zeta * nu)
        station = 1.0 - x / rr

        db_dnu = -(nu ** 2 + 1.0) / (2.0 * zeta * nu ** 2)
        db_dzeta = -b_ratio / zeta

        J['b1s_over_a1s', 'zeta'] = db_dzeta
        J['b1s_over_a1s', 'omega_n_ratio'] = db_dnu.reshape(-1, 1)

        J['dalpha_over_a1s', 'zeta'] = -db_dzeta * station
        J['dalpha_over_a1s', 'omega_n_ratio'] = (-db_dnu * station).reshape(-1, 1)
        J['dalpha_over_a1s', 'e_over_R'] = (b_ratio / rr).reshape(-1, 1)
        J['dalpha_over_a1s', 'r_over_R'] = (-b_ratio * x / rr ** 2).reshape(-1, 1)
