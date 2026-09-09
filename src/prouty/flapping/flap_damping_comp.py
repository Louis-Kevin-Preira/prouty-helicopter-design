"""Flapping damping, critical damping and damping ratio.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 458-459.
"""

import numpy as np
import openmdao.api as om


class FlapDampingComp(om.ExplicitComponent):
    """Aerodynamic damping of the flapping blade and its critical value.

    Critical damping lies on the boundary between oscillatory and
    non-oscillatory motion (p. 458)::

        c_crit = 2 I_b omega_n

    The aerodynamic damping follows from the hinge moment due to flapping
    velocity (p. 458)::

        M_A       = int_0^(R-e) r' (rho/2) a c r' beta_dot (r' + e) Omega dr'
        c_damp    = dM_A / dbeta_dot
                  = (I_b gamma Omega / 8) (1 - e/R)^4 (1 + e/3R) / (1 - e/R)

    and the damping ratio (p. 459)::

        c_damp / c_crit = (gamma/16) (1 - e/R)^4 (1 + e/3R) / (1 - e/R)
                          / (omega_n / Omega)

    Notes
    -----
    The printed grouping ``(1 - e/R)^4 (1 + e/3R) / (1 - e/R)`` collapses to
    ``(1 - e/R)^3 (1 + e/3R)``, which is exactly what the integral above gives.
    The code uses the collapsed form; the two are algebraically identical, so
    the recurring ``(1 - e/R)^4`` ambiguity does not arise here.

    Prouty warns on p. 458 that italic *c* is the chord and roman c the
    damping. The outputs are named ``c_damp`` and ``c_crit`` accordingly.

    The damping ratio is independent of both ``Omega`` and ``I_b``: they cancel
    between ``c_damp`` and ``c_crit``. It varies across nodes only through
    ``gamma``.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('I_b', val=1.0, units='slug*ft**2',
                       desc='Blade flapping inertia about the hinge.')
        self.add_input('e_over_R', val=0.0, desc='Hinge offset ratio, e/R.')
        self.add_input('omega_n_ratio', val=1.0,
                       desc='Undamped flapping frequency ratio.')
        self.add_input('gamma', val=np.full(nn, 8.0), desc='Lock number.')
        self.add_input('Omega', val=np.full(nn, 1.0), units='rad/s',
                       desc='Rotor angular speed.')

        self.add_output('c_damp', val=np.ones(nn), units='slug*ft**2/s',
                        desc='Aerodynamic flapping damping.')
        self.add_output('c_crit', val=np.ones(nn), units='slug*ft**2/s',
                        desc='Critical flapping damping.')
        self.add_output('zeta', val=np.zeros(nn),
                        desc='Damping ratio, c/c_crit.')
        self.add_output('omega_n', val=np.ones(nn), units='rad/s',
                        desc='Undamped flapping natural frequency.')

        ar = np.arange(nn)
        self.declare_partials('c_damp', ['gamma', 'Omega'], rows=ar, cols=ar)
        self.declare_partials('c_damp', ['I_b', 'e_over_R'])
        self.declare_partials('c_crit', 'Omega', rows=ar, cols=ar)
        self.declare_partials('c_crit', ['I_b', 'omega_n_ratio'])
        self.declare_partials('zeta', 'gamma', rows=ar, cols=ar)
        self.declare_partials('zeta', ['e_over_R', 'omega_n_ratio'])
        self.declare_partials('omega_n', 'Omega', rows=ar, cols=ar)
        self.declare_partials('omega_n', 'omega_n_ratio')

    @staticmethod
    def _shape_factor(x):
        """(1 - e/R)^3 (1 + e/3R) and its derivative w.r.t. e/R."""
        k = 1.0 - x
        K = k ** 3 * (1.0 + x / 3.0)
        dK = -4.0 / 3.0 * k ** 2 * (2.0 + x)
        return K, dK

    def compute(self, inputs, outputs):
        K, _ = self._shape_factor(inputs['e_over_R'])
        nu, I_b = inputs['omega_n_ratio'], inputs['I_b']

        outputs['omega_n'] = nu * inputs['Omega']
        outputs['c_damp'] = I_b * inputs['gamma'] * inputs['Omega'] * K / 8.0
        outputs['c_crit'] = 2.0 * I_b * outputs['omega_n']
        outputs['zeta'] = inputs['gamma'] * K / (16.0 * nu)

    def compute_partials(self, inputs, J):
        K, dK = self._shape_factor(inputs['e_over_R'])
        nu, I_b = inputs['omega_n_ratio'], inputs['I_b']
        gamma, Omega = inputs['gamma'], inputs['Omega']

        c_damp = I_b * gamma * Omega * K / 8.0
        zeta = gamma * K / (16.0 * nu)

        J['c_damp', 'gamma'] = I_b * Omega * K / 8.0
        J['c_damp', 'Omega'] = I_b * gamma * K / 8.0
        J['c_damp', 'I_b'] = (c_damp / I_b).reshape(-1, 1)
        J['c_damp', 'e_over_R'] = (c_damp * dK / K).reshape(-1, 1)

        J['c_crit', 'Omega'] = np.full(Omega.size, 2.0 * I_b * nu)
        J['c_crit', 'I_b'] = (2.0 * nu * Omega).reshape(-1, 1)
        J['c_crit', 'omega_n_ratio'] = (2.0 * I_b * Omega).reshape(-1, 1)

        J['zeta', 'gamma'] = K / (16.0 * nu)
        J['zeta', 'e_over_R'] = (zeta * dK / K).reshape(-1, 1)
        J['zeta', 'omega_n_ratio'] = (-zeta / nu).reshape(-1, 1)

        J['omega_n', 'Omega'] = np.full(Omega.size, nu[0])
        J['omega_n', 'omega_n_ratio'] = Omega.reshape(-1, 1)
