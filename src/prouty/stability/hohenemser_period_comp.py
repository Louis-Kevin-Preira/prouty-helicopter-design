"""The Hohenemser period of the hover oscillation.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", pp. 600-601, after Hohenemser
(reference 9.4). The radius form on p. 601 draws its ``C_T/sigma`` relation
from Chapter 3 with the tip speed ratio set to zero.
"""

import warnings

import numpy as np
import openmdao.api as om

TWO_PI = 2.0 * np.pi


class HohenemserPeriodComp(om.ExplicitComponent):
    """The hover period with the fuselage inertia thrown away.

    Hohenemser's observation, p. 600: the helicopter swings about a point far
    above itself, like a child on a swing, so ``I_yy`` about its own centre of
    gravity can be dropped. The cubic of p. 598 collapses to::

        -(dM/dq) s^2 + g (dM/dxdot) = 0

    a mass on a spring, whose natural frequency is what this component
    produces. There is no damping term left, so the reduction gives a period
    and nothing else -- not a time to double.

    Three forms, all emitted
    ------------------------
    p. 600 prints two::

        omega_N^2 = -g (dM/dxdot) / (dM/dq)                   `omega_N_squared`
                  = (g/(Omega R)) (da1s/dmu) / (-da1s/dq)     `..._flapping`

    and p. 601 a third, in terms of rotor radius alone::

        P = 2 pi sqrt(R) / sqrt(g (C_T_bar/sigma) gamma / a)  `period_radius`

    The section is about how far the analysis can be stripped and still say
    something useful, so all three are outputs rather than an option: the
    comparison is the content.

    The second form is exact, not an approximation
    ----------------------------------------------
    It looks as though moving from derivatives to flapping drops the mast
    height, since Table 9.2 builds

        dM/dxdot = (dM/da1s)(da1s/dmu)(dmu/dxdot) - (dX/dxdot) h_M
        dM/dq    = (dM/da1s)(da1s/dq)             - (dX/dq)    h_M

    It does not. Both ``dX`` derivatives carry the same
    ``-rho A_b (Omega R)^2 (dCH/sigma/da1s)`` in front of the same flapping
    derivative, so each moment factors as that flapping derivative times
    ``(dM/da1s + K h_M)``, and the bracket cancels in the ratio. The two forms
    agree to machine precision on exact inputs, and to 0.2 % on the rounded
    Table 9.4 values.

    The third form is not exact, and Prouty says so
    -----------------------------------------------
    p. 601 substitutes ``da1s/dmu = 16 (C_T/sigma)/a``, calling it "a small
    white lie concerning the coefficient of the induced velocity ratio". The
    lie is visible: Chapter 3 at ``mu = 0`` gives
    ``C_T/sigma = (a/16)[(8/3)theta_0 + 2 theta_1 - 4 v_1/(Omega R)]`` while
    ``da1s/dmu = (8/3)theta_0 + 2 theta_1 - 2 v_1/(Omega R)``, so the two
    differ by ``2 v_1/(Omega R)``. For the example helicopter that turns .34
    into .229 and stretches the period from 15.7 s to 17.8 s, 14 %.

    p. 601 also prints the damping derivative as ``da1s/dq = 16/(gamma R)``.
    That is a misprint for ``16/(gamma Omega)``, which is Table 9.1 at
    ``e/R = 0`` and the only version under which the printed ``P`` comes out
    proportional to ``sqrt(R)``. Entry C9-12 of
    ``docs/validation_stability.md``.

    When the mode does not oscillate
    --------------------------------
    ``omega_N^2`` is negative when ``dM/dxdot`` changes sign -- the teetering
    low-rotor case of p. 603. There is no period then, the mode is a
    divergence, and the three period outputs are set to zero with zero
    partials and a warning. ``omega_N_squared`` stays exact and differentiable
    throughout, so it, not the period, is what to use as a constraint.

    Options
    -------
    num_nodes : int

    Example helicopter
    ------------------
    ``omega_N = .4008 rad/s``, ``P = 15.67 s`` from the derivatives and
    ``15.69 s`` from the flapping form, against the printed 15.7 s.
    ``period_radius = 17.80 s``; with ``C_T/sigma = .086``, ``gamma = 8.1``
    and ``a = 6`` the coefficient is 3.25, which is the ``P = 3.2 sqrt(R)``
    rule of thumb p. 601 quotes.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('dM_dxdot', val=np.full(nn, 143.0), units='lbf*s')
        self.add_input('dM_dq', val=np.full(nn, -28659.0), units='lbf*ft*s/rad')
        self.add_input('d_a1s_d_mu', val=np.full(nn, 0.34))
        self.add_input('d_a1s_dq', val=np.full(nn, -0.105), units='s')
        self.add_input('Omega_R', val=np.full(nn, 650.0), units='ft/s')
        self.add_input('CT_sigma_bar', val=np.full(nn, 0.086))
        self.add_input('gamma', val=np.full(nn, 8.1))
        self.add_input('g', val=np.full(nn, 32.2), units='ft/s**2')
        self.add_input('a', val=6.0, units='1/rad')
        self.add_input('R', val=30.0, units='ft')

        for name in ('omega_N_squared', 'omega_N_squared_flapping'):
            self.add_output(name, val=np.zeros(nn), units='rad**2/s**2')
        for name in ('period', 'period_flapping', 'period_radius'):
            self.add_output(name, val=np.zeros(nn), units='s')

        self.declare_partials('omega_N_squared', ['dM_dxdot', 'dM_dq', 'g'],
                              rows=ar, cols=ar)
        self.declare_partials('period', ['dM_dxdot', 'dM_dq', 'g'],
                              rows=ar, cols=ar)
        flapping = ['d_a1s_d_mu', 'd_a1s_dq', 'Omega_R', 'g']
        self.declare_partials('omega_N_squared_flapping', flapping,
                              rows=ar, cols=ar)
        self.declare_partials('period_flapping', flapping, rows=ar, cols=ar)
        self.declare_partials('period_radius', ['CT_sigma_bar', 'gamma', 'g'],
                              rows=ar, cols=ar)
        self.declare_partials('period_radius', ['a', 'R'], rows=ar, cols=zeros)

    @staticmethod
    def _period(omega_squared):
        """2 pi / omega, zero where the mode is not oscillatory."""
        oscillatory = np.real(omega_squared) > 0.0
        if not np.all(oscillatory):
            warnings.warn('omega_N^2 is not positive at every node: the hover '
                          'mode is a divergence there, not an oscillation, and '
                          'the period outputs are set to zero.')
        safe = np.where(oscillatory, omega_squared, 1.0)
        return np.where(oscillatory, TWO_PI / np.sqrt(safe), 0.0), oscillatory

    def compute(self, inputs, outputs):
        g = inputs['g']
        w2 = -g * inputs['dM_dxdot'] / inputs['dM_dq']
        w2_f = -g * inputs['d_a1s_d_mu'] / (inputs['Omega_R']
                                            * inputs['d_a1s_dq'])

        outputs['omega_N_squared'] = w2
        outputs['omega_N_squared_flapping'] = w2_f
        outputs['period'] = self._period(w2)[0]
        outputs['period_flapping'] = self._period(w2_f)[0]
        outputs['period_radius'] = TWO_PI * np.sqrt(inputs['R']) / np.sqrt(
            g * inputs['CT_sigma_bar'] * inputs['gamma'] / inputs['a'][0])

    def compute_partials(self, inputs, J):
        g, Mu, Mq = inputs['g'], inputs['dM_dxdot'], inputs['dM_dq']
        a_mu, a_q, V = (inputs['d_a1s_d_mu'], inputs['d_a1s_dq'],
                        inputs['Omega_R'])

        w2 = -g * Mu / Mq
        J['omega_N_squared', 'dM_dxdot'] = -g / Mq
        J['omega_N_squared', 'dM_dq'] = g * Mu / Mq ** 2
        J['omega_N_squared', 'g'] = -Mu / Mq

        w2_f = -g * a_mu / (V * a_q)
        J['omega_N_squared_flapping', 'd_a1s_d_mu'] = -g / (V * a_q)
        J['omega_N_squared_flapping', 'd_a1s_dq'] = g * a_mu / (V * a_q ** 2)
        J['omega_N_squared_flapping', 'Omega_R'] = g * a_mu / (V ** 2 * a_q)
        J['omega_N_squared_flapping', 'g'] = -a_mu / (V * a_q)

        # dP/d(omega^2) = -pi (omega^2)^(-3/2), chained onto each input
        for period, squared, names in (
                ('period', w2, ('dM_dxdot', 'dM_dq', 'g')),
                ('period_flapping', w2_f,
                 ('d_a1s_d_mu', 'd_a1s_dq', 'Omega_R', 'g'))):
            oscillatory = squared > 0.0
            safe = np.where(oscillatory, squared, 1.0)
            chain = np.where(oscillatory, -np.pi * safe ** -1.5, 0.0)
            source = ('omega_N_squared' if period == 'period'
                      else 'omega_N_squared_flapping')
            for name in names:
                J[period, name] = chain * J[source, name]

        P_R = TWO_PI * np.sqrt(inputs['R']) / np.sqrt(
            g * inputs['CT_sigma_bar'] * inputs['gamma'] / inputs['a'][0])
        J['period_radius', 'g'] = -0.5 * P_R / g
        J['period_radius', 'CT_sigma_bar'] = -0.5 * P_R / inputs['CT_sigma_bar']
        J['period_radius', 'gamma'] = -0.5 * P_R / inputs['gamma']
        J['period_radius', 'a'] = 0.5 * P_R / inputs['a'][0]
        J['period_radius', 'R'] = 0.5 * P_R / inputs['R'][0]
