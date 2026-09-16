"""Thrust coefficient and induced velocity in forward flight, solved together.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 467-468.
"""

import numpy as np
import openmdao.api as om


class ThrustInflowComp(om.ExplicitComponent):
    """Thrust coefficient of a rotor with hinge offset, and its inflow.

    The thrust coefficient (p. 467)::

        C_T/sigma = (1 - e/R) (a/4) [ theta_0 (2/3 + mu^2)
                                    + theta_1 (1/2 + mu^2/2)
                                    + mu (alpha_s - B_1)
                                    - v_1/(Omega R) ]

    and the high-speed momentum inflow (p. 468)::

        v_1/(Omega R) = (C_T/sigma) sigma / (2 mu)

    The two form a linear loop, so no solver is needed. Writing
    ``K = (1 - e/R) a/4``, ``S`` for the square bracket without the inflow
    term and ``P = 2 mu + K sigma``::

        C_T/sigma     = 2 mu K S / P
        v_1/(Omega R) = sigma K S / P

    This grouping is algebraically identical to substituting one equation into
    the other, but has no division by ``mu``, so it stays finite as the
    advance ratio goes to zero.

    Options
    -------
    num_nodes : int
    inflow : {'internal', 'external'}
        ``'internal'``  solve the loop above and output both quantities.
        ``'external'``  take ``v1_over_OmegaR`` as an input, from a more
        complete inflow model, and output ``C_T/sigma`` only. ``sigma`` is
        then unused and not declared.

    Validity
    --------
    The p. 468 inflow is the high-speed momentum result, ``v_1 = T/(2 rho A
    V)``. It is not valid near hover. The code form above keeps
    ``v_1/(Omega R)`` bounded as ``mu -> 0`` (it tends to ``S``), so an
    optimiser will not blow up there, but the value is physically meaningless
    below roughly ``mu = 0.1``. Use ``inflow='external'`` with the Chapter 1
    or Chapter 3 inflow if low-speed flight matters.

    Twist reference
    ---------------
    Chapter 7 measures the blade pitch from the flapping hinge (p. 464)::

        theta = theta_0 + (r'/R) theta_1 - A_1 cos(psi) - B_1 sin(psi)

    with ``r' = r - e``. For a blade physically defined from the centre of
    rotation as ``theta = theta_root + (r/R) theta_tw``, this means
    ``theta_1 = theta_tw`` unchanged, while
    ``theta_0 = theta_root + (e/R) theta_tw``. Only the collective reference
    shifts; the twist coefficient does not rescale.

    Notes
    -----
    Setting ``e/R = 0`` recovers the Chapter 3 form of p. 166,
    ``C_T/sigma = (a/4)[theta_0 (2/3 + mu^2) + theta_1 (1/2 + mu^2/2)
    + lambda]`` with ``lambda = mu (alpha_s - B_1) - v_1/(Omega R)``.

    The ``(1 - e/R)`` prefactor is the exact leading term of the thrust
    integral over the hinge-to-tip span, which evaluates to
    ``(R^3/3)(1 - e/R)[1 + e/R + (e/R)^2]``. The bracket is dropped following
    the policy stated on p. 466; it is worth about 5 % of ``C_T`` at
    ``e/R = 0.05``.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('inflow', values=('internal', 'external'),
                             default='internal',
                             desc='Whether the induced velocity is solved here.')

    def setup(self):
        nn = self.options['num_nodes']
        internal = self.options['inflow'] == 'internal'

        self.add_input('e_over_R', val=0.0, desc='Hinge offset ratio, e/R.')
        self.add_input('a', val=np.full(nn, 6.0), units='1/rad',
                       desc='Blade section lift curve slope.')
        self.add_input('mu', val=np.full(nn, 0.3), desc='Advance ratio.')
        self.add_input('theta_0', val=np.zeros(nn), units='rad',
                       desc='Collective pitch at the flapping hinge.')
        self.add_input('theta_1', val=np.zeros(nn), units='rad',
                       desc='Linear blade twist.')
        self.add_input('alpha_s', val=np.zeros(nn), units='rad',
                       desc='Shaft angle of attack.')
        self.add_input('B_1', val=np.zeros(nn), units='rad',
                       desc='Longitudinal cyclic pitch.')

        self.add_output('CT_sigma', val=np.zeros(nn),
                        desc='Thrust coefficient over solidity.')

        vec = ['a', 'mu', 'theta_0', 'theta_1', 'alpha_s', 'B_1']
        ar = np.arange(nn)

        if internal:
            self.add_input('sigma', val=0.08, desc='Rotor solidity.')
            self.add_output('v1_over_OmegaR', val=np.zeros(nn),
                            desc='Mean induced velocity ratio.')
            outs = ['CT_sigma', 'v1_over_OmegaR']
            self.declare_partials(outs, ['e_over_R', 'sigma'])
        else:
            self.add_input('v1_over_OmegaR', val=np.zeros(nn),
                           desc='Mean induced velocity ratio.')
            vec = vec + ['v1_over_OmegaR']
            outs = ['CT_sigma']
            self.declare_partials(outs, 'e_over_R')

        self.declare_partials(outs, vec, rows=ar, cols=ar)

    def _terms(self, inputs):
        """K, S and dS/dmu, shared by both modes."""
        mu = inputs['mu']
        K = (1.0 - inputs['e_over_R']) * inputs['a'] / 4.0
        S = (inputs['theta_0'] * (2.0 / 3.0 + mu ** 2)
             + inputs['theta_1'] * (0.5 + mu ** 2 / 2.0)
             + mu * (inputs['alpha_s'] - inputs['B_1']))
        S_mu = (2.0 * inputs['theta_0'] * mu + inputs['theta_1'] * mu
                + inputs['alpha_s'] - inputs['B_1'])
        return K, S, S_mu

    def compute(self, inputs, outputs):
        K, S, _ = self._terms(inputs)
        mu = inputs['mu']

        if self.options['inflow'] == 'internal':
            P = 2.0 * mu + K * inputs['sigma']
            outputs['CT_sigma'] = 2.0 * mu * K * S / P
            outputs['v1_over_OmegaR'] = inputs['sigma'] * K * S / P
        else:
            outputs['CT_sigma'] = K * (S - inputs['v1_over_OmegaR'])

    def compute_partials(self, inputs, J):
        K, S, S_mu = self._terms(inputs)
        mu, a, x = inputs['mu'], inputs['a'], inputs['e_over_R']

        c_theta0 = 2.0 / 3.0 + mu ** 2
        c_theta1 = 0.5 + mu ** 2 / 2.0

        if self.options['inflow'] == 'external':
            J['CT_sigma', 'theta_0'] = K * c_theta0
            J['CT_sigma', 'theta_1'] = K * c_theta1
            J['CT_sigma', 'alpha_s'] = K * mu
            J['CT_sigma', 'B_1'] = -K * mu
            J['CT_sigma', 'v1_over_OmegaR'] = -K
            J['CT_sigma', 'mu'] = K * S_mu
            ct = K * (S - inputs['v1_over_OmegaR'])
            J['CT_sigma', 'a'] = ct / a
            J['CT_sigma', 'e_over_R'] = (-ct / (1.0 - x)).reshape(-1, 1)
            return

        sigma = inputs['sigma']
        P = 2.0 * mu + K * sigma
        N = K * S
        ct = 2.0 * mu * N / P
        v1 = sigma * N / P

        # Linear in the pitch and attitude inputs through S only
        for name, coeff in (('theta_0', c_theta0), ('theta_1', c_theta1),
                            ('alpha_s', mu), ('B_1', -mu)):
            J['CT_sigma', name] = 2.0 * mu * K * coeff / P
            J['v1_over_OmegaR', name] = sigma * K * coeff / P

        # K enters through both N and P
        J['CT_sigma', 'a'] = ct * 2.0 * mu / (a * P)
        J['v1_over_OmegaR', 'a'] = v1 * 2.0 * mu / (a * P)
        J['CT_sigma', 'e_over_R'] = (-ct * 2.0 * mu / ((1.0 - x) * P)).reshape(-1, 1)
        J['v1_over_OmegaR', 'e_over_R'] = (-v1 * 2.0 * mu / ((1.0 - x) * P)).reshape(-1, 1)

        J['CT_sigma', 'sigma'] = (-ct * K / P).reshape(-1, 1)
        J['v1_over_OmegaR', 'sigma'] = (2.0 * mu * N / P ** 2).reshape(-1, 1)

        J['CT_sigma', 'mu'] = (2.0 * N + 2.0 * mu * K * S_mu) / P - 4.0 * mu * N / P ** 2
        J['v1_over_OmegaR', 'mu'] = sigma * K * S_mu / P - 2.0 * sigma * N / P ** 2
