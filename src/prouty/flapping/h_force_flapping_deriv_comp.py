"""In-plane force derivatives with respect to flapping.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 478-479, with the
``C_H/sigma`` equation of Chapter 3, p. 176.
"""

import numpy as np
import openmdao.api as om


class HForceFlappingDerivComp(om.ExplicitComponent):
    """How much in-plane force one radian of flapping produces.

    Assuming the rotor force stays perpendicular to the tip path plane makes
    the in-plane force ``T a_1s``. p. 478 shows why that is too simple: the
    blade elements on the advancing and retreating sides are tilted
    symmetrically about the tip path plane but not about the shaft, so the
    inflow angle enters. Differentiating the Chapter 3 ``C_H/sigma`` equation
    with respect to ``a_1s`` and simplifying gives (p. 479)::

        d(C_H/sigma)/da_1s = C_T/sigma + (a/8) lambda'

    and a derivation of the Y-force shows the same behaviour::

        d(C_Y/sigma)/db_1s = C_T/sigma + (a/8) lambda'

    ``lambda'`` is the inflow ratio referred to the tip path plane, defined
    on p. 166 of Chapter 3::

        lambda' = mu alpha_TPP - v_1/(Omega R),   alpha_TPP = alpha_s + a_1s

    It is negative in forward flight and becomes more so with speed, which
    cuts the in-plane force and, with it, the flapping stiffness of the
    aircraft. For the example helicopter in hover the derivative falls from
    ``T = 20,000 lb`` to about 9,300 lb.

    Options
    -------
    num_nodes : int
    form : {'lambda', 'theta75'}
        ``'lambda'`` is the expression above, recommended by p. 479 for
        forward flight. ``'theta75'`` is the hover-convenient rewrite, got by
        assuming the thrust coefficient is set by the pitch at three-quarter
        radius, ``C_T/sigma = (a/4)((2/3) theta_75 + lambda')``::

            d(C_H/sigma)/da_1s = (3/2)(C_T/sigma)
                                 [1 - (a/18) theta_75 / (C_T/sigma)]

        The two are algebraically identical given that relation; they differ
        only in which quantity is supplied.

    What the simplification costs
    -----------------------------
    Prouty reaches the short form by saying the second term of the full
    derivative "for all practical purposes, is the equation for
    ``C_T/sigma``". It is not quite: the exact difference is
    ``-(a/4) mu a_1s``. At 115 kt with ``a_1s = 2.87 deg`` that is -0.0225
    against a retained value of 0.0679, a third of it.

    The table on p. 479 follows the short form, so this component does too.
    Entry C7-11 of ``docs/validation_flapping.md`` records the size of the gap.

    Notes
    -----
    The derivative is taken at fixed ``lambda'``, as Prouty does. Tilting the
    tip path plane in a real manoeuvre also changes ``alpha_TPP`` and hence
    ``lambda'``, which this partial does not account for.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('form', values=('lambda', 'theta75'),
                             default='lambda')

    def setup(self):
        nn = self.options['num_nodes']
        hover_form = self.options['form'] == 'theta75'

        self.add_input('CT_sigma', val=np.zeros(nn),
                       desc='Thrust coefficient over solidity.')
        self.add_input('a', val=np.full(nn, 6.0), units='1/rad',
                       desc='Blade section lift curve slope.')

        if hover_form:
            self.add_input('theta_75', val=np.zeros(nn), units='rad',
                           desc='Blade pitch at three-quarter radius.')
            sources = ['CT_sigma', 'a', 'theta_75']
        else:
            self.add_input('mu', val=np.full(nn, 0.3), desc='Advance ratio.')
            self.add_input('alpha_s', val=np.zeros(nn), units='rad',
                           desc='Shaft angle of attack.')
            self.add_input('a_1s', val=np.zeros(nn), units='rad',
                           desc='Longitudinal flapping.')
            self.add_input('v1_over_OmegaR', val=np.zeros(nn),
                           desc='Mean induced velocity ratio.')
            self.add_output('lambda_prime', val=np.zeros(nn),
                            desc='Inflow ratio about the tip path plane.')
            sources = ['CT_sigma', 'a', 'mu', 'alpha_s', 'a_1s',
                       'v1_over_OmegaR']

        self.add_output('dCHsigma_da1s', val=np.zeros(nn),
                        desc='d(C_H/sigma)/da_1s.')
        self.add_output('dCYsigma_db1s', val=np.zeros(nn),
                        desc='d(C_Y/sigma)/db_1s, equal to the H-force one.')

        ar = np.arange(nn)
        self.declare_partials(['dCHsigma_da1s', 'dCYsigma_db1s'], sources,
                              rows=ar, cols=ar)
        if not hover_form:
            self.declare_partials(
                'lambda_prime', ['mu', 'alpha_s', 'a_1s', 'v1_over_OmegaR'],
                rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        a = inputs['a']

        if self.options['form'] == 'theta75':
            deriv = 1.5 * inputs['CT_sigma'] - a * inputs['theta_75'] / 12.0
        else:
            lam = (inputs['mu'] * (inputs['alpha_s'] + inputs['a_1s'])
                   - inputs['v1_over_OmegaR'])
            outputs['lambda_prime'] = lam
            deriv = inputs['CT_sigma'] + a * lam / 8.0

        outputs['dCHsigma_da1s'] = deriv
        outputs['dCYsigma_db1s'] = deriv

    def compute_partials(self, inputs, J):
        a, nn = inputs['a'], self.options['num_nodes']

        if self.options['form'] == 'theta75':
            d = {'CT_sigma': np.full(nn, 1.5),
                 'a': -inputs['theta_75'] / 12.0,
                 'theta_75': -a / 12.0}
        else:
            mu = inputs['mu']
            tilt = inputs['alpha_s'] + inputs['a_1s']
            lam = mu * tilt - inputs['v1_over_OmegaR']

            J['lambda_prime', 'mu'] = tilt
            J['lambda_prime', 'alpha_s'] = mu
            J['lambda_prime', 'a_1s'] = mu
            J['lambda_prime', 'v1_over_OmegaR'] = -np.ones(nn)

            d = {'CT_sigma': np.ones(nn),
                 'a': lam / 8.0,
                 'mu': a * tilt / 8.0,
                 'alpha_s': a * mu / 8.0,
                 'a_1s': a * mu / 8.0,
                 'v1_over_OmegaR': -a / 8.0}

        for name, val in d.items():
            J['dCHsigma_da1s', name] = val
            J['dCYsigma_db1s', name] = val
