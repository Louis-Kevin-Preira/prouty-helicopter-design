"""Tail rotor tip path plane angle at trim.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 535. Inflow relation from Chapter 3,
p. 167.
"""

import numpy as np
import openmdao.api as om


class TailRotorTppAngleComp(om.ExplicitComponent):
    """Tail rotor tip path plane angle and flapping, p. 535::

        alpha_TPP_T = arctan[ lambda'/mu + sigma (C_T/sigma) / (2 mu^2) ]
        a1s_T       = alpha_TPP_T + beta

    The directional half of the control positions. Once the N equation has
    supplied the tail rotor thrust, the pedal position follows from the tail
    rotor collective, and p. 535 sends that to the Chapter 3 tail rotor
    method — which needs the angle of attack of the tail rotor disc, not the
    thrust.

    Where the first expression comes from
    -------------------------------------
    It is Chapter 3's inflow relation, p. 167, solved backwards::

        lambda' = mu tan(alpha_TPP) - v1/(Omega R)

    with the high-speed momentum induced velocity
    ``v1/(Omega R) = sigma (C_T/sigma) / (2 mu)``. Every quantity in it is a
    *tail rotor* quantity: its own tip speed ratio, its own solidity, its own
    thrust coefficient.

    It is singular in hover, ``mu -> 0``, twice over. That is not a defect to
    be patched: the tip path plane angle of a rotor in still air is not
    defined, and p. 534 handles hover separately through the constant ratio
    ``(C_T/sigma)_T / (C_Q/sigma)_M``, which is what Figure 8.30 plots.

    The second expression
    ---------------------
    ``a1s_T = alpha_TPP_T + beta`` is the same relation p. 489 uses for the
    horizontal stabiliser, in the tail rotor's own frame. Note what
    "longitudinal" means there: the tail rotor disc is vertical, so its
    ``a1s_T`` is fore-and-aft in the aircraft, along the flight path. Its
    ``b1s_T``, which ``TailRotorForcesComp`` uses for ``Z_T``, is the
    perpendicular one and is vertical in the aircraft. The two are different
    quantities and neither is the other's lateral counterpart.

    p. 535 writes it as ``alpha_TPP_T = a1s_T - beta``. Sideslip changes the
    direction the tail rotor sees its own flight path from, by ``beta``,
    which is why the term appears at all.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('lambda_p_T', shape=(nn,), val=0.0,
                       desc="tail rotor inflow ratio lambda'")
        self.add_input('mu_T', shape=(nn,), val=0.3,
                       desc='tail rotor tip speed ratio')
        self.add_input('sigma_T', shape=(nn,), val=0.1,
                       desc='tail rotor solidity')
        self.add_input('CT_sigma_T', shape=(nn,), val=0.0,
                       desc='tail rotor thrust coefficient over solidity')
        self.add_input('beta', shape=(nn,), val=0.0, units='rad',
                       desc='sideslip angle')

        self.add_output('alpha_TPP_T', shape=(nn,), units='rad',
                        desc='tail rotor tip path plane angle')
        self.add_output('a1s_T', shape=(nn,), units='rad',
                        desc='tail rotor longitudinal flapping')

        wrt = ['lambda_p_T', 'mu_T', 'sigma_T', 'CT_sigma_T']
        for out in ('alpha_TPP_T', 'a1s_T'):
            self.declare_partials(out, wrt, rows=ar, cols=ar)
        self.declare_partials('a1s_T', 'beta', rows=ar, cols=ar,
                              val=np.ones(nn))

    def _tangent(self, inputs):
        mu = inputs['mu_T']
        return (inputs['lambda_p_T'] / mu
                + inputs['sigma_T'] * inputs['CT_sigma_T'] / (2.0 * mu ** 2))

    def compute(self, inputs, outputs):
        alpha = np.arctan(self._tangent(inputs))
        outputs['alpha_TPP_T'] = alpha
        outputs['a1s_T'] = alpha + inputs['beta']

    def compute_partials(self, inputs, J):
        mu, sigma = inputs['mu_T'], inputs['sigma_T']
        CT_sigma, lambda_p = inputs['CT_sigma_T'], inputs['lambda_p_T']
        t = self._tangent(inputs)
        dalpha_dt = 1.0 / (1.0 + t ** 2)

        dt = {'lambda_p_T': 1.0 / mu,
              'sigma_T': CT_sigma / (2.0 * mu ** 2),
              'CT_sigma_T': sigma / (2.0 * mu ** 2),
              'mu_T': -lambda_p / mu ** 2 - sigma * CT_sigma / mu ** 3}

        for name, value in dt.items():
            J['alpha_TPP_T', name] = dalpha_dt * value
            J['a1s_T', name] = dalpha_dt * value
