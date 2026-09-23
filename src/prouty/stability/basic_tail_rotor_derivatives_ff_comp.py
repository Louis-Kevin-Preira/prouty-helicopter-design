"""Basic tail rotor derivatives in forward flight.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", Table 9.7, p. 578. The two chart
partials it consumes are the tail rotor column of Table 9.5, p. 574.
"""

import numpy as np
import openmdao.api as om

SCALAR_INPUTS = ('sigma',)

UNITS = {'d_mu_d_xdot': 's/ft', 'd_lambda_d_xdot': 's/ft',
         'd_lambda_d_ydot': 's/ft', 'd_beta_d_ydot': 'rad*s/ft'}


class BasicTailRotorDerivativesFFComp(om.ExplicitComponent):
    """Table 9.7: four rows, against the main rotor's fourteen.

    A tail rotor has no cyclic, so none of Table 9.6's flapping rows apply, and
    it has no hub spring row either -- Table 9.9 never needs one. What is left
    is how body motion moves the tail rotor's own ``mu``, ``lambda'`` and
    sideslip::

        dmu/dxdot     = 1/(Omega R)
        dlambda'/dxdot = (1/(Omega R))[a1s_bar
                         - (sigma/2mu)(dCT/sigma/dmu - (C_T/sigma)/mu)]
        dlambda'/dydot = -1/[(Omega R)(1 + (dCT/sigma/dlambda')(sigma/2mu))]
        dbeta/dydot   = 1/V

    Two differences from Table 9.6, both from the axis the disc faces
    ----------------------------------------------------------------
    A tail rotor thrusts along ``+y`` where the main rotor thrusts along
    ``-z``, so the two tables differ exactly where that matters and nowhere
    else.

    **The row that pairs with the main rotor's ``dlambda'/dzdot`` is
    ``dlambda'/dydot``, and it is negative.** Same expression, leading minus
    sign, which is the same asymmetry Table 9.1 prints for hover (entry C9-3)
    and which the forward-flight table repeats.

    **``dlambda'/dxdot`` uses ``a1s_bar`` where the main rotor uses
    ``alpha_TPP_bar``.** Forward speed is *in the plane* of a tail rotor disc,
    so it contributes no inflow directly; what tilts flow through the disc is
    the tail rotor's own longitudinal flapping. For the main rotor the same
    role is played by the tip-path-plane angle of attack. The induced term
    behind it is identical in both tables.

    Options
    -------
    num_nodes : int

    Example helicopter at 115 knots
    -------------------------------
    ================ ========= ========= =======
    output           model     Table 9.7
    ================ ========= ========= =======
    d_mu_d_xdot      .0015385  .00154
    d_lambda_d_xdot  .0001690  .000169
    d_lambda_d_ydot  -.0012275 -.00121   1.4 %
    d_beta_d_ydot    .0051520  .00515
    ================ ========= ========= =======

    ``dlambda'/dydot`` is the one row that does not close. It depends only on
    ``sigma_T``, ``mu`` and ``dCT/sigma/dlambda' = 1.04``, and the geometric
    solidity Table 9.3 implies -- ``A_b = 19.40 ft^2`` and ``R = 6.5 ft``, so
    ``sigma_T = .1461`` -- gives -.001228. Inverting the printed -.00121
    instead would need ``sigma_T = .157``, 7 % larger, which contradicts
    Table 9.3. The 1.4 % is taken to be rounding in Prouty's arithmetic rather
    than a different rotor.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('sigma', val=0.14614, desc='Tail rotor solidity.')
        self.add_input('Omega_R', val=np.full(nn, 650.0), units='ft/s')
        self.add_input('V', val=np.full(nn, 194.1), units='ft/s',
                       desc='True airspeed of the helicopter.')
        self.add_input('mu', val=np.full(nn, 0.30))
        self.add_input('CT_sigma_bar', val=np.full(nn, 0.033912))
        self.add_input('a1s_bar', val=np.full(nn, 0.065266), units='rad',
                       desc='Trim longitudinal flapping of the tail rotor.')
        self.add_input('dCT_sigma_dmu', val=np.full(nn, -0.070),
                       desc='Table 9.5, p. 574, tail rotor.')
        self.add_input('dCT_sigma_dlambda', val=np.full(nn, 1.04),
                       desc='Table 9.5, p. 574, tail rotor.')

        for name, units in UNITS.items():
            self.add_output(name, val=np.zeros(nn), units=units)

        self.declare_partials('d_mu_d_xdot', 'Omega_R', rows=ar, cols=ar)
        self.declare_partials('d_lambda_d_xdot',
                              ['Omega_R', 'mu', 'CT_sigma_bar', 'a1s_bar',
                               'dCT_sigma_dmu'], rows=ar, cols=ar)
        self.declare_partials('d_lambda_d_xdot', 'sigma', rows=ar, cols=zeros)
        self.declare_partials('d_lambda_d_ydot',
                              ['Omega_R', 'mu', 'dCT_sigma_dlambda'],
                              rows=ar, cols=ar)
        self.declare_partials('d_lambda_d_ydot', 'sigma', rows=ar, cols=zeros)
        self.declare_partials('d_beta_d_ydot', 'V', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        V_tip, mu, sig = inputs['Omega_R'], inputs['mu'], inputs['sigma'][0]
        half = 0.5 * sig / mu

        outputs['d_mu_d_xdot'] = 1.0 / V_tip
        outputs['d_lambda_d_xdot'] = (
            inputs['a1s_bar']
            - half * (inputs['dCT_sigma_dmu']
                      - inputs['CT_sigma_bar'] / mu)) / V_tip
        outputs['d_lambda_d_ydot'] = -1.0 / (
            V_tip * (1.0 + inputs['dCT_sigma_dlambda'] * half))
        outputs['d_beta_d_ydot'] = 1.0 / inputs['V']

    def compute_partials(self, inputs, J):
        V_tip, mu, sig = inputs['Omega_R'], inputs['mu'], inputs['sigma'][0]
        x_T, D_mu = inputs['CT_sigma_bar'], inputs['dCT_sigma_dmu']
        D_lam = inputs['dCT_sigma_dlambda']
        one = np.ones_like(mu)

        J['d_mu_d_xdot', 'Omega_R'] = -1.0 / V_tip ** 2

        # B = a1s_bar - (sigma/2mu) D_mu + (sigma/2mu^2) x_T
        B = inputs['a1s_bar'] - 0.5 * sig / mu * D_mu + 0.5 * sig / mu ** 2 * x_T
        J['d_lambda_d_xdot', 'Omega_R'] = -B / V_tip ** 2
        J['d_lambda_d_xdot', 'a1s_bar'] = one / V_tip
        J['d_lambda_d_xdot', 'dCT_sigma_dmu'] = -0.5 * sig / (mu * V_tip)
        J['d_lambda_d_xdot', 'CT_sigma_bar'] = 0.5 * sig / (mu ** 2 * V_tip)
        J['d_lambda_d_xdot', 'mu'] = (0.5 * sig / mu ** 2 * D_mu
                                      - sig / mu ** 3 * x_T) / V_tip
        J['d_lambda_d_xdot', 'sigma'] = (-0.5 * D_mu / mu
                                         + 0.5 * x_T / mu ** 2) / V_tip

        G = 1.0 + D_lam * 0.5 * sig / mu
        value = -1.0 / (V_tip * G)
        J['d_lambda_d_ydot', 'Omega_R'] = -value / V_tip
        J['d_lambda_d_ydot', 'dCT_sigma_dlambda'] = -value / G * 0.5 * sig / mu
        J['d_lambda_d_ydot', 'sigma'] = -value / G * 0.5 * D_lam / mu
        J['d_lambda_d_ydot', 'mu'] = value / G * 0.5 * D_lam * sig / mu ** 2

        J['d_beta_d_ydot', 'V'] = -1.0 / inputs['V'] ** 2
