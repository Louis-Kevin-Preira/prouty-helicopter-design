"""
ThrustDampingComp -- change of rotor thrust with axial velocity at constant pitch.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "Rotor Thrust Damping" pp. 101-102.

Differentiating C_T/sigma = (a/4)[theta_T - (v_1c + V_c)/(Omega R)] at constant
theta_T and at the initial C_T/sigma, with lambda_c = V_c/(Omega R):

    'small_climb'  (p. 102, as printed)
        g = 1 / [8/a + sqrt(sigma/2) / sqrt(C_T/sigma)]
    'exact'        (momentum of p. 94, u = sqrt(lambda_c^2/4 + sigma C_T/sigma / 2))
        g = (a/4)(1/2 + lambda_c/(4u)) / [1 + a sigma/(16u)]

    dCT_sigma_dlambda = g                        Chapter 9 sign (Table 9.1, p. 564)
    dCT_sigma_dVc     = -g / (Omega R)           p. 102
    dT_dVc            = rho sigma A (Omega R)^2 dCT_sigma_dVc = -rho sigma A (Omega R) g

The two options coincide at lambda_c = 0, where g is exactly the Table 9.1
entry of Chapter 9; they differ in climb and descent. Valid for small rates of
descent, in climb, and for the tail rotor in turns or sideward flight, outside
the vortex ring state (p. 102). dT_dVc is the Z_w of Chapter 9 in hover
(Table 9.2, dZ/dzdot = -182 lb/(ft/s) for the example helicopter).

    CT_sigma, [V_c] (nn,), a, sigma, V_tip, rho, A
        --> dCT_sigma_dlambda, dCT_sigma_dVc, dT_dVc (nn,)
"""

import numpy as np
import openmdao.api as om


class ThrustDampingComp(om.ExplicitComponent):
    """Rotor thrust damping, p. 102."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('inflow', default='exact', values=('small_climb', 'exact'))

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        exact = self.options['inflow'] == 'exact'

        self.add_input('CT_sigma', val=np.full(nn, 0.085), desc='initial C_T/sigma')
        if exact:
            self.add_input('V_c', val=np.zeros(nn), units='ft/s', desc='rate of climb')
        self.add_input('a', val=6.0, units='1/rad', desc='lift curve slope')
        self.add_input('sigma', val=0.085, desc='rotor solidity')
        self.add_input('V_tip', val=650.0, units='ft/s', desc='tip speed Omega R')
        self.add_input('rho', val=0.002377, units='slug/ft**3')
        self.add_input('A', val=1.0, units='ft**2', desc='disc area')

        self.add_output('dCT_sigma_dlambda', val=np.zeros(nn), desc='Chapter 9 sign, > 0')
        self.add_output('dCT_sigma_dVc', val=np.zeros(nn), units='s/ft')
        self.add_output('dT_dVc', val=np.zeros(nn), units='lbf*s/ft')

        per_node = ['CT_sigma', 'V_c'] if exact else ['CT_sigma']
        self.declare_partials('*', per_node, rows=ar, cols=ar)
        self.declare_partials('*', ['a', 'sigma'])
        self.declare_partials(['dCT_sigma_dVc', 'dT_dVc'], 'V_tip')
        if exact:
            self.declare_partials('dCT_sigma_dlambda', 'V_tip')
        self.declare_partials('dT_dVc', ['rho', 'A'])

    def _g(self, inputs):
        """g and its derivatives with respect to C_T/sigma, lambda_c, a, sigma."""
        x, a, sig = inputs['CT_sigma'], inputs['a'][0], inputs['sigma'][0]
        if self.options['inflow'] == 'small_climb':
            D = 8.0 / a + np.sqrt(0.5 * sig) / np.sqrt(x)
            g = 1.0 / D
            dg = {'x': g ** 2 * 0.5 * np.sqrt(0.5 * sig) * x ** -1.5,
                  'lam': 0.0 * x,
                  'a': g ** 2 * 8.0 / a ** 2,
                  'sigma': -g ** 2 / (4.0 * np.sqrt(0.5 * sig) * np.sqrt(x))}
            return g, dg

        lam = inputs['V_c'] / inputs['V_tip'][0]
        u = np.sqrt(0.25 * lam ** 2 + 0.5 * sig * x)
        N = 0.25 * a * (0.5 + lam / (4.0 * u))
        Dn = 1.0 + a * sig / (16.0 * u)
        g = N / Dn
        dN_du, dDn_du = -a * lam / (16.0 * u ** 2), -a * sig / (16.0 * u ** 2)
        dg_du = (dN_du - g * dDn_du) / Dn
        dg = {'x': dg_du * sig / (4.0 * u),
              'lam': (a / (16.0 * u)) / Dn + dg_du * lam / (4.0 * u),
              'a': (N / a - g * sig / (16.0 * u)) / Dn,
              'sigma': -g * a / (16.0 * u) / Dn + dg_du * x / (4.0 * u)}
        return g, dg

    def compute(self, inputs, outputs):
        g, _ = self._g(inputs)
        V_tip = inputs['V_tip'][0]
        k = inputs['rho'][0] * inputs['sigma'][0] * inputs['A'][0] * V_tip
        outputs['dCT_sigma_dlambda'] = g
        outputs['dCT_sigma_dVc'] = -g / V_tip
        outputs['dT_dVc'] = -k * g

    def compute_partials(self, inputs, partials):
        g, dg = self._g(inputs)
        rho, sig, A, V_tip = (inputs[n][0] for n in ('rho', 'sigma', 'A', 'V_tip'))
        k = rho * sig * A * V_tip
        exact = self.options['inflow'] == 'exact'
        lam = inputs['V_c'] / V_tip if exact else 0.0

        # total derivatives of g with respect to the component inputs
        G = {'CT_sigma': dg['x'], 'a': dg['a'], 'sigma': dg['sigma'],
             'V_tip': -dg['lam'] * lam / V_tip}
        if exact:
            G['V_c'] = dg['lam'] / V_tip

        for name, dgn in G.items():
            col = (lambda v: v) if name in ('CT_sigma', 'V_c') else (lambda v: v.reshape(-1, 1))
            d_dVc = -dgn / V_tip + (g / V_tip ** 2 if name == 'V_tip' else 0.0)
            d_T = -k * dgn - (k / sig * g if name == 'sigma' else 0.0) \
                  - (k / V_tip * g if name == 'V_tip' else 0.0)
            if name != 'V_tip' or exact:
                partials['dCT_sigma_dlambda', name] = col(dgn + 0.0 * g)
            partials['dCT_sigma_dVc', name] = col(d_dVc + 0.0 * g)
            partials['dT_dVc', name] = col(d_T + 0.0 * g)
        partials['dT_dVc', 'rho'] = (-k / rho * g).reshape(-1, 1)
        partials['dT_dVc', 'A'] = (-k / A * g).reshape(-1, 1)
