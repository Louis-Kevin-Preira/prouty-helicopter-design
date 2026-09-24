"""
AxialThrustIdealTwistComp -- thrust of an ideally twisted rotor in vertical flight.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "Rotor Thrust Damping" pp. 101-102.

A hovering rotor with ideal twist (Chapter 1), in vertical flight:

    C_T/sigma = (a/4) [theta_T - (v_1c + V_c)/(Omega R)]                     p. 101

with lambda_c = V_c/(Omega R) and the momentum flow through the disc:

    'small_climb'  v_1c + V_c = V_c/2 + sqrt(T/2 rho A)                      p. 102
                   C_T/sigma = (a/4) [theta_T - lambda_c/2 - sqrt(C_T/2)]
    'exact'        v_1c + V_c = V_c/2 + sqrt((V_c/2)^2 + v_1hov^2)           p. 94
                   C_T/sigma = (a/4) [theta_T - lambda_c/2 - u],  u = sqrt(lambda_c^2/4 + C_T/2)

Both are quadratics, in sqrt(C_T/sigma) and in u, solved in closed form.
Partials by implicit differentiation of F = C_T/sigma - (a/4)[...] = 0.
Valid for climb and low rates of descent, outside the vortex ring state (p. 102).
theta_T is the tip pitch of the ideal twist; a linear twist of the same thrust
has theta_75 = (3/2) theta_T (p. 114).

    theta_T, V_c (nn,), a, sigma, V_tip --> CT_sigma (nn,)
"""

import warnings

import numpy as np
import openmdao.api as om


class AxialThrustIdealTwistComp(om.ExplicitComponent):
    """C_T/sigma of an ideally twisted rotor at a given tip pitch and climb rate, pp. 101-102."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('inflow', default='exact', values=('small_climb', 'exact'))

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('theta_T', val=np.full(nn, 0.1), units='rad', desc='ideal twist tip pitch')
        self.add_input('V_c', val=np.zeros(nn), units='ft/s', desc='rate of climb')
        self.add_input('a', val=6.0, units='1/rad', desc='lift curve slope')
        self.add_input('sigma', val=0.085, desc='rotor solidity')
        self.add_input('V_tip', val=650.0, units='ft/s', desc='tip speed Omega R')
        self.add_output('CT_sigma', val=np.full(nn, 0.08), desc='thrust coefficient over solidity')
        self.declare_partials('CT_sigma', ['theta_T', 'V_c'], rows=ar, cols=ar)
        self.declare_partials('CT_sigma', ['a', 'sigma', 'V_tip'])

    def compute(self, inputs, outputs):
        th, a, sig = inputs['theta_T'], inputs['a'][0], inputs['sigma'][0]
        lam = inputs['V_c'] / inputs['V_tip'][0]
        if np.any(np.real(th - lam / 2) <= 0.0):
            warnings.warn('AxialThrustIdealTwistComp: theta_T - lambda_c/2 <= 0, no positive thrust')
        if self.options['inflow'] == 'small_climb':
            b, c = 0.25 * a * np.sqrt(0.5 * sig), 0.25 * a * (th - 0.5 * lam)
            outputs['CT_sigma'] = (0.5 * (-b + np.sqrt(b ** 2 + 4.0 * c))) ** 2
        else:
            K = 0.5 * lam ** 2 / sig + 0.25 * a * (th - 0.5 * lam)
            u = 0.25 * sig * (-0.25 * a + np.sqrt(a ** 2 / 16.0 + 8.0 * K / sig))
            outputs['CT_sigma'] = 2.0 * (u ** 2 - 0.25 * lam ** 2) / sig

    def compute_partials(self, inputs, partials):
        a, sig, V_tip = inputs['a'][0], inputs['sigma'][0], inputs['V_tip'][0]
        lam = inputs['V_c'] / V_tip
        outputs = {}
        self.compute(inputs, outputs)
        x = outputs['CT_sigma']

        if self.options['inflow'] == 'small_climb':
            r = np.sqrt(0.5 * sig * x)
            F_x = 1.0 + 0.25 * a * np.sqrt(0.5 * sig) / (2.0 * np.sqrt(x))
            F_lam = a / 8.0
        else:
            r = np.sqrt(0.25 * lam ** 2 + 0.5 * sig * x)
            F_x = 1.0 + 0.25 * a * sig / (4.0 * r)
            F_lam = 0.25 * a * (0.5 + lam / (4.0 * r))
        F_th, F_a, F_sig = -0.25 * a, -x / a, 0.25 * a * x / (4.0 * r)

        partials['CT_sigma', 'theta_T'] = -F_th / F_x
        partials['CT_sigma', 'V_c'] = -F_lam / (F_x * V_tip)
        partials['CT_sigma', 'V_tip'] = (F_lam * lam / (F_x * V_tip)).reshape(-1, 1)
        partials['CT_sigma', 'a'] = (-F_a / F_x).reshape(-1, 1)
        partials['CT_sigma', 'sigma'] = (-F_sig / F_x).reshape(-1, 1)
