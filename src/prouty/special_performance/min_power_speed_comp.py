"""
MinPowerSpeedComp -- G2d, speed for minimum power, without stall effects.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Generating the Deadman's Curve" pp. 356-357 (energy method of
Chapter 3).

Main rotor power, differentiated with respect to V and set to zero:

    R = V^4 + (1/2)(A_b/f) Omega R C_d V^3
        - (G.W./rho_0)^2 / (3 (rho/rho_0)^2 e f A) = 0        (V in ft/s)

e: induced efficiency factor, low-mu part of Figure 3.7 (Chapter 3).
The quartic has one positive root; it is solved by Newton inside the
component, and the linear system is solved analytically (scalar residuals).

    GW, rho_ratio, e, f, A, A_b, V_tip, C_d --> V_min (nn,)
"""

import numpy as np
import openmdao.api as om

RHO_0 = 0.002377                 # slug/ft^3, sea level standard


class MinPowerSpeedComp(om.ImplicitComponent):
    """Speed for minimum main rotor power, p. 357."""

    _INPUTS = ('GW', 'rho_ratio', 'e', 'f', 'A', 'A_b', 'V_tip', 'C_d')

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('tol', default=1e-12, desc='Newton relative tolerance')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('GW', val=20000.0 * np.ones(nn), units='lbf')
        self.add_input('rho_ratio', val=np.ones(nn))
        self.add_input('e', val=0.8 * np.ones(nn))
        self.add_input('f', val=20.0 * np.ones(nn), units='ft**2')
        self.add_input('A', val=2827.0 * np.ones(nn), units='ft**2')
        self.add_input('A_b', val=240.0 * np.ones(nn), units='ft**2')
        self.add_input('V_tip', val=650.0 * np.ones(nn), units='ft/s')
        self.add_input('C_d', val=0.01 * np.ones(nn))
        self.add_output('V_min', val=140.0 * np.ones(nn), units='ft/s', lower=1.0)
        self.declare_partials('V_min', ('V_min',) + self._INPUTS, rows=ar, cols=ar)

    @staticmethod
    def _coeffs(i):
        c = 0.5 * i['A_b'] * i['V_tip'] * i['C_d'] / i['f']
        K = (i['GW'] / RHO_0) ** 2 / (3.0 * i['rho_ratio'] ** 2 * i['e'] * i['f'] * i['A'])
        return c, K

    def apply_nonlinear(self, inputs, outputs, residuals):
        c, K = self._coeffs(inputs)
        V = outputs['V_min']
        residuals['V_min'] = V ** 4 + c * V ** 3 - K

    def solve_nonlinear(self, inputs, outputs):
        c, K = self._coeffs(inputs)
        V = K ** 0.25                         # root without profile term, an upper bound
        for _ in range(50):
            step = (V ** 4 + c * V ** 3 - K) / (4.0 * V ** 3 + 3.0 * c * V ** 2)
            V = V - step
            if np.all(np.abs(step) < self.options['tol'] * V):
                break
        outputs['V_min'] = V

    def linearize(self, inputs, outputs, J):
        c, K = self._coeffs(inputs)
        V = outputs['V_min']
        self._dR_dV = 4.0 * V ** 3 + 3.0 * c * V ** 2
        J['V_min', 'V_min'] = self._dR_dV
        J['V_min', 'GW'] = -2.0 * K / inputs['GW']
        J['V_min', 'rho_ratio'] = 2.0 * K / inputs['rho_ratio']
        J['V_min', 'e'] = K / inputs['e']
        J['V_min', 'A'] = K / inputs['A']
        J['V_min', 'f'] = -c * V ** 3 / inputs['f'] + K / inputs['f']
        for name in ('A_b', 'V_tip', 'C_d'):
            J['V_min', name] = c * V ** 3 / inputs[name]

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['V_min'] = d_residuals['V_min'] / self._dR_dV
        else:
            d_residuals['V_min'] = d_outputs['V_min'] / self._dR_dV
