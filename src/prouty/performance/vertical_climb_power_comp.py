"""
VerticalClimbPowerComp -- power needed above hover to climb vertically.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Vertical Climb" pp. 313-315 (the equation of p. 314 and its
definition of v_1c + V_c on p. 315); momentum method of Chapter 2.

The power in excess of hover at the same conditions is the change of induced
and download power, raised by the tail rotor that follows the main rotor
torque:

    dP = 1/550 { [G.W.(v_1c+V_c) + 4 (D_v/G.W.)_hov (rho/2) (v_1c+V_c)^3 A_M
                  + (dA_z C_D) (rho/2) V_c^3]
               - [G.W. v_1hov + 4 (D_v/G.W.)_hov (rho/2) v_1hov^3 A_M] } k_T

    k_T = 1 + v_1hov_T R_M / [(Omega R)_M l_T]

The download terms are consistent: with v_1hov^2 = G.W./(2 rho A_M),
4 (D_v/G.W.)(rho/2) v_1hov^3 A_M is exactly D_v v_1hov. dA_z C_D covers the
airframe outside the wake, which sees only the climb speed.

C4-3. The book prints the tail rotor factor against the hover bracket alone,
which would give a non-zero dP at V_c = 0. It multiplies the whole difference
here, as the physics requires: the tail rotor power follows the main rotor
torque increment.

    GW, v_hov, v_sum, V_c, Dv_GW (nn,), rho, A_M, dAz_CD, v_hov_T, R_M, V_tip_M, l_T
        --> dP (nn,), k_T
"""

import numpy as np
import openmdao.api as om

HP_TO_FT_LBF_PER_S = 550.0


class VerticalClimbPowerComp(om.ExplicitComponent):
    """Excess power of a vertical climb over hover, p. 314."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('GW', val=np.ones(nn), units='lbf', desc='gross weight')
        self.add_input('v_hov', val=np.ones(nn), units='ft/s', desc='hover induced velocity')
        self.add_input('v_sum', val=np.ones(nn), units='ft/s', desc='v_1c + V_c in climb')
        self.add_input('V_c', val=np.zeros(nn), units='ft/s', desc='rate of climb')
        self.add_input('Dv_GW', val=np.zeros(nn), desc='hover vertical drag over gross weight')
        self.add_input('rho', val=0.002377, units='slug/ft**3')
        self.add_input('A_M', val=1.0, units='ft**2', desc='main rotor disc area')
        self.add_input('dAz_CD', val=0.0, units='ft**2',
                       desc='drag area of the airframe outside the wake')
        self.add_input('v_hov_T', val=0.0, units='ft/s',
                       desc='tail rotor hover induced velocity')
        self.add_input('R_M', val=1.0, units='ft', desc='main rotor radius')
        self.add_input('V_tip_M', val=1.0, units='ft/s', desc='main rotor tip speed')
        self.add_input('l_T', val=1.0, units='ft', desc='main to tail rotor distance')

        self.add_output('dP', val=np.zeros(nn), units='hp', desc='power above hover')
        self.add_output('k_T', val=1.0, desc='tail rotor factor on the power increment')

        self.declare_partials('dP', ['GW', 'v_hov', 'v_sum', 'V_c', 'Dv_GW'], rows=ar, cols=ar)
        self.declare_partials('dP', ['rho', 'A_M', 'dAz_CD', 'v_hov_T', 'R_M', 'V_tip_M', 'l_T'])
        self.declare_partials('k_T', ['v_hov_T', 'R_M', 'V_tip_M', 'l_T'])

    def _terms(self, inputs):
        GW, v_h, v_s, V_c = (inputs[k] for k in ('GW', 'v_hov', 'v_sum', 'V_c'))
        rho, A, dAz = inputs['rho'][0], inputs['A_M'][0], inputs['dAz_CD'][0]
        d = inputs['Dv_GW']
        climb = GW * v_s + 4.0 * d * (rho / 2.0) * v_s ** 3 * A + dAz * (rho / 2.0) * V_c ** 3
        hover = GW * v_h + 4.0 * d * (rho / 2.0) * v_h ** 3 * A
        k_T = 1.0 + inputs['v_hov_T'][0] * inputs['R_M'][0] / (inputs['V_tip_M'][0]
                                                               * inputs['l_T'][0])
        return climb, hover, k_T

    def compute(self, inputs, outputs):
        climb, hover, k_T = self._terms(inputs)
        outputs['dP'] = (climb - hover) * k_T / HP_TO_FT_LBF_PER_S
        outputs['k_T'] = k_T

    def compute_partials(self, inputs, partials):
        GW, v_h, v_s, V_c = (inputs[k] for k in ('GW', 'v_hov', 'v_sum', 'V_c'))
        rho, A, dAz = inputs['rho'][0], inputs['A_M'][0], inputs['dAz_CD'][0]
        d = inputs['Dv_GW']
        climb, hover, k_T = self._terms(inputs)
        c = k_T / HP_TO_FT_LBF_PER_S
        vT, R, V_tip, l = (inputs[k][0] for k in ('v_hov_T', 'R_M', 'V_tip_M', 'l_T'))

        partials['dP', 'GW'] = c * (v_s - v_h)
        partials['dP', 'v_sum'] = c * (GW + 12.0 * d * (rho / 2.0) * v_s ** 2 * A)
        partials['dP', 'v_hov'] = -c * (GW + 12.0 * d * (rho / 2.0) * v_h ** 2 * A)
        partials['dP', 'V_c'] = c * 3.0 * dAz * (rho / 2.0) * V_c ** 2
        partials['dP', 'Dv_GW'] = c * 4.0 * (rho / 2.0) * A * (v_s ** 3 - v_h ** 3)

        d_rho = (4.0 * d * 0.5 * A * (v_s ** 3 - v_h ** 3) + dAz * 0.5 * V_c ** 3)
        partials['dP', 'rho'] = (c * d_rho).reshape(-1, 1)
        partials['dP', 'A_M'] = (c * 4.0 * d * (rho / 2.0) * (v_s ** 3 - v_h ** 3)).reshape(-1, 1)
        partials['dP', 'dAz_CD'] = (c * (rho / 2.0) * V_c ** 3).reshape(-1, 1)

        base = (climb - hover) / HP_TO_FT_LBF_PER_S
        dk = {'v_hov_T': R / (V_tip * l), 'R_M': vT / (V_tip * l),
              'V_tip_M': -vT * R / (V_tip ** 2 * l), 'l_T': -vT * R / (V_tip * l ** 2)}
        for name, val in dk.items():
            partials['dP', name] = (base * val).reshape(-1, 1)
            partials['k_T', name] = val
