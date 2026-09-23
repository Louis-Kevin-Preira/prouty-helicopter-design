"""
MainRotorHoverPowerComp -- main rotor power in hover with its two ground effects.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Hover Performance" pp. 308-309; pseudo ground effect of the
fuselage p. 280; ground effect p. 67.

The isolated rotor power of Chapter 1 is corrected by the torque increments of
the fuselage pseudo ground effect (G2) and of the real ground effect:

    P_MR = P_iso + (dCQ_sigma_pge + dCQ_sigma_ige) sigma rho A (Omega R)^3 / 550

Both increments are negative, so both save power. Out of ground effect
dCQ_sigma_ige is zero; in ground effect the book drops the pseudo one instead
(p. 309), which is done by the ground proximity option of G2.

    P_iso, dCQ_sigma_pge, dCQ_sigma_ige (nn,), sigma, rho, A, V_tip --> P_MR (nn,)
"""

import numpy as np
import openmdao.api as om

HP_TO_FT_LBF_PER_S = 550.0


class MainRotorHoverPowerComp(om.ExplicitComponent):
    """Isolated rotor power plus the ground effect increments, p. 309."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('P_iso', val=np.zeros(nn), units='hp', desc='isolated rotor power')
        self.add_input('dCQ_sigma_pge', val=np.zeros(nn), desc='fuselage pseudo ground effect')
        self.add_input('dCQ_sigma_ige', val=np.zeros(nn), desc='ground effect, Chapter 1 p. 67')
        self.add_input('sigma', val=0.085, desc='rotor solidity')
        self.add_input('rho', val=0.002377, units='slug/ft**3')
        self.add_input('A', val=1.0, units='ft**2', desc='disc area')
        self.add_input('V_tip', val=1.0, units='ft/s', desc='tip speed')
        self.add_output('P_MR', val=np.zeros(nn), units='hp', desc='main rotor power in hover')

        self.declare_partials('P_MR', ['P_iso', 'dCQ_sigma_pge', 'dCQ_sigma_ige'],
                              rows=ar, cols=ar)
        self.declare_partials('P_MR', ['sigma', 'rho', 'A', 'V_tip'])

    def _scale(self, inputs):
        return (inputs['sigma'][0] * inputs['rho'][0] * inputs['A'][0]
                * inputs['V_tip'][0] ** 3 / HP_TO_FT_LBF_PER_S)

    def compute(self, inputs, outputs):
        dCQ = inputs['dCQ_sigma_pge'] + inputs['dCQ_sigma_ige']
        outputs['P_MR'] = inputs['P_iso'] + dCQ * self._scale(inputs)

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        dCQ = inputs['dCQ_sigma_pge'] + inputs['dCQ_sigma_ige']
        scale = self._scale(inputs)

        partials['P_MR', 'P_iso'] = np.ones(nn)
        partials['P_MR', 'dCQ_sigma_pge'] = np.full(nn, scale)
        partials['P_MR', 'dCQ_sigma_ige'] = np.full(nn, scale)
        for name in ('sigma', 'rho', 'A'):
            partials['P_MR', name] = (dCQ * scale / inputs[name][0]).reshape(-1, 1)
        partials['P_MR', 'V_tip'] = (3.0 * dCQ * scale / inputs['V_tip'][0]).reshape(-1, 1)
