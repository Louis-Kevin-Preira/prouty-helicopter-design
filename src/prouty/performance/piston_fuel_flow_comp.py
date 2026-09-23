"""
PistonFuelFlowComp -- fuel flow of a piston engine installation.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4: fuel flow and 5 % deterioration p. 275; Willans-line shape of
Figure 4.3 p. 276; single-engine cruise p. 325, where the benefit is noted
to be smaller for reciprocating engines.

Prouty gives no piston data. The Willans line of Figure 4.3 is kept, with
coefficients from the engine specification (C4-5):

    FF_engine = a_w P_rated + bsfc P

a_w P_rated is the fuel burnt at zero power; a_w = 0 gives a constant
specific fuel consumption. With P_req shared by the n_eng operating engines:

    FF = (1 + k_det) [n_eng a_w P_rated + bsfc P_req]

    P_req, n_eng (nn,), P_rated, a_w, bsfc, k_det --> FF (nn,)
"""

import numpy as np
import openmdao.api as om


class PistonFuelFlowComp(om.ExplicitComponent):
    """Willans-line fuel flow, all operating engines."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('P_req', val=np.zeros(nn), units='hp', desc='shaft power, all engines')
        self.add_input('n_eng', val=np.ones(nn), desc='operating engines')
        self.add_input('P_rated', val=1.0, units='hp', desc='rated power per engine')
        self.add_input('a_w', val=0.0, units='lbm/h/hp', desc='zero-power fuel per rated hp')
        self.add_input('bsfc', val=0.5, units='lbm/h/hp', desc='incremental specific fuel consumption')
        self.add_input('k_det', val=0.0, desc='fuel flow deterioration allowance')
        self.add_output('FF', val=np.zeros(nn), units='lbm/h', desc='fuel flow, all engines')

        self.declare_partials('FF', ['P_req', 'n_eng'], rows=ar, cols=ar)
        self.declare_partials('FF', ['P_rated', 'a_w', 'bsfc', 'k_det'])

    def compute(self, inputs, outputs):
        outputs['FF'] = (1.0 + inputs['k_det'][0]) * self._base(inputs)

    def _base(self, inputs):
        return inputs['n_eng'] * inputs['a_w'] * inputs['P_rated'] + inputs['bsfc'] * inputs['P_req']

    def compute_partials(self, inputs, partials):
        c = 1.0 + inputs['k_det'][0]
        n = inputs['n_eng']

        partials['FF', 'P_req'] = c * inputs['bsfc'] * np.ones_like(n)
        partials['FF', 'n_eng'] = c * inputs['a_w'] * inputs['P_rated'] * np.ones_like(n)
        partials['FF', 'P_rated'] = (c * n * inputs['a_w']).reshape(-1, 1)
        partials['FF', 'a_w'] = (c * n * inputs['P_rated']).reshape(-1, 1)
        partials['FF', 'bsfc'] = (c * inputs['P_req']).reshape(-1, 1)
        partials['FF', 'k_det'] = self._base(inputs).reshape(-1, 1)
