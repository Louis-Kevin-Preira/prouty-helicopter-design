"""
RotorLoadingMatchComp -- blade loading match between main and tail rotors.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Hover Performance": Figure 4.28 p. 309 (calculated performance of
the isolated rotors), Figure 4.31 p. 312 (thrust coefficients of both rotors),
discussion p. 310.

Comparing the hover C_T/sigma of the two rotors of the example helicopter
"reveals a mismatch that would probably generate a redesign effort in an
actual project": at high gross weight the tail rotor is more heavily loaded
than the main rotor, especially relative to their own maximum capabilities, so
"the high gross weight or altitude performance will be limited by the tail
rotor rather than the main rotor" (p. 310). The cures listed are more tip
speed, chord or radius on the tail rotor.

The component turns that comparison into numbers usable as constraints:

    util   = (C_T/sigma) / (C_T/sigma)_max        1 at the rotor's own maximum
    margin = 1 - util                             positive while thrust is left
    mismatch = util_T / util_M                    above 1: the tail rotor limits

(C_T/sigma)_max is where the isolated rotor chart of Chapter 1 stops gaining
thrust with collective. Figure 4.28 gives 0.167 for the main rotor of the
example (theta_0 = 30 deg, sigma 0.085, -10 deg twist) and 0.155 for its tail
rotor (theta_0 = 25 deg, sigma 0.146, -5 deg twist); those are the defaults,
and they should be recomputed for another rotor.

    CT_sigma_M, CT_sigma_T (nn,), CT_sigma_max_M, CT_sigma_max_T
        --> util_M, util_T, margin_M, margin_T, mismatch (nn,)
"""

import numpy as np
import openmdao.api as om

CT_SIGMA_MAX_MAIN = 0.167       # Figure 4.28 p. 309, example main rotor
CT_SIGMA_MAX_TAIL = 0.155       # Figure 4.28 p. 309, example tail rotor


class RotorLoadingMatchComp(om.ExplicitComponent):
    """Blade loading used and left on each rotor, and the match between them, p. 310."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('CT_sigma_M', val=np.zeros(nn), desc='main rotor blade loading')
        self.add_input('CT_sigma_T', val=np.zeros(nn), desc='tail rotor blade loading')
        self.add_input('CT_sigma_max_M', val=CT_SIGMA_MAX_MAIN, desc='main rotor maximum')
        self.add_input('CT_sigma_max_T', val=CT_SIGMA_MAX_TAIL, desc='tail rotor maximum')

        for name, desc in (('util_M', 'main rotor blade loading over its maximum'),
                           ('util_T', 'tail rotor blade loading over its maximum'),
                           ('margin_M', 'main rotor blade loading margin'),
                           ('margin_T', 'tail rotor blade loading margin'),
                           ('mismatch', 'tail over main utilization, above 1 the tail limits')):
            self.add_output(name, val=np.zeros(nn), desc=desc)

        for out, tag in (('util_M', 'M'), ('util_T', 'T'), ('margin_M', 'M'), ('margin_T', 'T')):
            self.declare_partials(out, f'CT_sigma_{tag}', rows=ar, cols=ar)
            self.declare_partials(out, f'CT_sigma_max_{tag}')
        self.declare_partials('mismatch', ['CT_sigma_M', 'CT_sigma_T'], rows=ar, cols=ar)
        self.declare_partials('mismatch', ['CT_sigma_max_M', 'CT_sigma_max_T'])

    def compute(self, inputs, outputs):
        u_M = inputs['CT_sigma_M'] / inputs['CT_sigma_max_M'][0]
        u_T = inputs['CT_sigma_T'] / inputs['CT_sigma_max_T'][0]
        outputs['util_M'], outputs['util_T'] = u_M, u_T
        outputs['margin_M'], outputs['margin_T'] = 1.0 - u_M, 1.0 - u_T
        outputs['mismatch'] = u_T / u_M

    def compute_partials(self, inputs, partials):
        CT_M, CT_T = inputs['CT_sigma_M'], inputs['CT_sigma_T']
        max_M, max_T = inputs['CT_sigma_max_M'][0], inputs['CT_sigma_max_T'][0]
        u_M, u_T = CT_M / max_M, CT_T / max_T
        ones = np.ones_like(CT_M)

        partials['util_M', 'CT_sigma_M'] = ones / max_M
        partials['util_T', 'CT_sigma_T'] = ones / max_T
        partials['util_M', 'CT_sigma_max_M'] = (-u_M / max_M).reshape(-1, 1)
        partials['util_T', 'CT_sigma_max_T'] = (-u_T / max_T).reshape(-1, 1)
        partials['margin_M', 'CT_sigma_M'] = -ones / max_M
        partials['margin_T', 'CT_sigma_T'] = -ones / max_T
        partials['margin_M', 'CT_sigma_max_M'] = (u_M / max_M).reshape(-1, 1)
        partials['margin_T', 'CT_sigma_max_T'] = (u_T / max_T).reshape(-1, 1)

        partials['mismatch', 'CT_sigma_T'] = 1.0 / (max_T * u_M)
        partials['mismatch', 'CT_sigma_M'] = -u_T / (max_M * u_M ** 2)
        partials['mismatch', 'CT_sigma_max_T'] = (-u_T / (max_T * u_M)).reshape(-1, 1)
        partials['mismatch', 'CT_sigma_max_M'] = (u_T / (max_M * u_M)).reshape(-1, 1)
