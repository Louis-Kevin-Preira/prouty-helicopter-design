"""
GroundEffectPowerComp -- power saved by the ground effect at constant thrust.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, "Ground Effect" pp. 66-68, Figure 1.41 p. 66; used by the hover
performance of Chapter 4 (p. 309).

Holding the thrust while approaching the ground leaves the profile power
unchanged and cuts the induced power in the ratio of the induced velocities
(Figure 1.41, GroundEffectComp), p. 67:

    dCQ_sigma = -(C_T/sigma)^(3/2) sqrt(sigma/2) [1 - v_IGE/v_OGE]

An actual helicopter gains more, because the fuselage download falls in ground
effect. At constant weight, p. 67:

    dCQ_sigma = -(C_T/sigma_OGE)^(3/2) sqrt(sigma/2)
                {1 - (1 - D_v/G.W.)^(3/2) (v_IGE/v_OGE)}

vertical_drag = 'included' uses the second form (D_v/GW from G2, zero gives
the first), 'none' the first. The example of p. 309 uses no vertical drag in
ground effect, which is the same as D_v/GW = 0 here.

The thrust gained at constant power, p. 68, ignoring the profile power
difference, is also returned:

    (T_IGE/T_OGE)_const power = (v_IGE/v_OGE)^(-2/3)

Example helicopter, pp. 67-68: C_T/sigma_OGE = 0.085, D_v/GW = 0.04,
Z/D = 0.30 so v_IGE/v_OGE = 0.75, giving dCQ_sigma = -0.00151 and about 429 hp
out of 2,000. The book prints 0.000143 and 407 h.p.; 0.000143 is out by a
factor of ten against its own 407 h.p. (C4-26).

    CT_sigma (nn,), sigma, vi_IGE_OGE, Dv_GW --> dCQ_sigma, T_ratio_IGE
"""

import numpy as np
import openmdao.api as om


class GroundEffectPowerComp(om.ExplicitComponent):
    """Torque saved in ground effect at constant thrust, p. 67."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('vertical_drag', default='none', values=('none', 'included'))

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('CT_sigma', val=np.full(nn, 0.08), desc='blade loading, out of ground effect')
        self.add_input('sigma', val=0.085, desc='rotor solidity')
        self.add_input('vi_IGE_OGE', val=np.ones(nn), desc='Figure 1.41 induced velocity ratio')
        self.add_output('dCQ_sigma', val=np.zeros(nn), desc='change of C_Q/sigma, negative saves power')
        self.add_output('T_ratio_IGE', val=np.ones(nn), desc='thrust ratio at constant power, p. 68')

        self.declare_partials('dCQ_sigma', ['CT_sigma', 'vi_IGE_OGE'], rows=ar, cols=ar)
        self.declare_partials('dCQ_sigma', 'sigma')
        self.declare_partials('T_ratio_IGE', 'vi_IGE_OGE', rows=ar, cols=ar)

        if self.options['vertical_drag'] == 'included':
            self.add_input('Dv_GW', val=np.zeros(nn), desc='vertical drag over gross weight')
            self.declare_partials('dCQ_sigma', 'Dv_GW', rows=ar, cols=ar)

    def _gap(self, inputs):
        """Bracket {1 - (1 - Dv/GW)^(3/2) r} and its derivatives d/dr, d/d(Dv/GW)."""
        r = inputs['vi_IGE_OGE']
        if self.options['vertical_drag'] == 'none':
            return 1.0 - r, -np.ones_like(r), None
        w = 1.0 - inputs['Dv_GW']
        return 1.0 - w ** 1.5 * r, -w ** 1.5, 1.5 * w ** 0.5 * r

    def compute(self, inputs, outputs):
        CT, s = inputs['CT_sigma'], inputs['sigma'][0]
        outputs['dCQ_sigma'] = -CT ** 1.5 * np.sqrt(s / 2.0) * self._gap(inputs)[0]
        outputs['T_ratio_IGE'] = inputs['vi_IGE_OGE'] ** (-2.0 / 3.0)

    def compute_partials(self, inputs, partials):
        CT, s, r = inputs['CT_sigma'], inputs['sigma'][0], inputs['vi_IGE_OGE']
        gap, dgap_dr, dgap_dDv = self._gap(inputs)
        root = np.sqrt(s / 2.0)
        dCQ = -CT ** 1.5 * root * gap

        partials['dCQ_sigma', 'CT_sigma'] = 1.5 * dCQ / CT
        partials['dCQ_sigma', 'sigma'] = (0.5 * dCQ / s).reshape(-1, 1)
        partials['dCQ_sigma', 'vi_IGE_OGE'] = -CT ** 1.5 * root * dgap_dr
        if dgap_dDv is not None:
            partials['dCQ_sigma', 'Dv_GW'] = -CT ** 1.5 * root * dgap_dDv
        partials['T_ratio_IGE', 'vi_IGE_OGE'] = -2.0 / 3.0 * r ** (-5.0 / 3.0)
