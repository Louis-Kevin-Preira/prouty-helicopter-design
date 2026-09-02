"""
WakeRotationComp -- induced torque increment from the rotation of the wake.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, step 16, p. 72; Figure 1.29, "Power Losses Due to Rotation of Wake",
p. 52.

Simple momentum theory ignores the swirl the rotor leaves in its wake. The
energy in that swirl is real power the rotor must supply, and Figure 1.29 gives
it as a fraction of the thrust induced power, as a function of thrust
coefficient:

    C_Qi_rot = swirl_ratio(C_T) . C_Qi

The figure carries three curves. 'approximate' is the method consistent with
the rest of this calculation and is the default; 'wu' and 'durand_glauert' are
the two optimum solutions, kept for comparison. All three pass through the
origin and are nearly linear, the optimum solutions bracketing the approximate
one within about 15% over the useful range.

The tables were digitised from the figure, accurate to roughly +/- 0.002 in the
ordinate. The reading is anchored on the book's own numbers: at C_T = 0.00719
the approximate curve returns 0.017, the value printed in Figure 1.45, p. 76.

Interpolation is an Akima spline, which keeps the model differentiable through
the whole thrust range. It does not support complex step, so this component is
checked against finite differences.

    CT, CQi --> WakeRotationComp --> swirl_ratio, CQi_rot
"""

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

CT_TABLE = np.array([0.000, 0.005, 0.010, 0.015, 0.020, 0.025,
                     0.030, 0.035, 0.040, 0.045, 0.050])

CURVES = {
    'approximate': np.array([0.0000, 0.0125, 0.0235, 0.0345, 0.0450, 0.0560,
                             0.0665, 0.0775, 0.0880, 0.0985, 0.1090]),
    'wu': np.array([0.0000, 0.0155, 0.0285, 0.0400, 0.0510, 0.0615,
                    0.0715, 0.0815, 0.0915, 0.1010, 0.1105]),
    'durand_glauert': np.array([0.0000, 0.0105, 0.0200, 0.0290, 0.0380, 0.0465,
                                0.0555, 0.0645, 0.0740, 0.0835, 0.0930]),
}


class WakeRotationComp(om.ExplicitComponent):
    """Swirl induced power as a fraction of thrust induced power."""

    def initialize(self):
        self.options.declare('curve', values=tuple(CURVES),
                             default='approximate',
                             desc='which curve of Figure 1.29 to use')

    def setup(self):
        self._interp = InterpND(method='akima', points=CT_TABLE,
                               values=CURVES[self.options["curve"]],
                               extrapolate=True)

        self.add_input('CT', val=0.00719, desc='thrust coefficient, step 11')
        self.add_input('CQi', val=0.000459,
                       desc='induced torque coefficient, step 15')

        self.add_output('swirl_ratio', val=0.017,
                        desc='swirl induced power / thrust induced power')
        self.add_output('CQi_rot', val=0.0,
                        desc='induced torque increment from wake rotation')

        self.declare_partials('swirl_ratio', 'CT')
        self.declare_partials('CQi_rot', ['CT', 'CQi'])

    def _ratio(self, ct):
        val, deriv = self._interp.interpolate(np.atleast_1d(np.real(ct)),
                                              compute_derivative=True)
        return float(val[0]), float(np.ravel(deriv)[0])

    def compute(self, inputs, outputs):
        ratio, _ = self._ratio(inputs['CT'][0])
        outputs['swirl_ratio'] = ratio
        outputs['CQi_rot'] = ratio * inputs['CQi']

    def compute_partials(self, inputs, partials):
        ratio, dratio = self._ratio(inputs['CT'][0])
        partials['swirl_ratio', 'CT'] = dratio
        partials['CQi_rot', 'CT'] = dratio * inputs['CQi'][0]
        partials['CQi_rot', 'CQi'] = ratio
