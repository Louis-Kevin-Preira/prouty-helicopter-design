"""
WakeContractionComp -- empirical power correction for tip vortex interference.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, step 19, p. 72; Figure 1.34, "Error Due to Tip Vortex Interference",
p. 58.

The blade element calculation assumes each section sees a smooth inflow. It
does not: the wake contracts and the tip vortex shed by one blade passes close
under the next, raising the local inflow near the tip and costing power the
calculation cannot see. Figure 1.34 collects that error from 4.5 ft model rotor
tests and a full scale CH-53A rotor, and fits it as a single curve:

    measured power / calculated power = f( (D.L.)(C_T/sigma) )

The abscissa pairs the disc loading, which sets the strength of the wake, with
the blade loading, which sets how much of the disc the blades actually sweep.
The factor crosses unity near 0.37: below it the calculation is pessimistic,
above it optimistic, reaching about 8% optimistic at the right of the chart.

The table is digitised from the figure and interpolated with an Akima spline,
which stays monotone here and keeps the derivative continuous. Validation: at
0.61 it returns 1.049, against the 1.05 printed in the sample calculation of
Figure 1.45, p. 76.

Accuracy. The reading is good to about +/- 0.005, but that understates the real
uncertainty: the test points on Figure 1.34 scatter roughly +/- 0.03 about the
fitted curve, so the curve is a correlation through a cloud, not a law. Treat
the last digit of any power computed with it as decoration.

    DL, CT_sigma --> WakeContractionComp --> DL_CT_sigma, power_factor
"""

import warnings

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

X_TABLE = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                    0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20])
F_TABLE = np.array([0.955, 0.970, 0.982, 0.990, 0.996, 1.002, 1.012, 1.022,
                    1.032, 1.041, 1.048, 1.058, 1.065, 1.070, 1.074, 1.080])


class WakeContractionComp(om.ExplicitComponent):
    """Measured over calculated power, from Figure 1.34."""

    def setup(self):
        self._warned = False
        self._interp = InterpND(method='akima', points=X_TABLE,
                                values=F_TABLE, extrapolate=True)

        self.add_input('DL', val=7.2, units='lbf/ft**2', desc='disc loading')
        self.add_input('CT_sigma', val=0.0846, desc='blade loading C_T/sigma')

        self.add_output('DL_CT_sigma', val=0.61,
                        desc='abscissa of Figure 1.34')
        self.add_output('power_factor', val=1.05,
                        desc='measured power over calculated power')

        self.declare_partials('DL_CT_sigma', ['DL', 'CT_sigma'])
        self.declare_partials('power_factor', ['DL', 'CT_sigma'])

    def _lookup(self, x):
        if not self._warned and not (X_TABLE[0] <= np.real(x) <= X_TABLE[-1]):
            self._warned = True
            warnings.warn(f'(D.L.)(C_T/sigma) = {np.real(x):.3f} is outside the '
                          f'{X_TABLE[0]} to {X_TABLE[-1]} range of Figure 1.34; '
                          'the spline is extrapolating.')
        val, der = self._interp.interpolate(np.atleast_1d(np.real(x)),
                                            compute_derivative=True)
        return float(val[0]), float(np.ravel(der)[0])

    def compute(self, inputs, outputs):
        x = inputs['DL'][0] * inputs['CT_sigma'][0]
        outputs['DL_CT_sigma'] = x
        outputs['power_factor'] = self._lookup(x)[0]

    def compute_partials(self, inputs, partials):
        dl, cts = inputs['DL'][0], inputs['CT_sigma'][0]
        _, df = self._lookup(dl * cts)

        partials['DL_CT_sigma', 'DL'] = cts
        partials['DL_CT_sigma', 'CT_sigma'] = dl
        partials['power_factor', 'DL'] = df * cts
        partials['power_factor', 'CT_sigma'] = df * dl
