"""
FuselageDragComp -- equivalent flat plate area of the basic fuselage.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Parasite Drag in Forward Flight": surface imperfections p. 291,
Figure 4.17 pp. 292-294, procedure and example pp. 306-308.

    f_F = A_F C_DF                  A_F frontal area, C_DF on frontal area

Figure 4.17 gives the minimum skin friction and form drag of a streamline
fuselage against its fineness ratio l/d (R.N. = 7e7, a 45 ft fuselage at
150 kt), and the measured drag of ten airplanes and helicopters above it.
For a new design p. 294 advises reading the figure "at a level of cleanliness
corresponding to that for one of the known aircraft". Following p. 291, where
imperfections scale the computed friction drag, the level is a factor on the
minimum curve (C4-16):

    C_DF = k C_D,min(l/d),   k = C_D,ref / C_D,min(l/d ref)

mode
    'reference'   k from the aircraft chosen by the option reference
    'input'       C_DF given directly

Example helicopter, p. 306: A_F = 74 ft^2, l/d = 7, C_DF = 0.078, f_F = 5.8
ft^2 (between the P-51 and the L-286 levels: 0.075 and 0.080).

Digitization: minimum curve traced on a clean copy, quartic fit beyond l/d = 2.5
(rms 0.0014), row scan on the steep inboard branch; aircraft read as symbol
centres. Akima in l/d, held outside 1-10.2 with a warning.

    A_F, l_d, [C_DF] --> C_DF, f_F
"""

import warnings

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

_L_D = np.array([1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.50,
                 4.00, 4.50, 5.00, 5.50, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 9.50, 10.00,
                 10.20])
_CD_MIN = np.array([0.1020, 0.0839, 0.0657, 0.0595, 0.0548, 0.0513, 0.0440, 0.0394, 0.0375,
                    0.0356, 0.0363, 0.0370, 0.0383, 0.0398, 0.0414, 0.0431, 0.0450, 0.0472,
                    0.0497, 0.0524, 0.0555, 0.0589, 0.0627, 0.0670, 0.0716, 0.0767, 0.0788])
_CHART = InterpND(method='akima', points=_L_D, values=_CD_MIN, extrapolate=False)

# Figure 4.17: (fineness ratio, C_D on frontal area) of the known aircraft
REFERENCE_AIRCRAFT = {
    'OH-6A': (2.48, 0.0708), 'UH-1': (4.94, 0.0804), 'CH-47': (6.00, 0.1064),
    'L-286': (6.46, 0.0755), 'P-80': (6.95, 0.0749), 'DC-3': (7.02, 0.0813),
    'P-51': (7.13, 0.0764), 'F-104': (8.96, 0.0772), 'B-29': (9.16, 0.0933),
    'B-17': (10.17, 0.1077),
}


def minimum_fuselage_drag(l_d):
    """Figure 4.17 minimum skin friction and form drag, and its slope; held outside."""
    x = np.clip(np.real(l_d), _L_D[0], _L_D[-1])
    v, dv = _CHART.interpolate(np.atleast_1d(x), compute_derivative=True)
    inside = _L_D[0] < np.real(l_d) < _L_D[-1]
    return v[0], (dv.ravel()[0] if inside else 0.0)


class FuselageDragComp(om.ExplicitComponent):
    """Figure 4.17 fuselage drag at zero lift, p. 306."""

    def initialize(self):
        self.options.declare('mode', default='reference', values=('reference', 'input'))
        self.options.declare('reference', default='L-286', values=tuple(REFERENCE_AIRCRAFT),
                             desc='aircraft whose cleanliness level is used')

    def setup(self):
        self.add_input('A_F', val=1.0, units='ft**2', desc='fuselage frontal area')
        self.add_output('f_F', val=0.0, units='ft**2', desc='fuselage equivalent flat plate area')

        if self.options['mode'] == 'reference':
            self.add_input('l_d', val=5.0, desc='fuselage fineness ratio')
            self.add_output('C_DF', val=0.05, desc='drag coefficient on frontal area')
            l_ref, cd_ref = REFERENCE_AIRCRAFT[self.options['reference']]
            self._k = cd_ref / minimum_fuselage_drag(l_ref)[0]
            self.declare_partials('C_DF', 'l_d')
            self.declare_partials('f_F', ['A_F', 'l_d'])
        else:
            self.add_input('C_DF', val=0.05, desc='drag coefficient on frontal area')
            self.declare_partials('f_F', ['A_F', 'C_DF'])

    def compute(self, inputs, outputs):
        if self.options['mode'] == 'input':
            outputs['f_F'] = inputs['A_F'] * inputs['C_DF']
            return
        l_d = inputs['l_d'][0]
        if not _L_D[0] <= np.real(l_d) <= _L_D[-1]:
            warnings.warn(f'l/d = {np.real(l_d):.2f} outside Figure 4.17 (1-10.2); edge held.',
                          stacklevel=2)
        C_DF = self._k * minimum_fuselage_drag(l_d)[0]
        outputs['C_DF'] = C_DF
        outputs['f_F'] = inputs['A_F'] * C_DF

    def compute_partials(self, inputs, partials):
        if self.options['mode'] == 'input':
            partials['f_F', 'A_F'] = inputs['C_DF']
            partials['f_F', 'C_DF'] = inputs['A_F']
            return
        cd_min, slope = minimum_fuselage_drag(inputs['l_d'][0])
        partials['C_DF', 'l_d'] = self._k * slope
        partials['f_F', 'A_F'] = self._k * cd_min
        partials['f_F', 'l_d'] = inputs['A_F'] * self._k * slope
