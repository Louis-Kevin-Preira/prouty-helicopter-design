"""
LandingGearDragComp -- equivalent flat plate area of a non-retracting landing gear.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Bluff Body Drag" p. 301, Figure 4.26 p. 303 (from reference 4.2);
procedure and example p. 307.

Figure 4.26 gives drag coefficients for wheels, wheels with struts, nose and
tail wheels, and skids (LANDING_GEAR_CD, on the area stated for each), plus a
curve for a nose or tail wheel: C_D on wheel frontal area against e/d, the
strut length over wheel diameter, for a round or a faired strut. The wheel
partly shields the strut, so C_D rises to about 0.5 at e/d = 1 and then grows
again as the strut emerges.

    f_gear = A_gear C_D

Example helicopter, p. 307: main gear A = 4 ft^2, C_D = 0.3, f = 1.2 ft^2;
nose gear A = 1.5 ft^2, e/d = 3.4/2 = 1.7, C_D = 0.56, f = 0.8 ft^2.

mode
    'catalog'      C_D given directly (a value of LANDING_GEAR_CD)
    'nose_wheel'   C_D read on the e/d curve, strut = 'round' or 'faired'

Digitization (C4-21): clean copy, pixel columns, curves identical below
e/d = 1.25 where they overlap. At e/d = 1.7 the round strut reads 0.545,
where the book reads 0.56. Akima; held outside 0-3 with a warning.

    A_gear, [C_D | e, d] --> C_D, f_gear
"""

import warnings

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

# Figure 4.26 p. 303, with the area each coefficient is based on
LANDING_GEAR_CD = {
    'wheel, narrow (b x d)': 0.12, 'wheel, medium (b x d)': 0.25, 'wheel, wide (b x d)': 0.15,
    'wheels with struts, braced (frontal)': 0.55, 'wheels with struts, faired (frontal)': 0.36,
    'wheel with cantilever strut (frontal)': 0.25,
    'nose wheel installation, unfaired (frontal)': 0.58,
    'nose wheel installation, faired (frontal)': 0.27,
    'skid, tubular': 1.01, 'skid, faired': 0.40,
}

_E_D = np.round(np.arange(0.0, 3.001, 0.25), 2)
_CD_STRUT = {
    'round': np.array([0.000, 0.040, 0.120, 0.270, 0.461, 0.506, 0.526, 0.552, 0.594, 0.640,
                       0.693, 0.757, 0.812]),
    'faired': np.array([0.000, 0.040, 0.120, 0.270, 0.461, 0.496, 0.437, 0.422, 0.431, 0.453,
                        0.489, 0.520, 0.565]),
}
_CHARTS = {k: InterpND(method='akima', points=_E_D, values=v, extrapolate=False)
           for k, v in _CD_STRUT.items()}


def nose_wheel_drag_coefficient(e_d, strut='round'):
    """Figure 4.26 nose or tail wheel C_D and its slope; held outside 0 <= e/d <= 3."""
    x = np.clip(np.real(e_d), _E_D[0], _E_D[-1])
    v, dv = _CHARTS[strut].interpolate(np.atleast_1d(x), compute_derivative=True)
    inside = _E_D[0] < np.real(e_d) < _E_D[-1]
    return v[0], (dv.ravel()[0] if inside else 0.0)


class LandingGearDragComp(om.ExplicitComponent):
    """Landing gear drag, p. 307."""

    def initialize(self):
        self.options.declare('mode', default='catalog', values=('catalog', 'nose_wheel'))
        self.options.declare('strut', default='round', values=tuple(_CD_STRUT))

    def setup(self):
        self.add_input('A_gear', val=1.0, units='ft**2', desc='gear frontal area')
        self.add_output('f_gear', val=0.0, units='ft**2', desc='gear equivalent flat plate area')

        if self.options['mode'] == 'catalog':
            self.add_input('C_D', val=0.3, desc='drag coefficient, Figure 4.26')
            self.declare_partials('f_gear', ['A_gear', 'C_D'])
        else:
            self.add_input('e', val=1.0, units='ft', desc='strut length')
            self.add_input('d', val=1.0, units='ft', desc='wheel diameter')
            self.add_output('C_D', val=0.5, desc='drag coefficient on wheel frontal area')
            self.declare_partials('C_D', ['e', 'd'])
            self.declare_partials('f_gear', ['A_gear', 'e', 'd'])

    def compute(self, inputs, outputs):
        if self.options['mode'] == 'catalog':
            outputs['f_gear'] = inputs['A_gear'] * inputs['C_D']
            return
        e_d = inputs['e'][0] / inputs['d'][0]
        if not _E_D[0] <= np.real(e_d) <= _E_D[-1]:
            warnings.warn(f'e/d = {np.real(e_d):.2f} outside Figure 4.26 (0-3); edge held.',
                          stacklevel=2)
        C_D = nose_wheel_drag_coefficient(e_d, self.options['strut'])[0]
        outputs['C_D'] = C_D
        outputs['f_gear'] = inputs['A_gear'] * C_D

    def compute_partials(self, inputs, partials):
        A = inputs['A_gear'][0]
        if self.options['mode'] == 'catalog':
            partials['f_gear', 'A_gear'] = inputs['C_D']
            partials['f_gear', 'C_D'] = A
            return
        e, d = inputs['e'][0], inputs['d'][0]
        C_D, slope = nose_wheel_drag_coefficient(e / d, self.options['strut'])
        partials['C_D', 'e'], partials['C_D', 'd'] = slope / d, -slope * e / d ** 2
        partials['f_gear', 'A_gear'] = C_D
        partials['f_gear', 'e'] = A * slope / d
        partials['f_gear', 'd'] = -A * slope * e / d ** 2
