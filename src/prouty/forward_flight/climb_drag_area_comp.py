"""
ClimbDragAreaComp -- the climb folded into an equivalent flat plate area.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 240 (Table 3.5, case 6, step f) and p. 242 (step dd).

    f_climb = f + H_T/q + (G.W./q) tan(gamma_c)

Climbing costs the rotor the same as extra drag would, so the chart method
puts the weight component along the flight path into the abscissa rather than
into the force balance: step g then reads
f_climb/A_b + (sigma/mu^4)(C_T/sigma)^2 exactly as level flight reads f/A_b.

Note where the climb does NOT appear. Prouty's chart bookkeeping carries the
weight component in the INFLOW, through f_climb and hence lambda', and leaves
the thrust magnitude alone: p. 242 step bb computes
T_M = sqrt[(G.W. - L_F)^2 + (D_F + H_M + H_T)^2] with no G.W. sin(gamma) term.
That is the same choice Table 3.3's climb column makes, and the reason
TppAngleComp runs 1 % high against it -- see validation_notes section 1.

tan, not sin. p. 240 writes G.W. sin(gamma_c)/q and p. 242 writes
(G.W./q) tan(gamma_c); the worked example uses tan, 20,000 tan(4.9 deg)/45.2 =
37.9 ft^2. At 4.9 deg the two differ by 0.3 %, so nothing turns on it, but the
option records the disagreement rather than hiding it.

H_M is absent, and that is Prouty's, not an omission here. The main rotor
H-force is handled separately in the chart procedure (step x), so f_climb
covers the fuselage, the tail rotor and gravity only. A cross-check of
lambda' through ChartParameterComp will therefore not close exactly -- measured
at 18 % on the worked climb -- and the gap is H_M.

Worked check, p. 242: f = 20.6 ft^2, H_T = 139 lb, q = 45.2 psf,
G.W. = 20,000 lb, gamma_c = 4.9 deg gives f_climb = 61.6 ft^2.

    f, H_T, q, GW, gamma_fp --> f_climb
"""

import numpy as np
import openmdao.api as om


class ClimbDragAreaComp(om.ExplicitComponent):
    """Equivalent flat plate area including the climb, p. 240 and 242."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('weight_form', values=('tan', 'sin'),
                             default='tan',
                             desc="p. 242 writes tan, p. 240 writes sin; the "
                                  'worked example uses tan')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('f', shape=(nn,), units='ft**2',
                       desc='fuselage equivalent flat plate area')
        self.add_input('H_T', shape=(nn,), val=0.0, units='lbf')
        self.add_input('q', shape=(nn,), units='lbf/ft**2')
        self.add_input('gamma_fp', shape=(nn,), units='rad',
                       desc='flight path angle, positive in climb')
        self.add_input('GW', val=20000.0, units='lbf')

        self.add_output('f_climb', shape=(nn,), units='ft**2')

        for name in ('f', 'H_T', 'q', 'gamma_fp'):
            self.declare_partials('f_climb', name, rows=ar, cols=ar)
        self.declare_partials('f_climb', 'GW', rows=ar, cols=zeros)

    def _slope(self, gamma):
        if self.options['weight_form'] == 'tan':
            return np.tan(gamma), 1.0 / np.cos(gamma) ** 2
        return np.sin(gamma), np.cos(gamma)

    def compute(self, inputs, outputs):
        slope, _ = self._slope(inputs['gamma_fp'])
        outputs['f_climb'] = (inputs['f']
                              + (inputs['H_T'] + inputs['GW'][0] * slope)
                              / inputs['q'])

    def compute_partials(self, inputs, partials):
        q, GW = inputs['q'], inputs['GW'][0]
        slope, d_slope = self._slope(inputs['gamma_fp'])

        partials['f_climb', 'f'] = 1.0
        partials['f_climb', 'H_T'] = 1.0 / q
        partials['f_climb', 'GW'] = slope / q
        partials['f_climb', 'gamma_fp'] = GW * d_slope / q
        partials['f_climb', 'q'] = -(inputs['H_T'] + GW * slope) / q ** 2
