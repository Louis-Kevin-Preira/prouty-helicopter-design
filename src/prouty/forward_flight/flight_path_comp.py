"""
FlightPathComp -- flight path angle, rate of climb, and ideal climb power.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 194-195 (climb) and p. 198 (autorotative descent).

    sin(gamma) = (R/C) / (60 V)          V = mu Omega R, R/C in ft/min
    R/D        = -60 V sin(gamma)
    hp_ideal   = (R/C) G.W. / 33,000

The same relation is used in both directions, hence the mode option:

    'from_rate'   R/C given -> gamma. Climb and descent at a prescribed rate,
                  which is how Table 3.3's climb column is set up.
    'from_angle'  gamma given -> R/C. The autorotation sweep of p. 196 works
                  this way, and so does the dive of Table 3.5 case 7.

gamma is positive in climb and negative in descent, matching the sign
convention of TppAngleComp(form='forces'), where the G.W. sin(gamma) term
adds to the drag in a climb and subtracts from it in a descent.

hp_ideal is the power a perfect machine would need for the climb, G.W. times
vertical speed. Comparing it with the actual power increment gives Prouty's
climb efficiency (p. 195): Table 3.3 climbs at 1000 ft/min for 1760 - 1097 =
663 hp, against an ideal 606 hp, so 91 %. The missing 9 % is mostly the
increased fuselage download -- L_F goes from -746 to -1228 lb -- with a
smaller part from the tail rotor H-force. Prouty suggests using that
efficiency to scale rough climb estimates at other weights and speeds near
115 knots.

Note V is rebuilt as mu (Omega R) rather than taken as an input, for the same
reason as in DynamicPressureComp: mu is the independent variable of this
chapter and there must be a single source for the flight speed.

    mu, V_tip, GW, R_C     --> 'from_rate'  --> gamma_fp, hp_ideal
    mu, V_tip, GW, gamma_fp --> 'from_angle' --> R_C, hp_ideal
"""

import numpy as np
import openmdao.api as om

FT_LB_PER_MIN_PER_HP = 33000.0


class FlightPathComp(om.ExplicitComponent):
    """Flight path angle and rate of climb, p. 194-195."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('mode', values=('from_rate', 'from_angle'),
                             default='from_rate')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('mu', shape=(nn,), desc='tip speed ratio')
        self.add_input('V_tip', val=650.0, units='ft/s', desc='tip speed')
        self.add_input('GW', val=20000.0, units='lbf', desc='gross weight')

        self.solve_for = ('gamma_fp' if self.options['mode'] == 'from_rate'
                          else 'R_C')
        self.given = 'R_C' if self.solve_for == 'gamma_fp' else 'gamma_fp'
        units = {'gamma_fp': 'rad', 'R_C': 'ft/min'}

        self.add_input(self.given, shape=(nn,), val=0.0, units=units[self.given],
                       desc='rate of climb' if self.given == 'R_C'
                            else 'flight path angle, positive in climb')
        self.add_output(self.solve_for, shape=(nn,), units=units[self.solve_for])
        self.add_output('hp_ideal', shape=(nn,), units='hp',
                        desc='G.W. times vertical speed, p. 195')

        for name in ('mu', self.given):
            self.declare_partials([self.solve_for, 'hp_ideal'], name,
                                  rows=ar, cols=ar)
        self.declare_partials([self.solve_for, 'hp_ideal'], 'V_tip',
                              rows=ar, cols=zeros)
        self.declare_partials('hp_ideal', 'GW', rows=ar, cols=zeros)

    def _speed(self, inputs):
        V = inputs['mu'] * inputs['V_tip'][0]
        if np.any(np.real(V) <= 0.0):
            raise om.AnalysisError('FlightPathComp: mu and V_tip must be '
                                   f'positive, got V = {np.real(V)}.')
        return V

    def compute(self, inputs, outputs):
        V = self._speed(inputs)

        if self.solve_for == 'gamma_fp':
            R_C = inputs['R_C']
            s = R_C / (60.0 * V)
            if np.any(np.abs(np.real(s)) >= 1.0):
                raise om.AnalysisError(
                    'FlightPathComp: the requested rate of climb exceeds the '
                    f'flight speed, sin(gamma) = {np.real(s)}.')
            outputs['gamma_fp'] = np.arcsin(s)
        else:
            R_C = 60.0 * V * np.sin(inputs['gamma_fp'])
            outputs['R_C'] = R_C

        outputs['hp_ideal'] = R_C * inputs['GW'][0] / FT_LB_PER_MIN_PER_HP

    def compute_partials(self, inputs, partials):
        out, given = self.solve_for, self.given
        mu, V_tip, GW = inputs['mu'], inputs['V_tip'][0], inputs['GW'][0]
        V = self._speed(inputs)

        if out == 'gamma_fp':
            R_C = inputs['R_C']
            s = R_C / (60.0 * V)
            root = np.sqrt(1.0 - s ** 2)

            partials[out, 'R_C'] = 1.0 / (60.0 * V * root)
            partials[out, 'mu'] = -s / (mu * root)
            partials[out, 'V_tip'] = -s / (V_tip * root)

            dR_C = {'R_C': np.ones_like(R_C), 'mu': np.zeros_like(R_C),
                    'V_tip': np.zeros_like(R_C)}
        else:
            gamma = inputs['gamma_fp']
            R_C = 60.0 * V * np.sin(gamma)

            partials[out, 'gamma_fp'] = 60.0 * V * np.cos(gamma)
            partials[out, 'mu'] = R_C / mu
            partials[out, 'V_tip'] = R_C / V_tip

            dR_C = {'gamma_fp': partials[out, 'gamma_fp'],
                    'mu': R_C / mu, 'V_tip': R_C / V_tip}

        scale = GW / FT_LB_PER_MIN_PER_HP
        for name, value in dR_C.items():
            partials['hp_ideal', name] = scale * value
        partials['hp_ideal', 'GW'] = R_C / FT_LB_PER_MIN_PER_HP
