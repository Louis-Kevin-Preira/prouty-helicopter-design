"""
PropulsiveBalanceComp -- what has to push the helicopter along.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 244 (Table 3.5, case 7, steps u-w) and p. 246 (case 8, step u).

    F = q (f + f_M) + H_T

One equation, two readings, and Prouty writes the note himself at the end of
case 8: "note similarity to results of previous case". Both cases give
F = 2,764 lb on the same flight condition.

    dive        G.W. sin(gamma_D) = F,  R/D = 60 mu (Omega R) sin(gamma_D)
    auxiliary   T_aux = F

The rotor cannot always propel itself. In level flight it does -- f_M comes out
negative and F is zero by construction -- but at low collective, or when a
propeller is doing the pushing, f_M turns positive and something must supply
F. Case 7 lets gravity do it and asks how steeply the helicopter must descend;
case 8 puts a propeller there and asks how much thrust it needs. The number is
the same because the question is the same.

Sign conventions, which differ from FlightPathComp on purpose. gamma_D and R_D
are POSITIVE IN DESCENT here, following p. 244 -- "find angle of dive", "find
rate of descent". FlightPathComp uses gamma positive in climb. gamma_fp is
emitted as -gamma_D so the two can be wired together without a sign flip in
the middle of a group, which is exactly the sort of thing that produces a model
that converges to the wrong answer.

Worked check, p. 244: q = 45.2 psf, f = 19.5 ft^2, f_M = 41.3 ft^2, H_T = 16 lb
gives F = 2,764 lb, gamma_D = 7.9 deg and R/D = 1,608 ft/min at
G.W. = 20,000 lb, mu = 0.3, Omega R = 650 ft/s.

    q, f, f_M, H_T [, GW, mu, V_tip] --> F_prop [, gamma_D, R_D | T_aux]
"""

import numpy as np
import openmdao.api as om

SECONDS_PER_MINUTE = 60.0


class PropulsiveBalanceComp(om.ExplicitComponent):
    """Propulsive force balance, read as a dive angle or a thrust, p. 244-246."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('mode', values=('dive', 'auxiliary'),
                             default='dive',
                             desc="'dive' lets gravity supply the force, "
                                  "'auxiliary' a propeller")

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)
        dive = self.options['mode'] == 'dive'

        self.add_input('q', shape=(nn,), units='lbf/ft**2',
                       desc='dynamic pressure')
        self.add_input('f', shape=(nn,), units='ft**2',
                       desc='fuselage equivalent flat plate area')
        self.add_input('f_M', shape=(nn,), units='ft**2',
                       desc="rotor's equivalent flat plate area, p. 244")
        self.add_input('H_T', shape=(nn,), val=0.0, units='lbf',
                       desc='tail rotor H-force')

        self.add_output('F_prop', shape=(nn,), units='lbf',
                        desc='propulsive force the rotor cannot supply')

        for name in ('q', 'f', 'f_M', 'H_T'):
            self.declare_partials('F_prop', name, rows=ar, cols=ar)

        if dive:
            self.add_input('GW', val=20000.0, units='lbf',
                           desc='gross weight')
            self.add_input('mu', shape=(nn,))
            self.add_input('V_tip', val=650.0, units='ft/s')

            self.add_output('gamma_D', shape=(nn,), units='rad',
                            desc='dive angle, positive in descent')
            self.add_output('gamma_fp', shape=(nn,), units='rad',
                            desc='flight path angle, positive in climb')
            self.add_output('R_D', shape=(nn,), units='ft/min',
                            desc='rate of descent, positive down')

            for out in ('gamma_D', 'gamma_fp', 'R_D'):
                for name in ('q', 'f', 'f_M', 'H_T', 'mu'):
                    self.declare_partials(out, name, rows=ar, cols=ar)
                for name in ('GW', 'V_tip'):
                    self.declare_partials(out, name, rows=ar, cols=zeros)
        else:
            self.add_output('T_aux', shape=(nn,), units='lbf',
                            desc='auxiliary propulsive thrust required')
            for name in ('q', 'f', 'f_M', 'H_T'):
                self.declare_partials('T_aux', name, rows=ar, cols=ar)

    def _force(self, inputs):
        return inputs['q'] * (inputs['f'] + inputs['f_M']) + inputs['H_T']

    def compute(self, inputs, outputs):
        force = self._force(inputs)
        outputs['F_prop'] = force

        if self.options['mode'] == 'auxiliary':
            outputs['T_aux'] = force
            return

        sine = force / inputs['GW'][0]
        if np.any(np.abs(np.real(sine)) >= 1.0):
            raise om.AnalysisError(
                'PropulsiveBalanceComp: the propulsive force exceeds the '
                f'weight, sin(gamma_D) = {np.real(sine)}; no dive angle exists.')

        outputs['gamma_D'] = np.arcsin(sine)
        outputs['gamma_fp'] = -np.arcsin(sine)
        outputs['R_D'] = (SECONDS_PER_MINUTE * inputs['mu']
                          * inputs['V_tip'][0] * sine)

    def compute_partials(self, inputs, partials):
        q, f, f_M = inputs['q'], inputs['f'], inputs['f_M']
        force = self._force(inputs)

        d_force = {'q': f + f_M, 'f': q, 'f_M': q, 'H_T': np.ones_like(q)}
        for name, value in d_force.items():
            partials['F_prop', name] = value

        if self.options['mode'] == 'auxiliary':
            for name, value in d_force.items():
                partials['T_aux', name] = value
            return

        GW, mu, V_tip = inputs['GW'][0], inputs['mu'], inputs['V_tip'][0]
        sine = force / GW
        root = np.sqrt(1.0 - sine ** 2)
        speed = SECONDS_PER_MINUTE * mu * V_tip

        for name, value in d_force.items():
            partials['gamma_D', name] = value / (GW * root)
            partials['gamma_fp', name] = -value / (GW * root)
            partials['R_D', name] = speed * value / GW

        partials['gamma_D', 'GW'] = -sine / (GW * root)
        partials['gamma_fp', 'GW'] = sine / (GW * root)
        partials['R_D', 'GW'] = -speed * sine / GW

        partials['gamma_D', 'mu'] = 0.0
        partials['gamma_fp', 'mu'] = 0.0
        partials['R_D', 'mu'] = SECONDS_PER_MINUTE * V_tip * sine

        partials['gamma_D', 'V_tip'] = 0.0
        partials['gamma_fp', 'V_tip'] = 0.0
        partials['R_D', 'V_tip'] = SECONDS_PER_MINUTE * mu * sine
