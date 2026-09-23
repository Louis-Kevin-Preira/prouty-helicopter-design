"""
TipMachComp -- characteristic rotor Mach numbers in forward flight.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 232 (Table 3.5, case 1, step g) and p. 175.

    M_tip = V_tip / V_son                       hovering tip, reference
    M_190 = (1 + mu) V_tip / V_son              advancing tip at psi = 90 deg
    M_075 = 0.75 V_tip / V_son                  reference for mean drag, p. 175

M_190 is the compressibility parameter of the whole chapter: it is the third
input of the isolated rotor charts (built at M_190 = 0.7, p. 229) and the
quantity compared against the drag rise Mach number M_dd3 in Table 3.5.

M_075 is Prouty's convention for picking the average blade element drag
coefficient in the closed-form torque equation: "a Mach number corresponding to
75 % of the tip speed" (p. 175). It is 0.75 M_tip, not the Mach number at the
0.75 R station in forward flight, which would be azimuth dependent.

    mu, V_tip, V_son --> TipMachComp --> M_tip, M_190, M_075 (nn,)

All three outputs carry shape (nn,) even though M_tip and M_075 do not depend
on mu. Broadcasting here costs nothing and lets them feed the vectorised
airfoil components of Chapter 6 without an intermediate broadcast component.
"""

import numpy as np
import openmdao.api as om

M_REF = 0.75            # fraction of tip speed for the mean drag coefficient, p. 175


class TipMachComp(om.ExplicitComponent):
    """Hovering, advancing tip and reference Mach numbers."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('mu', shape=(nn,), desc='tip speed ratio')
        self.add_input('V_tip', val=650.0, units='ft/s', desc='tip speed Omega R')
        self.add_input('V_son', val=1116.0, units='ft/s', desc='speed of sound')

        self.add_output('M_tip', shape=(nn,), desc='hovering tip Mach number')
        self.add_output('M_190', shape=(nn,), desc='advancing tip Mach number')
        self.add_output('M_075', shape=(nn,), desc='Mach at 75 % of tip speed')

        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)
        self.declare_partials('M_190', 'mu', rows=ar, cols=ar)
        for out in ('M_tip', 'M_190', 'M_075'):
            self.declare_partials(out, 'V_tip', rows=ar, cols=zeros)
            self.declare_partials(out, 'V_son', rows=ar, cols=zeros)

    def compute(self, inputs, outputs):
        V_son = inputs['V_son'][0]
        if V_son <= 0.0:
            raise om.AnalysisError('TipMachComp: V_son must be positive, '
                                   f'got {V_son}.')

        M_tip = inputs['V_tip'][0] / V_son
        outputs['M_tip'] = M_tip
        outputs['M_190'] = (1.0 + inputs['mu']) * M_tip
        outputs['M_075'] = M_REF * M_tip

    def compute_partials(self, inputs, partials):
        mu = inputs['mu']
        V_tip = inputs['V_tip'][0]
        V_son = inputs['V_son'][0]
        M_tip = V_tip / V_son

        partials['M_190', 'mu'] = M_tip

        partials['M_tip', 'V_tip'] = 1.0 / V_son
        partials['M_190', 'V_tip'] = (1.0 + mu) / V_son
        partials['M_075', 'V_tip'] = M_REF / V_son

        partials['M_tip', 'V_son'] = -M_tip / V_son
        partials['M_190', 'V_son'] = -(1.0 + mu) * M_tip / V_son
        partials['M_075', 'V_son'] = -M_REF * M_tip / V_son
