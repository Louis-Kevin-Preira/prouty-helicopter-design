"""
ThrustCoefComp -- thrust coefficient divided by solidity.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 168:

    C_T/sigma = T / (rho A_b (Omega R)^2)

C_T/sigma, not C_T, is the working variable of the whole chapter: it is the
abscissa of every isolated rotor chart (p. 254-271) and the quantity that sets
blade stall, since it is proportional to the mean blade element lift
coefficient, C_T/sigma = cl_bar (1 + 1.5 mu^2) / 6 (p. 168).

A_b is the blade area, b c R, equal to sigma times the disc area. Prouty
tabulates it directly in Appendix A (240 ft^2 for the example helicopter main
rotor). The denominator is therefore a constant of the rotor at a given
density, which is why Table 3.5 works with pre-computed numbers:

    rho_0 A_b (Omega R)^2 = 241,100  main rotor   (p. 235)
    rho_0 A_b (Omega R)^2 =  19,550  tail rotor   (p. 236)

and divides by the density ratio rather than by rho.

Note that Prouty writes C_T/sigma = G.W. / rho A_b (Omega R)^2 as a first
approximation only (p. 168). The trim loop of p. 192 uses the true rotor
thrust, which differs from the gross weight by the fuselage lift and by the
in-plane forces -- 3 % at mu = 0.3, more in a climb. Feed T, not G.W.

    T, rho, A_b, V_tip --> ThrustCoefComp --> CT_sigma (nn,)
"""

import numpy as np
import openmdao.api as om


class ThrustCoefComp(om.ExplicitComponent):
    """Thrust coefficient over solidity, C_T/sigma."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('T', shape=(nn,), units='lbf', desc='rotor thrust')
        self.add_input('rho', val=0.002377, units='slug/ft**3', desc='air density')
        self.add_input('A_b', val=240.0, units='ft**2', desc='blade area b c R')
        self.add_input('V_tip', val=650.0, units='ft/s', desc='tip speed Omega R')

        self.add_output('CT_sigma', shape=(nn,), desc='C_T / sigma')

        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)
        self.declare_partials('CT_sigma', 'T', rows=ar, cols=ar)
        for name in ('rho', 'A_b', 'V_tip'):
            self.declare_partials('CT_sigma', name, rows=ar, cols=zeros)

    def _denom(self, inputs):
        rho, A_b, V_tip = inputs['rho'][0], inputs['A_b'][0], inputs['V_tip'][0]
        denom = rho * A_b * V_tip ** 2
        if denom <= 0.0:
            raise om.AnalysisError('ThrustCoefComp: rho, A_b and V_tip must all '
                                   f'be positive, got {rho}, {A_b}, {V_tip}.')
        return denom

    def compute(self, inputs, outputs):
        outputs['CT_sigma'] = inputs['T'] / self._denom(inputs)

    def compute_partials(self, inputs, partials):
        denom = self._denom(inputs)
        CT_sigma = inputs['T'] / denom

        partials['CT_sigma', 'T'] = 1.0 / denom
        partials['CT_sigma', 'rho'] = -CT_sigma / inputs['rho'][0]
        partials['CT_sigma', 'A_b'] = -CT_sigma / inputs['A_b'][0]
        partials['CT_sigma', 'V_tip'] = -2.0 * CT_sigma / inputs['V_tip'][0]
