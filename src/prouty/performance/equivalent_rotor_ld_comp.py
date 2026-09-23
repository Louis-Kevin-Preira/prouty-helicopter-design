"""
EquivalentRotorLDComp -- equivalent lift-to-drag ratio of the rotor.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Forward Flight Performance" p. 322, Figure 4.40 p. 323.

A rotor is compared with a wing by charging it with its own power and with
nothing else: the equivalent lift is the vertical component of the rotor
thrust, and the equivalent drag is the power it absorbs turned into a drag,
less the parasite drag of everything but the rotor hub and mast:

    L_e = T cos(alpha_TPP)
    D_e = 550 h.p._M / V - q f_rest
    L/D_e = L_e / D_e

f_rest is the parasite drag area of the airframe without the main rotor hub
and mast (f_total - f_M of G4), because that drag belongs to the rotor
installation, not to the rotor.

Example helicopter, Figure 4.40 (20,000 lb, sea level): the ratio peaks near
6.0 at about 100 kt, from 2.7 at 60 kt down to 2.9 at 160 kt. The chain of
this chapter reads higher, since its power is low at speed (C4-29).

    T, alpha_TPP, hp_M, V, q (nn,), f_rest --> L_e, D_e, L_D_e (nn,)
"""

import numpy as np
import openmdao.api as om

HP_TO_FT_LBF_PER_S = 550.0


class EquivalentRotorLDComp(om.ExplicitComponent):
    """Rotor lift and drag as if it were a wing, p. 322."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('T', val=np.ones(nn), units='lbf', desc='rotor thrust')
        self.add_input('alpha_TPP', val=np.zeros(nn), units='rad',
                       desc='tip path plane angle of attack')
        self.add_input('hp_M', val=np.ones(nn), units='hp', desc='main rotor power')
        self.add_input('V', val=np.ones(nn), units='ft/s', desc='flight speed')
        self.add_input('q', val=np.ones(nn), units='lbf/ft**2', desc='dynamic pressure')
        self.add_input('f_rest', val=0.0, units='ft**2',
                       desc='parasite drag area without the main rotor hub and mast')

        self.add_output('L_e', val=np.ones(nn), units='lbf', desc='equivalent lift')
        self.add_output('D_e', val=np.ones(nn), units='lbf', desc='equivalent drag')
        self.add_output('L_D_e', val=np.ones(nn), desc='equivalent lift-to-drag ratio')

        self.declare_partials('L_e', ['T', 'alpha_TPP'], rows=ar, cols=ar)
        self.declare_partials('D_e', ['hp_M', 'V', 'q'], rows=ar, cols=ar)
        self.declare_partials('D_e', 'f_rest')
        self.declare_partials('L_D_e', ['T', 'alpha_TPP', 'hp_M', 'V', 'q'], rows=ar, cols=ar)
        self.declare_partials('L_D_e', 'f_rest')

    def _values(self, inputs):
        L_e = inputs['T'] * np.cos(inputs['alpha_TPP'])
        D_e = (HP_TO_FT_LBF_PER_S * inputs['hp_M'] / inputs['V']
               - inputs['q'] * inputs['f_rest'][0])
        return L_e, D_e

    def compute(self, inputs, outputs):
        L_e, D_e = self._values(inputs)
        outputs['L_e'], outputs['D_e'] = L_e, D_e
        outputs['L_D_e'] = L_e / D_e

    def compute_partials(self, inputs, partials):
        T, alpha, hp, V, q = (inputs[k] for k in ('T', 'alpha_TPP', 'hp_M', 'V', 'q'))
        f_rest = inputs['f_rest'][0]
        L_e, D_e = self._values(inputs)

        dL_dT, dL_da = np.cos(alpha), -T * np.sin(alpha)
        dD_dhp = HP_TO_FT_LBF_PER_S / V
        dD_dV = -HP_TO_FT_LBF_PER_S * hp / V ** 2
        dD_dq, dD_df = -f_rest * np.ones_like(q), -q

        partials['L_e', 'T'], partials['L_e', 'alpha_TPP'] = dL_dT, dL_da
        partials['D_e', 'hp_M'], partials['D_e', 'V'] = dD_dhp, dD_dV
        partials['D_e', 'q'] = dD_dq
        partials['D_e', 'f_rest'] = dD_df.reshape(-1, 1)

        partials['L_D_e', 'T'] = dL_dT / D_e
        partials['L_D_e', 'alpha_TPP'] = dL_da / D_e
        for name, dD in (('hp_M', dD_dhp), ('V', dD_dV), ('q', dD_dq)):
            partials['L_D_e', name] = -L_e / D_e ** 2 * dD
        partials['L_D_e', 'f_rest'] = (-L_e / D_e ** 2 * dD_df).reshape(-1, 1)
