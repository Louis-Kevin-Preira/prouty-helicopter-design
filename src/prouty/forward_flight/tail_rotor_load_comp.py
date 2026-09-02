"""
Rotor power, and the tail rotor thrust it demands.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 236 (Table 3.5, case 3, step a).

    hp  = rho A_b (Omega R)^3 (C_Q/sigma) / 550
    T_T = 550 hp_M / [ (Omega R)_M (l_T / R_M) ]

Prouty writes the second as one long expression stacking the first inside it;
it is split here because the main rotor power is a headline result in its own
right -- it is what Table 3.3 tabulates -- while the tail rotor load is a
consequence of it.

Where T_T comes from. The tail rotor exists to balance the main rotor torque,
so T_T l_T = Q_M, and Q_M = 550 hp_M / Omega. Substituting gives the form
above, in which the main rotor radius cancels: only the tip speed and the
moment arm expressed in main rotor radii survive. That is why Appendix A
quotes l_T / R_M = 37/30 = 1.23 rather than the arm in feet.

Two consequences worth keeping in mind. First, T_T follows the sign of the
power: in autorotation the main rotor is driving the transmission rather than
absorbing from it, so the tail rotor thrust reverses -- Table 3.3 gives -27 lb
against +755 in level flight. Second, T_T is proportional to power, so it is
the tail rotor, not the main rotor, that sizes the anti-torque system at the
climb and hover extremes: 1211 lb in a 1000 ft/min climb against 755 level.

This is a one-way estimate, not a moment balance. It ignores the fin sideforce
and any vertical stabiliser unloading, which Chapter 8 handles.

    CQ_sigma, rho, A_b, V_tip --> RotorPowerComp    --> hp
    hp_M, V_tip, l_T_R        --> TailRotorLoadComp --> T_T, Q_M_over_R
"""

import numpy as np
import openmdao.api as om

FT_LB_PER_S_PER_HP = 550.0


def power_constant(rho, A_b, V_tip):
    """rho A_b (Omega R)^3 / 550, the 285,000 of p. 236 and 23,100 of p. 234.

    Shared with LossesTorqueComp so that the coefficient converting between
    horsepower and C_Q/sigma has one definition.
    """
    return rho * A_b * V_tip ** 3 / FT_LB_PER_S_PER_HP


class RotorPowerComp(om.ExplicitComponent):
    """Shaft power from the torque coefficient, p. 236."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('CQ_sigma', shape=(nn,), desc='C_Q / sigma')
        self.add_input('rho', val=0.002377, units='slug/ft**3', desc='density')
        self.add_input('A_b', val=240.0, units='ft**2', desc='blade area')
        self.add_input('V_tip', val=650.0, units='ft/s', desc='tip speed')

        self.add_output('hp', shape=(nn,), units='hp', desc='shaft power')

        self.declare_partials('hp', 'CQ_sigma', rows=ar, cols=ar)
        for name in ('rho', 'A_b', 'V_tip'):
            self.declare_partials('hp', name, rows=ar, cols=zeros)

    def _constant(self, inputs):
        return power_constant(inputs['rho'][0], inputs['A_b'][0],
                              inputs['V_tip'][0])

    def compute(self, inputs, outputs):
        outputs['hp'] = self._constant(inputs) * inputs['CQ_sigma']

    def compute_partials(self, inputs, partials):
        constant = self._constant(inputs)
        hp = constant * inputs['CQ_sigma']

        partials['hp', 'CQ_sigma'] = constant
        partials['hp', 'rho'] = hp / inputs['rho'][0]
        partials['hp', 'A_b'] = hp / inputs['A_b'][0]
        partials['hp', 'V_tip'] = 3.0 * hp / inputs['V_tip'][0]


class TailRotorLoadComp(om.ExplicitComponent):
    """Tail rotor thrust required to balance the main rotor torque, p. 236."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('hp_M', shape=(nn,), units='hp',
                       desc='main rotor shaft power')
        self.add_input('V_tip', val=650.0, units='ft/s',
                       desc='main rotor tip speed')
        self.add_input('l_T_R', val=1.23,
                       desc='tail rotor moment arm over main rotor radius')

        self.add_output('T_T', shape=(nn,), units='lbf',
                        desc='tail rotor thrust')

        self.declare_partials('T_T', 'hp_M', rows=ar, cols=ar)
        for name in ('V_tip', 'l_T_R'):
            self.declare_partials('T_T', name, rows=ar, cols=zeros)

    def compute(self, inputs, outputs):
        V_tip, l_T_R = inputs['V_tip'][0], inputs['l_T_R'][0]
        if V_tip <= 0.0 or l_T_R <= 0.0:
            raise om.AnalysisError('TailRotorLoadComp: V_tip and l_T_R must be '
                                   f'positive, got {V_tip} and {l_T_R}.')

        outputs['T_T'] = (FT_LB_PER_S_PER_HP * inputs['hp_M']
                          / (V_tip * l_T_R))

    def compute_partials(self, inputs, partials):
        V_tip, l_T_R = inputs['V_tip'][0], inputs['l_T_R'][0]
        T_T = FT_LB_PER_S_PER_HP * inputs['hp_M'] / (V_tip * l_T_R)

        partials['T_T', 'hp_M'] = FT_LB_PER_S_PER_HP / (V_tip * l_T_R)
        partials['T_T', 'V_tip'] = -T_T / V_tip
        partials['T_T', 'l_T_R'] = -T_T / l_T_R
