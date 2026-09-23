"""
LossesTorqueComp -- the torque coefficient an autorotating rotor must produce.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 197.

    C_Q/sigma = - [ (hp_T0 + hp_trans + hp_acc) 550 ] / [ rho A_b (Omega R)^3 ]

In powered flight C_Q/sigma is an output: the rotor absorbs whatever the
engines deliver. In autorotation it becomes an INPUT, and this component
supplies it. The rotor is no longer driven, so it must extract enough energy
from the airflow to drive the tail rotor at flat pitch and to overcome the
transmission and accessory losses -- nothing more. That fixes C_Q/sigma at a
small negative value, and the descent rate is then whatever makes the rotor
deliver it.

This is the pivot of the direct autorotation method of p. 197: with C_Q/sigma
imposed, the torque equation and the collective equation are solved
simultaneously for the inflow ratio lambda', instead of sweeping several rates
of descent and interpolating to zero power as on p. 196.

Sign. The output is negative, and it must be: a rotor delivering power to the
transmission has negative torque coefficient in Prouty's convention. Feeding
a positive value would describe a rotor still being driven.

What Prouty does not tabulate. Table 3.3 gives hp_M = -40 in autorotation and
hp_T = 25, which leaves about 15 hp for transmission and accessories -- but he
never splits it, and 15 hp is the number this module's defaults reproduce. The
three terms are kept separate rather than lumped because they scale
differently: tail rotor flat pitch power is essentially profile power and
follows density and tip speed, transmission loss follows the power actually
transmitted, which is near zero here, and accessory load is roughly fixed.

Note that 40 hp against 1097 hp in level flight is under 4 %, yet it is what
sets the autorotative descent rate. The sensitivity runs the other way from
what one might expect: a small absolute error in the losses moves C_Q/sigma by
little, and the descent rate by little -- 1795 ft/min came out of the p. 196
sweep against 1803 tabulated. It is the H-force, not the power, that the
autorotation column of Table 3.3 fails to reconcile.

    hp_T0, hp_trans, hp_acc, rho, A_b, V_tip --> CQ_sigma_target, hp_losses
"""

import numpy as np
import openmdao.api as om

from .tail_rotor_load_comp import power_constant

LOSS_TERMS = ('hp_T0', 'hp_trans', 'hp_acc')


class LossesTorqueComp(om.ExplicitComponent):
    """Target C_Q/sigma for autorotation, p. 197."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('hp_T0', shape=(nn,), val=25.0, units='hp',
                       desc='tail rotor power at flat pitch')
        self.add_input('hp_trans', shape=(nn,), val=10.0, units='hp',
                       desc='transmission loss')
        self.add_input('hp_acc', shape=(nn,), val=5.0, units='hp',
                       desc='accessory load')
        self.add_input('rho', val=0.002377, units='slug/ft**3', desc='density')
        self.add_input('A_b', val=240.0, units='ft**2', desc='blade area')
        self.add_input('V_tip', val=650.0, units='ft/s', desc='tip speed')

        self.add_output('CQ_sigma_target', shape=(nn,),
                        desc='C_Q/sigma the rotor must produce, negative')
        self.add_output('hp_losses', shape=(nn,), units='hp',
                        desc='total power the rotor must supply')

        for name in LOSS_TERMS:
            self.declare_partials(['CQ_sigma_target', 'hp_losses'], name,
                                  rows=ar, cols=ar)
        for name in ('rho', 'A_b', 'V_tip'):
            self.declare_partials('CQ_sigma_target', name, rows=ar, cols=zeros)

    def _losses(self, inputs):
        return sum(inputs[name] for name in LOSS_TERMS)

    def compute(self, inputs, outputs):
        constant = power_constant(inputs['rho'][0], inputs['A_b'][0],
                                  inputs['V_tip'][0])
        losses = self._losses(inputs)

        outputs['hp_losses'] = losses
        outputs['CQ_sigma_target'] = -losses / constant

    def compute_partials(self, inputs, partials):
        rho, A_b = inputs['rho'][0], inputs['A_b'][0]
        V_tip = inputs['V_tip'][0]
        constant = power_constant(rho, A_b, V_tip)
        target = -self._losses(inputs) / constant

        for name in LOSS_TERMS:
            partials['hp_losses', name] = 1.0
            partials['CQ_sigma_target', name] = -1.0 / constant

        partials['CQ_sigma_target', 'rho'] = -target / rho
        partials['CQ_sigma_target', 'A_b'] = -target / A_b
        partials['CQ_sigma_target', 'V_tip'] = -3.0 * target / V_tip
