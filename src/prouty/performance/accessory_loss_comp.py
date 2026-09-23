"""
AccessoryLossComp -- power absorbed by transmission-mounted accessories.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Power Required Losses", pp. 277-278.

Generators and hydraulic pumps absorb power according to their load in the
flight condition considered, with typical efficiencies, p. 278:

    generator loss      = load [W] / (0.75 x 746)                   hp
    hydraulic pump loss = pressure [psi] x flow [gpm] / (0.80 x 1,714)  hp

746 W/hp and 1,714 psi gpm/hp are the unit conversions. Shaft-driven cooling
fans (p. 277) have no formula in the book; they enter as P_other.

Example helicopter, p. 278: 2,200 W gives 4 hp, 1.3 gpm at 3,000 psi gives
3 hp; with the 49 hp of the gearboxes this is the 56 hp of the loss equation.

    load_elec, flow_hyd (nn,), p_hyd, eta_gen, eta_hyd, P_other --> P_loss_acc (nn,)
"""

import numpy as np
import openmdao.api as om

W_PER_HP = 746.0
PSI_GPM_PER_HP = 1714.0


class AccessoryLossComp(om.ExplicitComponent):
    """Generator, hydraulic pump and other accessory losses, p. 278."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('load_elec', val=np.zeros(nn), units='W', desc='generator electrical load')
        self.add_input('flow_hyd', val=np.zeros(nn), units='galUS/min', desc='hydraulic flow rate')
        self.add_input('p_hyd', val=0.0, units='psi', desc='hydraulic system design pressure')
        self.add_input('eta_gen', val=0.75, desc='generator efficiency')
        self.add_input('eta_hyd', val=0.80, desc='hydraulic pump efficiency')
        self.add_input('P_other', val=0.0, units='hp', desc='other shaft-driven accessories')
        self.add_output('P_loss_acc', val=np.zeros(nn), units='hp')

        self.declare_partials('P_loss_acc', ['load_elec', 'flow_hyd'], rows=ar, cols=ar)
        self.declare_partials('P_loss_acc', ['p_hyd', 'eta_gen', 'eta_hyd'])
        self.declare_partials('P_loss_acc', 'P_other', val=np.ones((nn, 1)))

    def _losses(self, inputs):
        gen = inputs['load_elec'] / (inputs['eta_gen'] * W_PER_HP)
        hyd = inputs['p_hyd'] * inputs['flow_hyd'] / (inputs['eta_hyd'] * PSI_GPM_PER_HP)
        return gen, hyd

    def compute(self, inputs, outputs):
        gen, hyd = self._losses(inputs)
        outputs['P_loss_acc'] = gen + hyd + inputs['P_other']

    def compute_partials(self, inputs, partials):
        gen, hyd = self._losses(inputs)
        eta_gen, eta_hyd = inputs['eta_gen'], inputs['eta_hyd']
        hyd_per_flow = inputs['p_hyd'] / (eta_hyd * PSI_GPM_PER_HP)

        partials['P_loss_acc', 'load_elec'] = np.full_like(gen, 1.0) / (eta_gen * W_PER_HP)
        partials['P_loss_acc', 'flow_hyd'] = np.full_like(hyd, 1.0) * hyd_per_flow
        partials['P_loss_acc', 'p_hyd'] = (inputs['flow_hyd'] / (eta_hyd * PSI_GPM_PER_HP)).reshape(-1, 1)
        partials['P_loss_acc', 'eta_gen'] = (-gen / eta_gen).reshape(-1, 1)
        partials['P_loss_acc', 'eta_hyd'] = (-hyd / eta_hyd).reshape(-1, 1)
