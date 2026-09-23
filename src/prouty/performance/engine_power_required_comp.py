"""
EnginePowerRequiredComp -- engine power required at the torquemeters.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4: gearbox losses p. 277, accessory losses and density scaling
p. 278, engine power in hover p. 311.

The engine delivers the rotor powers plus the drive system losses, p. 311:

    P_req = P_MR + P_TR + L_gearbox(P_req, P_MR, P_TR) + P_loss_acc

The gearbox losses (GearboxLossComp) are linear, with the nose boxes loaded
by P_req itself: L_gearbox = k_eng P_req + k_MR P_MR + k_TR P_TR + D,
D = sum of the design-power terms. The loop closes exactly (C4-9):

    P_req = [(1 + k_MR) P_MR + (1 + k_TR) P_TR + s (D + P_loss_acc)] / (1 - k_eng)

Density scaling, p. 278. Prouty takes the losses proportional to rho/rho_0
for convenience; his normalized hover equation p. 311,

    P_req/sigma = 56 + 1.0112 (P_MR/sigma) + 1.0075 (P_TR/sigma),

scales only the load-independent part (design terms and accessories, 56 hp)
and keeps the load terms on the actual powers (C4-10). With
density_scaled_losses=True, s = sigma; otherwise s = 1.

    P_MR, P_TR, P_loss_acc, [density_ratio] (nn,), P_design_<gearbox>
        --> P_req, P_loss (nn,)

P_req is the power required from the engines, all engines: it feeds the fuel
flow components of G0 directly by promotion.
"""

import numpy as np
import openmdao.api as om

from prouty.performance.gearbox_loss_comp import (EXAMPLE_GEARBOXES, SHAFTS, _check,
                                                  gearbox_coefficients)


class EnginePowerRequiredComp(om.ExplicitComponent):
    """Rotor powers plus gearbox and accessory losses, loop closed."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('gearboxes', types=dict, default=EXAMPLE_GEARBOXES,
                             check_valid=lambda name, value: _check(value))
        self.options.declare('density_scaled_losses', types=bool, default=False)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        slopes, self._designs = gearbox_coefficients(self.options['gearboxes'])
        self._k = {name: slopes.get(name, 0.0) for name in SHAFTS.values()}

        self.add_input('P_MR', val=np.zeros(nn), units='hp', desc='main rotor power')
        self.add_input('P_TR', val=np.zeros(nn), units='hp', desc='tail rotor power')
        self.add_input('P_loss_acc', val=np.zeros(nn), units='hp', desc='accessory losses')
        for design in self._designs:
            self.add_input(design, val=1.0, units='hp', desc='design max power per box')
        if self.options['density_scaled_losses']:
            self.add_input('density_ratio', val=np.ones(nn))

        self.add_output('P_req', val=np.zeros(nn), units='hp', desc='engine power, all engines')
        self.add_output('P_loss', val=np.zeros(nn), units='hp', desc='gearbox + accessory losses')

        per_node = ['P_MR', 'P_TR', 'P_loss_acc'] + (
            ['density_ratio'] if self.options['density_scaled_losses'] else [])
        self.declare_partials(['P_req', 'P_loss'], per_node, rows=ar, cols=ar)
        self.declare_partials(['P_req', 'P_loss'], list(self._designs))

    def _terms(self, inputs):
        s = inputs['density_ratio'] if self.options['density_scaled_losses'] \
            else np.ones_like(inputs['P_MR'])
        fixed = sum(c * inputs[d] for d, c in self._designs.items()) + inputs['P_loss_acc']
        den = 1.0 - self._k['P_req']
        return s, fixed, den

    def compute(self, inputs, outputs):
        s, fixed, den = self._terms(inputs)
        k = self._k
        P_req = ((1.0 + k['P_MR']) * inputs['P_MR'] + (1.0 + k['P_TR']) * inputs['P_TR']
                 + s * fixed) / den
        outputs['P_req'] = P_req
        outputs['P_loss'] = P_req - inputs['P_MR'] - inputs['P_TR']

    def compute_partials(self, inputs, partials):
        s, fixed, den = self._terms(inputs)
        k = self._k

        d_eng = {'P_MR': np.full_like(s, (1.0 + k['P_MR']) / den),
                 'P_TR': np.full_like(s, (1.0 + k['P_TR']) / den),
                 'P_loss_acc': s / den}
        if self.options['density_scaled_losses']:
            d_eng['density_ratio'] = fixed / den
        for name, val in d_eng.items():
            partials['P_req', name] = val
            partials['P_loss', name] = val - (1.0 if name in ('P_MR', 'P_TR') else 0.0)

        for design, c in self._designs.items():
            partials['P_req', design] = (s * c / den).reshape(-1, 1)
            partials['P_loss', design] = (s * c / den).reshape(-1, 1)
