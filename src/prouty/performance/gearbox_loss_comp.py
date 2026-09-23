"""
GearboxLossComp -- power lost in the gearboxes of the drive system.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Power Required Losses", pp. 277-278.

Losses between the torquemeter (or transmission input shaft, p. 276) and the
rotors are added to the rotor power required. For each gear stage:

    loss per stage = K (design max power + actual power)
    K = 0.0025 spur or bevel,  0.00375 planetary

A gearbox type g has K_g = 0.0025 n_spur_bevel + 0.00375 n_planetary per box.
Its n_series boxes in a row each carry the full shaft power P_g; its
n_parallel boxes side by side share it:

    L_g = n_series K_g (n_parallel P_design,g + P_g)

Example helicopter, p. 277 (EXAMPLE_GEARBOXES):

    engine nose     1 bevel,           2 parallel, 2,000 hp   0.0025  (4,000 + P_req)
    main rotor      2 spur + 1 planet, 1,          4,000 hp   0.00875 (4,000 + P_MR)
    tail rotor      1 bevel,           2 series,     750 hp   0.0050  (  750 + P_TR)

With P_req = P_MR + P_TR the book rounds this to 49 + 0.0112 P_MR + 0.0075 P_TR.
That simplification is not made here; EnginePowerRequiredComp closes the loop.

    P_req, P_MR, P_TR (nn,), P_design_<gearbox> --> P_loss_gearbox (nn,)

P_req is the engine shaft power, all engines (the P_req of the fuel flow
components); P_eng is kept for the uninstalled ratings of G0.
"""

import numpy as np
import openmdao.api as om

K_SPUR_BEVEL = 0.0025
K_PLANETARY = 0.00375

SHAFTS = {'engine': 'P_req', 'main_rotor': 'P_MR', 'tail_rotor': 'P_TR'}
SHAFT_DESC = {'P_req': 'engine shaft power, all engines',
              'P_MR': 'main rotor power', 'P_TR': 'tail rotor power'}

EXAMPLE_GEARBOXES = {
    'nose': dict(shaft='engine', n_spur_bevel=1, n_planetary=0, n_series=1, n_parallel=2),
    'main': dict(shaft='main_rotor', n_spur_bevel=2, n_planetary=1, n_series=1, n_parallel=1),
    'tail': dict(shaft='tail_rotor', n_spur_bevel=1, n_planetary=0, n_series=2, n_parallel=1),
}


def gearbox_coefficients(gearboxes):
    """
    Split L = sum n_series K_g (n_parallel P_design,g + P_g) into
    slopes {power input: dL/dP} and designs {design input: dL/dP_design}.
    """
    slopes, designs = {}, {}
    for name, g in gearboxes.items():
        k = g.get('n_series', 1) * (K_SPUR_BEVEL * g['n_spur_bevel'] + K_PLANETARY * g['n_planetary'])
        power = SHAFTS[g['shaft']]
        slopes[power] = slopes.get(power, 0.0) + k
        designs[f'P_design_{name}'] = k * g.get('n_parallel', 1)
    return slopes, designs


def _check(gearboxes):
    for name, g in gearboxes.items():
        if g['shaft'] not in SHAFTS:
            raise ValueError(f"gearbox '{name}': shaft must be one of {tuple(SHAFTS)}")


class GearboxLossComp(om.ExplicitComponent):
    """Sum of K (design + actual) over every gear stage, p. 277."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('gearboxes', types=dict, default=EXAMPLE_GEARBOXES,
                             check_valid=lambda name, value: _check(value),
                             desc='name -> shaft, n_spur_bevel, n_planetary, n_series, n_parallel')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self._slopes, self._designs = gearbox_coefficients(self.options['gearboxes'])

        self.add_output('P_loss_gearbox', val=np.zeros(nn), units='hp')

        for design, c in self._designs.items():
            self.add_input(design, val=1.0, units='hp', desc='design max power per box')
            self.declare_partials('P_loss_gearbox', design, val=np.full((nn, 1), c))

        for power, k in self._slopes.items():
            self.add_input(power, val=np.zeros(nn), units='hp', desc=SHAFT_DESC[power])
            self.declare_partials('P_loss_gearbox', power, rows=ar, cols=ar, val=k)

    def compute(self, inputs, outputs):
        outputs['P_loss_gearbox'] = (
            sum(k * inputs[power] for power, k in self._slopes.items())
            + sum(c * inputs[design] for design, c in self._designs.items()))
