"""
InstalledPowerComp -- installed power available, all engines.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Engine Installation Losses" pp. 276-277, transmission torque
limit p. 320; anchors pp. 311 and 320.

The losses between the engine and the torquemeter (turboshaft) or the
transmission input shaft (piston) reduce the power available, p. 276:

    inlet duct friction, particle separator, exhaust back pressure,
    infrared suppressor, compressor bleed        fractions of power
    engine-mounted accessories                   up to 100 hp

Prouty leaves their evaluation to an installation specialist and uses 2 %
inlet friction for the example helicopter, p. 277. The fractions are lumped
into k_inst:

    P_net   = (1 - k_inst) P_eng - P_acc           per engine
    P_avail = smoothmin(n_eng P_net, P_trans)      transmission_limit=True

n_eng is per node, so one-engine-inoperative points (pp. 325, 328) sit in
the same vector. The inlet temperature rise from exhaust reingestion (1-4 F)
is not a power fraction; it belongs to the T_air fed to the engine ratings.

    P_eng (nn, 3), n_eng (nn,), k_inst, P_acc, [P_trans] --> P_avail (nn, 3)
"""

import numpy as np
import openmdao.api as om

from prouty.performance._smooth import smoothmin
from prouty.performance.piston_power_lapse_comp import RATINGS


class InstalledPowerComp(om.ExplicitComponent):
    """Installation losses, operating engines and transmission limit."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('transmission_limit', types=bool, default=False)
        self.options.declare('blend_width', types=float, default=10.0,
                             desc='half width of the smooth minimum, hp')

    def setup(self):
        nn = self.options['num_nodes']
        nr = len(RATINGS)

        self.add_input('P_eng', val=np.ones((nn, nr)), units='hp',
                       desc='uninstalled ratings per engine: ' + ', '.join(RATINGS))
        self.add_input('n_eng', val=np.ones(nn), desc='operating engines')
        self.add_input('k_inst', val=0.0, desc='installation loss fraction')
        self.add_input('P_acc', val=0.0, units='hp', desc='engine-mounted accessories, per engine')
        self.add_output('P_avail', val=np.ones((nn, nr)), units='hp',
                        desc='installed ratings, all operating engines')

        rows = np.arange(nn * nr)
        self.declare_partials('P_avail', 'P_eng', rows=rows, cols=rows)
        self.declare_partials('P_avail', 'n_eng', rows=rows, cols=np.repeat(np.arange(nn), nr))
        self.declare_partials('P_avail', ['k_inst', 'P_acc'])

        if self.options['transmission_limit']:
            self.add_input('P_trans', val=1.0, units='hp', desc='transmission power limit')
            self.declare_partials('P_avail', 'P_trans')

    def _evaluate(self, inputs):
        n = inputs['n_eng'][:, None]
        k = inputs['k_inst'][0]
        P_net = (1.0 - k) * inputs['P_eng'] - inputs['P_acc'][0]
        P_tot = n * P_net

        if self.options['transmission_limit']:
            P, s_tot, s_trans = smoothmin(P_tot, inputs['P_trans'][0], self.options['blend_width'])
        else:
            P, s_tot, s_trans = P_tot, np.ones_like(P_tot), np.zeros_like(P_tot)
        return P, P_net, n, k, s_tot, s_trans

    def compute(self, inputs, outputs):
        outputs['P_avail'] = self._evaluate(inputs)[0]

    def compute_partials(self, inputs, partials):
        _, P_net, n, k, s_tot, s_trans = self._evaluate(inputs)

        partials['P_avail', 'P_eng'] = (s_tot * n * (1.0 - k)).ravel()
        partials['P_avail', 'n_eng'] = (s_tot * P_net).ravel()
        partials['P_avail', 'k_inst'] = (-s_tot * n * inputs['P_eng']).reshape(-1, 1)
        partials['P_avail', 'P_acc'] = (-s_tot * n * np.ones_like(P_net)).reshape(-1, 1)
        if self.options['transmission_limit']:
            partials['P_avail', 'P_trans'] = s_trans.reshape(-1, 1)
