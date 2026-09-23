"""
PseudoGroundEffectComp -- power saved by the fuselage acting as a ground plane.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Vertical Drag in Hover", pp. 280-283; Chapter 1, ground effect
in nondimensional form p. 67, Figure 1.41 p. 66.

The fuselage in the wake slows the induced velocity like a partial ground.
Its saving is the full ground effect at the mean fuselage position, p. 67,
times the fraction of the disc it covers, p. 280:

    dCQ_sigma = -k_PGE (A_wake/A) (C_T/sigma)^(3/2) sqrt(sigma/2) [1 - v_IGE/v_OGE]

with v_IGE/v_OGE from Figure 1.41 at Z/D, the mean depth of the fuselage
below the rotor over the rotor diameter (GroundEffectComp), and k_PGE the
ground proximity factor of GroundProximityDownloadComp (1 out of ground effect).

Example helicopter, p. 282: A_wake/A = 0.12, sigma = 0.085, Z/D = 0.12 gives
v_IGE/v_OGE = 0.62 (p. 281), so dCQ_sigma = -0.0094 (C_T/sigma)^(3/2); at
C_T/sigma = 0.086, -0.00024, or 68 hp (p. 283). The factor printed as 0.038
on p. 282 is 1 - 0.62 = 0.38 (C4-2).

    CT_sigma, k_PGE (nn,), sigma, A_wake_A, vi_IGE_OGE --> dCQ_sigma (nn,)
"""

import numpy as np
import openmdao.api as om


class PseudoGroundEffectComp(om.ExplicitComponent):
    """Torque coefficient saved by the fuselage pseudo ground effect, p. 280."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('CT_sigma', val=np.full(nn, 0.08), desc='blade loading C_T/sigma')
        self.add_input('sigma', val=0.085, desc='rotor solidity')
        self.add_input('A_wake_A', val=0.0, desc='airframe plan area in the wake over disc area')
        self.add_input('vi_IGE_OGE', val=1.0, desc='Figure 1.41 at the mean fuselage depth Z/D')
        self.add_input('k_PGE', val=np.ones(nn), desc='ground proximity factor, 1 out of ground effect')
        self.add_output('dCQ_sigma', val=np.zeros(nn), desc='change of C_Q/sigma, negative saves power')

        self.declare_partials('dCQ_sigma', ['CT_sigma', 'k_PGE'], rows=ar, cols=ar)
        self.declare_partials('dCQ_sigma', ['sigma', 'A_wake_A', 'vi_IGE_OGE'])

    def compute(self, inputs, outputs):
        CT, s, Aw, vi = (inputs[k] for k in ('CT_sigma', 'sigma', 'A_wake_A', 'vi_IGE_OGE'))
        outputs['dCQ_sigma'] = -inputs['k_PGE'] * Aw * CT ** 1.5 * np.sqrt(s / 2.0) * (1.0 - vi)

    def compute_partials(self, inputs, partials):
        CT, s, Aw, vi = (inputs[k] for k in ('CT_sigma', 'sigma', 'A_wake_A', 'vi_IGE_OGE'))
        k = inputs['k_PGE']
        root, gap = np.sqrt(s / 2.0), 1.0 - vi
        oge = -Aw * CT ** 1.5 * root * gap
        dCQ = k * oge

        partials['dCQ_sigma', 'CT_sigma'] = 1.5 * dCQ / CT
        partials['dCQ_sigma', 'k_PGE'] = oge
        partials['dCQ_sigma', 'sigma'] = (0.5 * dCQ / s).reshape(-1, 1)
        partials['dCQ_sigma', 'A_wake_A'] = (-k * CT ** 1.5 * root * gap).reshape(-1, 1)
        partials['dCQ_sigma', 'vi_IGE_OGE'] = (k * Aw * CT ** 1.5 * root).reshape(-1, 1)
