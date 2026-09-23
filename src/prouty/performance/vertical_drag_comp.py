"""
VerticalDragComp -- download of the airframe in the hover wake.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Vertical Drag in Hover": steps p. 278, equation p. 280,
Table 4.1 p. 282.

The plan view of the airframe is cut into segments n, each with a drag
coefficient C_Dn (Figure 4.5, based on plan area), a projected area A_n and a
wake dynamic pressure ratio (q/D.L.)_n. Segments cover one half of the plan
view, hence the factor 2:

    D_v / G.W. = 2 k_Dv sum C_Dn (q/D.L.)_n A_n / A
    T_target   = G.W. (1 + D_v / G.W.)          rotor thrust in hover, p. 278
    A_wake / A = 2 sum A_n / A                  for the pseudo ground effect, p. 282

Like Table 4.1, A_wake sums every segment, including those outboard of the
contracted wake. The equation scales q by G.W./A, as p. 280 does, although
the disc loading of the rotor is T/A; scaling by T/A would raise D_v by the
factor 1 + D_v/G.W. (about 4 %, 0.002 G.W. for the example helicopter).

k_Dv is the ground proximity factor of GroundProximityDownloadComp (1 out of
ground effect). T_target is named for the input of HoverRotorGroup in trim
mode.

    seg_CD, seg_A, q_DL (n,), A, GW, k_Dv (nn,) --> Dv_GW, D_v, T_target (nn,), A_wake_A
"""

import numpy as np
import openmdao.api as om


class VerticalDragComp(om.ExplicitComponent):
    """Sum of segment downloads, p. 280."""

    def initialize(self):
        self.options.declare('num_segments', types=int, default=1)
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        n, nn = self.options['num_segments'], self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('seg_CD', val=np.zeros(n), desc='segment drag coefficient, plan area')
        self.add_input('seg_A', val=np.zeros(n), units='ft**2', desc='segment plan area, half airframe')
        self.add_input('q_DL', val=np.zeros(n), desc='wake dynamic pressure over disc loading')
        self.add_input('A', val=1.0, units='ft**2', desc='disc area')
        self.add_input('GW', val=np.ones(nn), units='lbf', desc='gross weight')
        self.add_input('k_Dv', val=np.ones(nn), desc='ground proximity factor, 1 out of ground effect')

        self.add_output('Dv_GW', val=np.zeros(nn), desc='vertical drag over gross weight')
        self.add_output('D_v', val=np.zeros(nn), units='lbf', desc='vertical drag')
        self.add_output('T_target', val=np.ones(nn), units='lbf', desc='rotor thrust required in hover')
        self.add_output('A_wake_A', val=0.0, desc='airframe plan area over disc area')

        self.declare_partials(['Dv_GW', 'D_v', 'T_target'], ['seg_CD', 'seg_A', 'q_DL', 'A'])
        self.declare_partials(['Dv_GW', 'D_v', 'T_target'], 'k_Dv', rows=ar, cols=ar)
        self.declare_partials(['D_v', 'T_target'], 'GW', rows=ar, cols=ar)
        self.declare_partials('A_wake_A', ['seg_A', 'A'])

    def _base(self, inputs):
        """OGE ratio 2 sum C_D q A / A (scalar)."""
        return 2.0 * np.sum(inputs['seg_CD'] * inputs['q_DL'] * inputs['seg_A']) / inputs['A'][0]

    def compute(self, inputs, outputs):
        ratio = inputs['k_Dv'] * self._base(inputs)
        GW = inputs['GW']
        outputs['Dv_GW'] = ratio
        outputs['D_v'] = ratio * GW
        outputs['T_target'] = GW * (1.0 + ratio)
        outputs['A_wake_A'] = 2.0 * np.sum(inputs['seg_A']) / inputs['A'][0]

    def compute_partials(self, inputs, partials):
        CD, S, q, A = inputs['seg_CD'], inputs['seg_A'], inputs['q_DL'], inputs['A'][0]
        k, GW = inputs['k_Dv'][:, None], inputs['GW'][:, None]
        base = self._base(inputs)
        d_base = {'seg_CD': 2.0 * q * S / A, 'seg_A': 2.0 * CD * q / A,
                  'q_DL': 2.0 * CD * S / A, 'A': np.array([-base / A])}

        for name, d in d_base.items():
            partials['Dv_GW', name] = k * d
            partials['D_v', name] = k * d * GW
            partials['T_target', name] = k * d * GW
        partials['Dv_GW', 'k_Dv'] = np.full(k.size, base)
        partials['D_v', 'k_Dv'] = base * GW.ravel()
        partials['T_target', 'k_Dv'] = base * GW.ravel()
        partials['D_v', 'GW'] = base * k.ravel()
        partials['T_target', 'GW'] = 1.0 + base * k.ravel()
        partials['A_wake_A', 'seg_A'] = np.full(S.size, 2.0 / A)
        partials['A_wake_A', 'A'] = -2.0 * np.sum(S) / A ** 2
