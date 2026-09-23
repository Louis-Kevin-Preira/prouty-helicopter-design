"""
ChordForceComp -- chordwise force coefficient of a blade element.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 211-212.

The chordwise force is what makes torque and H-force, and it has three
sources: pressure drag, skin friction, and the tilt of the lift vector.

    c_c,0   = c_dp (U_T/U_B) + c_f U_T sqrt(U_T^2 + U_R^2) / U_B^2
    c_c,ind = - c_l (U_P/U_B)
    c_c     = c_c,0 + c_c,ind

Pressure drag and skin friction are separated, which the closed form does not
do. p. 211: "in the closed-form derivation, pressure drag and skin friction
were treated as one, being assumed to be governed by the component of velocity
perpendicular to the leading edge. This is strictly only true for pressure
drag. Skin friction is a function of the magnitude and direction of the total
local velocity." So skin friction responds to U_TR, not to U_B, and it points
along the total velocity rather than along the chord -- hence the factor
U_T/sqrt(U_T^2+U_R^2) that resolves it back into the chordwise direction, and
the U_TR that survives.

The split uses Prouty's own convention (p. 212): c_f = 0.006 and c_dp =
c_d - 0.006, for want of anything better. It is not a fit to the airfoil; it
is a constant lifted out of the measured drag so the two parts can be given
their different velocity dependences. If the airfoil model ever supplies its
own friction estimate, replace c_f and the subtraction together.

Why the two parts come out separately. The profile part and the induced part
are integrated over DIFFERENT spans (p. 212): "no root cutout or tip loss is
applied to the portion of the chordwise force produced by pressure drag". So
c_c,0 takes the full-blade weights and c_c,ind takes the weights running from
x_0 to B. That is the reason RotorDiscGridComp emits two weight vectors, and
the reason this component cannot hand its consumers a single c_c.

As in NormalForceComp, the exported quantities carry U_B^2 already:

    cc0_UB2   = c_dp U_T U_B + c_f U_T U_TR
    ccind_UB2 = - c_l U_P U_B

Both are regular; the raw c_c, which carries 1/U_B and reaches a hundred times
its natural size near the root, is exported only for plotting.

Sign check. Inside the reverse flow circle U_T is negative, so the pressure
drag term is negative: the drag on a reversed element pushes forward in the
chordwise sense, which is right, and the torque it makes changes sign with it.

    cl, cd, UT_bar, UP_bar, UB_bar, UTR_bar --> cc, cc0_UB2, ccind_UB2
"""

import numpy as np
import openmdao.api as om

SKIN_FRICTION = 0.006          # p. 212


class ChordForceComp(om.ExplicitComponent):
    """Chordwise force coefficient, split by source, p. 211-212."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('num_azimuth', types=int, default=24)
        self.options.declare('num_radial', types=int, default=21)

    def setup(self):
        nn = self.options['num_nodes']
        n_psi = self.options['num_azimuth']
        n_r = self.options['num_radial']
        field = (nn, n_psi, n_r)
        rows = np.arange(nn * n_psi * n_r)

        for name in ('cl', 'cd', 'UT_bar', 'UP_bar', 'UB_bar', 'UTR_bar'):
            self.add_input(name, shape=field)
        self.add_input('c_f', val=SKIN_FRICTION,
                       desc='skin friction coefficient, p. 212')

        self.add_output('cc', shape=field, desc='chordwise force coefficient')
        self.add_output('cc0_UB2', shape=field,
                        desc='U_B^2 times the profile part, full blade')
        self.add_output('ccind_UB2', shape=field,
                        desc='U_B^2 times the induced part, x_0 to B')

        # declare only the pairs that are actually non-zero: the profile part
        # ignores lift and U_P, the induced part ignores drag and U_TR
        depends = {
            'cc': ('cl', 'cd', 'UT_bar', 'UP_bar', 'UB_bar', 'UTR_bar', 'c_f'),
            'cc0_UB2': ('cd', 'UT_bar', 'UB_bar', 'UTR_bar', 'c_f'),
            'ccind_UB2': ('cl', 'UP_bar', 'UB_bar'),
        }
        zeros = np.zeros(rows.size, dtype=int)
        for out, names in depends.items():
            for name in names:
                cols = zeros if name == 'c_f' else rows
                self.declare_partials(out, name, rows=rows, cols=cols)

    def _parts(self, inputs):
        c_f = inputs['c_f'][0]
        c_dp = inputs['cd'] - c_f
        UT, UP = inputs['UT_bar'], inputs['UP_bar']
        UB, UTR = inputs['UB_bar'], inputs['UTR_bar']

        profile = c_dp * UT * UB + c_f * UT * UTR
        induced = -inputs['cl'] * UP * UB
        return profile, induced, c_dp

    def compute(self, inputs, outputs):
        profile, induced, _ = self._parts(inputs)

        outputs['cc0_UB2'] = profile
        outputs['ccind_UB2'] = induced
        outputs['cc'] = (profile + induced) / inputs['UB_bar'] ** 2

    def compute_partials(self, inputs, partials):
        c_f = inputs['c_f'][0]
        cl, cd = inputs['cl'], inputs['cd']
        UT, UP = inputs['UT_bar'], inputs['UP_bar']
        UB, UTR = inputs['UB_bar'], inputs['UTR_bar']
        profile, induced, c_dp = self._parts(inputs)

        d = {
            ('cc0_UB2', 'cd'): UT * UB,
            ('cc0_UB2', 'UT_bar'): c_dp * UB + c_f * UTR,
            ('cc0_UB2', 'UB_bar'): c_dp * UT,
            ('cc0_UB2', 'UTR_bar'): c_f * UT,
            ('cc0_UB2', 'cl'): np.zeros_like(UT),
            ('cc0_UB2', 'UP_bar'): np.zeros_like(UT),
            ('cc0_UB2', 'c_f'): UT * (UTR - UB),
            ('ccind_UB2', 'cl'): -UP * UB,
            ('ccind_UB2', 'UP_bar'): -cl * UB,
            ('ccind_UB2', 'UB_bar'): -cl * UP,
            ('ccind_UB2', 'cd'): np.zeros_like(UT),
            ('ccind_UB2', 'UT_bar'): np.zeros_like(UT),
            ('ccind_UB2', 'UTR_bar'): np.zeros_like(UT),
            ('ccind_UB2', 'c_f'): np.zeros_like(UT),
        }
        for key, value in d.items():
            if key in partials:
                partials[key] = value.ravel()

        total = profile + induced
        for name in ('cl', 'cd', 'UT_bar', 'UP_bar', 'UTR_bar', 'c_f'):
            partials['cc', name] = (
                (d[('cc0_UB2', name)] + d[('ccind_UB2', name)])
                / UB ** 2).ravel()
        partials['cc', 'UB_bar'] = (
            (d[('cc0_UB2', 'UB_bar')] + d[('ccind_UB2', 'UB_bar')]) / UB ** 2
            - 2.0 * total / UB ** 3).ravel()
