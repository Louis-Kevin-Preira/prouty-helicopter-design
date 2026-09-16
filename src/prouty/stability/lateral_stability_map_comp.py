"""The lateral-directional stability map of Figure 9.23.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", pp. 632-635. The spiral criterion
is p. 633 and the map itself Figure 9.23, p. 634.
"""

import numpy as np
import openmdao.api as om

from prouty.stability.modes import describe_modes

#: p. 633's table of the four derivatives the spiral mode turns on, as
#: (name, normal sign, value for the example helicopter).
SPIRAL_DERIVATIVES = (
    ('dR_dydot', 'dihedral effect', -1, -382.0),
    ('dN_dr', 'yaw damping', -1, -53913.0),
    ('dN_dydot', 'directional stability', +1, 1207.0),
    ('dR_dr', 'roll due to yaw rate', +1, 6912.0),
)


def classify_lateral(char_coeffs):
    """Which region of Figure 9.23 a characteristic equation falls in.

    The map has three: stable, unstable spiral dive above the ``E = 0`` line,
    and unstable Dutch roll below the Routh's discriminant line. The roll
    convergence is never the offender, so an unstable real root is the spiral
    and an unstable complex pair is the Dutch roll.

    Discrete, so this is a post-processing function rather than an output, as
    ``classify`` is on the longitudinal map.
    """
    modes = describe_modes(char_coeffs)
    unstable = [mode for mode in modes if mode.root.real > 0.0]
    if not unstable:
        return 'stable'
    if any(mode.oscillatory for mode in unstable):
        return 'unstable Dutch roll'
    return 'spiral dive'


class LateralStabilityMapComp(om.ExplicitComponent):
    """The spiral boundary of Figure 9.23, in closed form.

    p. 634 plots the aircraft in the plane of its directional stability
    ``dN/dydot`` against its dihedral effect ``dR/dydot``, with two curves
    bounding the regions: Routh's discriminant vanishing below, and ``E``, the
    constant term of the characteristic equation, vanishing above. p. 634 says
    which one matters here -- "the critical boundary is that associated with
    the constant term of the characteristic equation, ``E``, being zero".

    This component produces that one::

        E = (g / (I_xx I_zz)) [(dR/dydot)(dN/dr) - (dN/dydot)(dR/dr)]

    and the boundary itself, ``E = 0`` at::

        dN/dydot = (dR/dydot)(dN/dr)/(dR/dr)

    Routh's discriminant is not recomputed here; ``RouthDiscriminantComp``
    already has it from the characteristic equation.

    Exact, and derived rather than taken on trust
    ---------------------------------------------
    p. 633 prints the expression without derivation. It falls out of the
    determinant: the only entry of the lateral matrix carrying a constant term
    is ``+G.W.`` in the Y row, so two of the six Leibniz permutations reach the
    ``s^2`` coefficient of the degree-six determinant -- one odd, one even --
    and they give exactly the expression above. For the example helicopter it
    is **2.2544** against the ``2.2548`` p. 628 prints as the quartic's
    constant term.

    Why the spiral is stable here
    -----------------------------
    p. 633 tabulates the four derivatives and their normal signs::

        dR/dydot  dihedral effect        -    -382       } 20.59e6
        dN/dr     yaw damping            -    -53,913    }
        dN/dydot  directional stability  +    1,207      } 8.34e6
        dR/dr     roll due to yaw rate   +    6,912      }
                                              difference = 12.25e6

    Both products are positive because each pair has matching signs, and the
    spiral is stable because the first beats the second. The boundary sits at
    ``dN/dydot = 2,980`` and the aircraft is at **1,207**, well inside.

    The two ways to fix a spiral dive
    ---------------------------------
    Raising ``dN/dr`` in magnitude raises the boundary, which is the yaw damper
    or SAS of p. 635. Lowering ``dN/dydot`` moves the aircraft down the map,
    which is the Bell 212's vertical destabilizer ahead of the centre of
    gravity in Figure 9.24 -- deliberately *reducing* directional stability.
    Both show up in ``E_margin``.

    Options
    -------
    num_nodes : int
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('dR_dydot', val=np.full(nn, -382.0), units='lbf*s')
        self.add_input('dR_dr', val=np.full(nn, 6912.0), units='lbf*ft*s/rad')
        self.add_input('dN_dydot', val=np.full(nn, 1207.0), units='lbf*s')
        self.add_input('dN_dr', val=np.full(nn, -53913.0), units='lbf*ft*s/rad')
        self.add_input('I_xx', val=np.full(nn, 5000.0), units='slug*ft**2')
        self.add_input('I_zz', val=np.full(nn, 35000.0), units='slug*ft**2')
        self.add_input('g', val=np.full(nn, 32.2), units='ft/s**2')

        self.add_output('E', val=np.zeros(nn),
                        desc='Constant term of the characteristic equation.')
        self.add_output('dN_dydot_at_E_zero', val=np.zeros(nn), units='lbf*s',
                        desc='Spiral boundary of Figure 9.23.')
        self.add_output('E_margin', val=np.zeros(nn), units='lbf*s',
                        desc='Boundary minus dN/dydot; positive is stable.')

        self.declare_partials('E', ['dR_dydot', 'dR_dr', 'dN_dydot', 'dN_dr',
                                    'I_xx', 'I_zz', 'g'], rows=ar, cols=ar)
        for name in ('dN_dydot_at_E_zero', 'E_margin'):
            self.declare_partials(name, ['dR_dydot', 'dR_dr', 'dN_dr'],
                                  rows=ar, cols=ar)
        self.declare_partials('E_margin', 'dN_dydot', rows=ar, cols=ar,
                              val=-1.0)

    def compute(self, inputs, outputs):
        Rv, Rr = inputs['dR_dydot'], inputs['dR_dr']
        Nv, Nr = inputs['dN_dydot'], inputs['dN_dr']
        inertia = inputs['I_xx'] * inputs['I_zz']
        boundary = Rv * Nr / Rr

        outputs['E'] = inputs['g'] * (Rv * Nr - Nv * Rr) / inertia
        outputs['dN_dydot_at_E_zero'] = boundary
        outputs['E_margin'] = boundary - Nv

    def compute_partials(self, inputs, J):
        Rv, Rr = inputs['dR_dydot'], inputs['dR_dr']
        Nv, Nr = inputs['dN_dydot'], inputs['dN_dr']
        I_xx, I_zz, g = inputs['I_xx'], inputs['I_zz'], inputs['g']
        inertia = I_xx * I_zz
        numerator = Rv * Nr - Nv * Rr

        J['E', 'dR_dydot'] = g * Nr / inertia
        J['E', 'dN_dr'] = g * Rv / inertia
        J['E', 'dN_dydot'] = -g * Rr / inertia
        J['E', 'dR_dr'] = -g * Nv / inertia
        J['E', 'g'] = numerator / inertia
        J['E', 'I_xx'] = -g * numerator / (inertia * I_xx)
        J['E', 'I_zz'] = -g * numerator / (inertia * I_zz)

        J['dN_dydot_at_E_zero', 'dR_dydot'] = Nr / Rr
        J['dN_dydot_at_E_zero', 'dN_dr'] = Rv / Rr
        J['dN_dydot_at_E_zero', 'dR_dr'] = -Rv * Nr / Rr ** 2
        for name in ('dR_dydot', 'dN_dr', 'dR_dr'):
            J['E_margin', name] = J['dN_dydot_at_E_zero', name]
