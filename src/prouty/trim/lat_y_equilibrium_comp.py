"""Y-equilibrium equation in forward flight.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", Table 8.11 p. 536. Hover form on p. 531,
Figure 8.28. Anchors in Figure 8.31 p. 538.
"""

import numpy as np
import openmdao.api as om


class LatYEquilibriumComp(om.ExplicitComponent):
    """Sum of the Y forces, Table 8.11 p. 536::

        res_Y = Y_M + Y_T + Y_V + Y_F + G.W. cos(Theta) sin(Phi)

    The horizontal stabiliser has no row: a symmetric surface makes no side
    force, so only four components appear where the X and Z equations have
    five.

    The sign of the weight term
    ---------------------------
    Table 8.11 prints the row as ``-G.W. Phi`` and this component uses
    ``+G.W. Phi``. The hover form of p. 531 is unambiguous,
    ``T_M b1s_M + T_T = -G.W. Phi``, which rearranges to a sum with a plus,
    and the physics agrees: with ``Phi`` positive right-side-down, gravity
    leans toward positive body Y.

    Figure 8.31 p. 538 settles it numerically. Solving the printed R and N
    rows at ``beta = 0`` gives ``b1s_M = -0.779 deg`` and ``T_T = 660.7 lb``
    against the -0.78 and 661 annotated on that figure, so those two rows are
    right; the Y row then returns ``Phi = -1.92 deg`` with the plus and
    +1.92 deg with the minus, against a figure showing -1.9 deg, left.
    See C8-10 in ``docs/validation_trim.md``.

    ``linearized=True`` drops to ``+ G.W. Phi``, which is Table 8.11's row
    with its sign corrected, worth 20,000 Phi. The ``cos(Theta)`` it also
    drops is 0.9999 at the 0.9 degrees of pitch this helicopter trims to.

    Scale
    -----
    At 115 knots and zero sideslip the four aerodynamic contributions are
    about +415 of fin lift and rotor H-force, -280 from the lateral flapping,
    +535 of tail rotor thrust, against a weight term of -670. The tail rotor
    is half the equation, which is why the roll angle tracks it so closely.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('linearized', types=bool, default=False)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        one = np.ones(nn)

        for c in 'MTVF':
            self.add_input(f'Y_{c}', shape=(nn,), val=0.0, units='lbf')
        self.add_input('GW', shape=(nn,), val=0.0, units='lbf',
                       desc='gross weight')
        self.add_input('Phi', shape=(nn,), val=0.0, units='rad',
                       desc='roll angle, positive right side down')
        self.add_input('Theta', shape=(nn,), val=0.0, units='rad',
                       desc='fuselage pitch attitude')

        self.add_output('res_Y', shape=(nn,), units='lbf',
                        desc='Y-equilibrium residual')

        for c in 'MTVF':
            self.declare_partials('res_Y', f'Y_{c}', rows=ar, cols=ar, val=one)
        self.declare_partials('res_Y', ['GW', 'Phi'], rows=ar, cols=ar)
        if not self.options['linearized']:
            self.declare_partials('res_Y', 'Theta', rows=ar, cols=ar)

    def _tilt(self, inputs):
        if self.options['linearized']:
            return inputs['Phi'], np.ones_like(inputs['Phi']), None
        phi, theta = inputs['Phi'], inputs['Theta']
        return (np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi),
                -np.sin(theta) * np.sin(phi))

    def compute(self, inputs, outputs):
        tilt, _, _ = self._tilt(inputs)
        outputs['res_Y'] = sum(inputs[f'Y_{c}'] for c in 'MTVF') \
            + inputs['GW'] * tilt

    def compute_partials(self, inputs, J):
        tilt, dphi, dtheta = self._tilt(inputs)
        J['res_Y', 'GW'] = tilt
        J['res_Y', 'Phi'] = inputs['GW'] * dphi
        if dtheta is not None:
            J['res_Y', 'Theta'] = inputs['GW'] * dtheta
