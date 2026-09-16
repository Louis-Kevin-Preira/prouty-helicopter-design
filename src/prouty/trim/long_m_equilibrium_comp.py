"""M-equilibrium equation in forward flight.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", Table 8.4 pp. 520-521. Hover form on
p. 516. Moment arms in Table 8.5 p. 523.
"""

import numpy as np
import openmdao.api as om


class LongMEquilibriumComp(om.ExplicitComponent):
    """Sum of the pitching moments about the c.g., Table 8.4 pp. 520-521::

        res_M = M_M - X_M h_M + Z_M l_M
              + M_T - X_T h_T + Z_T l_T
                    - X_H h_H + Z_H l_H
                    - X_V h_V + Z_V l_V
              + M_F - X_F h_F + Z_F l_F

    ``l`` is positive aft, ``h`` positive above the c.g., and the moment is
    positive nose-up. The two checks that pin the convention: ``Z_M l_M``
    with ``Z_M = -T_M`` and ``l_M = -0.5`` gives ``+.5 T_M``, which is the
    row Table 8.4 prints; and ``-X_T h_T`` with ``X_T = -H_T`` gives
    ``+H_T h_T = 240``, also as printed.

    No weight term appears, the moments being taken about the centre of
    gravity the weight acts through.

    Table 8.4 has no ``Z_V l_V`` row
    --------------------------------
    Every other component contributes both a ``-X h`` and a ``+Z l`` term.
    The vertical stabiliser contributes only ``-X_V h_V``. The missing term
    is worth ``Z_V l_V = 4.93 x 35 = 173 ft-lb`` at 115 knots, against an
    equilibrium the book closes to about 129, so it is not below the noise
    of its own solution.

    ``linearized=True`` reproduces the table as printed and leaves the term
    out; ``linearized=False`` includes it. That follows the same rule as
    C8-1, where the tables drop the rotor torque coupling from ``R_M`` and
    ``M_M``: the linearised option means "the equations Prouty solved", not
    only "small angles". See C8-8 in ``docs/validation_trim.md``.

    Scale
    -----
    The large terms at 115 knots are the fuselage pitching moment at
    -11,757, the main rotor thrust arm at +10,276, the hub moment at -3,280
    and the H-force arm at -4,016. The horizontal stabiliser reaches about
    +9,000 through ``Z_H l_H``, which is what makes it the pitch trim device
    rather than the small correction its 273 lb of lift suggests.
    """

    _components = (('M', 'M_M'), ('T', 'M_T'), ('H', None), ('V', None),
                   ('F', 'M_F'))

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('linearized', types=bool, default=False)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)
        one = np.ones(nn)

        for station, moment in self._components:
            self.add_input(f'X_{station}', shape=(nn,), val=0.0, units='lbf')
            self.add_input(f'h_{station}', val=0.0, units='ft',
                           desc=f'{station} height above the c.g.')
            if moment is not None:
                self.add_input(moment, shape=(nn,), val=0.0, units='lbf*ft')

        for station in self._z_stations():
            self.add_input(f'Z_{station}', shape=(nn,), val=0.0, units='lbf')
            self.add_input(f'l_{station}', val=0.0, units='ft',
                           desc=f'{station} offset aft of the c.g.')

        self.add_output('res_M', shape=(nn,), units='lbf*ft',
                        desc='M-equilibrium residual')

        for station, moment in self._components:
            self.declare_partials('res_M', f'X_{station}', rows=ar, cols=ar)
            self.declare_partials('res_M', f'h_{station}', rows=ar, cols=zeros)
            if moment is not None:
                self.declare_partials('res_M', moment, rows=ar, cols=ar,
                                      val=one)
        for station in self._z_stations():
            self.declare_partials('res_M', f'Z_{station}', rows=ar, cols=ar)
            self.declare_partials('res_M', f'l_{station}', rows=ar, cols=zeros)

    def _z_stations(self):
        """Table 8.4 has no Z_V l_V row; the physics does."""
        return 'MTHF' if self.options['linearized'] else 'MTHVF'

    def compute(self, inputs, outputs):
        total = sum(inputs[m] for _, m in self._components if m is not None)
        for station, _ in self._components:
            total = total - inputs[f'X_{station}'] * inputs[f'h_{station}'][0]
        for station in self._z_stations():
            total = total + inputs[f'Z_{station}'] * inputs[f'l_{station}'][0]
        outputs['res_M'] = total

    def compute_partials(self, inputs, J):
        for station, _ in self._components:
            J['res_M', f'X_{station}'] = -inputs[f'h_{station}'][0]
            J['res_M', f'h_{station}'] = -inputs[f'X_{station}']
        for station in self._z_stations():
            J['res_M', f'Z_{station}'] = inputs[f'l_{station}'][0]
            J['res_M', f'l_{station}'] = inputs[f'Z_{station}']
