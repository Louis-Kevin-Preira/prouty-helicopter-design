"""Phase lag between maximum aerodynamic input and maximum flapping response.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", p. 459.
"""

import numpy as np
import openmdao.api as om


class PhaseAngleComp(om.ExplicitComponent):
    """Flapping phase angle, closed form replacing the Figure 7.3 chart.

    A rotor with no hinge offset is in resonance and responds a quarter cycle
    behind the input. Offset raises the natural frequency above the rotational
    frequency, so the lag falls below 90 degrees. Prouty writes (p. 459)::

        phi = acos{ [(omega_n/Omega)^2 - 1]
                    / sqrt([(omega_n/Omega)^2 - 1]^2
                           + 4 (c/c_crit)^2 (omega_n/Omega)^2) }

    Implementation
    --------------
    Since ``2 zeta (omega_n/Omega)`` is strictly positive for any real blade,
    the code evaluates the equivalent form::

        phi = pi/2 - atan[ ((omega_n/Omega)^2 - 1)
                           / (2 zeta (omega_n/Omega)) ]

    which is the same angle: the printed square root is exactly the hypotenuse
    of the two arguments. Two reasons for the rewrite. The arccos form has a
    derivative that blows up as its argument approaches +/-1, which happens
    whenever ``zeta (omega_n/Omega)`` gets small, while the arctangent stays
    smooth throughout. And unlike ``arctan2`` it accepts complex arguments, so
    the component remains verifiable by complex step.

    The argument of the arctangent is ``cot phi``, which p. 460 uses directly
    for the acceleration cross-coupling.

    The expression is the classical forced-oscillator lag written with the
    once-per-rev forcing frequency ``Omega``: with ``r = Omega/omega_n`` it
    reduces to ``tan phi = 2 zeta r / (1 - r^2)``.

    Notes
    -----
    ``zeta`` varies across nodes through the Lock number, so ``phi`` is a
    ``(num_nodes,)`` vector while ``omega_n_ratio`` stays scalar.

    The arctangent argument requires ``zeta > 0``, which holds for any blade
    with ``gamma > 0``.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('zeta', val=np.full(nn, 0.5),
                       desc='Damping ratio, c/c_crit.')
        self.add_input('omega_n_ratio', val=1.0,
                       desc='Undamped flapping frequency ratio.')

        self.add_output('phi', val=np.full(nn, 0.5 * np.pi), units='rad',
                        desc='Flapping phase angle.')

        ar = np.arange(nn)
        self.declare_partials('phi', 'zeta', rows=ar, cols=ar)
        self.declare_partials('phi', 'omega_n_ratio')

    def compute(self, inputs, outputs):
        nu = inputs['omega_n_ratio']
        cot_phi = (nu ** 2 - 1.0) / (2.0 * inputs['zeta'] * nu)
        outputs['phi'] = 0.5 * np.pi - np.arctan(cot_phi)

    def compute_partials(self, inputs, J):
        nu, zeta = inputs['omega_n_ratio'], inputs['zeta']

        y = 2.0 * zeta * nu       # sin side
        x = nu ** 2 - 1.0         # cos side
        d2 = x ** 2 + y ** 2

        J['phi', 'zeta'] = 2.0 * nu * x / d2
        J['phi', 'omega_n_ratio'] = (2.0 * (zeta * x - nu * y) / d2).reshape(-1, 1)
