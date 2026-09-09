"""Lock number of the blade.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", p. 458.
"""

import numpy as np
import openmdao.api as om


class LockNumberComp(om.ExplicitComponent):
    """Nondimensional ratio of aerodynamic to inertial blade forces (p. 458)::

        gamma = c rho a R^4 / I_b

    ``I_b`` is the flapping inertia about the hinge (p. 456), so ``gamma``
    inherits that definition. Prouty notes that most blades fall between 6 and
    10, and that a large tip mass such as a jet engine can drive it down to 2.

    Options
    -------
    mode : {'gamma', 'I_b'}
        ``'gamma'``  ``I_b`` is known, ``gamma`` is computed. ``rho`` and
        ``a`` may vary with the flight condition, so ``gamma`` is a
        ``(num_nodes,)`` vector.
        ``'I_b'``  ``gamma`` is a design choice and ``I_b`` is sized from it.
        A single blade cannot have one inertia per node, so this mode requires
        ``num_nodes=1`` and is meant for a reference condition.

    Notes
    -----
    ``a`` carries ``1/rad`` units. Declaring them is what makes OpenMDAO
    convert a value supplied in ``1/deg``; omitting them would pass the raw
    number through silently.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('mode', values=('gamma', 'I_b'), default='gamma',
                             desc='Which side of the definition is solved for.')

    def setup(self):
        nn = self.options['num_nodes']
        inverse = self.options['mode'] == 'I_b'

        if inverse and nn != 1:
            raise ValueError(
                "LockNumberComp: mode='I_b' sizes a single blade inertia and "
                f"requires num_nodes=1, got {nn}.")

        self.add_input('c', val=1.0, units='ft', desc='Blade chord.')
        self.add_input('R', val=1.0, units='ft', desc='Rotor radius.')
        self.add_input('rho', val=np.full(nn, 0.002378), units='slug/ft**3',
                       desc='Air density.')
        self.add_input('a', val=np.full(nn, 6.0), units='1/rad',
                       desc='Blade section lift curve slope.')

        if inverse:
            self.add_input('gamma', val=8.0, desc='Target Lock number.')
            self.add_output('I_b', val=1.0, units='slug*ft**2',
                            desc='Required blade flapping inertia.')
        else:
            self.add_input('I_b', val=1.0, units='slug*ft**2',
                           desc='Blade flapping inertia about the hinge.')
            self.add_output('gamma', val=np.full(nn, 8.0),
                            desc='Lock number.')

        out, swap = ('I_b', 'gamma') if inverse else ('gamma', 'I_b')
        ar = np.arange(nn)
        self.declare_partials(out, ['rho', 'a'], rows=ar, cols=ar)
        self.declare_partials(out, ['c', 'R', swap])

    def compute(self, inputs, outputs):
        num = inputs['c'] * inputs['rho'] * inputs['a'] * inputs['R'] ** 4

        if self.options['mode'] == 'I_b':
            outputs['I_b'] = num / inputs['gamma']
        else:
            outputs['gamma'] = num / inputs['I_b']

    def compute_partials(self, inputs, J):
        inverse = self.options['mode'] == 'I_b'
        out, swap = ('I_b', 'gamma') if inverse else ('gamma', 'I_b')

        num = inputs['c'] * inputs['rho'] * inputs['a'] * inputs['R'] ** 4
        val = num / inputs[swap]

        J[out, 'c'] = (val / inputs['c']).reshape(-1, 1)
        J[out, 'R'] = (4.0 * val / inputs['R']).reshape(-1, 1)
        J[out, swap] = (-val / inputs[swap]).reshape(-1, 1)
        J[out, 'rho'] = val / inputs['rho']
        J[out, 'a'] = val / inputs['a']
