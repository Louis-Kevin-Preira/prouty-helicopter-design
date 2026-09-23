"""Blade flapping time constant and the azimuth it corresponds to.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", p. 462.
"""

import numpy as np
import openmdao.api as om


class BladeTimeConstantComp(om.ExplicitComponent):
    """Time and azimuth for the blade to reach 63 % of a step response.

    For a step hinge moment the non-oscillatory part of the flapping is
    (p. 462)::

        beta = (M_st / k) [1 - exp(-(c/2) t / I_b)]

    which reaches 1 - 1/e = 63 % of its final value at::

        t_63 = I_b / (c/2) = 2 I_b / c_damp

    The corresponding azimuth travel, which Prouty calls the *azimuth
    constant*, is::

        psi_63 = Omega t_63

    Substituting the damping of p. 458 gives the closed forms printed on
    p. 462::

        t_63   = (16 / (gamma Omega)) / [(1 - e/R)^3 (1 + e/3R)]
        psi_63 = 917 / {gamma (1 - e/R)^3 (1 + e/3R)}   , deg

    The code uses the definition ``2 I_b / c_damp`` and takes ``c_damp`` from
    ``FlapDampingComp``, so the closed forms serve as tests rather than as a
    second code path. This also keeps the component free of any direct
    dependence on ``e/R``.

    Notes
    -----
    ``psi_63`` does not depend on ``Omega`` or on ``I_b``: both cancel against
    ``c_damp``. It varies across nodes only through the Lock number. This is
    the fast blade response that Prouty later invokes to justify neglecting
    flapping dynamics in the helicopter equations of motion.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('I_b', val=1.0, units='slug*ft**2',
                       desc='Blade flapping inertia about the hinge.')
        self.add_input('c_damp', val=np.ones(nn), units='slug*ft**2/s',
                       desc='Aerodynamic flapping damping.')
        self.add_input('Omega', val=np.ones(nn), units='rad/s',
                       desc='Rotor angular speed.')

        self.add_output('t_63', val=np.ones(nn), units='s',
                        desc='Blade flapping time constant.')
        self.add_output('psi_63', val=np.ones(nn), units='rad',
                        desc='Azimuth constant.')

        ar = np.arange(nn)
        self.declare_partials('t_63', 'c_damp', rows=ar, cols=ar)
        self.declare_partials('t_63', 'I_b')
        self.declare_partials('psi_63', ['c_damp', 'Omega'], rows=ar, cols=ar)
        self.declare_partials('psi_63', 'I_b')

    def compute(self, inputs, outputs):
        t_63 = 2.0 * inputs['I_b'] / inputs['c_damp']

        outputs['t_63'] = t_63
        outputs['psi_63'] = inputs['Omega'] * t_63

    def compute_partials(self, inputs, J):
        I_b, c, Om = inputs['I_b'], inputs['c_damp'], inputs['Omega']
        t_63 = 2.0 * I_b / c

        J['t_63', 'c_damp'] = -t_63 / c
        J['t_63', 'I_b'] = (2.0 / c).reshape(-1, 1)

        J['psi_63', 'c_damp'] = -Om * t_63 / c
        J['psi_63', 'Omega'] = t_63
        J['psi_63', 'I_b'] = (2.0 * Om / c).reshape(-1, 1)
