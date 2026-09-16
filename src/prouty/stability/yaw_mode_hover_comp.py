"""The yaw mode in hover.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", p. 605.
"""

import warnings

import numpy as np
import openmdao.api as om

LN_2 = np.log(2.0)


class YawModeHoverComp(om.ExplicitComponent):
    """One root, one damping time: the simplest mode in the chapter.

    p. 605 treats hover motion about the yaw axis as a mass and damper with no
    spring, since nothing in a hovering helicopter opposes a steady heading
    change::

        -I_zz r_dot + (dN/dr) r = 0        ->        s = (dN/dr) / I_zz

    There is no oscillation and no stiffness term, so the mode is a pure
    convergence whenever ``dN/dr`` is negative, which for a conventional
    helicopter it is: the tail rotor's -17,797 beats the main rotor's governed
    engine term of +4,471 (Table 9.4, p. 573).

    Options
    -------
    num_nodes : int

    When it diverges
    ----------------
    ``dN/dr`` positive means yaw-rate divergence and there is no time to half
    amplitude. ``time_to_half`` is then zero with zero partials and a warning
    is raised. ``root`` stays exact and differentiable throughout, so it is
    what to constrain -- and it is the more useful quantity anyway, being the
    reciprocal of the yaw time constant.

    Example helicopter
    ------------------
    With ``dN/dr = -13,326 ft lb/rad/sec`` from Table 9.4 and
    ``I_zz = 35,000 slug ft^2`` from Table 9.20's ``-35,000 s^2``,
    ``s = -.3807`` and ``t_half = 1.821 s``, against the printed -.38 and
    1.82 s.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('dN_dr', val=np.full(nn, -13326.0),
                       units='lbf*ft*s/rad')
        self.add_input('I_zz', val=np.full(nn, 35000.0), units='slug*ft**2')

        self.add_output('root', val=np.zeros(nn), units='1/s')
        self.add_output('time_to_half', val=np.zeros(nn), units='s')

        self.declare_partials(['root', 'time_to_half'], ['dN_dr', 'I_zz'],
                              rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        root = inputs['dN_dr'] / inputs['I_zz']
        outputs['root'] = root

        converging = np.real(root) < 0.0
        if not np.all(converging):
            warnings.warn('dN/dr is not negative at every node: the yaw mode '
                          'is a divergence there, and time_to_half is set to '
                          'zero.')
        safe = np.where(converging, root, -1.0)
        outputs['time_to_half'] = np.where(converging, -LN_2 / safe, 0.0)

    def compute_partials(self, inputs, J):
        dN_dr, I_zz = inputs['dN_dr'], inputs['I_zz']
        root = dN_dr / I_zz

        J['root', 'dN_dr'] = 1.0 / I_zz
        J['root', 'I_zz'] = -dN_dr / I_zz ** 2

        converging = root < 0.0
        safe = np.where(converging, root, -1.0)
        chain = np.where(converging, LN_2 / safe ** 2, 0.0)
        J['time_to_half', 'dN_dr'] = chain * J['root', 'dN_dr']
        J['time_to_half', 'I_zz'] = chain * J['root', 'I_zz']
