"""Pitch rate produced by a manoeuvre at a given load factor.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", p. 474.
"""

import numpy as np
import openmdao.api as om

G_STANDARD = 32.174  # ft/s**2


class ManeuverRateComp(om.ExplicitComponent):
    """Body pitch rate sustained in a steady turn or a pull-up.

    p. 474 quotes from Chapter 5, for a steady turn with no sideslip::

        q = (g/V) (n^2 - 1) / n

    Where it comes from. In a level turn at load factor ``n`` the bank angle
    satisfies ``cos(phi) = 1/n``, so ``sin(phi) = sqrt(n^2 - 1)/n`` and the
    turn rate is ``psi_dot = g sqrt(n^2 - 1) / V``. The body pitch rate is the
    component of that about the pitch axis, ``q = psi_dot sin(phi)``, which
    gives the expression above.

    Options
    -------
    num_nodes : int
    maneuver : {'level_turn', 'pull_up'}
        ``'level_turn'`` is the p. 474 formula. ``'pull_up'``, the other
        manoeuvre p. 474 names, is a pure vertical pull with no bank, where
        the load factor produces ``V q = (n - 1) g``, hence::

            q = (g/V) (n - 1)

        The two differ by more than they look: at ``n = 1.5`` a level turn
        needs 0.138 rad/s where a pull-up needs 0.083, a factor of 1.67.
    g : float
        Gravitational acceleration, ft/s**2.

    Notes
    -----
    ``V`` is declared in ``ft/s``; supplying it in ``kn`` is converted by
    OpenMDAO.

    Both expressions give ``q = 0`` at ``n = 1``, as they must.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('maneuver', values=('level_turn', 'pull_up'),
                             default='level_turn')
        self.options.declare('g', default=G_STANDARD,
                             desc='Gravitational acceleration, ft/s**2.')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('V', val=np.ones(nn), units='ft/s',
                       desc='True airspeed.')
        self.add_input('n', val=np.ones(nn), desc='Load factor.')

        self.add_output('q', val=np.zeros(nn), units='rad/s',
                        desc='Pitch rate, positive nose up.')

        ar = np.arange(nn)
        self.declare_partials('q', ['V', 'n'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        n = inputs['n']
        factor = n - 1.0 / n if self.options['maneuver'] == 'level_turn' \
            else n - 1.0

        outputs['q'] = self.options['g'] / inputs['V'] * factor

    def compute_partials(self, inputs, J):
        n, V = inputs['n'], inputs['V']
        g = self.options['g']

        if self.options['maneuver'] == 'level_turn':
            factor, d_factor = n - 1.0 / n, 1.0 + 1.0 / n ** 2
        else:
            factor, d_factor = n - 1.0, np.ones_like(n)

        J['q', 'V'] = -g * factor / V ** 2
        J['q', 'n'] = g * d_factor / V
