"""Lateral cyclic needed to suppress roll during a manoeuvre.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 474-475.
"""

import numpy as np
import openmdao.api as om


class LateralCyclicTurnComp(om.ExplicitComponent):
    """Lateral cyclic that keeps the lateral flapping at zero in a turn.

    Two effects push the disc sideways during a pitching manoeuvre, and they
    oppose each other (p. 474). The pitch rate tilts the disc down to the
    left, which costs::

        Delta_A_1 |pitch rate = q / Omega

    The rise in thrust raises the coning, which tilts the disc down to the
    right. From the Chapter 3 relation quoted on p. 474::

        A_1 - b_1s = -[(4/3) mu a_0 + v_1/(Omega R)] / (1 + mu^2/2)

    Options
    -------
    num_nodes : int
    coning_source : {'load_factor', 'direct'}
        ``'load_factor'`` is the p. 474 shortcut. Since ``a_0`` and ``v_1``
        are described as direct functions of rotor thrust::

            Delta_A_1 |manoeuvre = (n - 1) A_1 |level flight

        ``'direct'`` evaluates the Chapter 3 relation at the manoeuvre's own
        ``a_0`` and ``v_1`` instead::

            A_1 = q/Omega - [(4/3) mu a_0 + v_1/(Omega R)] / (1 + mu^2/2)

    Why the shortcut is approximate
    -------------------------------
    ``v_1`` does scale with thrust, but ``a_0`` does not: its blade weight
    term (p. 467) is independent of the load factor. Scaling the whole of
    ``a_0`` by ``n`` therefore leaves an error of
    ``(4/3) mu a_0,weight (n - 1)`` in the numerator. For the example
    helicopter the weight term is 0.19 deg of a 3.2 deg coning, about 6 % of
    it, but coning contributes only two thirds of the numerator, so the net
    error on ``A_1`` is 1.3 % at ``n = 1.5``. Minor here; it grows with
    ``mu``, with tip speed reductions and with light-blade rotors.

    Notes
    -----
    Sign convention as in Figure 7.11. For the example helicopter at 115 kt
    and ``n = 1.5``, p. 475 gets ``Delta_A_1 = 0.37 - 1.10 = -0.73 deg``: the
    coning effect wins, so the pilot holds left stick to stop a roll to the
    right.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('coning_source',
                             values=('load_factor', 'direct'),
                             default='load_factor')

    def setup(self):
        nn = self.options['num_nodes']
        by_load_factor = self.options['coning_source'] == 'load_factor'

        self.add_input('q', val=np.zeros(nn), units='rad/s',
                       desc='Pitch rate, positive nose up.')
        self.add_input('Omega', val=np.ones(nn), units='rad/s',
                       desc='Rotor angular speed.')
        self.add_input('A_1_level', val=np.zeros(nn), units='rad',
                       desc='Lateral cyclic in level flight.')

        if by_load_factor:
            self.add_input('n', val=np.ones(nn), desc='Load factor.')
            sources = ['q', 'Omega', 'n', 'A_1_level']
        else:
            self.add_input('mu', val=np.full(nn, 0.3), desc='Advance ratio.')
            self.add_input('a_0', val=np.zeros(nn), units='rad',
                           desc='Coning angle in the manoeuvre.')
            self.add_input('v1_over_OmegaR', val=np.zeros(nn),
                           desc='Mean induced velocity ratio.')
            sources = ['q', 'Omega', 'mu', 'a_0', 'v1_over_OmegaR',
                       'A_1_level']

        self.add_output('delta_A_1', val=np.zeros(nn), units='rad',
                        desc='Change in lateral cyclic for the manoeuvre.')
        self.add_output('A_1_maneuver', val=np.zeros(nn), units='rad',
                        desc='Lateral cyclic during the manoeuvre.')

        ar = np.arange(nn)
        self.declare_partials(['delta_A_1', 'A_1_maneuver'], sources,
                              rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        rate = inputs['q'] / inputs['Omega']
        A_L = inputs['A_1_level']

        if self.options['coning_source'] == 'load_factor':
            delta = rate + (inputs['n'] - 1.0) * A_L
            outputs['delta_A_1'] = delta
            outputs['A_1_maneuver'] = A_L + delta
        else:
            mu = inputs['mu']
            coning = ((4.0 / 3.0) * mu * inputs['a_0']
                      + inputs['v1_over_OmegaR']) / (1.0 + mu ** 2 / 2.0)
            outputs['A_1_maneuver'] = rate - coning
            outputs['delta_A_1'] = rate - coning - A_L

    def compute_partials(self, inputs, J):
        q, Om = inputs['q'], inputs['Omega']

        d = {'q': 1.0 / Om, 'Omega': -q / Om ** 2}

        if self.options['coning_source'] == 'load_factor':
            d['n'] = inputs['A_1_level']
            d['A_1_level'] = inputs['n'] - 1.0
            extra = {'A_1_level': inputs['n']}
        else:
            mu, a_0 = inputs['mu'], inputs['a_0']
            den = 1.0 + mu ** 2 / 2.0
            top = (4.0 / 3.0) * mu * a_0 + inputs['v1_over_OmegaR']

            d['a_0'] = -(4.0 / 3.0) * mu / den
            d['v1_over_OmegaR'] = -1.0 / den
            d['mu'] = -((4.0 / 3.0) * a_0 * den - top * mu) / den ** 2
            d['A_1_level'] = -np.ones_like(mu)
            extra = {'A_1_level': np.zeros_like(mu)}

        for name, val in d.items():
            J['delta_A_1', name] = np.broadcast_to(val, Om.shape).copy()
            J['A_1_maneuver', name] = np.broadcast_to(
                extra.get(name, val), Om.shape).copy()
