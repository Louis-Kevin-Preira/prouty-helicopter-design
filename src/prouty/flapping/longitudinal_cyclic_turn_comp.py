"""Longitudinal cyclic needed to suppress the flapping from a pitch rate.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", p. 475.
"""

import numpy as np
import openmdao.api as om

from prouty.flapping import flapping_2x2 as f2


class LongitudinalCyclicTurnComp(om.ExplicitComponent):
    """Longitudinal cyclic that cancels the flapping due to angular rates.

    A pitch rate makes the disc lag, and that flapping has to be suppressed
    with cyclic to keep zero pitching moment about the centre of gravity
    (p. 475, ignoring fuselage and stabiliser damping until Chapter 9)::

        Delta_B_1 = [1 / (da_1s/dB_1)] (16/gamma) (q/Omega)

    with the sensitivity read off the trim equation of longitudinal
    flapping::

        da_1s/dB_1 = -(1 + (3/2) mu^2) / (1 - mu^2/2)

    Implementation
    --------------
    The printed numerator ``(16/gamma)(q/Omega)`` is the *hover*, zero-offset
    value of ``-a_1s,rate`` from p. 473, while the denominator is the forward
    flight sensitivity. Mixing the two leaves a factor of ``1/(1 - mu^2/2)``
    unaccounted for: 4.7 % at ``mu = 0.3`` and 11 % at ``mu = 0.45``. See
    entry C7-10 of ``docs/validation_flapping.md``.

    The code takes ``a_1s_rate`` as an input instead, so numerator and
    denominator are evaluated at the same flight condition and the result
    stays valid with hinge offset and with a roll rate present::

        Delta_B_1 = -a_1s,rate / (da_1s/dB_1)

    The sensitivity is likewise taken from the 2x2 system of
    ``flapping_2x2``. Since ``B_1`` appears in ``N_1`` and not in ``N_2``::

        da_1s/dB_1 = -(1 + (3/2) mu^2)(1 + mu^2/2) / Delta

    which collapses to the printed expression when ``kappa = 0``, because
    ``Delta`` is then ``(1 - mu^2/2)(1 + mu^2/2)``.

    Options
    -------
    num_nodes : int
    exact_denominator : bool
        Keep ``kappa^2`` in ``Delta``, default True, consistent with the
        other flapping components. False reproduces p. 475 exactly, and
        makes the sensitivity independent of ``gamma`` and ``e/R``.

    Notes
    -----
    Suppressing ``a_1s`` with ``B_1`` also produces lateral flapping when the
    hinge offset is nonzero, ``db_1s/dB_1 = kappa (1 + (3/2) mu^2) / Delta``.
    That is the acceleration cross-coupling of p. 460 seen from the control
    side, and it adds to the lateral cyclic of ``LateralCyclicTurnComp``.

    For the example manoeuvre p. 475 gets ``Delta_B_1 = -0.62 deg`` against
    ``Delta_A_1 = -0.73 deg``, which is what Prouty calls an almost
    one-to-one coupling for the pilot.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('exact_denominator', types=bool, default=True,
                             desc='Keep kappa^2 in the determinant.')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('a_1s_rate', val=np.zeros(nn), units='rad',
                       desc='Longitudinal flapping due to rates.')
        self.add_input('mu', val=np.full(nn, 0.3), desc='Advance ratio.')
        self.add_input('gamma', val=np.full(nn, 8.0), desc='Lock number.')
        self.add_input('e_over_R', val=0.0, desc='Hinge offset ratio, e/R.')

        self.add_output('delta_B_1', val=np.zeros(nn), units='rad',
                        desc='Change in longitudinal cyclic.')
        self.add_output('da1s_dB1', val=-np.ones(nn),
                        desc='Sensitivity of longitudinal flapping to B_1.')
        self.add_output('db1s_dB1', val=np.zeros(nn),
                        desc='Lateral flapping produced by B_1.')

        outs = ['delta_B_1', 'da1s_dB1', 'db1s_dB1']
        ar = np.arange(nn)
        self.declare_partials('delta_B_1', 'a_1s_rate', rows=ar, cols=ar)
        self.declare_partials(outs, 'mu', rows=ar, cols=ar)
        if self.options['exact_denominator']:
            self.declare_partials(outs, 'gamma', rows=ar, cols=ar)
            self.declare_partials(outs, 'e_over_R')
        else:
            self.declare_partials('db1s_dB1', 'gamma', rows=ar, cols=ar)
            self.declare_partials('db1s_dB1', 'e_over_R')

    def _state(self, inputs):
        mu, x = inputs['mu'], inputs['e_over_R']
        kap = f2.kappa(inputs['gamma'], x)
        return {
            'mu': mu, 'x': x, 'kap': kap,
            'A': 1.0 + 1.5 * mu ** 2,
            'B': 1.0 + mu ** 2 / 2.0,
            'det': f2.determinant(mu, kap, self.options['exact_denominator']),
        }

    def compute(self, inputs, outputs):
        s = self._state(inputs)
        sens = -s['A'] * s['B'] / s['det']

        outputs['da1s_dB1'] = sens
        outputs['db1s_dB1'] = s['kap'] * s['A'] / s['det']
        outputs['delta_B_1'] = -inputs['a_1s_rate'] / sens

    def compute_partials(self, inputs, J):
        s = self._state(inputs)
        mu, x, kap, A, B, det = (s['mu'], s['x'], s['kap'], s['A'], s['B'],
                                 s['det'])
        a_r = inputs['a_1s_rate']
        gam = inputs['gamma']
        exact = self.options['exact_denominator']

        dkap_dgam = -kap / gam
        dkap_dx = f2.d_kappa_d_e_over_R(gam, x)
        two_kap = 2.0 * kap if exact else 0.0

        # delta_B_1 = a_1s_rate * det / (A B)
        J['delta_B_1', 'a_1s_rate'] = det / (A * B)

        for name, ddet, dkap in (('mu', -mu ** 3, 0.0),
                                 ('gamma', two_kap * dkap_dgam, dkap_dgam),
                                 ('e_over_R', two_kap * dkap_dx, dkap_dx)):
            if name == 'mu':
                dA, dB = 3.0 * mu, mu
            else:
                dA = dB = 0.0
                if not exact and name in ('gamma', 'e_over_R'):
                    ddet = 0.0

            d_sens = (-(dA * B + A * dB) * det + A * B * ddet) / det ** 2
            d_delta = a_r * (ddet * A * B - det * (dA * B + A * dB)) / (A * B) ** 2
            d_lat = (dkap * A + kap * dA) / det - kap * A * ddet / det ** 2

            scalar = name == 'e_over_R'
            store = ((lambda v: np.atleast_1d(v).reshape(-1, 1)) if scalar
                     else (lambda v: np.broadcast_to(v, mu.shape).copy()))

            if exact or name == 'mu':
                J['delta_B_1', name] = store(d_delta)
                J['da1s_dB1', name] = store(d_sens)
            J['db1s_dB1', name] = store(d_lat)
