"""Flapping produced by pitch and roll rates.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 469-473.
"""

import numpy as np
import openmdao.api as om

from prouty.flapping import flapping_2x2 as f2


class RateFlappingComp(om.ExplicitComponent):
    """Flapping increments due to pitch rate ``q`` and roll rate ``p``.

    A rotor being pitched or rolled sees an extra vertical velocity at each
    blade element, and the tip path plane also carries gyroscopic moments
    (p. 472)::

        M_gyro,sine   = -2 q Omega I_b
        M_gyro,cosine =  2 p Omega I_b

    Setting the total sine and cosine hinge moments to zero (p. 472) and
    dividing by ``Omega^2 I_b`` gives the same 2x2 system as steady flapping,
    with a different right-hand side::

        N_1 = p/Omega - (16/gamma) (q/Omega) / (1 - e/R)^2
        N_2 = -q/Omega - (16/gamma) (p/Omega) / (1 - e/R)^2

    The solution reproduces the two expressions printed on p. 473, which have
    been checked symbolically term by term.

    Options
    -------
    num_nodes : int
    exact_denominator : bool
        Keep ``kappa^2`` in the determinant, default True. p. 473 prints only
        the simplified form, but the underlying system is identical to
        pp. 468-469, so the default matches ``ClosedFormFlappingComp``. Set
        False to reproduce p. 473 verbatim.

    Notes
    -----
    Sign conventions are those of Figure 7.11: ``q`` nose up, ``p`` roll
    right, ``a_1s`` tip path plane back, ``b_1s`` tip path plane right.

    In hover with no hinge offset and pitch rate only, the system collapses
    to ``a_1s = -(16/gamma)(q/Omega)`` and ``b_1s = -(q/Omega)`` (p. 473): the
    disc lags nose down and rolls left. This is *rate* cross-coupling, as
    opposed to the acceleration cross-coupling of p. 460.

    p. 473 prints ``(1 - mu^4/4)`` on both ``a_1s`` and ``b_1s`` here. The
    ``(1 + mu^4/4)`` printed on ``a_1s`` on p. 469 is therefore the odd one
    out; see entry C7-4 of ``docs/validation_flapping.md``.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('exact_denominator', types=bool, default=True,
                             desc='Keep kappa^2 in the determinant.')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('e_over_R', val=0.0, desc='Hinge offset ratio, e/R.')
        self.add_input('gamma', val=np.full(nn, 8.0), desc='Lock number.')
        self.add_input('mu', val=np.full(nn, 0.3), desc='Advance ratio.')
        self.add_input('Omega', val=np.ones(nn), units='rad/s',
                       desc='Rotor angular speed.')
        self.add_input('p', val=np.zeros(nn), units='rad/s',
                       desc='Roll rate, positive right.')
        self.add_input('q', val=np.zeros(nn), units='rad/s',
                       desc='Pitch rate, positive nose up.')

        self.add_output('a_1s_rate', val=np.zeros(nn), units='rad',
                        desc='Longitudinal flapping due to rates.')
        self.add_output('b_1s_rate', val=np.zeros(nn), units='rad',
                        desc='Lateral flapping due to rates.')

        outs = ['a_1s_rate', 'b_1s_rate']
        ar = np.arange(nn)
        self.declare_partials(outs, ['gamma', 'mu', 'Omega', 'p', 'q'],
                              rows=ar, cols=ar)
        self.declare_partials(outs, 'e_over_R')

    def _state(self, inputs):
        x, gam, mu = inputs['e_over_R'], inputs['gamma'], inputs['mu']
        s = {'pr': inputs['p'] / inputs['Omega'],
             'qr': inputs['q'] / inputs['Omega'],
             'c16': 16.0 / (gam * (1.0 - x) ** 2)}

        s['N1'] = s['pr'] - s['c16'] * s['qr']
        s['N2'] = -s['qr'] - s['c16'] * s['pr']
        s['kap'] = f2.kappa(gam, x)
        s['det'] = f2.determinant(mu, s['kap'],
                                  self.options['exact_denominator'])
        return s

    def compute(self, inputs, outputs):
        s = self._state(inputs)
        a, b = f2.solve(s['N1'], s['N2'], inputs['mu'], s['kap'], s['det'])
        outputs['a_1s_rate'] = a
        outputs['b_1s_rate'] = b

    def compute_partials(self, inputs, J):
        s = self._state(inputs)
        x, gam, mu, Om = (inputs['e_over_R'], inputs['gamma'], inputs['mu'],
                          inputs['Omega'])
        N1, N2, kap, det, c16 = s['N1'], s['N2'], s['kap'], s['det'], s['c16']
        pr, qr = s['pr'], s['qr']

        a, b = f2.solve(N1, N2, mu, kap, det)

        dkap_dgam = -kap / gam
        dkap_dx = f2.d_kappa_d_e_over_R(gam, x)
        two_kap = 2.0 * kap if self.options['exact_denominator'] else 0.0

        # (dN1, dN2, dkappa, ddet, dq) for each input
        blocks = {
            'p': (1.0 / Om, -c16 / Om, 0.0, 0.0, 0.0),
            'q': (-c16 / Om, -1.0 / Om, 0.0, 0.0, 0.0),
            'Omega': (-N1 / Om, -N2 / Om, 0.0, 0.0, 0.0),
            'gamma': (c16 * qr / gam, c16 * pr / gam, dkap_dgam,
                      two_kap * dkap_dgam, 0.0),
            'mu': (0.0, 0.0, 0.0, -mu ** 3, mu),
            'e_over_R': (-2.0 * c16 * qr / (1.0 - x),
                         -2.0 * c16 * pr / (1.0 - x),
                         dkap_dx, two_kap * dkap_dx, 0.0),
        }

        for name, (dN1, dN2, dkap, ddet, dq) in blocks.items():
            da, db = f2.differentiate(N1, N2, a, b, mu, kap, det,
                                      dN1, dN2, dkap, ddet, dq)
            if name == 'e_over_R':
                J['a_1s_rate', name] = np.atleast_1d(da).reshape(-1, 1)
                J['b_1s_rate', name] = np.atleast_1d(db).reshape(-1, 1)
            else:
                J['a_1s_rate', name] = np.broadcast_to(da, mu.shape).copy()
                J['b_1s_rate', name] = np.broadcast_to(db, mu.shape).copy()
