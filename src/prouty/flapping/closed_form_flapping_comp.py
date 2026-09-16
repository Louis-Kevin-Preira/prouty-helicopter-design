"""Closed-form flapping of pp. 467-469, as a comparison path.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 467-469.
"""

import numpy as np
import openmdao.api as om

from prouty.flapping import flapping_2x2 as f2

G_STANDARD = 32.174  # ft/s**2


class ClosedFormFlappingComp(om.ExplicitComponent):
    """Coning and first-harmonic flapping from the book's closed forms.

    Coning follows the uniform-mass expression of p. 467::

        a_0 = (2/3) gamma (C_T/sigma) / a * (1 - e/R)^2 / (1 + e/2R)
              - (3/2) g R / (Omega R)^2 / (1 + e/2R)

    For ``a_1s`` and ``b_1s`` the book prints two nested fractions (p. 468 in
    full, p. 469 with the small term dropped). Both are the solution of one
    2x2 system, which is what the code solves::

        [ 1 - mu^2/2      -kappa   ] [a_1s]   [N_1]
        [   kappa      1 + mu^2/2  ] [b_1s] = [N_2]

        kappa = 12 (e/R) / [gamma (1 - e/R)^3]

        N_1 = (8/3) theta_0 mu + 2 theta_1 mu - B_1 (1 + 3 mu^2/2)
              + 2 mu [mu alpha_s - v_1/(Omega R)]
        N_2 = A_1 (1 + mu^2/2) + (4/3) mu a_0 + v_1/(Omega R)

    giving, with ``Delta = 1 - mu^4/4 + kappa^2``::

        a_1s = [N_1 (1 + mu^2/2) + kappa N_2] / Delta
        b_1s = [N_2 (1 - mu^2/2) - kappa N_1] / Delta

    This has been checked symbolically against the printed p. 468 expression
    (identical) and against p. 469 (identical once ``kappa^2`` is dropped).
    The ``144 (e/R)^2 / [gamma^2 (1 - e/R)^6 ...]`` of p. 468 is just
    ``kappa^2`` rearranged.

    Options
    -------
    num_nodes : int
    exact_denominator : bool
        Keep ``kappa^2`` in ``Delta`` (default, p. 468). False reproduces the
        p. 469 simplification.
    include_weight : bool
        Keep the blade weight term in the coning (default, p. 467). p. 468
        drops it when deriving the flapping equations. Matches the option of
        the same name on ``FlappingMatrixComp`` so the two paths compare
        like for like.

    Deviations from the printed text
    --------------------------------
    Two, both documented in ``docs/validation_flapping.md``:

    - C7-3, the ``v_1/(Omega R)`` term of ``N_2`` carries a coefficient of 1,
      not the ``4/3`` implied by the ``(4/3) C_T/sigma [... + sigma/2mu]``
      grouping of p. 469.
    - C7-4, ``a_1s`` uses ``(1 - mu^4/4)``, not the ``(1 + mu^4/4)`` printed.
      This falls out of the 2x2 solve above and matches the ``b_1s`` printed
      three lines below it.

    A third, smaller one: p. 469 writes the coning contribution to ``N_2`` as
    ``[(2/3) mu gamma/a] / (1 + (3/2) e/R)``, whereas the coning of p. 467
    carries ``(1 - e/R)^2 / (1 + e/2R)``. The code uses the actual ``a_0``,
    which is consistent; the printed factor is 5.6 % higher at ``e/R = 0.05``.

    Notes
    -----
    ``v1_over_OmegaR`` is an input rather than being rebuilt from
    ``(C_T/sigma) sigma / (2 mu)``. That keeps the component free of a
    division by ``mu`` and lets it take whichever inflow model is upstream.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('exact_denominator', types=bool, default=True,
                             desc='Keep kappa^2 in the determinant (p. 468).')
        self.options.declare('include_weight', types=bool, default=True,
                             desc='Keep the blade weight term in the coning.')
        self.options.declare('g', default=G_STANDARD,
                             desc='Gravitational acceleration, ft/s**2.')

    def setup(self):
        nn = self.options['num_nodes']
        weight = self.options['include_weight']

        self.add_input('e_over_R', val=0.0, desc='Hinge offset ratio, e/R.')
        self.add_input('gamma', val=np.full(nn, 8.0), desc='Lock number.')
        self.add_input('a', val=np.full(nn, 6.0), units='1/rad',
                       desc='Blade section lift curve slope.')
        self.add_input('mu', val=np.full(nn, 0.3), desc='Advance ratio.')
        self.add_input('CT_sigma', val=np.zeros(nn),
                       desc='Thrust coefficient over solidity.')
        self.add_input('v1_over_OmegaR', val=np.zeros(nn),
                       desc='Mean induced velocity ratio.')
        self.add_input('theta_0', val=np.zeros(nn), units='rad',
                       desc='Collective pitch at the flapping hinge.')
        self.add_input('theta_1', val=np.zeros(nn), units='rad',
                       desc='Linear blade twist.')
        self.add_input('alpha_s', val=np.zeros(nn), units='rad',
                       desc='Shaft angle of attack.')
        self.add_input('A_1', val=np.zeros(nn), units='rad',
                       desc='Lateral cyclic pitch.')
        self.add_input('B_1', val=np.zeros(nn), units='rad',
                       desc='Longitudinal cyclic pitch.')
        if weight:
            self.add_input('R', val=1.0, units='ft', desc='Rotor radius.')
            self.add_input('Omega', val=np.ones(nn), units='rad/s',
                           desc='Rotor angular speed.')

        self.add_output('a_0', val=np.zeros(nn), units='rad',
                        desc='Coning angle.')
        self.add_output('a_1s', val=np.zeros(nn), units='rad',
                        desc='Longitudinal flapping.')
        self.add_output('b_1s', val=np.zeros(nn), units='rad',
                        desc='Lateral flapping.')

        ar = np.arange(nn)
        coning_vec = ['gamma', 'a', 'CT_sigma'] + (['Omega'] if weight else [])
        coning_sca = ['e_over_R'] + (['R'] if weight else [])
        self.declare_partials('a_0', coning_vec, rows=ar, cols=ar)
        self.declare_partials('a_0', coning_sca)

        flap_vec = ['gamma', 'a', 'mu', 'CT_sigma', 'v1_over_OmegaR',
                    'theta_0', 'theta_1', 'alpha_s', 'A_1', 'B_1']
        flap_vec += ['Omega'] if weight else []
        self.declare_partials(['a_1s', 'b_1s'], flap_vec, rows=ar, cols=ar)
        self.declare_partials(['a_1s', 'b_1s'], coning_sca)

    # ------------------------------------------------------------------
    def _state(self, inputs):
        x, mu = inputs['e_over_R'], inputs['mu']
        s = {'x': x, 'mu': mu, 'q': mu ** 2 / 2.0}

        s['k1'] = (1.0 - x) ** 2 / (1.0 + x / 2.0)
        s['a0_aero'] = (2.0 / 3.0) * inputs['gamma'] / inputs['a'] \
            * inputs['CT_sigma'] * s['k1']

        if self.options['include_weight']:
            s['Wg'] = (1.5 * self.options['g']
                       / (inputs['Omega'] ** 2 * inputs['R'] * (1.0 + x / 2.0)))
        else:
            s['Wg'] = 0.0 * mu
        s['a0'] = s['a0_aero'] - s['Wg']

        v1 = inputs['v1_over_OmegaR']
        s['N1'] = ((8.0 / 3.0) * inputs['theta_0'] * mu
                   + 2.0 * inputs['theta_1'] * mu
                   - inputs['B_1'] * (1.0 + 1.5 * mu ** 2)
                   + 2.0 * mu * (mu * inputs['alpha_s'] - v1))
        s['N2'] = inputs['A_1'] * (1.0 + s['q']) + (4.0 / 3.0) * mu * s['a0'] + v1

        s['kap'] = f2.kappa(inputs['gamma'], x)
        s['det'] = f2.determinant(mu, s['kap'],
                                  self.options['exact_denominator'])
        return s

    def compute(self, inputs, outputs):
        s = self._state(inputs)
        a, b = f2.solve(s['N1'], s['N2'], s['mu'], s['kap'], s['det'])

        outputs['a_0'] = s['a0']
        outputs['a_1s'] = a
        outputs['b_1s'] = b

    # ------------------------------------------------------------------
    def compute_partials(self, inputs, J):
        s = self._state(inputs)
        x, mu, q = s['x'], s['mu'], s['q']
        N1, N2, kap, det = s['N1'], s['N2'], s['kap'], s['det']
        a1s, b1s = f2.solve(N1, N2, mu, kap, det)

        gam, a, v1 = inputs['gamma'], inputs['a'], inputs['v1_over_OmegaR']
        weight = self.options['include_weight']
        exact = self.options['exact_denominator']

        # --- coning ---------------------------------------------------------
        dk1 = -(1.0 - x) * (2.5 + 0.5 * x) / (1.0 + x / 2.0) ** 2
        da0 = {
            'CT_sigma': (2.0 / 3.0) * gam / a * s['k1'],
            'gamma': s['a0_aero'] / gam,
            'a': -s['a0_aero'] / a,
            'e_over_R': ((2.0 / 3.0) * gam / a * inputs['CT_sigma'] * dk1
                         + 0.5 * s['Wg'] / (1.0 + x / 2.0)),
        }
        if weight:
            da0['R'] = s['Wg'] / inputs['R']
            da0['Omega'] = 2.0 * s['Wg'] / inputs['Omega']

        for name, val in da0.items():
            scalar = name in ('e_over_R', 'R')
            J['a_0', name] = np.atleast_1d(val).reshape(-1, 1) if scalar else val

        # --- derivatives of the four building blocks ------------------------
        dkap_dx = f2.d_kappa_d_e_over_R(gam, x)
        two_kap = 2.0 * kap if exact else 0.0
        c43mu = (4.0 / 3.0) * mu

        # (dN1, dN2, dkappa, ddet, dq) for each input
        blocks = {
            'theta_0': ((8.0 / 3.0) * mu, 0.0, 0.0, 0.0, 0.0),
            'theta_1': (2.0 * mu, 0.0, 0.0, 0.0, 0.0),
            'B_1': (-(1.0 + 1.5 * mu ** 2), 0.0, 0.0, 0.0, 0.0),
            'alpha_s': (2.0 * mu ** 2, 0.0, 0.0, 0.0, 0.0),
            'A_1': (0.0, 1.0 + q, 0.0, 0.0, 0.0),
            'v1_over_OmegaR': (-2.0 * mu, 1.0, 0.0, 0.0, 0.0),
            'CT_sigma': (0.0, c43mu * da0['CT_sigma'], 0.0, 0.0, 0.0),
            'a': (0.0, c43mu * da0['a'], 0.0, 0.0, 0.0),
            'gamma': (0.0, c43mu * da0['gamma'], -kap / gam,
                      two_kap * (-kap / gam), 0.0),
            'mu': ((8.0 / 3.0) * inputs['theta_0'] + 2.0 * inputs['theta_1']
                   - 3.0 * inputs['B_1'] * mu + 4.0 * mu * inputs['alpha_s']
                   - 2.0 * v1,
                   inputs['A_1'] * mu + (4.0 / 3.0) * s['a0'],
                   0.0, -mu ** 3, mu),
            'e_over_R': (0.0, c43mu * da0['e_over_R'], dkap_dx,
                         two_kap * dkap_dx, 0.0),
        }
        if weight:
            blocks['R'] = (0.0, c43mu * da0['R'], 0.0, 0.0, 0.0)
            blocks['Omega'] = (0.0, c43mu * da0['Omega'], 0.0, 0.0, 0.0)

        for name, (dN1, dN2, dkap, ddet, dq) in blocks.items():
            da, db = f2.differentiate(N1, N2, a1s, b1s, mu, kap, det,
                                      dN1, dN2, dkap, ddet, dq)
            if name in ('e_over_R', 'R'):
                J['a_1s', name] = np.atleast_1d(da).reshape(-1, 1)
                J['b_1s', name] = np.atleast_1d(db).reshape(-1, 1)
            else:
                J['a_1s', name] = np.broadcast_to(da, mu.shape).copy()
                J['b_1s', name] = np.broadcast_to(db, mu.shape).copy()
