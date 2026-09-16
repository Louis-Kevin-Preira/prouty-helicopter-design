"""Basic rotor derivatives near hover.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", Table 9.1, pp. 564-565. The
underlying flapping algebra is Chapter 7, pp. 468-473; the two chart entries
come from the hover charts of Chapter 1, as p. 564 states.
"""

import numpy as np
import openmdao.api as om

from prouty.flapping import flapping_2x2 as f2

#: Which inputs each computed output depends on, as (per node, scalar).
DEPENDENCIES = {
    'd_mu_d_xdot': (['Omega_R'], []),
    'd_a1s_d_mu': (['theta_0', 'theta_1', 'v_1_over_Omega_R'], []),
    'd_b1s_d_mu': (['a_0'], []),
    'dCT_sigma_dlambda': (['CT_sigma'], ['a', 'sigma']),
    'dCQ_sigma_dlambda': (['theta_0', 'theta_1', 'v_1_over_Omega_R'], ['a']),
    'd_a1s_dq': (['Omega', 'gamma'], ['e_over_R']),
    'd_a1s_dp': (['Omega', 'gamma'], ['e_over_R']),
    'd_a1s_dA1': (['gamma'], ['e_over_R']),
    'd_a1s_dB1': (['gamma'], ['e_over_R']),
    'dCH_sigma_da1s': (['CT_sigma', 'theta_0', 'theta_1'], ['a']),
    'dM_da1s': (['rho', 'Omega_R', 'gamma'], ['e_over_R', 'A_b', 'R', 'a']),
}

#: Second entry of each Table 9.1 row, as (source output, sign).
MIRRORS = {
    'd_lambda_d_zdot': ('d_mu_d_xdot', 1.0),
    'd_lambda_d_ydot': ('d_mu_d_xdot', -1.0),
    'd_b1s_dp': ('d_a1s_dq', 1.0),
    'd_b1s_dq': ('d_a1s_dp', -1.0),
    'd_b1s_dB1': ('d_a1s_dA1', 1.0),
    'd_b1s_dA1': ('d_a1s_dB1', -1.0),
    'dCY_sigma_db1s': ('dCH_sigma_da1s', 1.0),
    'dR_db1s': ('dM_da1s', 1.0),
}

UNITS = {
    'd_mu_d_xdot': 's/ft', 'd_a1s_d_mu': None, 'd_b1s_d_mu': None,
    'dCT_sigma_dlambda': None, 'dCQ_sigma_dlambda': None,
    'd_a1s_dq': 's', 'd_a1s_dp': 's', 'd_a1s_dA1': None, 'd_a1s_dB1': None,
    'dCH_sigma_da1s': None, 'dM_da1s': 'lbf*ft/rad',
}
UNITS.update({name: UNITS[source] for name, (source, _) in MIRRORS.items()})


class BasicRotorDerivativesHoverComp(om.ExplicitComponent):
    """Table 9.1: the non-dimensional building blocks of the hover derivatives.

    Everything Tables 9.2 and 9.3 need, apart from geometry and the two
    entries p. 564 sends to the Chapter 1 hover charts. The same component
    serves both rotors: Table 9.1 prints one equation column and two value
    columns, so ``rotor`` is not an option here. What differs between the two
    rotors is which of these outputs the dimensional tables consume, and that
    belongs to Tables 9.2 and 9.3.

    Computed outputs, in the order of the printed table::

        d_mu_d_xdot      =  1/(Omega R)
        d_a1s_d_mu       =  (8/3) theta_0 + 2 theta_1 - 2 v_1/(Omega R)
        d_b1s_d_mu       =  (4/3) a_0
        dCT_sigma_dlambda = 1 / [8/a + sqrt(sigma/2)/sqrt(C_T/sigma)]
        dCQ_sigma_dlambda = -(a/4) [theta_75 - 2 v_1/(Omega R)]
        d_a1s_dq         = -16/[gamma Omega (1-e/R)^2] - 12(e/R)/[gamma Omega (1-e/R)^3]
        d_a1s_dp         =  (1/Omega) [1 - 192(e/R)/(gamma^2 (1-e/R)^5)]
        d_a1s_dA1        =  12(e/R)/[gamma (1-e/R)^3]
        d_a1s_dB1        = -1/[1 + 144(e/R)^2/(gamma^2 (1-e/R)^6)]
        dCH_sigma_da1s   =  (3/2)(C_T/sigma) [1 - (a/18) theta_75/(C_T/sigma)]
        dM_da1s          =  (3/4)(e/R) A_b rho R (Omega R)^2 a / gamma

    Most rows of the table carry a second derivative next to the first, equal
    to it up to a sign. Those are emitted as outputs in their own right rather
    than left to the caller, so that no sign has to be reapplied downstream::

        d_lambda_d_zdot =  d_mu_d_xdot        d_b1s_dB1      =  d_a1s_dA1
        d_lambda_d_ydot = -d_mu_d_xdot        d_b1s_dA1      = -d_a1s_dB1
        d_b1s_dp        =  d_a1s_dq           dCY_sigma_db1s =  dCH_sigma_da1s
        d_b1s_dq        = -d_a1s_dp           dR_db1s        =  dM_da1s

    Signs of the first two rows
    ---------------------------
    The two rows are *not* symmetric, and the asymmetry is easy to lose. p. 564
    prints, for the main rotor, ``(dmu/dxdot)_M`` alongside ``(dlambda'/dzdot)_M``
    with no minus sign, and for the tail rotor ``(dmu/dxdot)_T`` alongside
    ``(-dlambda'/dydot)_T``. Both equal ``1/(Omega R)``, so::

        dlambda'/dzdot =  1/(Omega R)   main rotor
        dlambda'/dydot = -1/(Omega R)   tail rotor

    This is the axis mapping, not a misprint: main rotor thrust acts along
    ``-z`` and tail rotor thrust along ``+y``, so only one of the two flips.
    The main rotor sign is what makes ``dZ/dzdot`` negative, i.e. heave
    damping, in Table 9.2 and again in Table 9.8 (p. 580, -261 lb/ft/sec).

    theta_75 is not an input
    ------------------------
    The table uses ``theta_0`` and ``theta_1`` in one row and ``theta_75`` in
    two others. Taking all three as inputs would let them drift apart, so
    ``theta_75 = theta_0 + 0.75 theta_1`` is formed internally, which is what
    the example helicopter's numbers reproduce.

    ``dCH_sigma_da1s`` is computed as ``(3/2)(C_T/sigma) - a theta_75/12``,
    algebraically identical to the printed form and free of its division by
    ``C_T/sigma``, which would blow up on an unloaded rotor.

    Options
    -------
    num_nodes : int
    rate_flapping : {'table', 'exact'}
        The rate derivatives of p. 565 drop a factor that the cyclic
        derivatives on the same page keep. Writing ``c16 = 16/[gamma(1-e/R)^2]``
        and ``kappa = 12(e/R)/[gamma(1-e/R)^3]``, the Chapter 7 2x2 system at
        ``mu = 0`` (pp. 468-473) gives::

            da_1s/dq = -(c16 + kappa) / [Omega (1 + kappa^2)]
            da_1s/dp =  (1 - kappa c16) / [Omega (1 + kappa^2)]

        and Table 9.1 prints both without the ``1 + kappa^2``, while
        ``da_1s/dB_1 = -1/(1 + kappa^2)`` on the very next line keeps it.
        ``'table'`` is the default and reproduces the printed numbers;
        ``'exact'`` restores the factor and agrees with ``RateFlappingComp``.
        For the example helicopter the two differ by 0.9 %. See entry C9-2 of
        ``docs/validation_stability.md``.

    Example helicopter, main rotor
    ------------------------------
    With ``Omega R = 650``, ``Omega = 21.67``, ``R = 30``, ``e/R = .05``,
    ``gamma = 8.1``, ``a = 6``, ``sigma = .085``, ``A_b = 240``,
    ``rho = .002378``, ``C_T/sigma = .0849``, ``theta_0 = .2794``,
    ``theta_1 = -.1396``, ``v_1/(Omega R) = .062``, ``a_0 = .075``:

    ================= ========= =========
    output            model     Table 9.1
    ================= ========= =========
    d_mu_d_xdot       .001538   .00154
    d_a1s_d_mu        .342      .34
    d_b1s_d_mu        .100      .10
    dCT_sigma_dlambda .490      .49
    dCQ_sigma_dlambda -.0757    -.076
    d_a1s_dq          -.1050    -.105
    d_a1s_dp          .0374     .037
    d_a1s_dA1         .0864     .086
    d_a1s_dB1         -.9926    -.993
    dCH_sigma_da1s    .0400     .040
    dM_da1s           200,941   200,940
    ================= ========= =========
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('rate_flapping', values=('table', 'exact'),
                             default='table')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('e_over_R', val=0.0, desc='Hinge offset ratio, e/R.')
        self.add_input('a', val=6.0, units='1/rad',
                       desc='Blade lift curve slope.')
        self.add_input('sigma', val=0.085, desc='Rotor solidity.')
        self.add_input('R', val=1.0, units='ft', desc='Rotor radius.')
        self.add_input('A_b', val=1.0, units='ft**2', desc='Total blade area.')

        self.add_input('rho', val=np.full(nn, 0.002378), units='slug/ft**3')
        self.add_input('Omega', val=np.ones(nn), units='rad/s')
        self.add_input('Omega_R', val=np.full(nn, 650.0), units='ft/s',
                       desc='Tip speed.')
        self.add_input('gamma', val=np.full(nn, 8.1), desc='Lock number.')
        self.add_input('CT_sigma', val=np.full(nn, 0.085),
                       desc='Thrust coefficient over solidity.')
        self.add_input('theta_0', val=np.zeros(nn), units='rad',
                       desc='Collective pitch at the blade root.')
        self.add_input('theta_1', val=np.zeros(nn), units='rad',
                       desc='Linear twist, negative for washout.')
        self.add_input('v_1_over_Omega_R', val=np.zeros(nn),
                       desc='Induced velocity ratio in hover.')
        self.add_input('a_0', val=np.zeros(nn), units='rad',
                       desc='Coning angle.')

        for name, (per_node, scalar) in self._all_outputs():
            self.add_output(name, val=np.zeros(nn), units=UNITS[name])
            if per_node:
                self.declare_partials(name, per_node, rows=ar, cols=ar)
            if scalar:
                self.declare_partials(name, scalar, rows=ar, cols=zeros)

    def _all_outputs(self):
        """(name, dependencies) for computed outputs, then mirrored ones."""
        for name, deps in DEPENDENCIES.items():
            yield name, deps
        for name, (source, _) in MIRRORS.items():
            yield name, DEPENDENCIES[source]

    def _state(self, inputs):
        """Quantities shared by compute and compute_partials."""
        x, gam = inputs['e_over_R'][0], inputs['gamma']
        s = {
            'x': x,
            'theta_75': inputs['theta_0'] + 0.75 * inputs['theta_1'],
            'c16': 16.0 / (gam * (1.0 - x) ** 2),
            'kap': f2.kappa(gam, x),
            'den': 8.0 / inputs['a'][0] + np.sqrt(0.5 * inputs['sigma'][0]
                                                  / inputs['CT_sigma']),
        }
        s['damp'] = (1.0 if self.options['rate_flapping'] == 'table'
                     else 1.0 / (1.0 + s['kap'] ** 2))
        return s

    def compute(self, inputs, outputs):
        s = self._state(inputs)
        nu, Om, V = inputs['v_1_over_Omega_R'], inputs['Omega'], inputs['Omega_R']

        outputs['d_mu_d_xdot'] = 1.0 / V
        outputs['d_a1s_d_mu'] = (8.0 / 3.0 * inputs['theta_0']
                                 + 2.0 * inputs['theta_1'] - 2.0 * nu)
        outputs['d_b1s_d_mu'] = 4.0 / 3.0 * inputs['a_0']
        outputs['dCT_sigma_dlambda'] = 1.0 / s['den']
        outputs['dCQ_sigma_dlambda'] = -0.25 * inputs['a'][0] * (s['theta_75']
                                                                - 2.0 * nu)
        outputs['d_a1s_dq'] = -(s['c16'] + s['kap']) * s['damp'] / Om
        outputs['d_a1s_dp'] = (1.0 - s['kap'] * s['c16']) * s['damp'] / Om
        outputs['d_a1s_dA1'] = s['kap']
        outputs['d_a1s_dB1'] = -1.0 / (1.0 + s['kap'] ** 2)
        outputs['dCH_sigma_da1s'] = (1.5 * inputs['CT_sigma']
                                     - inputs['a'][0] * s['theta_75'] / 12.0)
        outputs['dM_da1s'] = (0.75 * s['x'] * inputs['A_b'][0] * inputs['rho']
                              * inputs['R'][0] * V ** 2 * inputs['a'][0]
                              / inputs['gamma'])

        for name, (source, sign) in MIRRORS.items():
            outputs[name] = sign * outputs[source]

    def compute_partials(self, inputs, J):
        s = self._state(inputs)
        x, gam, Om, V = s['x'], inputs['gamma'], inputs['Omega'], inputs['Omega_R']
        a, sig, CTs = inputs['a'][0], inputs['sigma'][0], inputs['CT_sigma']
        c16, kap, den, damp = s['c16'], s['kap'], s['den'], s['damp']
        one = np.ones_like(V)

        J['d_mu_d_xdot', 'Omega_R'] = -1.0 / V ** 2

        J['d_a1s_d_mu', 'theta_0'] = 8.0 / 3.0 * one
        J['d_a1s_d_mu', 'theta_1'] = 2.0 * one
        J['d_a1s_d_mu', 'v_1_over_Omega_R'] = -2.0 * one
        J['d_b1s_d_mu', 'a_0'] = 4.0 / 3.0 * one

        d_inv = -1.0 / den ** 2
        J['dCT_sigma_dlambda', 'a'] = d_inv * (-8.0 / a ** 2)
        J['dCT_sigma_dlambda', 'sigma'] = d_inv / (2.0 * np.sqrt(2.0 * sig * CTs))
        J['dCT_sigma_dlambda', 'CT_sigma'] = (d_inv * -0.5 * np.sqrt(0.5 * sig)
                                              * CTs ** -1.5)

        J['dCQ_sigma_dlambda', 'a'] = -0.25 * (s['theta_75']
                                               - 2.0 * inputs['v_1_over_Omega_R'])
        J['dCQ_sigma_dlambda', 'theta_0'] = -0.25 * a * one
        J['dCQ_sigma_dlambda', 'theta_1'] = -0.1875 * a * one
        J['dCQ_sigma_dlambda', 'v_1_over_Omega_R'] = 0.5 * a * one

        # kappa and c16 sensitivities, then the two rate derivatives.
        dkap_dg, dc16_dg = -kap / gam, -c16 / gam
        dkap_dx = f2.d_kappa_d_e_over_R(gam, x)
        dc16_dx = 2.0 * c16 / (1.0 - x)
        ddamp_dkap = (0.0 if self.options['rate_flapping'] == 'table'
                      else -2.0 * kap / (1.0 + kap ** 2) ** 2)

        for name, core, dcore in (
                ('d_a1s_dq', -(c16 + kap), {'g': -(dc16_dg + dkap_dg),
                                            'x': -(dc16_dx + dkap_dx)}),
                ('d_a1s_dp', 1.0 - kap * c16,
                 {'g': -(dkap_dg * c16 + kap * dc16_dg),
                  'x': -(dkap_dx * c16 + kap * dc16_dx)})):
            J[name, 'Omega'] = -core * damp / Om ** 2
            J[name, 'gamma'] = (dcore['g'] * damp
                                + core * ddamp_dkap * dkap_dg) / Om
            J[name, 'e_over_R'] = (dcore['x'] * damp
                                   + core * ddamp_dkap * dkap_dx) / Om

        J['d_a1s_dA1', 'gamma'] = dkap_dg
        J['d_a1s_dA1', 'e_over_R'] = dkap_dx
        d_B1 = 2.0 * kap / (1.0 + kap ** 2) ** 2
        J['d_a1s_dB1', 'gamma'] = d_B1 * dkap_dg
        J['d_a1s_dB1', 'e_over_R'] = d_B1 * dkap_dx

        J['dCH_sigma_da1s', 'CT_sigma'] = 1.5 * one
        J['dCH_sigma_da1s', 'a'] = -s['theta_75'] / 12.0
        J['dCH_sigma_da1s', 'theta_0'] = -a / 12.0 * one
        J['dCH_sigma_da1s', 'theta_1'] = -0.0625 * a * one

        M = (0.75 * x * inputs['A_b'][0] * inputs['rho'] * inputs['R'][0]
             * V ** 2 * a / gam)
        J['dM_da1s', 'rho'] = M / inputs['rho']
        J['dM_da1s', 'Omega_R'] = 2.0 * M / V
        J['dM_da1s', 'gamma'] = -M / gam
        J['dM_da1s', 'A_b'] = M / inputs['A_b'][0]
        J['dM_da1s', 'R'] = M / inputs['R'][0]
        J['dM_da1s', 'a'] = M / a
        J['dM_da1s', 'e_over_R'] = (M / x if x != 0.0 else
                                    0.75 * inputs['A_b'][0] * inputs['rho']
                                    * inputs['R'][0] * V ** 2 * a / gam)

        for name, (source, sign) in MIRRORS.items():
            per_node, scalar = DEPENDENCIES[source]
            for inp in per_node + scalar:
                J[name, inp] = sign * J[source, inp]
