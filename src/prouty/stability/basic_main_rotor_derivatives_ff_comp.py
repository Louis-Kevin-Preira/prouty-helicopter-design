"""Basic main rotor derivatives in forward flight.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", Table 9.6, pp. 576-577. The two
chart partials it consumes are Table 9.5, p. 574, read off the Chapter 3
performance charts as Figure 9.7 (p. 575) illustrates.
"""

import numpy as np
import openmdao.api as om

from prouty.flapping import flapping_2x2 as f2

#: Second entry of the rows Table 9.6 prints as a pair, as (source, sign).
MIRRORS = {'dCY_sigma_db1s': ('dCH_sigma_da1s', 1.0),
           'dR_db1s': ('dM_da1s', 1.0)}

SCALAR_INPUTS = ('e_over_R', 'a', 'sigma', 'R', 'A_b')

UNITS = {
    'd_mu_d_xdot': 's/ft', 'd_lambda_d_xdot': 's/ft',
    'd_lambda_d_zdot': 's/ft', 'd_beta_d_ydot': 'rad*s/ft',
    'dCH_sigma_da1s': None,
    'd_a1s_dq': 's', 'd_a1s_dp': 's', 'd_a1s_dA1': None, 'd_a1s_dB1': None,
    'd_b1s_dq': 's', 'd_b1s_dp': 's', 'd_b1s_dA1': None, 'd_b1s_dB1': None,
    'dM_da1s': 'lbf*ft/rad',
}
UNITS.update({name: UNITS[source] for name, (source, _) in MIRRORS.items()})


class BasicMainRotorDerivativesFFComp(om.ExplicitComponent):
    """Table 9.6: what Table 9.8 needs before it can be dimensionalised.

    Fourteen rows. Four relate body motion to the rotor's own variables, one
    is the H-force sensitivity to disc tilt, eight are flapping derivatives and
    the last is the hub spring.

    The flapping rows, rewritten
    ---------------------------
    Table 9.6 prints eight flapping derivatives as separate fractions. They
    share three rotor factors and four advance-ratio factors::

        c16  = 16/[gamma Omega (1-e/R)^2]      f1 = 1 - mu^2/2
        kappa = 12(e/R)/[gamma (1-e/R)^3]      f2 = 1 + mu^2/2
        k192 = kappa c16                       f3 = 1 - mu^4/4
                                               f4 = 1 + (3/2) mu^2

    and then read::

        da1s/dq  = -c16/f1 - (kappa/Omega)/f3    db1s/dq  = -1/(Omega f2) + k192/f3
        da1s/dp  =  1/(Omega f1) - k192/f3       db1s/dp  = -c16/f2 - (kappa/Omega)/f3
        da1s/dA1 =  kappa f2/f3                  db1s/dA1 =  1
        da1s/dB1 = -f4/f1                        db1s/dB1 =  kappa f4/f3

    ``kappa`` and ``k192 = kappa c16`` are the same quantities Chapter 7 uses,
    so ``flapping_2x2`` supplies them rather than this module recomputing them.

    Where it disagrees with Table 9.1
    ---------------------------------
    Set ``mu = 0`` and six of the eight collapse onto the hover table exactly.
    Two do not::

                        Table 9.6 at mu = 0     Table 9.1 (p. 565)
        da1s/dB1        -1                      -1/(1 + kappa^2)
        db1s/dA1        +1                      +1/(1 + kappa^2)

    Table 9.1 keeps the flapping determinant on its cyclic rows and drops it on
    its rate rows (entry C9-2); Table 9.6 drops it everywhere. For the example
    helicopter that is 0.7 %, and this component reproduces Table 9.6 as
    printed. Entry C9-13 of ``docs/validation_stability.md``.

    The two chart partials
    ----------------------
    ``dCT_sigma_dmu`` and ``dCT_sigma_dlambda`` are Table 9.5, p. 574:
    -.140 and .79 for the main rotor at ``mu = .30``. p. 576 says the ``mu``
    partial was taken as a difference between the ``mu = .25`` and ``mu = .35``
    charts and the others read at ``mu = .30``. They are inputs here, as the
    hover chart entries are in ``BasicRotorDerivativesHoverComp``.

    Options
    -------
    num_nodes : int

    Example helicopter at 115 knots
    -------------------------------
    ================ ========= =========
    output           model     Table 9.6
    ================ ========= =========
    d_mu_d_xdot      .001538   .00154
    d_lambda_d_xdot  .0000370  .000037
    d_lambda_d_zdot  .001384   .00138
    d_beta_d_ydot    .005152   .00515
    dCH_sigma_da1s   .0693     .069
    d_a1s_dq         -.10977   -.1098
    d_a1s_dp         .03958    .0396
    d_a1s_dA1        .09047    .0905
    d_a1s_dB1        -1.1885   -1.188
    d_b1s_dq         -.03542   -.0354
    d_b1s_dp         -.10066   -.101
    d_b1s_dA1        1         1
    d_b1s_dB1        .09826    .0983
    dM_da1s          200,941   200,940
    ================ ========= =========
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('e_over_R', val=0.05)
        self.add_input('a', val=6.0, units='1/rad')
        self.add_input('sigma', val=0.085)
        self.add_input('R', val=30.0, units='ft')
        self.add_input('A_b', val=240.0, units='ft**2')

        self.add_input('rho', val=np.full(nn, 0.002378), units='slug/ft**3')
        self.add_input('Omega', val=np.full(nn, 650.0 / 30.0), units='rad/s')
        self.add_input('Omega_R', val=np.full(nn, 650.0), units='ft/s')
        self.add_input('V', val=np.full(nn, 194.1), units='ft/s',
                       desc='True airspeed.')
        self.add_input('gamma', val=np.full(nn, 8.1))
        self.add_input('mu', val=np.full(nn, 0.30))
        self.add_input('CT_sigma_bar', val=np.full(nn, 0.0865))
        self.add_input('lambda_bar', val=np.full(nn, -0.023),
                       desc="Trim inflow ratio lambda'.")
        self.add_input('alpha_TPP_bar', val=np.full(nn, -0.036631), units='rad')
        self.add_input('dCT_sigma_dmu', val=np.full(nn, -0.140),
                       desc='Table 9.5, p. 574.')
        self.add_input('dCT_sigma_dlambda', val=np.full(nn, 0.79),
                       desc='Table 9.5, p. 574.')

        for name, units in UNITS.items():
            self.add_output(name, val=np.zeros(nn), units=units)

        node, scalar = (ar, ar), (ar, zeros)
        deps = {
            'd_mu_d_xdot': (['Omega_R'], []),
            'd_lambda_d_xdot': (['Omega_R', 'mu', 'CT_sigma_bar',
                                 'alpha_TPP_bar', 'dCT_sigma_dmu'], ['sigma']),
            'd_lambda_d_zdot': (['Omega_R', 'mu', 'dCT_sigma_dlambda'],
                                ['sigma']),
            'd_beta_d_ydot': (['V'], []),
            'dCH_sigma_da1s': (['CT_sigma_bar', 'lambda_bar'], ['a']),
            'd_a1s_dq': (['gamma', 'Omega', 'mu'], ['e_over_R']),
            'd_a1s_dp': (['gamma', 'Omega', 'mu'], ['e_over_R']),
            'd_a1s_dA1': (['gamma', 'mu'], ['e_over_R']),
            'd_a1s_dB1': (['mu'], []),
            'd_b1s_dq': (['gamma', 'Omega', 'mu'], ['e_over_R']),
            'd_b1s_dp': (['gamma', 'Omega', 'mu'], ['e_over_R']),
            'd_b1s_dA1': ([], []),
            'd_b1s_dB1': (['gamma', 'mu'], ['e_over_R']),
            'dM_da1s': (['rho', 'Omega_R', 'gamma'],
                        ['e_over_R', 'A_b', 'R', 'a']),
        }
        self._deps = dict(deps)
        for name, (source, _) in MIRRORS.items():
            self._deps[name] = deps[source]

        for name, (per_node, scalars) in self._deps.items():
            for inp in per_node:
                self.declare_partials(name, inp, rows=node[0], cols=node[1])
            for inp in scalars:
                self.declare_partials(name, inp, rows=scalar[0], cols=scalar[1])

    def _factors(self, inputs):
        """Shared rotor and advance-ratio factors, with their derivatives."""
        x, gam, Om, mu = (inputs['e_over_R'][0], inputs['gamma'],
                          inputs['Omega'], inputs['mu'])
        c16 = 16.0 / (gam * Om * (1.0 - x) ** 2)
        kap = f2.kappa(gam, x)
        dkap_dx = f2.d_kappa_d_e_over_R(gam, x)
        return {
            'c16': c16, 'kap': kap, 'k192': kap * c16,
            'dc16_dg': -c16 / gam, 'dc16_dOm': -c16 / Om,
            'dc16_dx': 2.0 * c16 / (1.0 - x),
            'dkap_dg': -kap / gam, 'dkap_dx': dkap_dx,
            'dk192_dg': -2.0 * kap * c16 / gam,
            'dk192_dOm': -kap * c16 / Om,
            'dk192_dx': dkap_dx * c16 + kap * 2.0 * c16 / (1.0 - x),
            'f1': 1.0 - 0.5 * mu ** 2, 'df1': -mu,
            'f2': 1.0 + 0.5 * mu ** 2, 'df2': mu,
            'f3': 1.0 - 0.25 * mu ** 4, 'df3': -mu ** 3,
            'f4': 1.0 + 1.5 * mu ** 2, 'df4': 3.0 * mu,
        }

    def compute(self, inputs, outputs):
        s = self._factors(inputs)
        V_tip, Om, mu = inputs['Omega_R'], inputs['Omega'], inputs['mu']
        sig, x_T = inputs['sigma'][0], inputs['CT_sigma_bar']
        half = 0.5 * sig / mu

        outputs['d_mu_d_xdot'] = 1.0 / V_tip
        outputs['d_lambda_d_xdot'] = (
            inputs['alpha_TPP_bar']
            - half * (inputs['dCT_sigma_dmu'] - x_T / mu)) / V_tip
        outputs['d_lambda_d_zdot'] = 1.0 / (
            V_tip * (1.0 + inputs['dCT_sigma_dlambda'] * half))
        outputs['d_beta_d_ydot'] = 1.0 / inputs['V']
        outputs['dCH_sigma_da1s'] = x_T + 0.125 * inputs['a'][0] * inputs['lambda_bar']

        outputs['d_a1s_dq'] = -s['c16'] / s['f1'] - s['kap'] / (Om * s['f3'])
        outputs['d_a1s_dp'] = 1.0 / (Om * s['f1']) - s['k192'] / s['f3']
        outputs['d_a1s_dA1'] = s['kap'] * s['f2'] / s['f3']
        outputs['d_a1s_dB1'] = -s['f4'] / s['f1']
        outputs['d_b1s_dq'] = -1.0 / (Om * s['f2']) + s['k192'] / s['f3']
        outputs['d_b1s_dp'] = -s['c16'] / s['f2'] - s['kap'] / (Om * s['f3'])
        outputs['d_b1s_dA1'] = np.ones_like(mu)
        outputs['d_b1s_dB1'] = s['kap'] * s['f4'] / s['f3']

        outputs['dM_da1s'] = (0.75 * inputs['e_over_R'][0] * inputs['A_b'][0]
                              * inputs['rho'] * inputs['R'][0] * V_tip ** 2
                              * inputs['a'][0] / inputs['gamma'])

        for name, (source, sign) in MIRRORS.items():
            outputs[name] = sign * outputs[source]

    def compute_partials(self, inputs, J):
        s = self._factors(inputs)
        V_tip, Om, mu = inputs['Omega_R'], inputs['Omega'], inputs['mu']
        sig, a, gam = inputs['sigma'][0], inputs['a'][0], inputs['gamma']
        x_T, D_mu = inputs['CT_sigma_bar'], inputs['dCT_sigma_dmu']
        D_lam = inputs['dCT_sigma_dlambda']
        one = np.ones_like(mu)

        J['d_mu_d_xdot', 'Omega_R'] = -1.0 / V_tip ** 2

        # lambda' with forward speed: B = alpha - (sigma/2mu) D_mu + (sigma/2mu^2) x_T
        B = (inputs['alpha_TPP_bar'] - 0.5 * sig / mu * D_mu
             + 0.5 * sig / mu ** 2 * x_T)
        J['d_lambda_d_xdot', 'Omega_R'] = -B / V_tip ** 2
        J['d_lambda_d_xdot', 'alpha_TPP_bar'] = one / V_tip
        J['d_lambda_d_xdot', 'dCT_sigma_dmu'] = -0.5 * sig / (mu * V_tip)
        J['d_lambda_d_xdot', 'CT_sigma_bar'] = 0.5 * sig / (mu ** 2 * V_tip)
        J['d_lambda_d_xdot', 'mu'] = (0.5 * sig / mu ** 2 * D_mu
                                      - sig / mu ** 3 * x_T) / V_tip
        J['d_lambda_d_xdot', 'sigma'] = (-0.5 * D_mu / mu
                                         + 0.5 * x_T / mu ** 2) / V_tip

        G = 1.0 + D_lam * 0.5 * sig / mu
        value = 1.0 / (V_tip * G)
        J['d_lambda_d_zdot', 'Omega_R'] = -value / V_tip
        J['d_lambda_d_zdot', 'dCT_sigma_dlambda'] = -value / G * 0.5 * sig / mu
        J['d_lambda_d_zdot', 'sigma'] = -value / G * 0.5 * D_lam / mu
        J['d_lambda_d_zdot', 'mu'] = value / G * 0.5 * D_lam * sig / mu ** 2

        J['d_beta_d_ydot', 'V'] = -1.0 / inputs['V'] ** 2

        J['dCH_sigma_da1s', 'CT_sigma_bar'] = one
        J['dCH_sigma_da1s', 'lambda_bar'] = 0.125 * a * one
        J['dCH_sigma_da1s', 'a'] = 0.125 * inputs['lambda_bar']

        # -c16/f1 - kap/(Omega f3)
        J['d_a1s_dq', 'gamma'] = (-s['dc16_dg'] / s['f1']
                                  - s['dkap_dg'] / (Om * s['f3']))
        J['d_a1s_dq', 'Omega'] = (-s['dc16_dOm'] / s['f1']
                                  + s['kap'] / (Om ** 2 * s['f3']))
        J['d_a1s_dq', 'e_over_R'] = (-s['dc16_dx'] / s['f1']
                                     - s['dkap_dx'] / (Om * s['f3']))
        J['d_a1s_dq', 'mu'] = (s['c16'] * s['df1'] / s['f1'] ** 2
                               + s['kap'] * s['df3'] / (Om * s['f3'] ** 2))

        # 1/(Omega f1) - k192/f3
        J['d_a1s_dp', 'gamma'] = -s['dk192_dg'] / s['f3']
        J['d_a1s_dp', 'Omega'] = (-1.0 / (Om ** 2 * s['f1'])
                                  - s['dk192_dOm'] / s['f3'])
        J['d_a1s_dp', 'e_over_R'] = -s['dk192_dx'] / s['f3']
        J['d_a1s_dp', 'mu'] = (-s['df1'] / (Om * s['f1'] ** 2)
                               + s['k192'] * s['df3'] / s['f3'] ** 2)

        # kappa f2/f3
        J['d_a1s_dA1', 'gamma'] = s['dkap_dg'] * s['f2'] / s['f3']
        J['d_a1s_dA1', 'e_over_R'] = s['dkap_dx'] * s['f2'] / s['f3']
        J['d_a1s_dA1', 'mu'] = s['kap'] * (s['df2'] / s['f3']
                                           - s['f2'] * s['df3'] / s['f3'] ** 2)

        J['d_a1s_dB1', 'mu'] = -s['df4'] / s['f1'] + s['f4'] * s['df1'] / s['f1'] ** 2

        # -1/(Omega f2) + k192/f3
        J['d_b1s_dq', 'gamma'] = s['dk192_dg'] / s['f3']
        J['d_b1s_dq', 'Omega'] = (1.0 / (Om ** 2 * s['f2'])
                                  + s['dk192_dOm'] / s['f3'])
        J['d_b1s_dq', 'e_over_R'] = s['dk192_dx'] / s['f3']
        J['d_b1s_dq', 'mu'] = (s['df2'] / (Om * s['f2'] ** 2)
                               - s['k192'] * s['df3'] / s['f3'] ** 2)

        # -c16/f2 - kap/(Omega f3)
        J['d_b1s_dp', 'gamma'] = (-s['dc16_dg'] / s['f2']
                                  - s['dkap_dg'] / (Om * s['f3']))
        J['d_b1s_dp', 'Omega'] = (-s['dc16_dOm'] / s['f2']
                                  + s['kap'] / (Om ** 2 * s['f3']))
        J['d_b1s_dp', 'e_over_R'] = (-s['dc16_dx'] / s['f2']
                                     - s['dkap_dx'] / (Om * s['f3']))
        J['d_b1s_dp', 'mu'] = (s['c16'] * s['df2'] / s['f2'] ** 2
                               + s['kap'] * s['df3'] / (Om * s['f3'] ** 2))

        # kappa f4/f3
        J['d_b1s_dB1', 'gamma'] = s['dkap_dg'] * s['f4'] / s['f3']
        J['d_b1s_dB1', 'e_over_R'] = s['dkap_dx'] * s['f4'] / s['f3']
        J['d_b1s_dB1', 'mu'] = s['kap'] * (s['df4'] / s['f3']
                                           - s['f4'] * s['df3'] / s['f3'] ** 2)

        x = inputs['e_over_R'][0]
        M = (0.75 * x * inputs['A_b'][0] * inputs['rho'] * inputs['R'][0]
             * V_tip ** 2 * a / gam)
        J['dM_da1s', 'rho'] = M / inputs['rho']
        J['dM_da1s', 'Omega_R'] = 2.0 * M / V_tip
        J['dM_da1s', 'gamma'] = -M / gam
        J['dM_da1s', 'A_b'] = M / inputs['A_b'][0]
        J['dM_da1s', 'R'] = M / inputs['R'][0]
        J['dM_da1s', 'a'] = M / a
        J['dM_da1s', 'e_over_R'] = (M / x if x != 0.0 else
                                    0.75 * inputs['A_b'][0] * inputs['rho']
                                    * inputs['R'][0] * V_tip ** 2 * a / gam)

        for name, (source, sign) in MIRRORS.items():
            per_node, scalars = self._deps[name]
            for inp in per_node + scalars:
                J[name, inp] = sign * J[source, inp]
