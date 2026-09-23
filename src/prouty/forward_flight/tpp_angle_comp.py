"""
TppAngleComp -- angle of attack of the tip path plane.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 167 (approximation) and p. 192 / 194 (equilibrium form).

Two forms of the same angle, selected by the `form` option.

form = 'parasite'  (p. 167, initialisation)

    alpha_TPP = - (f / A_b) (mu^2 / 2) / (C_T/sigma)       rad

    Prouty derives it from alpha_TPP = -atan(D_F / G.W.) by dropping the
    in-plane forces and the fuselage lift. The two are algebraically identical
    up to the arctangent, since q f / T = (f/A_b)(mu^2/2)/(C_T/sigma) exactly.
    Note the reference force is the rotor thrust, not the gross weight, because
    C_T/sigma carries it -- at mu = 0.3 that is a 3 % difference.

form = 'forces'  (p. 192, extended to climb and descent p. 194)

    alpha_TPP = -atan[(D_F + H_M + H_T + G.W. sin gamma) / (G.W. - L_F)]
    T         = sqrt[(G.W. - L_F)^2 + (D_F + H_M + H_T + G.W. sin gamma)^2]

    The two share the same numerator and denominator: alpha_TPP is the
    direction of the resultant of the forces the rotor must balance, T is its
    magnitude. They are computed together because splitting them into two
    components would duplicate the force sum and let the two drift apart --
    p. 192 prints them as a pair for exactly that reason.

    A book discrepancy in climb. p. 194 extends the numerator of alpha_TPP by
    G.W. sin(gamma) but does not restate T, and Table 3.3's climb column is
    consistent with T having been left in its level flight form: 21,297 lb
    without the climb term against 21,290 tabulated, where the shared
    numerator gives 21,502. Everything downstream in that column follows the
    21,290 -- it is the C_T/sigma that reproduces the tabulated theta_0 of
    18.6 deg. This component keeps the shared numerator, which is the
    physically correct one: in a steady climb the rotor must balance the
    weight component along the flight path like any other in-plane force. The
    1 % it costs against Table 3.3's climb thrust is documented rather than
    matched. Level flight and autorotation are unaffected, agreeing to 0.01 %
    and 0.09 %.

    This is the equilibrium relation the trim loop of p. 192 iterates on. The
    climb angle gamma is positive in climb, negative in descent, and zero in
    level flight, which recovers the p. 192 form exactly. Nothing else changes
    between the three flight conditions of Table 3.3.

Sign convention: alpha_TPP is negative nose down, the usual level flight case
(-3.7 deg for the example helicopter at mu = 0.3, Table 3.2 p. 193), and turns
positive in autorotation (+4.3 deg, Table 3.3 p. 196).

    mu, CT_sigma, f, A_b                         --> 'parasite' --> alpha_TPP
    GW, L_F, D_F, H_M, H_T, gamma_fp             --> 'forces'   --> alpha_TPP

The 'forces' form is what G1b uses once the H-forces and the fuselage
aerodynamics exist; 'parasite' only ever serves to start that loop.
"""

import numpy as np
import openmdao.api as om


class TppAngleComp(om.ExplicitComponent):
    """Tip path plane angle of attack, initialisation or equilibrium form."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('form', values=('parasite', 'forces'),
                             default='parasite',
                             desc="'parasite': p. 167 initialisation; "
                                  "'forces': p. 192/194 equilibrium")

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_output('alpha_TPP', shape=(nn,), units='rad',
                        desc='tip path plane angle of attack')

        if self.options['form'] == 'parasite':
            self.add_input('mu', shape=(nn,), desc='tip speed ratio')
            self.add_input('CT_sigma', shape=(nn,), desc='C_T / sigma')
            self.add_input('f', shape=(nn,), units='ft**2',
                           desc='fuselage equivalent flat plate area')
            self.add_input('A_b', val=240.0, units='ft**2', desc='blade area')

            for name in ('mu', 'CT_sigma', 'f'):
                self.declare_partials('alpha_TPP', name, rows=ar, cols=ar)
            self.declare_partials('alpha_TPP', 'A_b', rows=ar, cols=zeros)
        else:
            self.add_input('GW', val=20000.0, units='lbf', desc='gross weight')
            self.add_input('L_F', shape=(nn,), units='lbf', desc='fuselage lift')
            self.add_input('D_F', shape=(nn,), units='lbf', desc='fuselage drag')
            self.add_input('H_M', shape=(nn,), units='lbf', desc='main rotor H-force')
            self.add_input('H_T', shape=(nn,), units='lbf', desc='tail rotor H-force')
            self.add_input('gamma_fp', shape=(nn,), val=0.0, units='rad',
                           desc='flight path angle, positive in climb')

            self.add_output('T', shape=(nn,), units='lbf',
                            desc='rotor thrust, p. 192')

            for name in ('L_F', 'D_F', 'H_M', 'H_T', 'gamma_fp'):
                self.declare_partials(['alpha_TPP', 'T'], name,
                                      rows=ar, cols=ar)
            self.declare_partials(['alpha_TPP', 'T'], 'GW', rows=ar, cols=zeros)

    def compute(self, inputs, outputs):
        if self.options['form'] == 'parasite':
            outputs['alpha_TPP'] = -(inputs['f'] / inputs['A_b'][0]) \
                * 0.5 * inputs['mu'] ** 2 / inputs['CT_sigma']
        else:
            num, den = self._forces(inputs)
            outputs['alpha_TPP'] = -np.arctan(num / den)
            outputs['T'] = np.sqrt(den ** 2 + num ** 2)

    def _forces(self, inputs):
        """Numerator and denominator of the equilibrium relation, p. 192/194.

        arctan2 would be the robust choice, but it does not accept complex
        arguments and would block check_partials under complex step. The plain
        arctan is safe here because the denominator G.W. - L_F stays positive:
        it would only change sign if the fuselage lifted more than the whole
        helicopter weighs.
        """
        GW = inputs['GW'][0]
        num = (inputs['D_F'] + inputs['H_M'] + inputs['H_T']
               + GW * np.sin(inputs['gamma_fp']))
        den = GW - inputs['L_F']
        if np.any(np.real(den) <= 0.0):
            raise om.AnalysisError('TppAngleComp: G.W. - L_F must stay positive, '
                                   f'got {np.real(den)}.')
        return num, den

    def compute_partials(self, inputs, partials):
        if self.options['form'] == 'parasite':
            f, A_b = inputs['f'], inputs['A_b'][0]
            mu, CT_sigma = inputs['mu'], inputs['CT_sigma']
            alpha = -(f / A_b) * 0.5 * mu ** 2 / CT_sigma

            partials['alpha_TPP', 'f'] = alpha / f
            partials['alpha_TPP', 'A_b'] = -alpha / A_b
            partials['alpha_TPP', 'mu'] = 2.0 * alpha / mu
            partials['alpha_TPP', 'CT_sigma'] = -alpha / CT_sigma
        else:
            GW = inputs['GW'][0]
            num, den = self._forces(inputs)
            r = den ** 2 + num ** 2
            d_num, d_den = -den / r, num / r          # d alpha / d num, d alpha / d den

            T = np.sqrt(r)
            t_num, t_den = num / T, den / T        # d T / d num, d T / d den
            cos_g = np.cos(inputs['gamma_fp'])
            sin_g = np.sin(inputs['gamma_fp'])

            for name in ('D_F', 'H_M', 'H_T'):
                partials['alpha_TPP', name] = d_num
                partials['T', name] = t_num
            partials['alpha_TPP', 'L_F'] = -d_den
            partials['T', 'L_F'] = -t_den
            partials['alpha_TPP', 'gamma_fp'] = d_num * GW * cos_g
            partials['T', 'gamma_fp'] = t_num * GW * cos_g
            partials['alpha_TPP', 'GW'] = d_num * sin_g + d_den
            partials['T', 'GW'] = t_num * sin_g + t_den
