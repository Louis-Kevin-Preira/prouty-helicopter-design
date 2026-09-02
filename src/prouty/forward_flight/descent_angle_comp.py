"""
DescentAngleComp -- autorotative descent angle written in rotor coefficients.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 239 (Table 3.5, case 5, steps g through o).

    C_D_F/sigma = mu^2 (f/A_b) / 2
    alpha_TPP   = atan[ lambda'/mu + sigma (C_T/sigma) / (2 mu^2) ]
    sin(gamma_D) = [ alpha_TPP (C_T/sigma) / 57.3 + C_D_F/sigma + C_H/sigma ]
                   / (C_T/sigma)
    R/D = 60 mu (Omega R) sin(gamma_D)

The 57.3 is there because Prouty carries alpha_TPP in degrees; in radians the
whole thing collapses to

    sin(gamma_D) = alpha_TPP + (C_D_F/sigma + C_H/sigma) / (C_T/sigma)

which reads directly: the flight path is the disc tilt plus whatever drag the
thrust has to drag along, expressed as a fraction of that thrust.

C_D_F/sigma is the fuselage drag written as a rotor coefficient. From
D_F = q f with q = (rho/2)(mu Omega R)^2, dividing by rho A_b (Omega R)^2
leaves mu^2 (f/A_b)/2 exactly -- no approximation.

alpha_TPP is not taken as an input. It is lambda' inverted:
lambda' = mu alpha_TPP - v_1/Omega R with the high speed momentum inflow
v_1/Omega R = sigma (C_T/sigma) / 2 mu gives the expression above, and Prouty's
step j computes it that way, with 1/(2 mu^2) printed as 5.56 at mu = 0.3.

The same descent, two ways. PropulsiveBalanceComp gets sin(gamma_D) from
F/G.W. with F = q(f + f_M) + H_T; this component gets it from rotor
coefficients. They are the same balance: q f_M / T works out to tan(alpha_TPP),
q f / T to (C_D_F/sigma)/(C_T/sigma), and H_T/T to its own coefficient ratio.
The two agree to the small angle approximation and to whatever fuselage lift
does to the difference between T and G.W. Running both is a real cross-check
on a chapter where several tabulated columns turned out not to satisfy their
own equations.

Worked check, p. 239: mu = 0.3, C_T/sigma = 0.082, lambda' = 0.021,
sigma = 0.085, f = 19.7 ft^2, A_b = 240 ft^2, C_H/sigma = 0.0004 gives
C_D_F/sigma = 0.0037, alpha_TPP = 6.4 deg, sin(gamma_D) = 0.162 and
R/D = 1,895 ft/min.

    lambda_p, CT_sigma, CH_sigma, mu, sigma, f, A_b, V_tip
        --> CD_F_sigma, alpha_TPP, sin_gamma_D, gamma_D, gamma_fp, R_D
"""

import numpy as np
import openmdao.api as om

SECONDS_PER_MINUTE = 60.0


class DescentAngleComp(om.ExplicitComponent):
    """Autorotative descent angle from rotor coefficients, p. 239."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('lambda_p', shape=(nn,))
        self.add_input('CT_sigma', shape=(nn,))
        self.add_input('CH_sigma', shape=(nn,), val=0.0)
        self.add_input('mu', shape=(nn,))
        self.add_input('sigma', val=0.084883)
        self.add_input('f', shape=(nn,), units='ft**2',
                       desc='fuselage equivalent flat plate area')
        self.add_input('A_b', val=240.0, units='ft**2', desc='blade area')
        self.add_input('V_tip', val=650.0, units='ft/s')

        self.add_output('CD_F_sigma', shape=(nn,),
                        desc='fuselage drag as a rotor coefficient')
        self.add_output('alpha_TPP', shape=(nn,), units='rad')
        self.add_output('sin_gamma_D', shape=(nn,))
        self.add_output('gamma_D', shape=(nn,), units='rad',
                        desc='descent angle, positive down')
        self.add_output('gamma_fp', shape=(nn,), units='rad',
                        desc='flight path angle, positive in climb')
        self.add_output('R_D', shape=(nn,), units='ft/min',
                        desc='rate of descent, positive down')

        vector = ('lambda_p', 'CT_sigma', 'CH_sigma', 'mu', 'f')
        scalar = ('sigma', 'A_b', 'V_tip')
        angles = ('sin_gamma_D', 'gamma_D', 'gamma_fp', 'R_D')

        for name in ('mu', 'f'):
            self.declare_partials('CD_F_sigma', name, rows=ar, cols=ar)
        self.declare_partials('CD_F_sigma', 'A_b', rows=ar, cols=zeros)
        for name in ('lambda_p', 'CT_sigma', 'mu'):
            self.declare_partials('alpha_TPP', name, rows=ar, cols=ar)
        self.declare_partials('alpha_TPP', 'sigma', rows=ar, cols=zeros)
        for out in angles:
            for name in vector:
                self.declare_partials(out, name, rows=ar, cols=ar)
            for name in scalar:
                self.declare_partials(out, name, rows=ar, cols=zeros)

    def _pieces(self, inputs):
        mu, CT = inputs['mu'], inputs['CT_sigma']
        sigma, A_b = inputs['sigma'][0], inputs['A_b'][0]

        argument = inputs['lambda_p'] / mu + sigma * CT / (2.0 * mu ** 2)
        CD_F = mu ** 2 * inputs['f'] / (2.0 * A_b)
        alpha = np.arctan(argument)
        sine = alpha + (CD_F + inputs['CH_sigma']) / CT
        return argument, CD_F, alpha, sine

    def compute(self, inputs, outputs):
        _, CD_F, alpha, sine = self._pieces(inputs)

        if np.any(np.abs(np.real(sine)) >= 1.0):
            raise om.AnalysisError(
                'DescentAngleComp: no descent angle exists, '
                f'sin(gamma_D) = {np.real(sine)}.')

        outputs['CD_F_sigma'] = CD_F
        outputs['alpha_TPP'] = alpha
        outputs['sin_gamma_D'] = sine
        outputs['gamma_D'] = np.arcsin(sine)
        outputs['gamma_fp'] = -np.arcsin(sine)
        outputs['R_D'] = (SECONDS_PER_MINUTE * inputs['mu']
                          * inputs['V_tip'][0] * sine)

    def compute_partials(self, inputs, partials):
        mu, CT = inputs['mu'], inputs['CT_sigma']
        sigma, A_b = inputs['sigma'][0], inputs['A_b'][0]
        V_tip = inputs['V_tip'][0]
        argument, CD_F, alpha, sine = self._pieces(inputs)

        d_CD_F = {'mu': mu * inputs['f'] / A_b,
                  'f': mu ** 2 / (2.0 * A_b),
                  'A_b': -CD_F / A_b}
        for name, value in d_CD_F.items():
            partials['CD_F_sigma', name] = value

        d_argument = {'lambda_p': 1.0 / mu,
                      'CT_sigma': sigma / (2.0 * mu ** 2),
                      'mu': -inputs['lambda_p'] / mu ** 2 - sigma * CT / mu ** 3,
                      'sigma': CT / (2.0 * mu ** 2)}
        d_alpha = {name: value / (1.0 + argument ** 2)
                   for name, value in d_argument.items()}
        for name, value in d_alpha.items():
            partials['alpha_TPP', name] = value

        drag = CD_F + inputs['CH_sigma']
        d_sine = {name: d_alpha.get(name, 0.0)
                  + d_CD_F.get(name, 0.0) / CT
                  for name in ('lambda_p', 'CT_sigma', 'mu', 'f', 'sigma',
                               'A_b')}
        d_sine['CT_sigma'] = d_sine['CT_sigma'] - drag / CT ** 2
        d_sine['CH_sigma'] = 1.0 / CT
        d_sine['V_tip'] = np.zeros_like(CT)

        root = np.sqrt(1.0 - sine ** 2)
        speed = SECONDS_PER_MINUTE * mu * V_tip
        for name, value in d_sine.items():
            partials['sin_gamma_D', name] = value
            partials['gamma_D', name] = value / root
            partials['gamma_fp', name] = -value / root
            partials['R_D', name] = speed * value

        partials['R_D', 'mu'] = (speed * d_sine['mu']
                                 + SECONDS_PER_MINUTE * V_tip * sine)
        partials['R_D', 'V_tip'] = SECONDS_PER_MINUTE * mu * sine
