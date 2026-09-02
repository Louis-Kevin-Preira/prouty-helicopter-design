"""
TunnelWallCorrectionComp -- wind tunnel wall correction to the disc angle.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 247-248 (Table 3.5, case 9), with the wall interference factor
from reference 3.47.

    chi = arccot( 2 v_1 / V ) = arccot[ (C_T/sigma) sigma / mu^2 ]
    delta_alpha_TPP = - delta_WL (A_M/A_tun) (C_T/sigma) sigma / (2 mu^2)
    alpha_TPP_corr  = alpha_TPP_uncorr + delta_alpha_TPP

A rotor in a closed test section pushes its wake into the walls, which turn it
back and change the inflow the rotor sees. The correction is written as an
equivalent change of disc angle, proportional to the induced velocity ratio and
to how much of the test section the disc fills.

chi is the wake skew angle, measured from the disc plane: 90 deg means the wake
goes straight aft, 0 deg straight down. Writing 2 v_1/V in coefficients,
v_1/Omega R = sigma (C_T/sigma)/2 mu and V/Omega R = mu give
2 v_1/V = sigma (C_T/sigma)/mu^2 with nothing left over. At the worked
condition it is 86 deg -- the wake is nearly in the plane of the disc, which is
why the walls matter at all.

delta_WL is NOT computed here. It comes from reference 3.47 as a function of
the tunnel width-to-height ratio, the model height over half the tunnel height,
and chi; that reference is not available, so the factor is an input. The worked
example gives delta_WL = -0.54 for gamma_tun = 2, zeta = 1 and chi = 86 deg.
Its sign convention matters: negative delta_WL with the minus sign in front
produces a POSITIVE correction, tilting the disc further back, which is the
right direction for a closed section -- the walls constrain the wake and the
rotor behaves as if at a larger angle of attack.

Worked check, p. 247-248: sigma = 0.062, mu = 0.3, C_T/sigma = 0.105,
A_M/A_tun = 1,820/3,200 = 0.57, delta_WL = -0.54 give chi = 86 deg,
delta_alpha_TPP = 0.6 deg and alpha_TPP_corr = 5.3 + 0.6 = 5.9 deg.

The correction is small here and it is not negligible: 0.6 deg on 5.3 moves
C_T/sigma from 0.105 to 0.106, and the whole point of case 9 is that the
measured value is 0.110.

    alpha_TPP_uncorr, CT_sigma, mu, sigma, area_ratio, delta_WL
        --> chi, delta_alpha_TPP, alpha_TPP_corr
"""

import numpy as np
import openmdao.api as om


class TunnelWallCorrectionComp(om.ExplicitComponent):
    """Wake skew angle and the wall correction to alpha_TPP, p. 247-248."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('alpha_TPP_uncorr', shape=(nn,), units='rad',
                       desc='shaft angle plus longitudinal flapping')
        self.add_input('CT_sigma', shape=(nn,))
        self.add_input('mu', shape=(nn,))
        self.add_input('sigma', val=0.062)
        self.add_input('area_ratio', val=0.57,
                       desc='disc area over test section area')
        self.add_input('delta_WL', val=-0.54,
                       desc='wall interference factor, reference 3.47')

        self.add_output('chi', shape=(nn,), units='rad',
                        desc='wake skew angle from the disc plane')
        self.add_output('inflow_ratio', shape=(nn,),
                        desc='2 v_1 / V in coefficients')
        self.add_output('delta_alpha_TPP', shape=(nn,), units='rad')
        self.add_output('alpha_TPP_corr', shape=(nn,), units='rad')

        for out in ('chi', 'inflow_ratio'):
            self.declare_partials(out, ['CT_sigma', 'mu'], rows=ar, cols=ar)
            self.declare_partials(out, 'sigma', rows=ar, cols=zeros)
        for out in ('delta_alpha_TPP', 'alpha_TPP_corr'):
            self.declare_partials(out, ['CT_sigma', 'mu'], rows=ar, cols=ar)
            self.declare_partials(out, ['sigma', 'area_ratio', 'delta_WL'],
                                  rows=ar, cols=zeros)
        self.declare_partials('alpha_TPP_corr', 'alpha_TPP_uncorr', rows=ar,
                              cols=ar, val=1.0)

    def _ratio(self, inputs):
        """2 v_1 / V, and the pieces the derivatives need."""
        sigma, mu = inputs['sigma'][0], inputs['mu']
        return sigma * inputs['CT_sigma'] / mu ** 2

    def compute(self, inputs, outputs):
        ratio = self._ratio(inputs)
        scale = -0.5 * inputs['delta_WL'][0] * inputs['area_ratio'][0]

        outputs['inflow_ratio'] = ratio
        # chi = pi/2 - arctan(ratio) rather than arctan(1/ratio): the same
        # branch, regular as C_T goes to zero where the wake lies in the disc
        # plane and chi tends to 90 deg, and unlike arctan2 it accepts the
        # complex arguments check_partials needs
        outputs['chi'] = 0.5 * np.pi - np.arctan(ratio)
        outputs['delta_alpha_TPP'] = scale * ratio
        outputs['alpha_TPP_corr'] = inputs['alpha_TPP_uncorr'] + scale * ratio

    def compute_partials(self, inputs, partials):
        sigma, mu = inputs['sigma'][0], inputs['mu']
        CT = inputs['CT_sigma']
        ratio = self._ratio(inputs)
        scale = -0.5 * inputs['delta_WL'][0] * inputs['area_ratio'][0]

        d_ratio = {'CT_sigma': sigma / mu ** 2,
                   'mu': -2.0 * sigma * CT / mu ** 3,
                   'sigma': CT / mu ** 2}
        d_chi = -1.0 / (1.0 + ratio ** 2)   # d(chi)/d(ratio)

        for name, value in d_ratio.items():
            partials['inflow_ratio', name] = value
            partials['chi', name] = d_chi * value
            for out in ('delta_alpha_TPP', 'alpha_TPP_corr'):
                partials[out, name] = scale * value

        for out in ('delta_alpha_TPP', 'alpha_TPP_corr'):
            partials[out, 'area_ratio'] = (
                -0.5 * inputs['delta_WL'][0] * ratio)
            partials[out, 'delta_WL'] = -0.5 * inputs['area_ratio'][0] * ratio
