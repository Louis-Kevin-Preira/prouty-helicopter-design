"""
CompressibilityTorqueComp -- torque penalty due to compressibility.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 183-185 (Figure 3.43) and p. 233 (Table 3.5, case 2, steps i-k).

    dCQ_comp = f(mu, M_ratio) * M_tip^3          M_ratio = M_190 / M_dr3

Figure 3.43 gives (delta C_Q/sigma)_comp / M_OmegaR^3 against M_190/M_dr3,
one curve per tip speed ratio. The six curves were digitised from the plot by
tracking dark pixel clusters row by row -- 223 rows in which all six curves
separate cleanly, spanning 0.0013 to 0.0199 in ordinate -- then fitted with

    f = 0.020 [ (M_ratio - 1) / s(mu) ]^p(mu)        M_ratio > 1
    f = 0                                            M_ratio <= 1

    s(mu) = 0.331957 - 0.313071 mu + 0.043571 mu^2
    p(mu) = 2.441500 + 2.010929 mu - 1.760714 mu^2

s(mu) is the abscissa offset at which a curve reaches the top of the plot,
and it falls almost exactly linearly with mu. The exponent is close to 3 for
mu >= 0.3 and drops towards 2.5 at mu = 0, so a single exponent does not fit
the family -- an earlier attempt with one exponent left the mu = 0 curve 70 %
out at mid-range.

Accuracy against the digitised curves: worst residual 0.00086 and rms 0.00025
in units of the ordinate, over mu = 0 to 0.5. At Prouty's own worked point,
(mu, M_ratio) = (0.3, 1.143), the fit gives 0.00438 where he reads 0.0043 off
the same figure; the digitised curve itself passes through 0.00453 there. The
spread between those three numbers, about 4 %, is the intrinsic accuracy of
reading a hand-drawn plot, and nothing in this module is better than that.

Why a fit and not the integral of p. 183. Prouty gives the underlying double
integral,

    (1/2 pi) int int (r/R + mu sin psi)^2 (r/R) delta_c_d d(r/R) dpsi,
    delta_c_d = K1 M_OmegaR^3 (r/R + mu sin psi - M_dr/M_OmegaR)^3,

with the spanwise M_dr of p. 184. Evaluating it (scripts/compressibility_
study.py) reproduces Figure 3.43 at mu = 0 within 15 % but falls a factor
2 to 4 short for every mu > 0, and -- worse -- gets the trend backwards: the
integral makes the penalty DECREASE with mu at fixed M_ratio, because the
advancing blade spends less of the revolution above the drag rise Mach, while
the figure makes it increase by a factor 2.7 between mu = 0 and mu = 0.3. The
two cannot both be right, and Figure 3.43 is what Table 3.5 and the isolated
rotor charts were built on, so the figure wins. The integral is kept in
scripts/ as a documented disagreement, not as a fallback.

Note also that the printed integral carries no factor 1/2, although the
elemental torque (rho/2) U_T^2 delta_c_d c r does. Restoring it would halve
the result and double the disagreement, so Figure 3.43 appears to have been
computed from the equation as printed.

Smoothness. The exponent exceeds 1 everywhere in 0 <= mu <= 0.5, so both the
value and its first derivative vanish as M_ratio approaches 1 from above. No smoothstep blending
is needed at the threshold -- which matters, because the example helicopter
sits at M_ratio = 0.99 in level flight and an optimiser will spend its time
right on that boundary.

    mu, M_ratio, M_tip --> CompressibilityTorqueComp --> dCQ_sigma_comp (nn,)
"""

import numpy as np
import openmdao.api as om

F_REF = 0.020                                   # top of the plotted range
SCALE = (0.043571, -0.313071, 0.331957)         # s(mu), highest power first
EXPON = (-1.760714, 2.010929, 2.441500)         # p(mu), highest power first


class CompressibilityTorqueComp(om.ExplicitComponent):
    """Compressibility torque increment, Figure 3.43."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('mu', shape=(nn,), desc='tip speed ratio')
        self.add_input('M_ratio', shape=(nn,), desc='M_190 / M_dr3')
        self.add_input('M_tip', shape=(nn,), desc='hovering tip Mach number')

        self.add_output('dCQ_sigma_comp', shape=(nn,),
                        desc='compressibility increment to C_Q/sigma')
        self.add_output('f_chart', shape=(nn,),
                        desc='Figure 3.43 ordinate, dCQ/sigma_comp / M_tip^3')

        for out in ('dCQ_sigma_comp', 'f_chart'):
            self.declare_partials(out, ['mu', 'M_ratio'], rows=ar, cols=ar)
        self.declare_partials('dCQ_sigma_comp', 'M_tip', rows=ar, cols=ar)

    def _chart(self, inputs):
        """Figure 3.43 ordinate and the pieces needed for its derivatives."""
        mu = inputs['mu']
        scale = np.polyval(SCALE, mu)
        expon = np.polyval(EXPON, mu)
        excess = inputs['M_ratio'] - 1.0
        active = np.real(excess) > 0.0
        z = np.where(active, excess / scale, 1.0)
        f = np.where(active, F_REF * z ** expon, 0.0)
        return f, z, scale, expon, active

    def compute(self, inputs, outputs):
        f, *_ = self._chart(inputs)
        outputs['f_chart'] = f
        outputs['dCQ_sigma_comp'] = f * inputs['M_tip'] ** 3

    def compute_partials(self, inputs, partials):
        f, z, scale, expon, active = self._chart(inputs)
        mu, M_tip = inputs['mu'], inputs['M_tip']

        # f = F_REF z^p with z = (M_ratio-1)/s; both s and p depend on mu
        df_dz = np.where(active, expon * f / z, 0.0)
        d_scale = np.polyval(np.polyder(SCALE), mu)
        d_expon = np.polyval(np.polyder(EXPON), mu)

        df_dratio = df_dz / scale
        df_dmu = np.where(
            active,
            -df_dz * z * d_scale / scale + f * np.log(z) * d_expon, 0.0)

        partials['f_chart', 'M_ratio'] = df_dratio
        partials['f_chart', 'mu'] = df_dmu

        partials['dCQ_sigma_comp', 'M_ratio'] = df_dratio * M_tip ** 3
        partials['dCQ_sigma_comp', 'mu'] = df_dmu * M_tip ** 3
        partials['dCQ_sigma_comp', 'M_tip'] = 3.0 * f * M_tip ** 2
