"""
DragRiseMachComp -- drag rise Mach number and its ratio to the advancing tip.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 184 and p. 233 (Table 3.5, case 2, steps g and h).

    M_dr3   = M_dr3_cl0 [ 1 - 6 (C_T/sigma)^2 ]
    M_ratio = M_190 / M_dr3

M_dr3 is the THREE-dimensional drag rise Mach number at the blade tip. It is
higher than the two-dimensional value because of tip relief: near the tip the
flow can escape spanwise, so shocks form later. Prouty takes
M_dr2 / M_dr3 = 0.92 for the 0012 (p. 184), and Figure 3.42 fits the
two-dimensional 0012 data with c_d = 0.0080 + 12.5 (M - 0.74)^3 above
M = 0.74. Those two numbers give M_dr3_cl0 = 0.74 / 0.92 = 0.804, which is
the 0.80 used in Table 3.5 -- worth knowing, because 0.80 otherwise looks
like a round number pulled from nowhere.

The [1 - 6 (C_T/sigma)^2] factor is empirical (p. 184): Prouty fits it to the
full-scale wind tunnel results of reference 3.27 and the flight tests of
reference 3.28, having noted that a loaded rotor stalls and shocks earlier
than an unloaded one. It costs 4.4 % at C_T/sigma = 0.086 and 14 % at
C_T/sigma = 0.15, so it is not a detail near the thrust limit.

M_ratio is the abscissa of Figure 3.43, which returns
(delta C_Q/sigma)_comp / M_OmegaR^3. This component stops there: the figure
itself is a digitised surface and belongs in CompressibilityTorqueComp.

Note that no compressibility loss exists while M_190 < M_dr3, that is
M_ratio < 1. For the example helicopter at OmegaR = 650 ft/s and mu = 0.3,
M_190 = 0.757 against M_dr3 = 0.764, so the rotor sits just below the
threshold -- which is why case 1 of Table 3.5 ignores compressibility and
case 2, at OmegaR = 750 ft/s, cannot.

The spanwise variation of M_dr between M_dr2 inboard and M_dr3 at the tip
(p. 184) is not modelled here: Figure 3.43 has already integrated it.

    CT_sigma, M_190, M_dr3_cl0 --> M_dr3, M_ratio (nn,)
"""

import numpy as np
import openmdao.api as om

THRUST_FACTOR = 6.0             # empirical coefficient of p. 184


class DragRiseMachComp(om.ExplicitComponent):
    """Drag rise Mach number corrected for thrust, p. 184."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('CT_sigma', shape=(nn,), desc='C_T / sigma')
        self.add_input('M_190', shape=(nn,), desc='advancing tip Mach number')
        self.add_input('M_dr3_cl0', val=0.80,
                       desc='three-dimensional drag rise Mach at zero lift')

        self.add_output('M_dr3', shape=(nn,), desc='drag rise Mach number')
        self.add_output('M_ratio', shape=(nn,), desc='M_190 / M_dr3')

        for out in ('M_dr3', 'M_ratio'):
            self.declare_partials(out, 'CT_sigma', rows=ar, cols=ar)
            self.declare_partials(out, 'M_dr3_cl0', rows=ar, cols=zeros)
        self.declare_partials('M_ratio', 'M_190', rows=ar, cols=ar)

    def _M_dr3(self, inputs):
        M_dr3 = inputs['M_dr3_cl0'][0] * (
            1.0 - THRUST_FACTOR * inputs['CT_sigma'] ** 2)
        if np.any(np.real(M_dr3) <= 0.0):
            raise om.AnalysisError(
                'DragRiseMachComp: the thrust correction drove M_dr3 to zero; '
                f'C_T/sigma = {np.real(inputs["CT_sigma"])} is out of range.')
        return M_dr3

    def compute(self, inputs, outputs):
        M_dr3 = self._M_dr3(inputs)
        outputs['M_dr3'] = M_dr3
        outputs['M_ratio'] = inputs['M_190'] / M_dr3

    def compute_partials(self, inputs, partials):
        CT_sigma, M_190 = inputs['CT_sigma'], inputs['M_190']
        M_dr3_cl0 = inputs['M_dr3_cl0'][0]
        M_dr3 = self._M_dr3(inputs)

        d_CT = -2.0 * THRUST_FACTOR * M_dr3_cl0 * CT_sigma

        partials['M_dr3', 'CT_sigma'] = d_CT
        partials['M_dr3', 'M_dr3_cl0'] = M_dr3 / M_dr3_cl0

        partials['M_ratio', 'M_190'] = 1.0 / M_dr3
        partials['M_ratio', 'CT_sigma'] = -M_190 * d_CT / M_dr3 ** 2
        partials['M_ratio', 'M_dr3_cl0'] = -M_190 / (M_dr3 * M_dr3_cl0)


# Upper panel of Figure 3.43, p. 185: M_dr3 at zero lift against effective
# thickness, 2 x max(y_upper)/c in per cent. Digitised from the plot (126
# points, cubic fit, worst residual 0.003) and consistent with the 0.80 that
# Prouty uses for the NACA 0012.
#
#     t/c %      0     6     9    12    15    18    20
#     M_dr3   1.003 0.882 0.837 0.800 0.770 0.745 0.729
#
M_DR3_VS_THICKNESS = (-0.0000119, 0.000780, -0.024565, 1.00342)


def m_dr3_from_thickness(t_c_percent):
    """Zero lift three-dimensional drag rise Mach number, Figure 3.43 top."""
    return np.polyval(M_DR3_VS_THICKNESS, t_c_percent)
