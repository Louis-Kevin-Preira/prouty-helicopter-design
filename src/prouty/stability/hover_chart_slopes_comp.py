"""The two hover chart slopes of Table 9.1, in closed form.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*. Table 9.1,
p. 564, sends ``dCT/sigma/dtheta0`` and ``dCQ/sigma/dtheta0`` to the hover
charts of Chapter 1 rather than giving equations for them. The equations here
are derived from Chapter 1's own combined momentum and blade element theory:
the thrust relation of p. 19, the linear-to-ideal twist conversion of p. 20,
and the torque relation of p. 22.
"""

import numpy as np
import openmdao.api as om


class HoverChartSlopesComp(om.ExplicitComponent):
    """``dCT/sigma/dtheta0`` and ``dCQ/sigma/dtheta0`` without the charts.

    Both follow from the one derivative Table 9.1 *does* give an equation for,
    ``dCT/sigma/dlambda'``, which is why that is an input here rather than
    something this component recomputes.

    Where the equations come from
    -----------------------------
    Chapter 1, p. 19, for a blade with ideal twist::

        C_T/sigma = (a/4) (theta_t - phi_t),   phi_t = sqrt(C_T/2)

    Perturb the tip pitch and let the induced angle respond. Writing
    ``x = C_T/sigma``, ``lambda_i = sqrt(sigma x/2)`` and
    ``k = dlambda_i/dx = lambda_i/(2x)``::

        dx/dtheta_t = 1 / [4/a + k]

    p. 20 relates the collective of a linearly twisted blade to the tip pitch
    of an ideally twisted one, ``theta_0 = (3/2) theta_t - (3/4) theta_1``, so
    ``dtheta_t/dtheta_0 = 2/3`` and::

        dx/dtheta_0 = (2/3) / [4/a + k] = (4/3) / [8/a + sqrt(sigma/2)/sqrt(x)]
                    = (4/3) dx/dlambda'

    The second form is Table 9.1's own ``dCT/sigma/dlambda'`` denominator, so
    the whole thing collapses to a factor of 4/3 on a derivative the table
    already supplies.

    Torque follows from p. 22, ``C_Q/sigma = lambda_i x + c_d/8``, and from
    p. 24, ``alpha_bar = 6 x / a``::

        dCQ/sigma/dtheta_0 = (3/2) k_induced lambda_i dx/dtheta_0
                             + (1/8)(dcd/dalpha)(6/a) dx/dtheta_0

    using ``x dlambda_i/dx = lambda_i/2`` to collapse the induced part.

    Why lambda' carries a factor the naive derivation misses
    -------------------------------------------------------
    Worth recording, because getting it wrong is easy and silent. Momentum
    theory in climb gives ``v_1 = -V_c/2 + sqrt((V_c/2)^2 + T/(2 rho A))``, so
    a climb rate changes the *total* inflow through the disc by only half of
    itself -- the induced velocity absorbs the other half. Treating the climb
    inflow as simply added to a frozen induced velocity gives
    ``1/[4/a + k]``, exactly twice Table 9.1's value. The table is right.

    How close this gets
    -------------------
    For the example helicopter at ``C_T/sigma = .086``, against the Chapter 1
    charts and against the repository's own Chapter 1 model:

    ==================== ========== =========== =========
    quantity             this comp  Chapter 1   Table 9.1
    ==================== ========== =========== =========
    dCT/sigma/dlambda'   .4911      --          .490
    dCT/sigma/dtheta0    .6548      .5986       .610
    dCQ/sigma/dtheta0    .0641      .0748       .078
    ==================== ========== =========== =========

    Thrust lands 7 % high. The gap is tip loss and the empirical corrections
    of Chapter 1, pp. 69-76, which the closed form has no way to carry.

    Torque lands 18 % low at ``k_induced = 1``, which is the ideal momentum
    value. Real rotors need more induced torque than momentum theory predicts;
    ``k_induced = 1.2`` puts this row at .0760, within 2.6 % of the chart. The
    default is 1.0 rather than 1.2 because 1.2 is a number fitted to this one
    rotor, not a result.

    So this component is for letting an optimiser see the slopes move with the
    design, not for reproducing Table 9.4. Keep
    ``derivative_source='table'`` when the book's numbers are the target.

    Options
    -------
    num_nodes : int
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('dCT_sigma_dlambda', val=np.full(nn, 0.49),
                       desc='Table 9.1, p. 564.')
        self.add_input('CT_sigma', val=np.full(nn, 0.086))
        self.add_input('sigma', val=0.085, desc='Rotor solidity.')
        self.add_input('a', val=6.0, units='1/rad',
                       desc='Blade lift curve slope.')
        self.add_input('k_induced', val=np.ones(nn),
                       desc='Induced torque factor; 1 is ideal momentum.')
        self.add_input('dcd_dalpha', val=np.full(nn, 0.0573), units='1/rad',
                       desc='Profile drag slope, Figure 1.10, p. 23.')

        self.add_output('dCT_sigma_dtheta0', val=np.zeros(nn))
        self.add_output('dCQ_sigma_dtheta0', val=np.zeros(nn))

        self.declare_partials('dCT_sigma_dtheta0', 'dCT_sigma_dlambda',
                              rows=ar, cols=ar, val=4.0 / 3.0)
        self.declare_partials(
            'dCQ_sigma_dtheta0',
            ['dCT_sigma_dlambda', 'CT_sigma', 'k_induced', 'dcd_dalpha'],
            rows=ar, cols=ar)
        self.declare_partials('dCQ_sigma_dtheta0', ['sigma', 'a'],
                              rows=ar, cols=zeros)

    def compute(self, inputs, outputs):
        lambda_i = np.sqrt(0.5 * inputs['sigma'][0] * inputs['CT_sigma'])
        outputs['dCT_sigma_dtheta0'] = 4.0 / 3.0 * inputs['dCT_sigma_dlambda']
        outputs['dCQ_sigma_dtheta0'] = inputs['dCT_sigma_dlambda'] * (
            2.0 * inputs['k_induced'] * lambda_i
            + inputs['dcd_dalpha'] / inputs['a'][0])

    def compute_partials(self, inputs, J):
        sigma, a = inputs['sigma'][0], inputs['a'][0]
        CTs, L = inputs['CT_sigma'], inputs['dCT_sigma_dlambda']
        k, dcd = inputs['k_induced'], inputs['dcd_dalpha']
        lambda_i = np.sqrt(0.5 * sigma * CTs)

        J['dCQ_sigma_dtheta0', 'dCT_sigma_dlambda'] = 2.0 * k * lambda_i + dcd / a
        J['dCQ_sigma_dtheta0', 'CT_sigma'] = L * k * sigma / (2.0 * lambda_i)
        J['dCQ_sigma_dtheta0', 'k_induced'] = 2.0 * L * lambda_i
        J['dCQ_sigma_dtheta0', 'dcd_dalpha'] = L / a
        J['dCQ_sigma_dtheta0', 'sigma'] = L * k * CTs / (2.0 * lambda_i)
        J['dCQ_sigma_dtheta0', 'a'] = -L * dcd / a ** 2
