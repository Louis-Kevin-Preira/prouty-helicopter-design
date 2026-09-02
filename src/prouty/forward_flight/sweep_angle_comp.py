"""
SweepAngleComp -- local sweep angle from the spanwise flow.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 217-219 (Figure 3.56).

    tan(Lambda) = U_R / U_T

A blade element in forward flight is a yawed wing: the free stream has a
spanwise component U_R = mu cos(psi), so the section behaves as if swept back
by Lambda. Sweep delays stall, and Prouty gives the two ways of putting that
into an airfoil model (p. 218):

    tabulated data       alpha_ref = alpha_actual cos(Lambda)
    equation form        alpha_L   = alpha_L,0 sqrt(1 + (U_R/U_T)^2)

The second is 1/cos(Lambda), so the two are the same statement: the stall
angle rises as sec(Lambda), or equivalently the section responds to the
reduced angle alpha cos(Lambda). This package uses the equation form of
Chapter 6, so sec_Lambda is what feeds the airfoil model through its external
stall angle, and cos_Lambda is provided for the tabulated route.

Both are exported rather than Lambda itself for the force path, because it is
the trigonometric functions the models want and going through the angle only
adds a call.

Where it matters. Sweep vanishes at psi = 90 and 270 deg, where the flow is
purely chordwise, and peaks fore and aft where U_R = +/- mu. Figure 3.56 marks
the regions where Lambda exceeds 30 deg: two lobes on the fore-aft line
reaching in to r/R = mu/tan(30 deg) = 0.52 at mu = 0.3, growing to 0.78 at
mu = 0.45. Inside those lobes the stall angle is raised by 15 % or more.

Prouty adds that sweep under 60 deg has little effect on drag and pitching
moment (p. 219), so only the lift path uses this.

Sign and singularity. Lambda is signed, positive where U_R is: that keeps it
smooth through psi = 90 and 270 deg, where taking the magnitude would put a
kink on two whole radial lines of the grid. cos and sec depend on the sign
only through Lambda^2, so nothing downstream sees it.

sec(Lambda) is unbounded where U_T vanishes and is NOT capped here, which
follows Prouty: he lets the stall angle run and then bounds the lift
coefficient itself (p. 221, Figure 3.58), because that is where the real
problem is. Know the magnitudes before reading any diagnostic, though.

U_T vanishes on the reverse flow boundary, r/R = -mu sin(psi), and that curve
passes exactly through grid nodes whenever mu sin(psi) lands on a station --
at mu = 0.3 with 41 stations it does so at psi = 210 and 330 deg, r/R = 0.15.
There sec(Lambda) is mu/epsilon, 2.6e7 with the default, and no amount of grid
refinement removes it; refining only moves which nodes are hit. Over the
lift-carrying stations, 7.5 % of the disc has sec(Lambda) above 2 at mu = 0.3,
which is the honest number to quote.

Those nodes carry almost no load -- U_B there is what is left of U_P, about
0.03, so their dynamic pressure is a thousandth of a normal element's -- but
they hand the airfoil model a stall angle of 1e7 radians, meaning nothing
there ever stalls. That is exactly the situation Prouty's lift coefficient
bounds are for, and this component is unusable without them.

    UT_bar, UR_bar --> SweepAngleComp --> Lambda, cos_Lambda, sec_Lambda
"""

import numpy as np
import openmdao.api as om


class SweepAngleComp(om.ExplicitComponent):
    """Local sweep angle and its trigonometric functions, p. 218."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('num_azimuth', types=int, default=24)
        self.options.declare('num_radial', types=int, default=21)
        self.options.declare('epsilon', types=float, default=1.0e-8)

    def setup(self):
        nn = self.options['num_nodes']
        n_psi = self.options['num_azimuth']
        n_r = self.options['num_radial']
        field = (nn, n_psi, n_r)

        self.add_input('UT_bar', shape=field)
        self.add_input('UR_bar', shape=(nn, n_psi))

        self.add_output('Lambda', shape=field, units='rad',
                        desc='local sweep angle, signed')
        self.add_output('cos_Lambda', shape=field,
                        desc='for alpha_ref = alpha cos(Lambda), p. 218')
        self.add_output('sec_Lambda', shape=field,
                        desc='for alpha_L = alpha_L0 sec(Lambda), p. 218')

        rows = np.arange(nn * n_psi * n_r)
        sweep = np.repeat(np.arange(nn * n_psi), n_r)
        for out in ('Lambda', 'cos_Lambda', 'sec_Lambda'):
            self.declare_partials(out, 'UT_bar', rows=rows, cols=rows)
            self.declare_partials(out, 'UR_bar', rows=rows, cols=sweep)

    def _tan(self, inputs):
        """tan(Lambda) and the softened |U_T| it is built on."""
        eps = self.options['epsilon']
        s = np.sqrt(inputs['UT_bar'] ** 2 + eps ** 2)
        UR = inputs['UR_bar'][:, :, np.newaxis]
        return UR / s, s, UR

    def compute(self, inputs, outputs):
        t, _, _ = self._tan(inputs)
        root = np.sqrt(1.0 + t ** 2)

        outputs['Lambda'] = np.arctan(t)
        outputs['cos_Lambda'] = 1.0 / root
        outputs['sec_Lambda'] = root

    def compute_partials(self, inputs, partials):
        shape = inputs['UT_bar'].shape
        t, s, UR = self._tan(inputs)
        UT = inputs['UT_bar']
        root = np.sqrt(1.0 + t ** 2)

        dt_UT = -UR * UT / s ** 3
        dt_UR = np.broadcast_to(1.0 / s, shape)

        chain = {'Lambda': 1.0 / (1.0 + t ** 2),
                 'cos_Lambda': -t / root ** 3,
                 'sec_Lambda': t / root}
        for out, d in chain.items():
            partials[out, 'UT_bar'] = (d * dt_UT).ravel()
            partials[out, 'UR_bar'] = (d * dt_UR).ravel()
