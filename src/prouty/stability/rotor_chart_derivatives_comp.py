"""Table 9.5, the rotor derivatives read off the Chapter 3 performance charts.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", Table 9.5, p. 574, with Figure 9.7
(p. 575) showing the graphical extraction. The charts themselves are
Chapter 3, pp. 254-271, described on pp. 229-231.
"""

import numpy as np
import openmdao.api as om

#: Table 9.5, p. 574. Keyed by output name, then by rotor.
TABLE_9_5 = {
    # trim condition of the chart reading
    'mu': {'main': 0.30, 'tail': 0.30},
    'theta_0_chart': {'main': np.deg2rad(13.5), 'tail': np.deg2rad(7.1)},
    'lambda_bar': {'main': -0.023, 'tail': 0.0051},
    # partials with respect to mu
    'dCT_sigma_dmu': {'main': -0.140, 'tail': -0.070},
    'dCH_sigma_dmu': {'main': 0.008, 'tail': 0.004},
    'dCQ_sigma_dmu': {'main': -0.005, 'tail': -0.001},
    'd_a1s_d_mu': {'main': 0.33, 'tail': 0.12},
    'd_b1s_d_mu': {'main': -0.05, 'tail': -0.02},
    # partials with respect to the chart collective
    'dCT_sigma_dtheta0': {'main': 0.46, 'tail': 0.659},
    'dCH_sigma_dtheta0': {'main': -0.04, 'tail': 0.001},
    'dCQ_sigma_dtheta0': {'main': 0.052, 'tail': 0.005},
    'd_a1s_d_theta0': {'main': 1.1, 'tail': 0.60},
    'd_b1s_d_theta0': {'main': 0.25, 'tail': 0.12},
    # partials with respect to lambda'
    'dCT_sigma_dlambda': {'main': 0.79, 'tail': 1.04},
    'dCH_sigma_dlambda': {'main': -0.07, 'tail': -0.002},
    'dCQ_sigma_dlambda': {'main': 0.010, 'tail': -0.026},
    'd_a1s_d_lambda': {'main': 1.2, 'tail': 0.70},
    'd_b1s_d_lambda': {'main': 0.59, 'tail': 0.17},
}

UNITS = {'theta_0_chart': 'rad'}

#: What each column is a derivative of, for the docstring and the tests.
DEPENDENT = ('CT_sigma', 'CH_sigma', 'CQ_sigma', 'a1s', 'b1s')
INDEPENDENT = ('mu', 'theta0', 'lambda')


class RotorChartDerivativesComp(om.IndepVarComp):
    """The thirty numbers of Table 9.5, as a component you can swap out.

    Fifteen derivatives per rotor, plus the three trim values the reading was
    taken at. They are *outputs*, not inputs, because this component is the
    swap point: replacing it with a differentiated Chapter 3 model is what a
    ``derivative_source='model'`` would mean for forward flight, and an
    ``IndepVarComp`` is what a model subsystem substitutes for cleanly.
    Override any value with ``prob.set_val`` as usual.

    Options
    -------
    num_nodes : int
    rotor : {'main', 'tail'}
        Which column of Table 9.5 to emit. p. 574 prints both.

    What the charts are, and are not
    --------------------------------
    They are the non-dimensional map of **one blade**, not of a helicopter.
    p. 570 states the premise: for a given rotor the forces and flapping are
    uniquely determined by ``mu``, ``theta`` and ``lambda'``. The chart blade
    is fixed at (Chapter 3, pp. 229-230)::

        twist              -5 deg linear
        airfoil            NACA 0012
        advancing tip Mach  0.7
        chord/radius        0.079
        tip loss factor     0.97

    Solidity is **not** in that list, and it does not belong there: pushing
    ``sigma`` from .085 to .120 in the Chapter 3 model moves ``C_T/sigma`` at
    fixed ``(mu, theta_0, lambda')`` by 0.02 %. Solidity enters downstream, in
    Table 9.6's inflow rows, as the ``sigma/(2 mu)`` of ``dlambda'/dzdot``.
    Blade aerodynamics in the charts, momentum closure in Table 9.6.

    ``theta_0_chart`` is the collective referenced to the chart's twist
    ---------------------------------------------------------------
    Table 9.5's row is labelled ``theta_0`` subscript ``theta_1 = -5 deg``.
    p. 230 gives the conversion for a blade of different twist::

        theta_0_chart = theta_0 + .75 (theta_1 + 5 deg)

    Since that is a shift and not a scaling, ``dtheta_0_chart/dtheta_0 = 1``
    and the *derivatives* carry over to another twist unchanged -- only the
    trim point moves. Checked against the Chapter 3 model: at -10 deg twist,
    entering the equivalent chart collective reproduces ``C_T/sigma`` to 5 %.

    These are secants, not tangents
    -------------------------------
    p. 576 says the ``mu`` partials were taken as the difference between the
    ``mu = .25`` and ``mu = .35`` charts, a secant over ``Delta mu = .10``, and
    Figure 9.7 marks ``Delta lambda' = .020`` and ``Delta theta_0 = 2 deg`` on
    the other two. So Table 9.5 holds finite differences over wide windows.
    It matters: differentiating the Chapter 3 model with a narrow step gives
    ``dCQ/sigma/dlambda' = -.0035``, and over Prouty's own +/-.020 window the
    secant is ``+.0141`` against a printed ``+.010``. The sign of that row is a
    step-size artefact, not a disagreement. See the validation notes.

    Example helicopter at 115 knots
    -------------------------------
    Main rotor trimmed at ``mu = .30``, ``theta_0 = 13.5 deg``,
    ``lambda' = -.023``; tail rotor at ``mu = .30``, ``theta_0 = 7.1 deg``,
    ``lambda' = .0051``.
    """

    def initialize(self):
        super().initialize()
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('rotor', values=('main', 'tail'), default='main')

    def setup(self):
        nn = self.options['num_nodes']
        rotor = self.options['rotor']
        for name, columns in TABLE_9_5.items():
            self.add_output(name, val=np.full(nn, columns[rotor]),
                            units=UNITS.get(name),
                            desc=f'Table 9.5, p. 574, {rotor} rotor.')
