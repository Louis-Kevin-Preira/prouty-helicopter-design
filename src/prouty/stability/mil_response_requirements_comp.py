"""MIL-H-8501A response and damping requirements in hover.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", Table 9.18 and the
single-degree-of-freedom solution, p. 611, read against the envelopes of
Figure 9.13, p. 610. The specification paragraphs are 3.2.13, 3.2.14, 3.3.5,
3.3.15, 3.3.18, 3.3.19 and 3.6.1.1 of MIL-H-8501A.
"""

import numpy as np
import openmdao.api as om

#: Table 9.18, p. 611. Minimum response is the numerator over the cube root of
#: ``G.W. + 1,000``; damping is the coefficient times ``inertia**0.7``.
TABLE_9_18 = {
    'longitudinal': dict(time=1.0, response_visual=45.0,
                         response_instrument=73.0, max_rate=None,
                         damping_visual=8.0, damping_instrument=15.0),
    'lateral': dict(time=0.5, response_visual=27.0,
                    response_instrument=32.0, max_rate=20.0,
                    damping_visual=18.0, damping_instrument=25.0),
    'directional': dict(time=1.0, response_visual=110.0,
                        response_instrument=110.0, max_rate=None,
                        damping_visual=27.0, damping_instrument=27.0),
}

#: The one footnote in Table 9.18: the visual directional damping figure is
#: "not a requirement, only a preference".
PREFERENCE_ONLY = (('directional', 'visual'),)

#: Control power per inch of stick, in ft lb/in, read off Figure 9.13's
#: horizontal axis for the example helicopter. Chapter 9 never prints the
#: control linkage gearing, so these are the only route to the figure's
#: abscissa; see the class docstring.
FIGURE_9_13_CONTROL_POWER = {'longitudinal': 12800.0, 'lateral': 7500.0,
                             'directional': 63000.0}

#: Figure 9.14, p. 613: acceptance boxes in the (steady rate, time constant)
#: plane, as (rate_min, rate_max, time_constant_max). These are *not* Table
#: 9.18 recast -- p. 613 calls them "simplified approximations of the
#: boundaries that are generally accepted today as the result of several flight
#: test and simulator studies", references 9.10 and 9.11. Read off the figure,
#: so good to about half a division.
FIGURE_9_14 = {
    'longitudinal': {'any': (5.0, 12.3, 1.0)},
    'lateral': {'any': (9.0, 20.5, 0.5)},
    'directional': {'utility': (15.0, 24.0, 0.5),
                    'armed': (30.0, 50.0, 0.24)},
}

RAD_TO_DEG = 180.0 / np.pi


def figure_9_14_violations(steady_rate, time_constant, axis, role='any'):
    """Which of Figure 9.14's boundaries a point falls outside, p. 613.

    Returns a tuple of labels, empty when the point is inside the desirable
    box. More than one can apply: the example helicopter in pitch is both
    oversensitive and too slow.

    The plane is the one p. 612 arrives at -- steady rate on the abscissa and
    time constant, the inverse of damping over inertia, on the ordinate. p. 613
    makes the case for it: "flight test data in the form of time histories
    following step control inputs can yield the information required to judge
    the flying qualities directly", without any derivative estimation at all.

    The yaw axis carries two boxes, for a utility and an armed helicopter;
    ``role`` picks one. The other two axes take ``'any'``.
    """
    rate_min, rate_max, time_max = FIGURE_9_14[axis][role]
    violations = []
    if steady_rate < rate_min:
        violations.append('sluggish')
    if steady_rate > rate_max:
        violations.append('oversensitive')
    if time_constant > time_max:
        violations.append('takes too long to respond')
    return tuple(violations)


class MilResponseRequirementsComp(om.ExplicitComponent):
    """Table 9.18 and the one-second displacement that is compared with it.

    p. 610 explains what the specification asks for: "the damping and the
    response to one-inch control steps for both visual and instrument flight
    conditions in all three axes". Table 9.18 collects the numbers::

        axis          t     min response (deg)      max rate   damping
                            visual   instrument     (deg/s)    (ft lb/rad/s)
        longitudinal  1     45/R     73/R           --         8 I^.7 / 15 I^.7
        lateral       .5    27/R     32/R           20         18 I^.7 / 25 I^.7
        directional   1     110/R    110/R          --         27 I^.7 / 27 I^.7

    with ``R`` the cube root of ``G.W. + 1,000`` -- 27.59 for the example
    helicopter's 20,000 lb. The visual directional damping figure carries the
    table's only footnote: "not a requirement, only a preference".

    The displacement it is compared with
    ------------------------------------
    p. 611 treats each moment equation as a single degree of freedom::

        (dM/dB1) B1 = I_yy Theta_ddot - (dM/dq) Theta_dot

    which integrates to

        Theta = [(dM/dB1)B1 / -(dM/dq)]
                [t + (I_yy / -(dM/dq))(exp((dM/dq/I_yy) t) - 1)]

    and, per inch of stick and in degrees, to the form p. 611 prints::

        Theta/inch = 57.3 (CP/I)/(D/I) [t + (1/(D/I))(exp(-(D/I) t) - 1)]

    p. 611 folds ``t = 1`` into its printed version; this component keeps
    ``t`` because the lateral axis is specified at half a second.

    Both parameters of Figure 9.13 are outputs, so a point can be placed on it
    directly: ``control_power_over_inertia`` on the abscissa and
    ``damping_over_inertia`` on the ordinate. So are Figure 9.14's, p. 613:
    ``steady_rate`` and ``time_constant``, the latter being the inverse of
    ``damping_over_inertia`` as p. 612 notes. :func:`figure_9_14_violations`
    reads the verdict off that plane.

    p. 612 gives two independent checks on the displacement formula, and both
    hold: as damping goes to zero it reduces to ``2 Theta/t^2`` -- schoolboy
    ``s = at^2/2`` -- and as damping goes to infinity to
    ``Theta (Damping/Inertia)``, each recovering ``CP/I`` exactly.

    Control power per inch is an input, and has to be
    ------------------------------------------------
    The abscissa of Figure 9.13 needs moment per **inch of stick**, and
    Chapter 9 nowhere prints the control linkage gearing that converts degrees
    of cyclic into inches. ``FIGURE_9_13_CONTROL_POWER`` holds the values read
    back off the figure for the example helicopter; they are defaults, not
    derived quantities, and they are the one place in this chapter's
    implementation where a number comes from a figure rather than from an
    equation.

    Options
    -------
    num_nodes : int
    axis : {'longitudinal', 'lateral', 'directional'}
    instrument : bool
        Which column of Table 9.18 to require.

    Example helicopter in hover
    ---------------------------
    ====================== =========== =========== ===========
    axis                   damping/I   required/I  verdict
    ====================== =========== =========== ===========
    longitudinal           0.717       0.333       passes
    lateral                5.83        1.94        passes
    directional            0.381       1.17        **fails**
    ====================== =========== =========== ===========

    Pitch and roll clear the instrument-flight damping requirement, but by
    very different margins: 28,659 against a required 24,977 in pitch, **15
    per cent**, and 29,127 against 9,707 in roll, three to one. Pitch is the
    tight axis, which is the same conclusion the longitudinal stability map
    reaches from the other direction.

    **Yaw fails it by a factor of three** -- 13,326 against a
    required 40,948 ft lb/rad/sec -- and that is with the hover tail rotor
    doing all the work against a main rotor term of the opposite sign (entry
    C9-15). p. 612 states the example helicopter "would satisfy the instrument
    flight requirements", which is hard to reconcile with the yaw axis on these
    numbers.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('axis', values=tuple(TABLE_9_18), default='longitudinal')
        self.options.declare('instrument', types=bool, default=True)

    def setup(self):
        nn = self.options['num_nodes']
        axis = self.options['axis']
        rules = TABLE_9_18[axis]
        ar = np.arange(nn)

        self.add_input('G_W', val=np.full(nn, 20000.0), units='lbf')
        self.add_input('inertia', val=np.full(nn, 40000.0), units='slug*ft**2',
                       desc=f'Moment of inertia about the {axis} axis.')
        self.add_input('damping', val=np.full(nn, 28659.0),
                       units='lbf*ft*s/rad',
                       desc='Magnitude of the rate damping derivative.')
        self.add_input('control_power', units='lbf*ft',
                       val=np.full(nn, FIGURE_9_13_CONTROL_POWER[axis]),
                       desc='Moment per inch of stick; see the docstring.')

        self.add_output('response_required', val=np.zeros(nn), units='deg')
        self.add_output('damping_required', val=np.zeros(nn),
                        units='lbf*ft*s/rad')
        self.add_output('displacement', val=np.zeros(nn), units='deg',
                        desc=f'Attitude {rules["time"]} s after a one-inch step.')
        self.add_output('steady_rate', val=np.zeros(nn), units='deg/s')
        self.add_output('time_constant', val=np.zeros(nn), units='s',
                        desc='Inertia over damping; Figure 9.14 ordinate.')
        self.add_output('response_margin', val=np.zeros(nn), units='deg')
        self.add_output('damping_margin', val=np.zeros(nn),
                        units='lbf*ft*s/rad')
        self.add_output('control_power_over_inertia', val=np.zeros(nn),
                        units='rad/s**2', desc='Figure 9.13 abscissa.')
        self.add_output('damping_over_inertia', val=np.zeros(nn), units='1/s',
                        desc='Figure 9.13 ordinate.')

        self.declare_partials('response_required', 'G_W', rows=ar, cols=ar)
        self.declare_partials('damping_required', 'inertia', rows=ar, cols=ar)
        for name in ('displacement', 'response_margin'):
            self.declare_partials(name, ['damping', 'inertia', 'control_power'],
                                  rows=ar, cols=ar)
        self.declare_partials('response_margin', 'G_W', rows=ar, cols=ar)
        self.declare_partials('steady_rate', ['damping', 'control_power'],
                              rows=ar, cols=ar)
        self.declare_partials('time_constant', ['damping', 'inertia'],
                              rows=ar, cols=ar)
        self.declare_partials('damping_margin', ['damping', 'inertia'],
                              rows=ar, cols=ar)
        self.declare_partials('control_power_over_inertia',
                              ['control_power', 'inertia'], rows=ar, cols=ar)
        self.declare_partials('damping_over_inertia', ['damping', 'inertia'],
                              rows=ar, cols=ar)

    @property
    def _rules(self):
        rules = TABLE_9_18[self.options['axis']]
        column = 'instrument' if self.options['instrument'] else 'visual'
        return (rules['time'], rules[f'response_{column}'],
                rules[f'damping_{column}'], rules['max_rate'])

    @property
    def is_preference_only(self):
        """True where Table 9.18's footnote applies, p. 611."""
        column = 'instrument' if self.options['instrument'] else 'visual'
        return (self.options['axis'], column) in PREFERENCE_ONLY

    def compute(self, inputs, outputs):
        time, response, damping_factor, _ = self._rules
        D, I = inputs['damping'], inputs['inertia']
        power = inputs['control_power']
        ratio = D / I
        shape = RAD_TO_DEG * (power / D) * (
            time + (np.exp(-ratio * time) - 1.0) / ratio)

        outputs['response_required'] = response * (inputs['G_W']
                                                   + 1000.0) ** (-1.0 / 3.0)
        outputs['damping_required'] = damping_factor * I ** 0.7
        outputs['displacement'] = shape
        outputs['steady_rate'] = RAD_TO_DEG * power / D
        outputs['time_constant'] = I / D
        outputs['response_margin'] = shape - outputs['response_required']
        outputs['damping_margin'] = D - outputs['damping_required']
        outputs['control_power_over_inertia'] = power / I
        outputs['damping_over_inertia'] = ratio

    def compute_partials(self, inputs, J):
        time, response, damping_factor, _ = self._rules
        D, I = inputs['damping'], inputs['inertia']
        power = inputs['control_power']
        ratio = D / I
        decay = np.exp(-ratio * time)
        bracket = time + (decay - 1.0) / ratio

        J['response_required', 'G_W'] = (-response / 3.0
                                         * (inputs['G_W'] + 1000.0) ** (-4.0 / 3.0))
        J['damping_required', 'inertia'] = 0.7 * damping_factor * I ** -0.3

        # displacement = 57.3 (power/D) [t + (exp(-ratio t) - 1)/ratio]
        d_bracket_d_ratio = (-time * decay / ratio
                             - (decay - 1.0) / ratio ** 2)
        J['displacement', 'control_power'] = RAD_TO_DEG * bracket / D
        J['displacement', 'damping'] = RAD_TO_DEG * power * (
            -bracket / D ** 2 + d_bracket_d_ratio / (D * I))
        J['displacement', 'inertia'] = RAD_TO_DEG * power / D * (
            d_bracket_d_ratio * -ratio / I)

        J['steady_rate', 'control_power'] = RAD_TO_DEG / D
        J['steady_rate', 'damping'] = -RAD_TO_DEG * power / D ** 2
        J['time_constant', 'inertia'] = 1.0 / D
        J['time_constant', 'damping'] = -I / D ** 2

        for name in ('damping', 'inertia', 'control_power'):
            J['response_margin', name] = J['displacement', name]
        J['response_margin', 'G_W'] = -J['response_required', 'G_W']

        J['damping_margin', 'damping'] = np.ones_like(D)
        J['damping_margin', 'inertia'] = -J['damping_required', 'inertia']

        J['control_power_over_inertia', 'control_power'] = 1.0 / I
        J['control_power_over_inertia', 'inertia'] = -power / I ** 2
        J['damping_over_inertia', 'damping'] = 1.0 / I
        J['damping_over_inertia', 'inertia'] = -D / I ** 2
