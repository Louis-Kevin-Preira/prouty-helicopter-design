"""Handling qualities criteria for the longitudinal motion in forward flight.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis". Table 9.21, p. 622, summarises
MIL-H-8501A's damping requirements; Figure 9.21, p. 629, is the Boeing-Vertol
stabilizer sizing criterion, after Blake and Alansky, *JAHS* 22-1, 1977.
"""

import numpy as np
import openmdao.api as om

#: Table 9.21, p. 622. Each band gives (upper period bound, visual rule,
#: instrument rule). The rules are named in :func:`mil_h_8501a`.
TABLE_9_21 = (
    (5.0, 'half in 2 cycles', 'half in 1 cycle'),
    (10.0, 'lightly damped', 'half in 2 cycles'),
    (20.0, 'not double in 10 s', 'lightly damped'),
    (np.inf, 'no requirement', 'not double in 20 s'),
)

#: Figure 9.21, p. 629: horizontal stabilizer areas the criterion is drawn for.
STABILIZER_AREAS = (18.0, 36.0, 54.0, 72.0, 90.0)


def mil_h_8501a(period, real_part, instrument=False):
    """Does an oscillatory mode meet Table 9.21?

    Returns ``(passes, rule)``. ``period`` is in seconds and ``real_part`` is
    the root's real part, positive for a growing oscillation.

    The requirement depends on the period band, so this is a lookup with
    discontinuous thresholds and cannot serve as a gradient-based constraint.
    It is a classifier, like ``classify`` on the stability map, and the rule
    that fired is returned so the caller can see which clause decided.

    A caution on the last band
    --------------------------
    Table 9.21 imposes **no** visual-flight requirement above 20 seconds, on
    the argument (p. 622) that "the time is so long that the pilot
    instinctively corrects for any instability with his normal control
    motions". Applied literally that lets anything through, and the example
    helicopter with a 36 ft² stabilizer exploits it: period 41.6 seconds,
    amplitude doubling in **1.6 seconds**. That is a divergence wearing a very
    slow oscillation, not the slow wallow the clause was written for. The
    function reports the rule rather than silently patching it.
    """
    for upper, visual, instrument_rule in TABLE_9_21:
        if period < upper:
            rule = instrument_rule if instrument else visual
            break

    if rule == 'no requirement':
        passes = True
    elif rule == 'lightly damped':
        passes = real_part < 0.0
    elif rule.startswith('half in'):
        cycles = 1.0 if '1 cycle' in rule else 2.0
        passes = (real_part < 0.0
                  and np.log(2.0) / -real_part <= cycles * period)
    else:
        seconds = 10.0 if '10 s' in rule else 20.0
        passes = (real_part <= 0.0
                  or np.log(2.0) / real_part >= seconds)
    return bool(passes), rule


class BoeingVertolParameterComp(om.ExplicitComponent):
    """Figure 9.21's short-period stability parameter, p. 629.

    One number for sizing the horizontal stabilizer::

        (dZ/dzdot)/(G.W./g) x (dM/dq)/I_yy - V_bar (dM/dzdot)/I_yy

    Blake and Alansky plotted it against pilot Handling Qualities Rating for
    the YUH-61A and found an acceptable/unacceptable boundary; Prouty draws
    the example helicopter's five stabilizer areas on it.

    Unlike everything else in the chapter's handling-qualities material, this
    is a single smooth scalar built from derivatives the tables already
    produce, so it is the one criterion here that can be a gradient-based
    constraint on stabilizer area.

    It is the short-period stiffness in disguise
    --------------------------------------------
    Expand the short-period quadratic of p. 625 and its constant term is

        [(dZ/dzdot)(dM/dq) - (dZ/dq + m V)(dM/dzdot)] / [I_yy (m - dZ/dzddot)]

    Drop ``dZ/dq`` and ``dZ/dzddot`` -- for the example helicopter -217 against
    a ``m V`` of 120,555, and zero -- and it becomes exactly the expression
    above. So the Boeing-Vertol parameter is the **square of the short-period
    natural frequency**, and the criterion is that the short period be a real
    oscillation of at least about 1 rad/sec rather than a divergence.

    For the example helicopter the two differ by 0.2 %: -2.6487 against the
    exact -2.6427.

    The last term of the printed formula
    ------------------------------------
    Figure 9.21 prints its numerator as ``dM/dz``, without the dot. It is
    ``dM/dzdot``: the figure is set in a typewriter face throughout, and
    ``dM/dz`` would make the term ``1/s^3`` where the first product is
    ``1/s^2``.

    Options
    -------
    num_nodes : int

    Example helicopter at 115 knots
    -------------------------------
    ========= ========== ===================================
    area      parameter  longitudinal oscillation
    ========= ========== ===================================
    18 ft²    -2.649     none -- four real roots
    36 ft²    -1.489     P = 41.6 s, doubling in 1.6 s
    54 ft²    -0.325     P = 17.1 s, doubling in 3.1 s
    72 ft²    +0.843     P = 17.0 s, doubling in 46.6 s
    90 ft²    +2.015     P = 7.5 s, halving in 0.7 s
    ========= ========== ===================================

    The sign change between 54 and 72 square feet is where the short period
    stops being a divergence, and it agrees with Table 9.21 read on the same
    configurations: 72 ft² passes visual flight and fails instrument flight,
    which is what p. 622 says in words.

    The values sit about 1.2 below the marks on Figure 9.21. The stabilizer
    contributions are scaled linearly with area here and the aircraft is not
    re-trimmed at each size, which Prouty would have done.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('dZ_dzdot', val=np.full(nn, -287.0), units='lbf*s/ft')
        self.add_input('dM_dq', val=np.full(nn, -43752.0), units='lbf*ft*s/rad')
        self.add_input('dM_dzdot', val=np.full(nn, 650.0), units='lbf*s')
        self.add_input('G_W', val=np.full(nn, 20000.0), units='lbf')
        self.add_input('I_yy', val=np.full(nn, 40000.0), units='slug*ft**2')
        self.add_input('g', val=np.full(nn, 32.2), units='ft/s**2')
        self.add_input('V', val=np.full(nn, 194.1), units='ft/s')
        self.add_input('acceptable', val=np.full(nn, 1.0), units='1/s**2',
                       desc='Boundary read off Figure 9.21, p. 629.')

        self.add_output('short_period_parameter', val=np.zeros(nn),
                        units='1/s**2')
        self.add_output('handling_margin', val=np.zeros(nn), units='1/s**2',
                        desc='Parameter minus the acceptable boundary.')

        wrt = ['dZ_dzdot', 'dM_dq', 'dM_dzdot', 'G_W', 'I_yy', 'g', 'V']
        self.declare_partials('short_period_parameter', wrt, rows=ar, cols=ar)
        self.declare_partials('handling_margin', wrt, rows=ar, cols=ar)
        self.declare_partials('handling_margin', 'acceptable', rows=ar,
                              cols=ar, val=-1.0)

    def _parameter(self, inputs):
        mass = inputs['G_W'] / inputs['g']
        return (inputs['dZ_dzdot'] / mass * inputs['dM_dq'] / inputs['I_yy']
                - inputs['V'] * inputs['dM_dzdot'] / inputs['I_yy'])

    def compute(self, inputs, outputs):
        parameter = self._parameter(inputs)
        outputs['short_period_parameter'] = parameter
        outputs['handling_margin'] = parameter - inputs['acceptable']

    def compute_partials(self, inputs, J):
        G_W, g, I_yy = inputs['G_W'], inputs['g'], inputs['I_yy']
        V, Zw, Mq, Mw = (inputs['V'], inputs['dZ_dzdot'], inputs['dM_dq'],
                         inputs['dM_dzdot'])
        mass = G_W / g
        damping = Zw * Mq / (mass * I_yy)
        lift = V * Mw / I_yy

        J['short_period_parameter', 'dZ_dzdot'] = Mq / (mass * I_yy)
        J['short_period_parameter', 'dM_dq'] = Zw / (mass * I_yy)
        J['short_period_parameter', 'dM_dzdot'] = -V / I_yy
        J['short_period_parameter', 'V'] = -Mw / I_yy
        J['short_period_parameter', 'G_W'] = -damping / G_W
        J['short_period_parameter', 'g'] = damping / g
        J['short_period_parameter', 'I_yy'] = (-damping + lift) / I_yy

        for name in ('dZ_dzdot', 'dM_dq', 'dM_dzdot', 'G_W', 'I_yy', 'g', 'V'):
            J['handling_margin', name] = J['short_period_parameter', name]
