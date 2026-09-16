"""Cockpit control positions from the Appendix A rigging curves.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*, Appendix A
"Characteristics of the Example Helicopter", Figure A.5 p. 682. Anchors in
Table 8.8 p. 530 and Figure 8.31 p. 538.

All four panels of Figure A.5 are straight lines, so the figure reduces to
four linear relations and no table is stored. Each was tracked column by
column and fitted, with these residuals over the printed span:

======================= =============== ================
panel                    residual        points
======================= =============== ================
longitudinal cyclic      0.26 %          121
lateral cyclic           0.047 deg        75
tail rotor collective    0.069 deg        51
main rotor collective    0.29 %           63
======================= =============== ================
"""

import numpy as np
import openmdao.api as om

# Figure A.5 p. 682, fitted. Percentages run from full forward, full left
# and full down; the control travels are printed on the figure itself.
LONG_PCT_0, LONG_PCT_SLOPE, LONG_TRAVEL = 67.03, -3.2851, 10.0
LAT_A1_0, LAT_A1_SLOPE, LAT_TRAVEL = -10.333, 0.17144, 9.0
PEDAL_TH_0, PEDAL_TH_SLOPE, PEDAL_TRAVEL = 32.300, -0.47790, 5.5
COLL_PCT_0, COLL_PCT_SLOPE, COLL_TRAVEL = -3.735, 5.6861, 12.0

# channel -> (blade angle, position output, zero parameter, slope parameter,
#             whether the chart reads position from angle or angle from
#             position)
_CHANNELS = {
    'long': ('B_1', 'pct_B1', 'long_pct_0', 'long_pct_slope', 'direct'),
    'lat': ('A_1', 'pct_A1', 'lat_A1_0', 'lat_A1_slope', 'inverse'),
    'pedal': ('theta_75_T', 'pct_pedal', 'pedal_th_0', 'pedal_th_slope',
              'inverse'),
    'collective': ('theta_75_M', 'pct_coll', 'coll_pct_0', 'coll_pct_slope',
                   'direct'),
}
_DEFAULTS = {
    'long': (LONG_PCT_0, LONG_PCT_SLOPE, None, '1/deg'),
    'lat': (LAT_A1_0, LAT_A1_SLOPE, 'deg', 'deg'),
    'pedal': (PEDAL_TH_0, PEDAL_TH_SLOPE, 'deg', 'deg'),
    'collective': (COLL_PCT_0, COLL_PCT_SLOPE, None, '1/deg'),
}


class ControlPositionsComp(om.ExplicitComponent):
    """Where the pilot's hands and feet are, Figure A.5 p. 682::

        pct_B1       = 67.03 - 3.285 B_1          % from full forward
        pct_A1       = (A_1 + 10.333) / 0.17144   % from full left
        pct_pedal    = (32.30 - th_75T) / 0.4779  % from full left
        pct_coll     = -3.735 + 5.686 th_75M      % from full down

    The last step of the chapter. Trim gives blade angles; a pilot flies a
    stick, and the flying qualities specifications that p. 530 cites are
    written in inches of control travel, not degrees of cyclic. This is the
    kinematic linkage that connects the two, and it is the only place in the
    package where the answer depends on the airframe's mechanical design
    rather than on its aerodynamics.

    Note which way each one runs. More ``B_1`` is more forward stick, so the
    longitudinal percentage *falls* as the cyclic rises. More ``A_1`` is more
    right stick and the lateral percentage rises. More tail rotor pitch is
    more left pedal, so that percentage falls too.

    Options
    -------
    num_nodes : int
    channels : tuple of {'long', 'lat', 'pedal', 'collective'}
        Which of the four panels to evaluate. Only the requested inputs and
        outputs are declared, so a group that solves one axis does not carry
        three controls stuck at zero. ``LongitudinalTrimGroup`` asks for
        ``('long',)`` and ``LateralDirectionalTrimGroup`` for
        ``('lat', 'pedal')``.

    Agreement with the book
    -----------------------
    Table 8.8 p. 530 gives three longitudinal positions in level flight,
    autorotation and climb, and they check the fit end to end:

    ========== ========== ============ =========
    ``B_1``     printed    this fit     error
    ========== ========== ============ =========
    8.5 deg     38 %       39.1 %       +1.1
    8.8 deg     37 %       38.1 %       +1.1
    9.4 deg     35 %       36.2 %       +1.2
    ========== ========== ============ =========

    A constant offset of 1.1 points and a slope of -3.285 against the -3.33
    the three printed values imply between them: the line is right and the
    zero is a point low, which is digitising precision on a chart whose
    ordinate is 2.2 pixels per per cent.

    Table 8.7 p. 529 does not agree with either. See C8-14 in
    ``docs/validation_trim.md``.

    Notes
    -----
    The rigging constants are inputs, not literals, so another helicopter
    can be described without touching this file. The defaults are Figure A.5.

    ``in_B1``, declared with the ``'long'`` channel, is the longitudinal
    stick position in inches from full forward, on the 10 inch travel the
    figure prints. p. 530 puts the minimum acceptable control shift between
    full-power climb and autorotation at 3 inches, and that criterion cannot
    be checked in per cent.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('channels', default=('long', 'lat', 'pedal',
                                                  'collective'),
                             types=tuple)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        for name in self.options['channels']:
            if name not in _CHANNELS:
                raise ValueError('unknown control channel %r' % name)
            angle, pct, zero, slope, _ = _CHANNELS[name]
            z_val, s_val, z_units, s_units = _DEFAULTS[name]

            self.add_input(angle, shape=(nn,), val=0.0, units='deg')
            self.add_input(zero, val=z_val, units=z_units)
            self.add_input(slope, val=s_val, units=s_units)
            self.add_output(pct, shape=(nn,), desc='per cent of travel')

            self.declare_partials(pct, angle, rows=ar, cols=ar)
            self.declare_partials(pct, [zero, slope], rows=ar, cols=zeros)

        if 'long' in self.options['channels']:
            self.add_input('long_travel', val=LONG_TRAVEL, units='inch')
            self.add_output('in_B1', shape=(nn,), units='inch',
                            desc='longitudinal stick from full forward')
            self.declare_partials('in_B1', 'B_1', rows=ar, cols=ar)
            self.declare_partials('in_B1', ['long_pct_0', 'long_pct_slope',
                                            'long_travel'],
                                  rows=ar, cols=zeros)

    def _percent(self, inputs, name):
        angle, _, zero, slope, form = _CHANNELS[name]
        if form == 'direct':
            return inputs[zero][0] + inputs[slope][0] * inputs[angle]
        return (inputs[angle] - inputs[zero][0]) / inputs[slope][0]

    def compute(self, inputs, outputs):
        for name in self.options['channels']:
            outputs[_CHANNELS[name][1]] = self._percent(inputs, name)
        if 'long' in self.options['channels']:
            outputs['in_B1'] = (outputs['pct_B1'] / 100.0
                                * inputs['long_travel'][0])

    def compute_partials(self, inputs, J):
        for name in self.options['channels']:
            angle, pct, zero, slope, form = _CHANNELS[name]
            ones = np.ones_like(inputs[angle])
            if form == 'direct':
                J[pct, angle] = inputs[slope][0] * ones
                J[pct, zero] = ones
                J[pct, slope] = inputs[angle]
            else:
                value = self._percent(inputs, name)
                J[pct, angle] = ones / inputs[slope][0]
                J[pct, zero] = -ones / inputs[slope][0]
                J[pct, slope] = -value / inputs[slope][0]

        if 'long' in self.options['channels']:
            travel = inputs['long_travel'][0]
            J['in_B1', 'B_1'] = J['pct_B1', 'B_1'] / 100.0 * travel
            J['in_B1', 'long_pct_0'] = travel / 100.0 * np.ones_like(
                inputs['B_1'])
            J['in_B1', 'long_pct_slope'] = inputs['B_1'] * travel / 100.0
            J['in_B1', 'long_travel'] = self._percent(inputs, 'long') / 100.0
