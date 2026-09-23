"""Stability metrics from a swept longitudinal trim.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", pp. 525-530: speed stability p. 525,
angle of attack stability p. 529, power effects p. 530. Anchors in
Figure 8.27 p. 528, Table 8.7 p. 529 and Table 8.8 p. 530.
"""

import numpy as np
import openmdao.api as om


class TrimGradientComp(om.ExplicitComponent):
    """How the longitudinal cyclic moves across a swept trim, pp. 525-530.

    Three subsections of the chapter run the same experiment: solve the
    longitudinal trim at two or three conditions and look at what happened
    to ``B_1``. Run ``LongitudinalTrimGroup`` with ``num_nodes`` set to the
    number of conditions and hand the result here::

        dB1_dsweep = diff(B_1) / diff(sweep_var)
        B1_shift   = B_1[-1] - B_1[0]

    The arithmetic is a finite difference. What this component actually
    carries is the sign convention, which is the content of those pages and
    is not the same in all three.

    More ``B_1`` means more forward stick
    -------------------------------------
    ============== ================================= =================
    ``sweep``       stable when                        p.
    ============== ================================= =================
    speed           ``B_1`` rises with speed           525-528
    load_factor     ``B_1`` falls with load factor     529
    climb           no sign; the shift itself is       530
                    the quantity of interest
    ============== ================================= =================

    Speed stability is a pilot holding *more forward* stick to fly faster:
    let go and the helicopter pitches up and slows back down. Manoeuvre
    stability is the opposite sense — p. 529 asks whether the stick is *aft*
    of the level flight position in a descending turn — because there the
    disturbance being resisted is an increase in angle of attack, not in
    speed. Getting these two backwards is easy and produces a model that
    calls an unstable helicopter stable.

    ``sweep_var`` is declared dimensionless. A gradient with respect to
    speed, load factor or climb angle has no single unit, so the input
    carries whatever the caller passes — knots, g, degrees — and
    ``dB1_dsweep`` comes back in radians per that. The sign is unaffected,
    and the sign is the point.

    Nodes must be in order, since ``B1_shift`` is taken end to end rather
    than as a maximum minus a minimum: that keeps it differentiable, and
    Table 8.8 p. 530 is already written from autorotation through level
    flight to climb.

    Anchors
    -------
    ==================== ================ ============== ==========
    subsection            sweep            ``B_1``        verdict
    ==================== ================ ============== ==========
    speed, p. 527         115, 135 kt      8.9, 9.3 deg   stable
    load factor, p. 529   1.0, 1.3 g       8.9, 11.4 deg  unstable
    power, p. 530         -9.2, 0, 9.7 deg 8.5, 8.8, 9.4  0.9 deg
    ==================== ================ ============== ==========

    p. 530 reads the last row as a good result: 0.9 degrees of cyclic across
    the whole power range, "practically no stick motion", because the change
    in ``B_1 + a1s_M`` needed to trim the rotor nearly cancels the change in
    ``a1s_M`` needed to trim the airframe. It also says why, and the reason
    is uncomfortable: it is a benefit of having a stabiliser too small to
    give positive angle of attack stability. The same helicopter fails the
    p. 529 test, and Table 8.7 shows the stick moving *forward* in the turn.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=2)
        self.options.declare('sweep', values=('speed', 'load_factor',
                                              'climb'), default='speed')

    def setup(self):
        nn = self.options['num_nodes']
        if nn < 2:
            raise ValueError('TrimGradientComp needs at least two nodes')
        ar = np.arange(nn - 1)

        self.add_input('B_1', shape=(nn,), val=0.0, units='rad',
                       desc='longitudinal cyclic pitch at each condition')
        self.add_input('sweep_var', shape=(nn,), val=np.arange(nn),
                       desc='the swept quantity, in the caller units')

        self.add_output('dB1_dsweep', shape=(nn - 1,),
                        desc='rad of cyclic per unit of sweep_var')
        self.add_output('B1_shift', shape=(1,), units='rad',
                        desc='B_1 at the last node minus the first')

        self.declare_partials('dB1_dsweep', ['B_1', 'sweep_var'],
                              rows=np.repeat(ar, 2),
                              cols=np.column_stack([ar, ar + 1]).ravel())
        self.declare_partials('B1_shift', 'B_1',
                              rows=np.zeros(2, dtype=int),
                              cols=np.array([0, nn - 1]),
                              val=np.array([-1.0, 1.0]))

        if self.options['sweep'] != 'climb':
            self.add_output('stability', shape=(nn - 1,),
                            desc='dB1_dsweep signed so positive is stable')
            self.declare_partials('stability', ['B_1', 'sweep_var'],
                                  rows=np.repeat(ar, 2),
                                  cols=np.column_stack([ar, ar + 1]).ravel())

    def _sign(self):
        return -1.0 if self.options['sweep'] == 'load_factor' else 1.0

    def compute(self, inputs, outputs):
        gradient = np.diff(inputs['B_1']) / np.diff(inputs['sweep_var'])
        outputs['dB1_dsweep'] = gradient
        outputs['B1_shift'] = inputs['B_1'][-1] - inputs['B_1'][0]
        if self.options['sweep'] != 'climb':
            outputs['stability'] = self._sign() * gradient

    def compute_partials(self, inputs, J):
        dx = np.diff(inputs['sweep_var'])
        gradient = np.diff(inputs['B_1']) / dx

        dB = np.column_stack([-1.0 / dx, 1.0 / dx]).ravel()
        dx_part = np.column_stack([gradient / dx, -gradient / dx]).ravel()

        J['dB1_dsweep', 'B_1'] = dB
        J['dB1_dsweep', 'sweep_var'] = dx_part
        if self.options['sweep'] != 'climb':
            J['stability', 'B_1'] = self._sign() * dB
            J['stability', 'sweep_var'] = self._sign() * dx_part
