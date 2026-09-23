"""
FuelFlowSlopeComp -- fuel flow and its slope with speed, from a sub-problem.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Cruising Flight" pp. 323-325, Figures 4.42 and 4.43.

The best range speed is the tangency condition FF = (V - V_wind) dFF/dV
(BestRangeSpeedBalance), so the slope of the fuel flow with speed is needed as
a value, not only as a derivative of the model. It cannot be read anywhere in
the chain: the fuel flow comes out of the engine, which follows the power
required, which comes out of the Chapter 3 trim.

This component therefore carries its own sub-problem, a copy of the cruise
chain, runs it at the current point and asks OpenMDAO for the total
derivative:

    FF      = sub-problem output
    dFF/dV  = compute_totals(of=fuel, wrt=speed)

so the slope is the analytic derivative through the trim, not a difference
quotient. Its own partials are finite differenced, which is a second
derivative of the chain: the value the balance closes on stays exact, only the
Newton step is approximate.

    V, <pass-through inputs> --> FF, dFF_dV
"""

import numpy as np
import openmdao.api as om
from openmdao.utils.units import convert_units


class FuelFlowSlopeComp(om.ExplicitComponent):
    """Fuel flow and dFF/dV from a sub-problem of the cruise chain."""

    def initialize(self):
        self.options.declare('problem', types=om.Problem,
                             desc='cruise chain, not yet set up: speed in, fuel flow out')
        self.options.declare('speed_name', default='V')
        self.options.declare('fuel_name', default='FF')
        self.options.declare('speed_units', default='ft/s')
        self.options.declare('fuel_units', default='lbm/h')
        self.options.declare('passthrough', types=dict, default={},
                             desc='name -> units of the other sub-problem inputs to expose')
        self.options.declare('reset', types=dict, default={},
                             desc='name -> value forced before each sub-problem run, to give '
                                  'its own solvers a fresh start')

    def setup(self):
        self._prob = self.options['problem']
        try:                                      # an already set up problem keeps its values
            self._prob.get_val(self.options['speed_name'])
        except Exception:
            self._prob.setup()
            self._prob.final_setup()

        self.add_input('V', val=100.0, units='kn', desc='true airspeed')
        for name, units in self.options['passthrough'].items():
            self.add_input(name, val=self._prob.get_val(name, units=units)[0]
                           if np.ndim(self._prob.get_val(name, units=units)) else
                           self._prob.get_val(name, units=units), units=units)

        self.add_output('FF', val=500.0, units=self.options['fuel_units'])
        self.add_output('dFF_dV', val=0.0,
                        units=f"{self.options['fuel_units']}/kn")

        self.declare_partials('*', '*', method='fd', step=0.05, form='central')

    def _run(self, inputs):
        prob, opt = self._prob, self.options
        for name, val in opt['reset'].items():
            prob.set_val(name, val)
        prob.set_val(opt['speed_name'], np.real(inputs['V'][0]), units='kn')
        for name, units in opt['passthrough'].items():
            prob.set_val(name, np.real(inputs[name][0]), units=units)
        prob.run_model()
        return prob

    def compute(self, inputs, outputs):
        prob, opt = self._run(inputs), self.options
        FF = prob.get_val(opt['fuel_name'], units=opt['fuel_units'])
        totals = prob.compute_totals(of=[opt['fuel_name']], wrt=[opt['speed_name']])
        # compute_totals works in the sub-problem's own units
        per_knot = convert_units(1.0, 'kn', opt['speed_units'])
        outputs['FF'] = np.sum(FF)
        outputs['dFF_dV'] = np.sum(totals[opt['fuel_name'], opt['speed_name']]) * per_knot
