"""
Mean blade element drag coefficient for the closed-form equations.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 175 (prescription) and p. 206 (choice of the representative
condition).

The closed-form torque and H-force equations of p. 174-176 carry a single
c_d for the whole rotor. Prouty prescribes where to evaluate it (p. 175):
"the average obtained as a function of C_T/sigma as it was in hover -- that
is, as a function of angle of attack equal to 57.3 (C_T/sigma) degrees and at
a Mach number corresponding to 75 % of the tip speed."

That splits into three things, each with its own home:

    AvgLiftCoefComp         C_T/sigma, mu  ->  cl_bar          p. 168
    MeanDragConditionComp   cl_bar, a      ->  alpha_cd        p. 175
    the Chapter 6 airfoil   alpha_cd, M    ->  cd_bar

MeanDragConditionComp is therefore thin on purpose: alpha = cl_bar / a. It
does not recompute cl_bar, so the 'hover versus forward' choice lives in one
place only, on AvgLiftCoefComp. p. 175 calls for the hover form, which is
what MeanDragCoefGroup builds by default.

Where 57.3 comes from. It is not a units conversion dressed up: alpha in
radians is cl_bar / a = 6 (C_T/sigma) / a, which is exactly C_T/sigma when
a = 6 /rad, hence 57.3 (C_T/sigma) in degrees. The literal 57.3 therefore
silently assumes a = 6.0, the same value that reproduces the Lock number of
Appendix A and the flapping moment of p. 477. Computing cl_bar / a keeps that
assumption visible and lets a different lift curve slope propagate.

Airfoil model interface assumed by MeanDragCoefGroup:

    input   'alpha'  units='rad'   shape (nn,)
    input   'M'                    shape (nn,)
    output  'cd'                   shape (nn,)

Rename the promotes in the group if prouty.airfoil uses other names.
"""

import numpy as np
import openmdao.api as om

from .avg_lift_coef_comp import AvgLiftCoefComp


class MeanDragConditionComp(om.ExplicitComponent):
    """Reference angle of attack for the mean drag coefficient, p. 175."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('cl_bar', shape=(nn,), desc='mean lift coefficient')
        self.add_input('a', val=6.0, units='1/rad', desc='lift curve slope')

        self.add_output('alpha_cd', shape=(nn,), units='rad',
                        desc='reference angle of attack for c_d')

        self.declare_partials('alpha_cd', 'cl_bar', rows=ar, cols=ar)
        self.declare_partials('alpha_cd', 'a', rows=ar,
                              cols=np.zeros(nn, dtype=int))

    def compute(self, inputs, outputs):
        outputs['alpha_cd'] = inputs['cl_bar'] / inputs['a'][0]

    def compute_partials(self, inputs, partials):
        a = inputs['a'][0]
        partials['alpha_cd', 'cl_bar'] = 1.0 / a
        partials['alpha_cd', 'a'] = -inputs['cl_bar'] / a ** 2


class MeanDragCoefGroup(om.Group):
    """cl_bar, its reference angle of attack, and an airfoil model."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('basis', values=('hover', 'forward'),
                             default='hover',
                             desc="form of cl_bar; p. 175 calls for 'hover'")
        self.options.declare('airfoil', types=om.Group, recordable=False,
                             desc='Chapter 6 airfoil model, see module header')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('cl_bar',
                           AvgLiftCoefComp(num_nodes=nn,
                                           form=self.options['basis']),
                           promotes=['*'])
        self.add_subsystem('condition', MeanDragConditionComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem(
            'airfoil', self.options['airfoil'],
            promotes_inputs=[('alpha', 'alpha_cd'), ('M', 'M_075')],
            promotes_outputs=[('cd', 'cd_bar')])
