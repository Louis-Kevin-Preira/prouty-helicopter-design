"""
ClosedFormRotorGroup (G1) -- closed-form rotor equations.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 163-207.

Given the non-dimensional flight condition produced by G0, this group returns
the rotor's trim angles and its torque and H-force coefficients:

    a0          coning                                       p. 169-171
    theta_0     collective, or C_T/sigma in 'thrust' mode     p. 167-168
    B1_a1s      longitudinal control plane tilt               p. 168
    A1_b1s      lateral control plane tilt                    p. 169
    cl_bar      average blade element lift coefficient        p. 168
    CQ_sigma    torque coefficient, profile plus inflow       p. 174-175
    CH_sigma    H-force coefficient                           p. 175-176
    CQ_sigma_total  torque including compressibility          p. 184-185

Execution is feed-forward in both modes, no solver:

    'collective'  C_T/sigma given -> theta_0. This is the trim direction, the
                  one G1b and Table 3.5 need.
    'thrust'      theta_0 given -> C_T/sigma. This is the chart direction: the
                  isolated rotor charts are drawn as families of theta_0
                  curves (p. 254). In this mode C_T/sigma is an OUTPUT, so G0's
                  ThrustCoefComp must not also be connected to it.

Options

    airfoil          a Chapter 6 group; when given, cd_bar is computed at the
                     p. 175 reference condition instead of being an input.
                     Interface: alpha (rad), M -> cd. See mean_drag_coef.
    compressibility  adds the Figure 3.43 penalty and exposes CQ_sigma_total.
                     Below the drag rise threshold the increment is exactly
                     zero, with zero derivative, so leaving it on costs
                     nothing.
    drag_basis       'hover' follows p. 175, 'forward' uses the p. 168 mean
                     lift coefficient. They differ by 14 % at mu = 0.3 and
                     30 % at mu = 0.45 -- see AvgLiftCoefComp.

Two inputs that look like they should be outputs, and are not:

  * a1s. Only the sums B_1 + a_1s and A_1 - b_1s are set by the aerodynamics;
    splitting them needs the pitching and rolling moment balance of Chapter 8.
    Prouty's Table 3.2 sets a_1s = b_1s = 0, which is the default here and is
    valid for performance but not for stability and control.
  * B and x_0. Tip loss and root cutout enter as ordinary inputs rather than a
    discrete option, so that B = 1, x_0 = 0 reproduces the assumptions-held
    equations exactly while keeping the model differentiable with respect to
    B, which p. 199 obtains from the hover equation rather than fixing.

Still outside this group: the reverse-flow region (p. 202-204) and the
three-term drag polar (p. 205), which belong to the eliminating-the-
assumptions pass, and InducedVelocityComp and ConingComp still carry their
B = 1 forms rather than those of p. 200.
"""

import numpy as np
import openmdao.api as om

from .avg_lift_coef_comp import AvgLiftCoefComp
from .collective_pitch_comp import CollectivePitchComp
from .compressibility_torque_comp import CompressibilityTorqueComp
from .coning_comp import ConingComp
from .drag_rise_mach_comp import DragRiseMachComp
from .h_force_coef_comp import HForceCoefComp
from .lat_flapping_comp import LatFlappingComp
from .long_flapping_comp import LongFlappingComp
from .mean_drag_coef import MeanDragConditionComp
from .torque_coef_comp import TorqueCoefComp


class ClosedFormRotorGroup(om.Group):
    """G1 -- closed-form thrust, flapping, torque and H-force."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('mode', values=('collective', 'thrust'),
                             default='collective')
        self.options.declare('compressibility', types=bool, default=True)
        self.options.declare('drag_basis', values=('hover', 'forward'),
                             default='hover')
        self.options.declare('airfoil', default=None, recordable=False,
                             desc='Chapter 6 airfoil group, or None to take '
                                  'cd_bar as an input')

    def setup(self):
        nn = self.options['num_nodes']
        mode = self.options['mode']

        # C_T/sigma must exist before coning and the drag reference, so in
        # 'thrust' mode the collective relation runs first
        pitch = CollectivePitchComp(num_nodes=nn, mode=mode)
        if mode == 'thrust':
            self.add_subsystem('collective', pitch, promotes=['*'])

        self.add_subsystem('coning', ConingComp(num_nodes=nn, form='ct'),
                           promotes=['*'])
        self.add_subsystem('lift_coef',
                           AvgLiftCoefComp(num_nodes=nn,
                                           form=self.options['drag_basis']),
                           promotes=['*'])

        if self.options['airfoil'] is not None:
            self.add_subsystem('drag_condition',
                               MeanDragConditionComp(num_nodes=nn),
                               promotes=['*'])
            self.add_subsystem(
                'airfoil', self.options['airfoil'],
                promotes_inputs=[('alpha', 'alpha_cd'), ('M', 'M_075')],
                promotes_outputs=[('cd', 'cd_bar')])

        if mode == 'collective':
            self.add_subsystem('collective', pitch, promotes=['*'])

        self.add_subsystem('long_flapping',
                           LongFlappingComp(num_nodes=nn, form='tpp'),
                           promotes=['*'])
        self.add_subsystem('lat_flapping', LatFlappingComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem('torque', TorqueCoefComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem('h_force',
                           HForceCoefComp(num_nodes=nn, form='direct'),
                           promotes=['*'])

        if self.options['compressibility']:
            self.add_subsystem('drag_rise', DragRiseMachComp(num_nodes=nn),
                               promotes=['*'])
            self.add_subsystem('compressibility',
                               CompressibilityTorqueComp(num_nodes=nn),
                               promotes=['*'])
            self.add_subsystem(
                'torque_total',
                om.ExecComp('CQ_sigma_total = CQ_sigma + dCQ_sigma_comp',
                            CQ_sigma_total=np.zeros(nn),
                            CQ_sigma=np.zeros(nn),
                            dCQ_sigma_comp=np.zeros(nn)),
                promotes=['*'])
