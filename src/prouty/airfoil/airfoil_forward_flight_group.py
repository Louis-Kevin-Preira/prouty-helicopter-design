"""
AirfoilForwardFlightGroup -- NACA 0012 section coefficients over 0-360 deg.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 6, "Representing Airfoil Data with Equations", p. 426-434,
Figure 6.47 p. 434; quadrant convention p. 214.

    alpha_raw --> AlphaWrapComp        --> alpha_360, alpha_sig   (p. 214, 433)

    M         --> LiftModelCoefsComp   --> a, alpha_L, K1, K2     (p. 427-430)
    alpha_sig --> LiftCoefComp         --> cl_gen                 (p. 428, 430)
    alpha_360 --> LiftCoefFwdComp      --> cl                     (p. 433)

    M         --> DragModelCoefsComp   --> alpha_D, K3, K4, delta_cd_M (p. 432-433)
    alpha_sig --> IncompDragFwdComp    --> cd_incomp              (p. 433)
    alpha_sig --> DragCoefHoverComp    --> cd_gen                 (p. 432-433)
    alpha_360 --> DragCoefFwdComp      --> cd                     (p. 434)

Three components are reused unchanged from the hover group: LiftModelCoefsComp,
LiftCoefComp and DragModelCoefsComp. Only the incompressible drag series and
the 0-360 deg extension are specific to forward flight.

Option stall_angles (added for Chapter 3, p. 218 and 221)

    'internal'  the stall angles come straight from the Mach models, which is
                the behaviour this group has always had.
    'external'  StallAngleComp is inserted between the Mach models and the
                coefficient generators, and the group additionally accepts
                sec_Lambda and d_alpha_stall. The numerical rotor method of
                Chapter 3 needs this: yawed flow raises the lift stall angle
                by sec(Lambda) (p. 218) and dynamic overshoot delays both lift
                stall and drag rise (p. 221). With the neutral defaults,
                sec_Lambda = 1 and d_alpha_stall = 0, the two modes agree
                exactly.

Inputs : alpha_raw [deg], M [, sec_Lambda, d_alpha_stall]
Outputs: cl, cd [, alpha_L_static, alpha_D_static]
"""

import openmdao.api as om

from prouty.airfoil.alpha_wrap_comp import AlphaWrapComp
from prouty.airfoil.lift_model_coefs_comp import LiftModelCoefsComp
from prouty.airfoil.lift_coef_comp import LiftCoefComp
from prouty.airfoil.lift_coef_fwd_comp import LiftCoefFwdComp
from prouty.airfoil.drag_model_coefs_comp import DragModelCoefsComp
from prouty.airfoil.incomp_drag_fwd_comp import IncompDragFwdComp
from prouty.airfoil.drag_coef_hover_comp import DragCoefHoverComp
from prouty.airfoil.drag_coef_fwd_comp import DragCoefFwdComp
from prouty.airfoil.stall_angle_comp import StallAngleComp


class AirfoilForwardFlightGroup(om.Group):
    """NACA 0012 lift and drag coefficients over the full azimuth range."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('clip_K1', types=bool, default=True,
                             desc='see LiftCoefComp')
        self.options.declare('stall_angles', values=('internal', 'external'),
                             default='internal',
                             desc='see the module header')

    def setup(self):
        nn = self.options['num_nodes']
        external = self.options['stall_angles'] == 'external'

        # in external mode the two stall angles leave the Mach models under
        # a _static name and come back corrected from StallAngleComp
        lift_coefs_out = (['a', 'K1', 'K2', ('alpha_L', 'alpha_L_static')]
                          if external else ['*'])
        drag_coefs_out = (['K3', 'K4', 'delta_cd_M',
                           ('alpha_D', 'alpha_D_static')]
                          if external else ['*'])

        self.add_subsystem('wrap', AlphaWrapComp(num_nodes=nn), promotes=['*'])

        # --- Mach-dependent coefficients, p. 427-430 and 432-433 -------
        self.add_subsystem('lift_coefs', LiftModelCoefsComp(num_nodes=nn),
                           promotes_inputs=['M'],
                           promotes_outputs=lift_coefs_out)
        self.add_subsystem('drag_coefs', DragModelCoefsComp(num_nodes=nn),
                           promotes_inputs=['M'],
                           promotes_outputs=drag_coefs_out)

        # The stall angles must be corrected BEFORE the generators consume
        # them; adding this subsystem after them leaves lift_gen reading the
        # default alpha_L of 1 deg and every post-stall coefficient wrong.
        if external:
            self.add_subsystem('stall_angles', StallAngleComp(num_nodes=nn),
                               promotes=['*'])

        # --- lift chain, p. 428-430 then p. 433 ------------------------
        self.add_subsystem('lift_gen',
                           LiftCoefComp(num_nodes=nn,
                                        clip_K1=self.options['clip_K1']),
                           promotes_inputs=['a', 'alpha_L', 'K1', 'K2'],
                           promotes_outputs=[('cl', 'cl_gen')])
        self.add_subsystem('lift', LiftCoefFwdComp(num_nodes=nn), promotes=['*'])

        # --- drag chain, p. 432-433 then p. 434 ------------------------
        self.add_subsystem('drag_incomp', IncompDragFwdComp(num_nodes=nn),
                           promotes_outputs=['cd_incomp'])
        self.add_subsystem(
            'drag_gen', DragCoefHoverComp(num_nodes=nn, symmetric_stall=True),
            promotes_inputs=['cd_incomp', 'alpha_D', 'K3', 'K4', 'delta_cd_M'],
            promotes_outputs=[('cd', 'cd_gen')])
        self.add_subsystem('drag', DragCoefFwdComp(num_nodes=nn), promotes=['*'])

        self.connect('alpha_sig',
                     ['lift_gen.alpha', 'drag_incomp.alpha', 'drag_gen.alpha'])
