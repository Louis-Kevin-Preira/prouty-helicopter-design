"""
AirfoilHoverGroup -- NACA 0012 section coefficients for hover analysis.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 6, "Representing Airfoil Data with Equations", p. 426-433.

    M     --> LiftModelCoefsComp   --> a, alpha_L, K1, K2   (p. 427-430)
    alpha --> LiftCoefComp         --> cl                   (p. 428, 430)
    M     --> DragModelCoefsComp   --> alpha_D, K3, K4, delta_cd_M (p. 432-433)
    alpha --> IncompDragHoverComp  --> cd_incomp            (p. 432)
              DragCoefHoverComp    --> cd                   (p. 432-433)

Inputs : M, alpha [deg]      Outputs : cl, cd
"""

import openmdao.api as om

from prouty.airfoil.lift_model_coefs_comp import LiftModelCoefsComp
from prouty.airfoil.lift_coef_comp import LiftCoefComp
from prouty.airfoil.drag_model_coefs_comp import DragModelCoefsComp
from prouty.airfoil.incomp_drag_hover_comp import IncompDragHoverComp
from prouty.airfoil.drag_coef_hover_comp import DragCoefHoverComp


class AirfoilHoverGroup(om.Group):
    """NACA 0012 lift and drag coefficients, hover form."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('clip_K1', types=bool, default=True,
                             desc='see LiftCoefComp')
        self.options.declare('symmetric_stall', types=bool, default=False,
                             desc='see DragCoefHoverComp')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('lift_coefs', LiftModelCoefsComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem('lift', LiftCoefComp(num_nodes=nn,
                                                clip_K1=self.options['clip_K1']),
                           promotes=['*'])
        self.add_subsystem('drag_coefs', DragModelCoefsComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem('drag_incomp', IncompDragHoverComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem(
            'drag',
            DragCoefHoverComp(num_nodes=nn,
                              symmetric_stall=self.options['symmetric_stall']),
            promotes=['*'])
