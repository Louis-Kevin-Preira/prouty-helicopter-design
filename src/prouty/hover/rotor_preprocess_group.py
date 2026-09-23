"""
RotorPreprocessGroup -- steps 1 to 3 of the combined momentum and blade element
method.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, p. 69.

    step 1  RotorGeometryComp   dimensional geometry -> ratios, A, Omega
            AtmosphereComp      pressure altitude    -> rho, V_son   (p. 703-704)
    step 2  BladeGridComp       radial stations r/R
    step 3  ChordDistComp       c/R, sigma, sigma_T                  (p. 17)
            TwistDistComp       twist increment d_theta              (p. 14)
            LocalMachComp       local Mach number M

Everything here depends on geometry and test conditions only, never on the
collective or the inflow, so the whole group evaluates once per design point
and sits upstream of the inflow / airfoil cycle.

Inputs  : R, c_root, c_tip, r_1, r_cutout, V_tip, altitude, dT, b, theta_1
Outputs : r_R, c_R, d_theta, M (nn,); sigma, sigma_T, A, Omega, rho, V_son
"""

import openmdao.api as om

from prouty.hover.rotor_geometry_comp import RotorGeometryComp
from prouty.hover.atmosphere_comp import AtmosphereComp
from prouty.hover.blade_grid_comp import BladeGridComp
from prouty.hover.chord_dist_comp import ChordDistComp
from prouty.hover.twist_dist_comp import TwistDistComp
from prouty.hover.local_mach_comp import LocalMachComp


class RotorPreprocessGroup(om.Group):
    """Rotor geometry, atmosphere and blade element tabulation."""

    def initialize(self):
        self.options.declare('num_elements', types=int, default=10,
                             desc='blade elements; p. 69 recommends 5 to 15')
        self.options.declare('distribution', values=('uniform', 'cosine'),
                             default='uniform')
        self.options.declare('twist_reference', values=('center', 'cutout'),
                             default='center')

    def setup(self):
        ne = self.options['num_elements']
        nn = ne + 1

        self.add_subsystem('geometry', RotorGeometryComp(), promotes=['*'])
        self.add_subsystem('atmosphere', AtmosphereComp(), promotes=['*'])
        self.add_subsystem('grid', BladeGridComp(
            num_elements=ne, distribution=self.options['distribution']),
            promotes=['*'])
        self.add_subsystem('chord', ChordDistComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('twist', TwistDistComp(
            num_nodes=nn, reference=self.options['twist_reference']),
            promotes=['*'])
        self.add_subsystem('mach', LocalMachComp(num_nodes=nn), promotes=['*'])
