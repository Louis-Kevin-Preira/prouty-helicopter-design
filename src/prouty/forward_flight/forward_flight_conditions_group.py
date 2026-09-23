"""
ForwardFlightConditionsGroup (G0) -- forward flight conditions and
non-dimensionalisation.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 142, 162, 166-168, 175, 213, 232.

This group turns a flight condition and a rotor description into the six
non-dimensional quantities the rest of Chapter 3 is written in:

    mu        tip speed ratio                        p. 142, 162
    M_190     advancing tip Mach number              p. 232
    M_075     Mach at 75 % of tip speed              p. 175
    gamma     Lock number                            p. 31, 213
    CT_sigma  thrust coefficient over solidity       p. 168
    lambda'   inflow ratio, tip path plane           p. 166-167

Execution is feed-forward, no solver needed:

    c, R           --> BladeAreaComp        --> A_b, A, sigma
    V, V_tip       --> AdvanceRatioComp     --> mu
    mu, V_tip,V_son--> TipMachComp          --> M_tip, M_190, M_075
    rho,a,c,R,I_b  --> LockNumberComp       --> gamma
    T,rho,A_b,V_tip--> ThrustCoefComp       --> CT_sigma
    ...            --> TppAngleComp         --> alpha_TPP
    CT_sigma,sigma,mu-> InducedVelocityComp --> vi_OR
    mu,alpha_TPP,vi_OR-> InflowComp         --> lambda_p

Two options carry the modelling choices settled component by component:

    tpp_form = 'parasite'  (p. 167)  alpha_TPP from parasite drag alone.
                                     Acyclic, this is the initialisation branch
                                     and Prouty's own first iteration (p. 193).
             = 'forces'    (p. 192)  alpha_TPP from the force balance. Needs
                                     L_F, D_F, H_M, H_T, which are outputs of
                                     G1 and G1b, so choosing it closes a loop
                                     through this group and requires a solver
                                     at the level above.

    vi_form  = 'high_speed' (p. 167) v1/OmegaR = C_T/(2 mu), the form used
                                     throughout the chapter. Singular at mu=0.
             = 'exact'      (p. 123) regular everywhere, 5.8 % away from the
                                     other at mu = 0.10.

Cross-chapter connections still to be made when the package is assembled:

  * `a` must arrive in 1/rad from AirfoilHoverGroup evaluated once at M_075.
    Prouty's example helicopter is consistent with a = 6.0 /rad, which is the
    compressible slope of the 0012 at M = 0.44 (see LockNumberComp).
  * `rho`, `V_son` come from the Chapter 1 AtmosphereComp.
  * `T` must be the rotor thrust from the trim loop (p. 192), not the gross
    weight. The difference is 3 % at mu = 0.3 and more in a climb.
  * If prouty.hover already exposes A_b and sigma from its own geometry
    component, drop BladeAreaComp and promote from there instead.
"""

import openmdao.api as om

from .advance_ratio_comp import AdvanceRatioComp
from .blade_area_comp import BladeAreaComp
from .induced_velocity_comp import InducedVelocityComp
from .inflow_comp import InflowComp
from .lock_number_comp import LockNumberComp
from .thrust_coef_comp import ThrustCoefComp
from .tip_mach_comp import TipMachComp
from .tpp_angle_comp import TppAngleComp


class ForwardFlightConditionsGroup(om.Group):
    """G0 -- flight conditions and non-dimensionalisation."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('num_blades', types=int, default=4)
        self.options.declare('tpp_form', values=('parasite', 'forces'),
                             default='parasite')
        self.options.declare('vi_form', values=('high_speed', 'exact'),
                             default='high_speed')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('geometry',
                           BladeAreaComp(num_blades=self.options['num_blades']),
                           promotes=['*'])
        self.add_subsystem('advance_ratio', AdvanceRatioComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem('tip_mach', TipMachComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('lock_number', LockNumberComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem('thrust_coef', ThrustCoefComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem('tpp_angle',
                           TppAngleComp(num_nodes=nn,
                                        form=self.options['tpp_form']),
                           promotes=['*'])
        self.add_subsystem('induced_velocity',
                           InducedVelocityComp(num_nodes=nn,
                                               form=self.options['vi_form']),
                           promotes=['*'])
        self.add_subsystem('inflow', InflowComp(num_nodes=nn), promotes=['*'])
