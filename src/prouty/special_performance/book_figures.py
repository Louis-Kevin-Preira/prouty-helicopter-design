"""
Chapter 5 validation data: figures digitized for validation only, and the
example helicopter values the anchors use.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, Figures 5.5-5.17; Appendix A (example helicopter).
Figures used by the models themselves (5.2, 5.8, 5.9) live in their components.
"""

import numpy as np

KT = 1.6878   # ft/s per knot

# Figure 5.5 (p. 351): rate of descent in autorotation, V [kt] -> R/D [ft/min]
FIG_5_5 = {70: 1626, 80: 1605, 90: 1637, 100: 1711, 120: 2047, 140: 2358}
# Figure 5.6 (p. 353), failure at 160 kt: V_1 [kt] -> (Delta_h, Delta_d) [ft]
FIG_5_6 = {60: (473, None), 70: (453, 2007), 80: (430, 2178), 90: (397, 2236),
           100: (353, 2018), 120: (249, 1457), 140: (125, 785)}
# Figure 5.10 (p. 359), sea level, 20,000 lb: time delay -> (nose V_CR [kt], top h_hi [ft])
FIG_5_10 = {'faa': (80.0, 1330.0), 'military': (101.0, 1390.0)}
# Figure 5.12 (p. 362): C_W/sigma -> {flare angle [deg]: mu_auto}
FIG_5_12 = {0.05: {10: .2245, 15: .156, 20: .1193, 30: .0837, 45: .0663},
            0.10: {15: .205, 20: .154, 30: .1094, 45: .0901}}
# Figures 5.14 and 5.15 (pp. 365-366): V [kt] -> acceleration / deceleration [ft/s^2]
FIG_5_14 = {40: 22.9, 60: 18.1, 80: 14.0, 100: 10.2, 120: 6.6, 140: 3.3, 160: 0.0}
FIG_5_15 = {40: 17.8, 60: 12.4, 80: 8.7, 100: 6.4, 120: 5.7, 140: 5.7, 160: 6.4}
# Figure 5.16 (p. 369), 28,000 lb: obstacle [ft] -> (optimum V_rot [kt], distance [ft])
FIG_5_16 = {50: (26.0, 325.0), 250: (30.0, 1150.0), 500: (38.0, 1850.0)}
# Figure 5.17 (p. 370), 115 kt start, 20,000 lb
FIG_5_17 = dict(t_total=23.0, t_turn=12.5, V_min_kt=27.0, x_max=1010.0, y_max=965.0)

# Example helicopter (Appendix A), as the Chapter 3 trim tests set it
EXAMPLE_ROTOR = dict(V_tip=650.0, rho=0.002377, A_b=240.0, sigma=0.084883,
                     theta_1=np.deg2rad(-10.0), a=6.0, gamma=8.05033, R=30.0, GW=20000.0,
                     i_s=0.0, a1s=0.0, l_T_R=1.23, cd_bar=0.0100, delta_3=np.deg2rad(-30.0))
# Chapter 4 values the anchors take as inputs (see docs/validation_special_performance.md)
EXAMPLE_28K = dict(P_hover=4209.0, P_avail=4077.0, T_max_IGE=31574.0, V_max_kt=214.4)   # TakeoffCapabilityGroup
# Chapter 4 level flight power inputs of the example (design ratings, accessories)
EXAMPLE_POWER = dict(V_son=1116.0, P_design_nose=2000.0, P_design_main=4000.0,
                     P_design_tail=750.0, load_elec=2200.0, flow_hyd=1.3, p_hyd=3000.0)
