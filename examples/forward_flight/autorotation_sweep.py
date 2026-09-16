"""Diagnostic: rebuild the p. 196 sweep method for autorotation.

Not a validation script -- the fuselage aerodynamics are reconstructed from
the three columns of Table 3.3 itself, so this can only test SELF-consistency
of the closed-form equations, not agreement with Appendix A.
"""

import numpy as np
import openmdao.api as om

from prouty.forward_flight.collective_pitch_comp import CollectivePitchComp
from prouty.forward_flight.coning_comp import ConingComp
from prouty.forward_flight.h_force_coef_comp import HForceCoefComp
from prouty.forward_flight.torque_coef_comp import TorqueCoefComp

Q, P = 241028.0, 284851.0            # rho A_b (OmegaR)^2 and .../550
SIGMA, MU, A, TH1 = 0.084883, 0.3, 6.0, np.deg2rad(-10.0)
GW, V, DYN_Q, L_T_R, V_TIP = 20000.0, 195.0, 45.19, 1.23, 650.0
GAMMA_LOCK = 8.05033

# fuselage fitted on the three Table 3.3 columns
C_F = np.array([0.01638, 0.00706, 19.437])
C_L = np.array([1.9483, -4.4685])


def _rotor():
    p = om.Problem()
    m = p.model
    m.add_subsystem('pitch', CollectivePitchComp(), promotes=['*'])
    m.add_subsystem('cone', ConingComp(), promotes=['*'])
    m.add_subsystem('torque', TorqueCoefComp(), promotes=['*'])
    m.add_subsystem('hforce', HForceCoefComp(form='direct'), promotes=['*'])
    p.setup()
    p.set_val('mu', MU)
    p.set_val('a', A)
    p.set_val('theta_1', TH1)
    p.set_val('gamma', GAMMA_LOCK)
    p.set_val('R', 30.0)
    p.set_val('V_tip', V_TIP)
    p.set_val('a1s', 0.0)
    return p


ROTOR = _rotor()


def rotor(CT, lam, vi, cd):
    ROTOR.set_val('CT_sigma', CT)
    ROTOR.set_val('lambda_p', lam)
    ROTOR.set_val('vi_OR', vi)
    ROTOR.set_val('cd_bar', cd)
    ROTOR.run_model()
    return (ROTOR.get_val('theta_0')[0], ROTOR.get_val('a0')[0],
            ROTOR.get_val('CQ_sigma')[0], ROTOR.get_val('CH_sigma')[0])


def trim(gamma_deg, cd, n=80):
    """Iteration of p. 193, with the climb angle term of p. 194."""
    g = np.deg2rad(gamma_deg)
    aF, H_M, H_T = 0.0, 0.0, 12.0

    for _ in range(n):
        f, LFq = np.polyval(C_F, aF), np.polyval(C_L, aF)
        D_F, L_F = DYN_Q * f, DYN_Q * LFq
        num = D_F + H_M + H_T + GW * np.sin(g)
        aTPP = -np.arctan(num / (GW - L_F))
        T = np.hypot(GW - L_F, num)

        CT = T / Q
        vi = SIGMA * CT / (2 * MU)
        lam = MU * aTPP - vi
        th0, a0, cq, ch = rotor(CT, lam, vi, cd)
        H_M = ch * Q
        aF = np.degrees(lam / MU)

    return dict(aF=aF, L_F=L_F, D_F=D_F, T=T, aTPP=np.degrees(aTPP), lam=lam,
                a0=np.degrees(a0), th0=np.degrees(th0), H_M=H_M, hp=cq * P,
                T_T=550.0 * cq * P / (V_TIP * L_T_R))


def gamma_of(rc):
    return np.degrees(np.arcsin(rc / (60.0 * V)))
