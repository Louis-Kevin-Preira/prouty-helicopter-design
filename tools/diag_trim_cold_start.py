"""
Diagnostic D -- cold start of the Chapter 3 trim (level and climb).

Runs TrimConditionsGroup from a fresh state with the book's example helicopter,
with the closed-form rotor, for both settings of solve_subsystems, and prints
whether it converges, in how many Newton iterations, and where it lands.

    python tools/diag_trim_cold_start.py
"""

import warnings

import numpy as np
import openmdao.api as om

from prouty.forward_flight import TrimConditionsGroup

warnings.filterwarnings('ignore')
KT = 1.68781
BASE = dict(V_tip=650.0, rho=0.002377, A_b=240.0, sigma=0.084883, theta_1=np.deg2rad(-10.0),
            a=6.0, gamma=8.05033, R=30.0, GW=20000.0, i_s=0.0, a1s=0.0, l_T_R=1.23,
            cd_bar=0.0100, delta_3=np.deg2rad(-30.0))


def cold(mode, V_kt, R_C=None, solve_subsystems=False, alpha0_deg=-6.0, CT0=0.085):
    p = om.Problem()
    p.model.add_subsystem('trim', TrimConditionsGroup(mode=mode), promotes=['*'])
    p.setup()
    newton = p.model.trim.nonlinear_solver
    newton.options['solve_subsystems'] = solve_subsystems
    newton.options['iprint'] = -1
    for name, val in BASE.items():
        p.set_val(name, val)
    p.set_val('mu', V_kt * KT / BASE['V_tip'])
    if R_C is not None:
        p.set_val('R_C', R_C, units='ft/min')
    p.set_val('alpha_F', np.deg2rad(alpha0_deg))
    p.set_val('CT_sigma', CT0)
    converged = True
    try:
        p.run_model()
    except om.AnalysisError:
        converged = False
    return dict(converged=converged, iterations=newton._iter_count,
                alpha_F_deg=float(np.degrees(p.get_val('alpha_F')[0])),
                hp_M=float(p.get_val('hp_M')[0]),
                T_GW=float(p.get_val('T')[0] / BASE['GW']))


if __name__ == '__main__':
    for sub in (False, True):
        print(f'--- solve_subsystems = {sub}')
        for V in (80, 140, 175):
            r = cold('level', V, solve_subsystems=sub)
            print(f"level {V:3d} kt           {r}")
        for rc in (0.0, 500.0, 1500.0, 3000.0):
            r = cold('climb', 80, rc, solve_subsystems=sub)
            print(f"climb  80 kt {rc:6.0f} fpm {r}")
