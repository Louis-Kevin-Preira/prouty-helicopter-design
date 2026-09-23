"""
Diagnostic A -- where does the forward flight power go?

One flight point, the book's example helicopter at 20,000 lb, sea level,
standard day, 140 kt, trimmed by the Chapter 3 closed-form and numerical
rotors. The main rotor power of each trim is split by the energy method into
induced, profile and parasite parts, and set against the main rotor power the
book's own curves imply (Figure 4.48 read at 140 kt, through the drive
equation of p. 311).

    python tools/diag_power_breakdown.py
"""

import time
import warnings

import numpy as np
import openmdao.api as om

from prouty.forward_flight import TrimConditionsGroup

warnings.filterwarnings('ignore')
KT, HP = 1.68781, 550.0
BASE = dict(V_tip=650.0, rho=0.002377, A_b=240.0, sigma=0.084883, theta_1=np.deg2rad(-10.0),
            a=6.0, gamma=8.05033, R=30.0, GW=20000.0, i_s=0.0, a1s=0.0, l_T_R=1.23,
            cd_bar=0.0100, delta_3=np.deg2rad(-30.0))
NUMERICAL = dict(c_R=2.0 / 30.0, B=4.0, x_0=0.15)
V_KT = 140.0
BOOK_ENGINE_HP = 2337.0          # Figure 4.48, level flight, 20,000 lb, sea level


def trim(rotor):
    p = om.Problem()
    p.model.add_subsystem('trim', TrimConditionsGroup(mode='level', rotor=rotor),
                          promotes=['*'])
    p.setup()
    p.model.trim.nonlinear_solver.options['solve_subsystems'] = True
    p.model.trim.nonlinear_solver.options['iprint'] = -1
    for name, val in BASE.items():
        try:
            p.set_val(name, val)
        except KeyError:
            pass                          # the numerical rotor has no mean c_d
    if rotor == 'numerical':
        for name, val in NUMERICAL.items():
            try:
                p.set_val(name, val)
            except KeyError:
                pass
    p.set_val('mu', V_KT * KT / BASE['V_tip'])
    p.set_val('alpha_F', np.deg2rad(-7.0))
    p.set_val('CT_sigma', 0.088)
    t0 = time.time()
    p.run_model()
    return p, time.time() - t0


def breakdown(p):
    V = V_KT * KT
    rho, A_b, V_tip = BASE['rho'], BASE['A_b'], BASE['V_tip']
    mu = V / V_tip
    T = p.get_val('T')[0]
    v_i = p.get_val('vi_OR')[0] * V_tip
    q = p.get_val('q')[0]
    f = p.get_val('f')[0]
    hp_M = p.get_val('hp_M')[0]
    induced = T * v_i / HP
    parasite = q * f * V / HP
    profile = BASE['cd_bar'] / 8.0 * (1.0 + 4.65 * mu ** 2) * rho * A_b * V_tip ** 3 / HP
    return dict(mu=mu, T_GW=T / BASE['GW'], f=f, hp_M=hp_M, induced=induced,
                parasite=parasite, profile_cd010=profile,
                rest=hp_M - induced - parasite, hp_T=p.get_val('hp_T')[0])


if __name__ == '__main__':
    rows = {}
    for rotor in ('closed_form', 'numerical'):
        p, dt = trim(rotor)
        rows[rotor] = breakdown(p)
        rows[rotor]['seconds'] = dt

    hp_T = rows['closed_form']['hp_T']
    book_hp_M = (BOOK_ENGINE_HP - 56.0 - 1.0075 * hp_T) / 1.0112          # p. 311
    ref = rows['closed_form']
    book_rest = book_hp_M - ref['induced'] - ref['parasite']
    unit_profile = ref['profile_cd010'] / 0.0100

    print(f"point: 20,000 lb, sea level, {V_KT:.0f} kt, mu = {ref['mu']:.3f}")
    print(f"{'':28s}{'closed form':>14s}{'numerical':>14s}{'book':>12s}")
    for key, label in (('hp_M', 'main rotor hp'), ('induced', '  induced T v_i'),
                       ('parasite', '  parasite q f V'), ('rest', '  profile (remainder)')):
        book = {'hp_M': book_hp_M, 'induced': ref['induced'], 'parasite': ref['parasite'],
                'rest': book_rest}[key]
        print(f"{label:28s}{rows['closed_form'][key]:14.0f}{rows['numerical'][key]:14.0f}"
              f"{book:12.0f}")
    print(f"{'profile at c_d = 0.010':28s}{ref['profile_cd010']:14.0f}")
    print(f"{'equivalent mean c_d':28s}{rows['closed_form']['rest'] / unit_profile:14.4f}"
          f"{rows['numerical']['rest'] / unit_profile:14.4f}{book_rest / unit_profile:12.4f}")
    print(f"{'or equivalent f, ft^2':28s}{'':14s}{'':14s}"
          f"{ref['f'] * (book_hp_M - ref['induced'] - ref['profile_cd010']) / ref['parasite']:12.1f}")
    print(f"trim f = {ref['f']:.1f} ft^2, T/GW = {ref['T_GW']:.3f}, "
          f"time: closed {rows['closed_form']['seconds']:.1f} s, "
          f"numerical {rows['numerical']['seconds']:.1f} s")


def speed_sweep():
    """Equivalent mean c_d implied by Figure 4.48 at each speed, closed-form trim."""
    fig_4_48 = {60: 1131, 80: 1059, 100: 1189, 120: 1577, 140: 2337, 160: 3182}
    global V_KT
    print('\nspeed  mu     trim hp_M  book hp_M  gap   c_d trim  c_d book')
    for V, engine in fig_4_48.items():
        V_KT = float(V)
        p, _ = trim('closed_form')
        b = breakdown(p)
        book = (engine - 56.0 - 1.0075 * b['hp_T']) / 1.0112
        unit = b['profile_cd010'] / 0.0100
        print(f"{V:5d}  {b['mu']:.3f}  {b['hp_M']:9.0f}  {book:9.0f}  "
              f"{book / b['hp_M'] - 1:+5.0%}  {b['rest'] / unit:8.4f}  "
              f"{(book - b['induced'] - b['parasite']) / unit:8.4f}")


if __name__ == '__main__':
    speed_sweep()
