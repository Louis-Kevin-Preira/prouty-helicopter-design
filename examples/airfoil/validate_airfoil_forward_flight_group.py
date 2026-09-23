"""
Validation of AirfoilForwardFlightGroup against Prouty, Chapter 6,
Figure 6.47 (p. 434) and the segment tables of p. 433-434.
"""

import pathlib

import numpy as np
import openmdao.api as om
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prouty.airfoil.airfoil_forward_flight_group import AirfoilForwardFlightGroup

M_REF = 0.30                       # Fig. 6.47 high-alpha data is low Mach, p. 433
ALPHA = np.linspace(0.0, 360.0, 1441)


def run(alpha, M):
    nn = alpha.size
    p = om.Problem()
    p.model.add_subsystem('af', AirfoilForwardFlightGroup(num_nodes=nn),
                          promotes=['*'])
    p.setup()
    p.set_val('alpha_raw', alpha)
    p.set_val('M', np.full(nn, M))
    p.run_model()
    return p


p = run(ALPHA, M_REF)
cl, cd = p.get_val('cl'), p.get_val('cd')

# --- anchor 1: segment values quoted p. 433-434 ---------------------------
print('--- Segment anchors, M = 0.30 ---')
checks = [(45.0, 1.15, 1.03), (90.0, 0.0, 2.05), (167.0, -0.70, None),
          (180.0, 0.0, 0.01), (193.0, 0.70, None), (270.0, 0.0, 2.05),
          (315.0, -1.15, 1.03)]
print(f"{'alpha':>7}{'cl':>9}{'cl exp':>9}{'cd':>9}{'cd exp':>9}")
for a, cl_e, cd_e in checks:
    i = np.argmin(np.abs(ALPHA - a))
    cd_s = f'{cd_e:9.2f}' if cd_e is not None else f"{'-':>9}"
    print(f"{a:7.1f}{cl[i]:9.4f}{cl_e:9.2f}{cd[i]:9.4f}{cd_s}")

# --- anchor 2: antisymmetry of cl, symmetry of cd ------------------------
mirror = np.interp((360.0 - ALPHA) % 360.0, ALPHA, np.arange(ALPHA.size))
idx = mirror.astype(int)
print(f'\nmax |cl(a) + cl(360-a)| = {np.abs(cl + cl[idx]).max():.2e}')
print(f'max |cd(a) - cd(360-a)| = {np.abs(cd - cd[idx]).max():.2e}')

# --- anchor 3: extrema ---------------------------------------------------
print(f'\ncl range : [{cl.min():.3f}, {cl.max():.3f}]')
print(f'cd range : [{cd.min():.4f}, {cd.max():.4f}]')

# --- anchor 4: totals on the assembled group -----------------------------
print('\n--- check_totals on the group ---')
p2 = run(np.array([10.0, 45.0, 150.0, 250.0, 350.0]), 0.4)
p2.model.add_design_var('alpha_raw')
p2.model.add_design_var('M')
p2.model.add_constraint('cl', lower=-3.0)
p2.model.add_constraint('cd', lower=0.0)
p2.setup()
p2.set_val('alpha_raw', np.array([10.0, 45.0, 150.0, 250.0, 350.0]))
p2.set_val('M', np.full(5, 0.4))
p2.run_model()
p2.check_totals(method='fd', compact_print=True)

# --- figure --------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax[0].plot(ALPHA, cl, lw=1.3, color='tab:blue')
ax[0].axhline(0, color='k', lw=0.5)
ax[0].set(ylabel='$c_l$', ylim=(-1.5, 1.5))
ax[1].plot(ALPHA, cd, lw=1.3, color='tab:red')
ax[1].set(xlabel='Angle of attack, deg', ylabel='$c_d$',
          xlim=(0, 360), ylim=(0, 2.5))
for a in ax:
    a.grid(alpha=0.3)
    for b in (20, 161, 173, 187, 199, 340):
        a.axvline(b, color='gray', ls=':', lw=0.7)
    a.set_xticks(np.arange(0, 361, 40))
fig.suptitle('AirfoilForwardFlightGroup at M = 0.30 '
             '(compare Prouty Fig. 6.47, p. 434)')
fig.tight_layout()
output = pathlib.Path(__file__).with_name('fig647_check.png')
fig.savefig(output, dpi=140)
print(f'\nfigure written to {output}')
