"""
Validation of AirfoilHoverGroup against Prouty, Chapter 6, Figure 6.43 (p. 427)
and the anchor values quoted in the text (p. 428-433).
"""

import pathlib

import numpy as np
import openmdao.api as om
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prouty.airfoil.airfoil_hover_group import AirfoilHoverGroup

MACHS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85])
ALPHA = np.linspace(0.0, 18.0, 361)


def run(M_grid, a_grid):
    """Evaluate the group on a flattened (M, alpha) mesh."""
    MM, AA = np.meshgrid(M_grid, a_grid, indexing='ij')
    nn = MM.size

    p = om.Problem()
    p.model.add_subsystem('af', AirfoilHoverGroup(num_nodes=nn), promotes=['*'])
    p.setup()
    p.set_val('M', MM.ravel())
    p.set_val('alpha', AA.ravel())
    p.run_model()

    shape = MM.shape
    return (p.get_val('cl').reshape(shape), p.get_val('cd').reshape(shape),
            p.get_val('alpha_L').reshape(shape), p.get_val('alpha_D').reshape(shape))


cl, cd, aL, aD = run(MACHS, ALPHA)

# --- anchor 1: documented coefficient values, p. 429-430 -------------------
print('--- Coefficients vs. text, p. 429-430 ---')
p1 = om.Problem()
p1.model.add_subsystem('af', AirfoilHoverGroup(num_nodes=3), promotes=['*'])
p1.setup()
p1.set_val('M', [0.2, 0.5, 0.7])
p1.set_val('alpha', [0.0, 0.0, 0.0])
p1.run_model()
book_K1 = [0.0233, 0.0257, 0.0497]
book_K2 = [1.95, 1.57, 1.38]
print(f"{'M':>5}{'K1 model':>10}{'K1 book':>9}{'K2 model':>10}{'K2 book':>9}")
for i, m in enumerate([0.2, 0.5, 0.7]):
    print(f"{m:5.1f}{p1.get_val('K1')[i]:10.4f}{book_K1[i]:9.4f}"
          f"{p1.get_val('K2')[i]:10.2f}{book_K2[i]:9.2f}")

# --- anchor 2: cl_max and stall behaviour vs. Figure 6.43 -----------------
print('\n--- cl_max and cd at zero angle ---')
print(f"{'M':>6}{'alpha_L':>9}{'cl_max':>9}{'a@clmax':>9}{'alpha_D':>9}{'cd(0)':>9}")
for i, m in enumerate(MACHS):
    k = np.argmax(cl[i])
    print(f"{m:6.2f}{aL[i, 0]:9.2f}{cl[i, k]:9.3f}{ALPHA[k]:9.2f}"
          f"{aD[i, 0]:9.2f}{cd[i, 0]:9.5f}")

# --- anchor 3: derivative check on the assembled group -------------------
print('\n--- check_totals on the group ---')
p2 = om.Problem()
p2.model.add_subsystem('af', AirfoilHoverGroup(num_nodes=4), promotes=['*'])
p2.model.add_design_var('alpha')
p2.model.add_design_var('M')
p2.model.add_objective('cl', index=0)
p2.model.add_constraint('cd', lower=0.0)
p2.setup()
p2.set_val('M', [0.3, 0.5, 0.7, 0.8])
p2.set_val('alpha', [5.0, 9.0, 6.0, 3.0])
p2.run_model()
p2.check_totals(method='fd', compact_print=True)

# --- figure ---------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for i, m in enumerate(MACHS):
    ax[0].plot(ALPHA, cl[i], lw=1.2, label=f'{m:.2f}')
    ax[1].plot(ALPHA, cd[i], lw=1.2, label=f'{m:.2f}')

ax[0].set(xlabel='Angle of attack, deg', ylabel='$c_l$', xlim=(0, 18), ylim=(0, 1.5))
ax[1].set(xlabel='Angle of attack, deg', ylabel='$c_d$', xlim=(0, 18), ylim=(0, 0.16))
for a in ax:
    a.grid(alpha=0.3)
    a.legend(title='M', fontsize=7, ncol=2)
fig.suptitle('AirfoilHoverGroup - generated NACA 0012 characteristics '
             '(compare Prouty Fig. 6.43, p. 427)')
fig.tight_layout()
output = pathlib.Path(__file__).with_name('fig643_check.png')
fig.savefig(output, dpi=140)
print(f'\nfigure written to {output}')
