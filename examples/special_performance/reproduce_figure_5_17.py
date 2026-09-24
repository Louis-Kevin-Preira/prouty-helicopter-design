"""
Figure 5.17, p. 370: return-to-target maneuver, 115 kt, 20,000 lb, sea level.

Writes figure_5_17_model.png: ground track of G6. The powered turn uses the
fixed fallback with n_p = 1.89, the value PoweredTurnGroup finds with the
Chapter 4 hover chain (test_g6_return_to_target.py), to keep this script
free of the Chapter 4 hover inputs.
"""

import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import openmdao.api as om

from prouty.special_performance import ReturnToTargetChainGroup
from prouty.special_performance.book_figures import FIG_5_17, KT

HERE = pathlib.Path(__file__).parent
V_GRID = np.array([26, 30, 35, 40, 50, 60, 70, 80, 90, 100, 115, 125.0]) * KT

p = om.Problem(reports=False)
p.model.add_subsystem('g', ReturnToTargetChainGroup(V_grid=V_GRID, powered_turn='fixed'),
                      promotes=['*'])
p.setup()
p.set_val('sigma', 0.0849)
p.set_val('cd_bar', 0.01 * np.ones(len(V_GRID)))
p.set_val('R', 30.0)
p.set_val('V_0', 115.0, units='kn')
p.set_val('n_p', 1.89)
p.run_model()

x, y = p.get_val('x'), p.get_val('y')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(x, y, 'k', label=f'G6 turn ({p.get_val("t1")[0]:.1f} s)')
ax.plot([x[-1], 0.0], [y[-1], 0.0], 'k--',
        label=f'return ({p.get_val("t2")[0]:.1f} s), total {p.get_val("t_total")[0]:.1f} s')
ax.set(title=f'Figure 5.17: total {FIG_5_17["t_total"]} s, loop about '
             f'{FIG_5_17["x_max"]:.0f} x {FIG_5_17["y_max"]:.0f} ft',
       xlabel='x distance, ft', ylabel='y distance, ft', aspect='equal')
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / 'figure_5_17_model.png', dpi=130)
