"""
Figure 5.10, p. 359: height-velocity diagrams of the example helicopter,
sea level, 20,000 lb, FAA and military time delays.

Writes figure_5_10_model.png: boundaries of G2d (V_min from the energy method
of p. 357) with the noses and tops read on the figure.
"""

import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import openmdao.api as om

from prouty.special_performance import HeightVelocityGroup
from prouty.special_performance.book_figures import FIG_5_10

HERE = pathlib.Path(__file__).parent
EXAMPLE = dict(GW=20000.0, f=20.0, A=np.pi * 30.0 ** 2, A_b=240.0, V_tip=650.0, C_d=0.01,
               e=0.8, CW_sigma=0.083, V_LG=6.0, P_IGE=1500.0)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for ax, td in zip(axes, ('faa', 'military')):
    p = om.Problem(reports=False)
    p.model.add_subsystem('g', HeightVelocityGroup(time_delay=td, num_points=41), promotes=['*'])
    p.setup()
    for k, v in EXAMPLE.items():
        p.set_val(k, v)
    p.run_model()
    ax.plot(p.get_val('V_up'), p.get_val('h_up'), 'k')
    ax.plot(p.get_val('V_lo'), p.get_val('h_lo_branch'), 'k', label='G2d')
    v_cr, h_hi = FIG_5_10[td]
    ax.plot([v_cr, 0.0], [p.get_val('h_CR')[0], h_hi], 'o', mfc='none', label='Figure 5.10 nose, top')
    ax.set(title=f'sea level, {td} time delay', xlabel='velocity, kt', xlim=(0, 130))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel('height, ft')
fig.tight_layout()
fig.savefig(HERE / 'figure_5_10_model.png', dpi=130)
