"""
Figure 2.14, p. 114: rate of descent and collective pitch in vertical autorotation
against tip speed, example helicopter (20,000 lb, sea level, theta_1 = -10 deg).

G7 with the Chapter 6 NACA 0012 drag, and with the drag p. 112 implies (c_d = 0.00865),
against the digitized figure. Writes figure_2_14.png next to this script.
"""

import pathlib
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import openmdao.api as om

from prouty.vertical import VerticalAutorotationGroup

HERE = pathlib.Path(__file__).parent
BOOK = np.array([[450, 4550, 16.33], [490, 4533, 15.00], [530, 4524, 13.78],
                 [550, 4524, 13.25], [590, 4529, 12.28], [630, 4547, 11.48],
                 [650, 4559, 11.14], [690, 4592, 10.55]])


def sweep(V_tip, drag):
    nn = V_tip.size
    p = om.Problem()
    p.model.add_subsystem('g7', VerticalAutorotationGroup(num_nodes=1, drag=drag), promotes=['*'])
    p.setup()
    p.set_val('T', 20000.0, units='lbf')
    p.set_val('A', 2827.4, units='ft**2')
    p.set_val('a', 5.73)
    p.set_val('theta_1', -10.0, units='deg')
    if drag == 'input':
        p.set_val('cd_bar', 0.00865)
    rd, th = np.zeros(nn), np.zeros(nn)
    for i, v in enumerate(V_tip):
        p.set_val('V_tip', v, units='ft/s')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p.run_model()
        rd[i] = p.get_val('V_D_auto', units='ft/min')[0]
        th[i] = p.get_val('theta_0_auto', units='deg')[0]
    return rd, th


if __name__ == '__main__':
    V = np.linspace(440.0, 720.0, 57)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    for drag, style in (('airfoil', 'b-'), ('input', 'g--')):
        rd, th = sweep(V, drag)
        label = 'G7, Chapter 6 drag' if drag == 'airfoil' else 'G7, c_d = 0.00865'
        ax1.plot(V, rd, style, label=label)
        ax2.plot(V, th, style, label=label)
    ax1.plot(BOOK[:, 0], BOOK[:, 1], 'ko', mfc='none', label='Figure 2.14')
    ax2.plot(BOOK[:, 0], BOOK[:, 2], 'ko', mfc='none', label='Figure 2.14')
    ax1.set_ylabel('rate of descent, ft/min')
    ax2.set(xlabel='tip speed, ft/s', ylabel='collective pitch, deg')
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / 'figure_2_14.png', dpi=130)
