"""
Figure 2.13, p. 113, and the G0 induced velocity over the whole axial range.

Writes two figures next to this script:
  figure_2_13_model.png    v1_bar(V_D_bar) against both momentum branches and the chart points
  figure_2_13_overlay.png  digitized points and model drawn on the book scan (optional:
                           pass the p. 113 page image as first argument)

Scan calibration: column = 493.9 + 142.8 (V_D_bar - v1_bar), row = 212.9 + 160.4 V_D_bar,
for the 924 x 1316 page image of the project scans.
"""

import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from prouty.vertical import FIGURE_2_13, axial_inflow

HERE = pathlib.Path(__file__).parent


def v1_bar(x, theta_1_deg):
    return axial_inflow(x, np.radians(theta_1_deg))[0]


def plot_model():
    x = np.linspace(-2.0, 5.0, 1400)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, x / 2 + np.sqrt(x ** 2 / 4 + 1), 'k--', lw=0.8, label='momentum, climb / low descent')
    xw = x[x >= 2.0]
    ax.plot(xw, xw / 2 - np.sqrt(xw ** 2 / 4 - 1), 'k:', label='momentum, windmill brake')
    for th, c in ((0.0, 'tab:red'), (-10.0, 'tab:green'), (-12.0, 'tab:blue')):
        ax.plot(x, v1_bar(x, th), c, label=f'G0, theta_1 = {th:g} deg')
        if th in FIGURE_2_13:
            xt, yt = FIGURE_2_13[th]
            ax.plot(xt, xt - yt, 'o', ms=3, mfc='none', color=c)
    for edge in (0.0, 0.25, 2.6, 3.6):
        ax.axvline(edge, color='0.8', lw=0.6)
    ax.set(xlabel='V_D / v_1hov (descent > 0)', ylabel='v_1 / v_1hov', ylim=(0, 3.5))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / 'figure_2_13_model.png', dpi=130)


def plot_overlay(scan):
    from PIL import Image, ImageDraw
    im = Image.open(scan).convert('RGB')
    draw = ImageDraw.Draw(im)
    px = lambda x, y: (493.9 + 142.8 * y, 212.9 + 160.4 * x)
    x = np.linspace(0.0, 2.75, 600)
    for th, col in ((0.0, (220, 0, 0)), (-12.0, (0, 0, 220))):
        draw.line([px(a, a - b) for a, b in zip(x, v1_bar(x, th))], fill=col, width=1)
        for a, b in zip(*FIGURE_2_13[th]):
            u, v = px(a, b)
            draw.ellipse((u - 2, v - 2, u + 2, v + 2), outline=col)
    im.crop((250, 150, 800, 700)).resize((1100, 1100)).save(HERE / 'figure_2_13_overlay.png')


if __name__ == '__main__':
    plot_model()
    if len(sys.argv) > 1:
        plot_overlay(sys.argv[1])
