"""Compare the two routes to the compressibility torque correction.

Route A -- integrate p. 183 directly:

    dCQ/sigma_comp / M_OmegaR^3
        = (K1 / 2 pi) int int (x + mu sin psi)^2 x
                              [(x + mu sin psi) - m_dr(x)]^3 dx dpsi

    over the region where the local Mach exceeds the local drag rise Mach,
    with the spanwise m_dr(x) = M_dr(x)/M_OmegaR of p. 184:

        x <  1 - c/R :  m_dr = (M_dr2/M_dr3) (M_dr3/M_OmegaR)
        x >  1 - c/R :  m_dr = [1 - (1 - M_dr2/M_dr3)(1-x)/(c/R)]
                               (M_dr3/M_OmegaR)

    and M_dr3/M_OmegaR = (1 + mu) / M_ratio, since the abscissa of the figure
    is M_ratio = M_190/M_dr3 = (1+mu) M_OmegaR / M_dr3.

Route B -- digitise Figure 3.43.

Constants behind Figure 3.43: K1 = 12.5 and M_dr2 = 0.74 from Figure 3.42,
M_dr2/M_dr3 = 0.92 from p. 184, c/R = 2/30 for the example helicopter.
"""

import numpy as np

K1 = 12.5
M_DR2_OVER_M_DR3 = 0.92
C_OVER_R = 2.0 / 30.0


def m_dr_span(x, M_dr3_over_M_OR, ratio=M_DR2_OVER_M_DR3, c_R=C_OVER_R):
    """Local drag rise Mach divided by the tip speed Mach, p. 184."""
    inboard = ratio * M_dr3_over_M_OR
    tip = (1.0 - (1.0 - ratio) * (1.0 - x) / c_R) * M_dr3_over_M_OR
    return np.where(x < 1.0 - c_R, inboard, tip)


def integral(mu, M_ratio, n_x=2001, n_psi=2001, c_R=C_OVER_R):
    """Route A. Returns dCQ/sigma_comp / M_OmegaR^3."""
    M_dr3_over_M_OR = (1.0 + mu) / M_ratio

    x = np.linspace(0.0, 1.0, n_x)
    psi = np.linspace(0.0, 2.0 * np.pi, n_psi)
    X, PSI = np.meshgrid(x, psi, indexing='ij')

    M_local = X + mu * np.sin(PSI)
    excess = M_local - m_dr_span(X, M_dr3_over_M_OR, c_R=c_R)
    excess = np.where(excess > 0.0, excess, 0.0)

    integrand = M_local ** 2 * X * excess ** 3
    return K1 / (2.0 * np.pi) * np.trapezoid(
        np.trapezoid(integrand, x, axis=0), psi)


# Figure 3.43 read off the plot, p. 185. Abscissa at which each mu curve
# crosses the given ordinate; digitised visually, so a few percent at best.
FIGURE_343 = {
    0.0: [(1.00, 0.0), (1.10, 0.0009), (1.15, 0.0022), (1.20, 0.0045),
          (1.25, 0.0080), (1.30, 0.0130), (1.34, 0.0200)],
    0.1: [(1.00, 0.0), (1.10, 0.0012), (1.15, 0.0029), (1.20, 0.0058),
          (1.25, 0.0102), (1.29, 0.0160), (1.32, 0.0200)],
    0.2: [(1.00, 0.0), (1.10, 0.0018), (1.15, 0.0042), (1.20, 0.0082),
          (1.24, 0.0135), (1.27, 0.0200)],
    0.3: [(1.00, 0.0), (1.10, 0.0025), (1.143, 0.0043), (1.15, 0.0059),
          (1.20, 0.0115), (1.23, 0.0170), (1.25, 0.0200)],
    0.4: [(1.00, 0.0), (1.10, 0.0034), (1.15, 0.0080), (1.18, 0.0130),
          (1.21, 0.0200)],
    0.5: [(1.00, 0.0), (1.10, 0.0045), (1.15, 0.0105), (1.17, 0.0150),
          (1.19, 0.0200)],
}
