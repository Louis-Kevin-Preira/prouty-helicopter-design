"""Control response in hover: transfer functions and step time histories.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", pp. 605-609. The transfer function
in determinant form is p. 606, the Heaviside expansion p. 607, and the closed
form and its plot pp. 608-609.
"""

import numpy as np
import openmdao.api as om


def heaviside_step_response(numerator, denominator, times):
    """Time history of a transfer function following a unit step.

    p. 607's expansion::

        response(t) = N(0)/D(0) + sum_r N(s_r) exp(s_r t) / (s_r D'(s_r))

    over the roots of the denominator. Both polynomials are given in the
    ascending-power convention the rest of the package uses. Returns the real
    part, the imaginary parts cancelling in conjugate pairs.

    p. 607 offers it as a way "to make simple checks of computer results", and
    that is what it is here: root-finding and complex exponentials, so a
    post-processing function rather than a component.

    Repeated roots break the expansion, as they do in the book. The hover
    cubic has three distinct ones.
    """
    numerator = np.asarray(numerator, dtype=float)[::-1]
    denominator = np.asarray(denominator, dtype=float)[::-1]
    roots = np.roots(denominator)
    derivative = np.polyder(denominator)
    times = np.asarray(times, dtype=float)

    steady = np.polyval(numerator, 0.0) / np.polyval(denominator, 0.0)
    transient = sum(np.polyval(numerator, root) * np.exp(root * times)
                    / (root * np.polyval(derivative, root))
                    for root in roots)
    return np.real(steady + transient)


class HoverControlColumnComp(om.ExplicitComponent):
    """The right-hand side of p. 606's two equations, as a control column.

    p. 605 describes the construction: "the numerator is identical except that
    the control column on the right-hand side of the equations of motion is
    substituted for the column representing the degree of freedom of
    interest". p. 606 writes those equations for longitudinal cyclic::

        -(G.W./g) x_ddot + (dX/dxdot) xdot + (dX/dq) q - G.W. Theta
                                                        = -(dX/dB1) B1
        (dM/dxdot) xdot - I_yy q_dot + (dM/dq) q         = -(dM/dB1) B1

    so the column is ``[-dX/dB1, -dM/dB1]``, both entries constant in ``s``.
    Feed it and ``HoverLongTwoDofMatrixComp``'s matrix to
    ``TransferFunctionGroup(n=2, degree=2, column=1)`` for ``Theta/B1``.

    Why the numerator comes out as one term
    ---------------------------------------
    Substituting the column into the attitude column and expanding gives

        (dM/dB1)(m s - dX/dxdot) + (dX/dB1)(dM/dxdot)

    and p. 606 notes that "some terms in the numerator cancel themselves out
    just as they did in the derivation of the characteristic equation". They
    do, and for the same reason: Table 9.2 builds ``dM/dB1`` and ``dM/dxdot``
    from the same ``dM/da1s``, and ``dX/dB1`` and ``dX/dxdot`` from the same
    ``-rho A_b (Omega R)^2 (dCH/sigma/da1s)``, so
    ``(dM/dB1)(dX/dxdot) = (dX/dB1)(dM/dxdot)`` identically. What survives is

        Theta(s)/B1(s) = (1/I_yy)(dM/dB1) s / [cubic of p. 598]

    Nothing here enforces the cancellation; on the rounded Table 9.4 values the
    residual is 5e-5 of the surviving term.

    Pitch rate costs one more ``s``
    -------------------------------
    p. 606: "the operation represented by ``s`` is differentiation with respect
    to time", so ``q(s)/B1(s)`` is the attitude transfer function times ``s``,
    numerator ``(1/I_yy)(dM/dB1) s^2``. For the example helicopter that
    coefficient is **-6.78**, which is ``dM/dB1 = -270,578`` over
    ``I_yy = 40,000``.

    Options
    -------
    num_nodes : int

    Example helicopter in hover
    ---------------------------
    The step response of pitch rate to longitudinal cyclic, p. 608::

        q/B1 = 5.78 exp(-.874 t) - 6.85 exp(.075 t) sin(20.34 t + 57.54)

    with the angles in degrees. :func:`heaviside_step_response` reproduces it
    to four figures over the whole history Figure 9.12 plots.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        shape = (nn, 2, 3)

        self.add_input('dX_dB1', val=np.full(nn, 9516.0), units='lbf/rad')
        self.add_input('dM_dB1', val=np.full(nn, -270578.0), units='lbf*ft/rad')

        self.add_output('control_coeffs', shape=shape)

        def flat(row):
            return np.ravel_multi_index(
                (ar, np.full(nn, row), np.zeros(nn, dtype=int)), shape)

        self.declare_partials('control_coeffs', 'dX_dB1', rows=flat(0),
                              cols=ar, val=-1.0)
        self.declare_partials('control_coeffs', 'dM_dB1', rows=flat(1),
                              cols=ar, val=-1.0)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        column = np.zeros((nn, 2, 3), dtype=inputs._get_data().dtype)
        column[:, 0, 0] = -inputs['dX_dB1']
        column[:, 1, 0] = -inputs['dM_dB1']
        outputs['control_coeffs'] = column
