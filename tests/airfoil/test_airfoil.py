"""Regression tests for prouty.airfoil against the values printed in the book."""

import numpy as np
import openmdao.api as om
import pytest

from prouty.airfoil import AirfoilHoverGroup, AirfoilForwardFlightGroup


def build(group, nn, **inputs):
    p = om.Problem()
    p.model.add_subsystem('af', group(num_nodes=nn), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in inputs.items():
        p.set_val(k, v)
    p.run_model()
    return p


# --- book anchors ---------------------------------------------------------

def test_lift_coefficients_match_book_tables():
    """K1 values tabulated p. 429-430."""
    p = build(AirfoilHoverGroup, 3, M=[0.2, 0.5, 0.7], alpha=[0.0, 0.0, 0.0])
    np.testing.assert_allclose(p.get_val('K1'), [0.0233, 0.0257, 0.0497],
                               atol=5e-4)


def test_zero_angle_drag_is_book_value():
    """cd0 = 0.0081 below drag divergence, p. 432."""
    p = build(AirfoilHoverGroup, 3, M=[0.1, 0.4, 0.7], alpha=np.zeros(3))
    np.testing.assert_allclose(p.get_val('cd'), 0.0081, atol=1e-12)


def test_drag_divergence_line_crosses_zero_at_break():
    """alpha_D = 17 - 23.4M passes through zero at M = 0.725, p. 432."""
    p = build(AirfoilHoverGroup, 1, M=[0.725], alpha=[0.0])
    assert abs(p.get_val('alpha_D')[0]) < 0.05


def test_forward_flight_segment_values():
    """Segment table of p. 433-434."""
    alpha = np.array([45.0, 90.0, 167.0, 180.0, 193.0, 270.0, 315.0])
    p = build(AirfoilForwardFlightGroup, alpha.size,
              alpha_raw=alpha, M=np.full(alpha.size, 0.3))
    np.testing.assert_allclose(
        p.get_val('cl'), [1.15, 0.0, -0.7, 0.0, 0.7, 0.0, -1.15], atol=1e-9)
    np.testing.assert_allclose(
        p.get_val('cd')[[0, 1, 3, 5, 6]], [1.03, 2.05, 0.01, 2.05, 1.03],
        atol=1e-9)


# --- structural properties ------------------------------------------------

def test_lift_is_antisymmetric_and_drag_symmetric():
    alpha = np.linspace(0.0, 360.0, 721)
    p = build(AirfoilForwardFlightGroup, alpha.size,
              alpha_raw=alpha, M=np.full(alpha.size, 0.3))
    cl, cd = p.get_val('cl'), p.get_val('cd')
    mirror = cl[::-1]
    assert np.abs(cl + mirror).max() < 1e-12
    assert np.abs(cd - cd[::-1]).max() < 1e-12


def test_hover_cl_max_decreases_with_mach_below_break():
    alpha = np.linspace(0.0, 18.0, 181)
    peaks = []
    for M in (0.1, 0.3, 0.5, 0.7):
        p = build(AirfoilHoverGroup, alpha.size,
                  alpha=alpha, M=np.full(alpha.size, M))
        peaks.append(p.get_val('cl').max())
    assert all(np.diff(peaks) < 0.0)


def test_alpha_wrapping_is_idempotent():
    """-135 deg and 225 deg must give identical coefficients, p. 214."""
    p = build(AirfoilForwardFlightGroup, 2,
              alpha_raw=[-135.0, 225.0], M=[0.3, 0.3])
    assert p.get_val('cl')[0] == pytest.approx(p.get_val('cl')[1])
    assert p.get_val('cd')[0] == pytest.approx(p.get_val('cd')[1])


# --- derivatives ----------------------------------------------------------

@pytest.mark.parametrize('group,inputs', [
    (AirfoilHoverGroup, dict(M=[0.3, 0.5, 0.8], alpha=[5.0, 9.0, 3.0])),
    (AirfoilForwardFlightGroup, dict(M=[0.3, 0.4, 0.5],
                                     alpha_raw=[10.0, 150.0, 250.0])),
])
def test_totals_are_consistent(group, inputs):
    p = build(group, 3, **inputs)
    data = p.check_totals(of=['cl', 'cd'], wrt=list(inputs),
                          method='fd', out_stream=None)
    for entry in data.values():
        assert np.abs(entry['J_fwd'] - entry['J_fd']).max() < 1e-4
