"""Chart slope tests -- the two Table 9.1 entries p. 564 sends to Chapter 1.

The reference values are the Chapter 1 charts as Table 9.1 reads them, and the
repository's own Chapter 1 model differentiated at the same operating point.
"""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability.hover_chart_slopes_comp import HoverChartSlopesComp
from prouty.stability.hover_derivatives_group import HoverDerivativesGroup

from test_g1_group import EXAMPLE, SCALARS

# Example helicopter main rotor at C_T/sigma = .086 (Chapter 1, p. 18).
MAIN = dict(dCT_sigma_dlambda=0.4911, CT_sigma=0.086, sigma=0.085, a=6.0,
            k_induced=1.0, dcd_dalpha=0.0573)
SCALAR_INPUTS = ('sigma', 'a')

# Table 9.1, p. 564, main rotor column.
CHART = {'dCT_sigma_dtheta0': 0.61, 'dCQ_sigma_dtheta0': 0.078}

# The same two slopes from the repository's Chapter 1 model, central
# differenced at theta_0 = 17.82 deg where it trims to C_T/sigma = .086.
CHAPTER_1_MODEL = {'dCT_sigma_dtheta0': 0.5986, 'dCQ_sigma_dtheta0': 0.07484}


def run(nn=1, **overrides):
    values = dict(MAIN, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('s', HoverChartSlopesComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, value if name in SCALAR_INPUTS
                     else np.full(nn, value))
    prob.run_model()
    return prob


def test_thrust_slope_is_four_thirds_of_the_inflow_slope():
    """dx/dtheta0 = (4/3) dx/dlambda', from p. 19 and the p. 20 conversion."""
    prob = run()
    assert_near_equal(prob.get_val('dCT_sigma_dtheta0')[0],
                      4.0 / 3.0 * MAIN['dCT_sigma_dlambda'], 1e-13)


def test_thrust_slope_against_the_chart():
    """.655 here against .610 read off the Chapter 1 chart: 7 % high."""
    got = run().get_val('dCT_sigma_dtheta0')[0]
    assert 1.05 < got / CHART['dCT_sigma_dtheta0'] < 1.09


def test_thrust_slope_against_the_chapter_1_model():
    """.655 against .599 from the model itself: 9 % high, same direction."""
    got = run().get_val('dCT_sigma_dtheta0')[0]
    assert 1.05 < got / CHAPTER_1_MODEL['dCT_sigma_dtheta0'] < 1.12


def test_torque_slope_at_ideal_momentum_is_low():
    """k_induced = 1 is the ideal value and lands 18 % under the chart."""
    got = run().get_val('dCQ_sigma_dtheta0')[0]
    assert 0.78 < got / CHART['dCQ_sigma_dtheta0'] < 0.85


def test_torque_slope_with_a_realistic_induced_factor():
    """k_induced = 1.2 puts the torque slope within 3 % of the chart."""
    got = run(k_induced=1.2).get_val('dCQ_sigma_dtheta0')[0]
    assert abs(got - CHART['dCQ_sigma_dtheta0']) < 0.03 * CHART['dCQ_sigma_dtheta0']


def test_torque_slope_splits_into_induced_and_profile():
    """C_Q/sigma = lambda_i x + c_d/8, p. 22, so the two parts add."""
    induced = run(dcd_dalpha=0.0).get_val('dCQ_sigma_dtheta0')[0]
    profile = run(k_induced=0.0).get_val('dCQ_sigma_dtheta0')[0]
    total = run().get_val('dCQ_sigma_dtheta0')[0]

    assert_near_equal(induced + profile, total, 1e-13)
    assert profile / total < 0.2                     # profile is the small part


def test_inflow_slope_is_half_the_naive_one():
    """The factor the naive derivation misses, recorded as an assertion.

    Freezing the induced velocity and adding the climb inflow to it gives
    1/[4/a + k]. Momentum theory in climb halves that, and Table 9.1's .49 is
    the halved value.
    """
    a, sigma, x = MAIN['a'], MAIN['sigma'], MAIN['CT_sigma']
    lambda_i = np.sqrt(0.5 * sigma * x)
    k = lambda_i / (2.0 * x)

    naive = 1.0 / (4.0 / a + k)
    prouty = 1.0 / (8.0 / a + np.sqrt(0.5 * sigma) / np.sqrt(x))

    assert_near_equal(naive / prouty, 2.0, 1e-12)
    assert_near_equal(prouty, 0.49, 3e-3)


def test_partials():
    prob = run(nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-9, rtol=1e-9)


# ------------------------------------------------------ group integration

def group(source, **overrides):
    values = dict(EXAMPLE, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('g', HoverDerivativesGroup(derivative_source=source),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        if source == 'model' and name.startswith(('dCT_sigma_dtheta0',
                                                  'dCQ_sigma_dtheta0')):
            continue
        prob.set_val(name, value if name in SCALARS else np.full(1, value))
    prob.run_model()
    return prob


def test_model_source_replaces_the_chart_inputs():
    """With derivative_source='model' the four chart inputs are gone."""
    prob = group('model')
    promoted = {m['prom_name'] for _, m
                in prob.model.list_inputs(out_stream=None, prom_name=True,
                                          val=False)}
    for name in ('dCT_sigma_dtheta0_M', 'dCQ_sigma_dtheta0_M',
                 'dCT_sigma_dtheta0_T', 'dCQ_sigma_dtheta0_T'):
        assert name not in promoted
    assert 'k_induced_M' in promoted and 'dcd_dalpha_T' in promoted


def test_model_source_feeds_the_dimensional_tables():
    """The slope component's outputs reach Table 9.2, not a stray default."""
    prob = group('model')
    assert_near_equal(
        prob.get_val('g.main_table.dCT_sigma_dtheta0')[0],
        prob.get_val('g.main_slopes.dCT_sigma_dtheta0')[0], 1e-13)


def test_model_source_costs_accuracy_against_the_book():
    """dZ/dtheta0 is proportional to the thrust slope, so it moves 7 %."""
    table = group('table').get_val('dZ_dtheta0_M')[0]
    model = group('model').get_val('dZ_dtheta0_M')[0]
    assert 1.05 < model / table < 1.09


def test_table_source_is_still_the_default():
    prob = om.Problem()
    prob.model.add_subsystem('g', HoverDerivativesGroup(), promotes=['*'])
    prob.setup()
    prob.final_setup()
    assert_near_equal(prob.get_val('dCT_sigma_dtheta0_M')[0], 0.61, 1e-13)


def test_model_source_totals():
    """The added layer differentiates end to end."""
    prob = group('model')
    data = prob.check_totals(of=['dZ_dtheta0_M', 'dN_dtheta0_M'],
                             wrt=['sigma_M', 'k_induced_M'],
                             method='cs', out_stream=None)
    for key, entry in data.items():
        scale = max(1.0, abs(float(np.atleast_1d(entry['J_fwd']).ravel()[0])))
        assert entry['abs error'].forward < 1e-6 * scale, key
