"""G7 tests -- MIL-H-8501A Table 9.21 and Figure 9.21 (pp. 622, 629)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability import PolyDeterminantComp, describe_modes
from prouty.stability.handling_qualities_ff import (
    STABILIZER_AREAS,
    TABLE_9_21,
    BoeingVertolParameterComp,
    mil_h_8501a,
)
from prouty.stability.long_matrix_ff_comp import LongMatrixFFComp
from prouty.stability.long_mode_approximations import ShortPeriodMatrixComp

from test_g7_longitudinal import EXAMPLE as FULL
from test_g7_mode_approximations import HS

#: Figure 9.21 and Table 9.21, evaluated on the five stabilizer areas.
EXPECTED = {
    18.0: (-2.649, None),
    36.0: (-1.489, (41.6, 1.6)),
    54.0: (-0.325, (17.1, 3.1)),
    72.0: (0.843, (17.0, 46.6)),
    90.0: (2.015, (7.5, -0.7)),        # negative time means halving
}


def sized(area):
    """The example helicopter with a stabilizer of the given area."""
    extra = area / STABILIZER_AREAS[0] - 1.0
    return {name: FULL[name] + extra * HS.get(name, 0.0) for name in FULL}


def parameter(values, nn=1, **overrides):
    prob = om.Problem()
    prob.model.add_subsystem('bv', BoeingVertolParameterComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name in ('dZ_dzdot', 'dM_dq', 'dM_dzdot', 'G_W', 'I_yy', 'g', 'V'):
        prob.set_val(name, np.full(nn, values[name]))
    for name, value in overrides.items():
        prob.set_val(name, np.full(nn, value))
    prob.run_model()
    return prob


def oscillation(values):
    """The oscillatory mode of the full longitudinal quartic, or None."""
    prob = om.Problem()
    prob.model.add_subsystem('matrix', LongMatrixFFComp(), promotes=['*'])
    prob.model.add_subsystem('det', PolyDeterminantComp(n=3, degree=2,
                                                        n_zero_roots=2),
                             promotes=['*'])
    prob.setup()
    for name, value in values.items():
        prob.set_val(name, np.full(1, value))
    prob.run_model()
    modes = [m for m in describe_modes(prob.get_val('char_coeffs')[0])
             if m.oscillatory]
    return modes[0] if modes else None


# ------------------------------------------------------ Figure 9.21, p. 629

@pytest.mark.parametrize('area', STABILIZER_AREAS)
def test_the_parameter_across_the_five_areas(area):
    expected = EXPECTED[area][0]
    got = parameter(sized(area)).get_val('short_period_parameter')[0]
    assert abs(got - expected) < 5e-3, f'{area}: {got} vs {expected}'


def test_it_changes_sign_between_54_and_72_square_feet():
    """Where the short period stops being a divergence."""
    assert parameter(sized(54.0)).get_val('short_period_parameter')[0] < 0.0
    assert parameter(sized(72.0)).get_val('short_period_parameter')[0] > 0.0


def test_it_increases_monotonically_with_area():
    values = [parameter(sized(a)).get_val('short_period_parameter')[0]
              for a in STABILIZER_AREAS]
    assert values == sorted(values)


def test_it_is_the_short_period_stiffness():
    """The parameter is the short-period quadratic's constant term.

    Drop dZ/dq and dZ/dzddot from p. 625's expression -- -217 against an m V
    of 120,555, and zero -- and the two coincide. So the criterion is that the
    square of the short-period natural frequency exceed about 1.
    """
    values = FULL
    prob = om.Problem()
    prob.model.add_subsystem('m', ShortPeriodMatrixComp(), promotes=['*'])
    prob.model.add_subsystem('det', PolyDeterminantComp(n=2, degree=2,
                                                        n_zero_roots=2),
                             promotes=['*'])
    prob.setup()
    for name in ('dZ_dzdot', 'dZ_dzddot', 'dZ_dq', 'dM_dzdot', 'dM_dzddot',
                 'dM_dq', 'G_W', 'I_yy', 'g', 'V'):
        prob.set_val(name, np.full(1, values[name]))
    prob.run_model()

    stiffness = prob.get_val('char_coeffs')[0][0]
    bv = parameter(values).get_val('short_period_parameter')[0]

    assert_near_equal(stiffness, -2.6427, 1e-3)
    assert_near_equal(bv, -2.6487, 1e-3)
    assert abs(bv / stiffness - 1.0) < 3e-3


def test_neglecting_dZ_dq_is_what_separates_them():
    """The 0.2 % gap is exactly the dZ/dq term."""
    values = FULL
    mass = values['G_W'] / values['g']
    correction = (values['dZ_dq'] * values['dM_dzdot']
                  / (values['I_yy'] * mass))
    bv = parameter(values).get_val('short_period_parameter')[0]
    assert_near_equal(bv - correction, -2.6427, 1e-3)


def test_the_margin_is_against_the_figure_9_21_boundary():
    prob = parameter(sized(90.0))
    assert_near_equal(prob.get_val('handling_margin')[0],
                      prob.get_val('short_period_parameter')[0] - 1.0, 1e-12)
    assert prob.get_val('handling_margin')[0] > 0.0
    assert parameter(sized(72.0)).get_val('handling_margin')[0] < 0.0


# ------------------------------------------------------- Table 9.21, p. 622

def test_the_table_has_four_period_bands():
    assert len(TABLE_9_21) == 4
    assert TABLE_9_21[0][0] == 5.0 and TABLE_9_21[-1][0] == np.inf


def test_seventy_two_square_feet_passes_visual_and_fails_instrument():
    """p. 622, in words: "would allow the aircraft to satisfy the visual flight
    requirement, but a somewhat larger tail ... would be needed for instrument
    flight".

    Reproduced from the derivative tables: the 72 ft² configuration oscillates
    at 17.0 seconds with amplitude doubling in 46.6, which clears the 10-20
    second band's visual rule of "not double in 10 sec" and fails its
    instrument rule of "at least lightly damped".
    """
    mode = oscillation(sized(72.0))
    assert_near_equal(mode.period, 17.0, 2e-2)
    assert mode.root.real > 0.0

    visual, visual_rule = mil_h_8501a(mode.period, mode.root.real)
    instrument, instrument_rule = mil_h_8501a(mode.period, mode.root.real,
                                              instrument=True)
    assert visual and visual_rule == 'not double in 10 s'
    assert not instrument and instrument_rule == 'lightly damped'


def test_ninety_square_feet_passes_both():
    """"A somewhat larger tail" -- and it is enough."""
    mode = oscillation(sized(90.0))
    assert mode.root.real < 0.0
    assert mil_h_8501a(mode.period, mode.root.real)[0]
    assert mil_h_8501a(mode.period, mode.root.real, instrument=True)[0]


def test_fifty_four_square_feet_fails_both():
    mode = oscillation(sized(54.0))
    assert not mil_h_8501a(mode.period, mode.root.real)[0]
    assert not mil_h_8501a(mode.period, mode.root.real, instrument=True)[0]


def test_the_no_requirement_band_lets_a_divergence_through():
    """A caution, not a bug. Table 9.21 imposes no visual requirement above
    20 seconds, and the 36 ft² configuration exploits it: a 41.6 second period
    with amplitude doubling in 1.6 seconds passes.

    That is a divergence wearing a very slow oscillation, not the slow wallow
    p. 622's justification has in mind. The function reports the rule so the
    caller can see which clause decided.
    """
    mode = oscillation(sized(36.0))
    assert_near_equal(mode.period, 41.6, 2e-2)
    assert mode.time_to_double < 2.0

    passes, rule = mil_h_8501a(mode.period, mode.root.real)
    assert passes and rule == 'no requirement'
    assert not mil_h_8501a(mode.period, mode.root.real, instrument=True)[0]


@pytest.mark.parametrize('period, real_part, instrument, expected', [
    (3.0, -0.5, False, True),          # half in 2 cycles: t_half 1.4 <= 6
    (3.0, -0.15, False, True),         # 4.6 <= 6
    (3.0, -0.10, False, False),        # 6.9 > 6
    (3.0, -0.15, True, False),         # instrument needs 1 cycle: 4.6 > 3
    (7.0, -0.01, False, True),         # lightly damped
    (7.0, 0.01, False, False),
    (15.0, 0.05, False, True),         # not double in 10: 13.9 >= 10
    (15.0, 0.10, False, False),        # 6.9 < 10
    (15.0, -0.01, True, True),         # lightly damped
    (25.0, 0.20, False, True),         # no requirement
    (25.0, 0.02, True, True),          # not double in 20: 34.7 >= 20
    (25.0, 0.05, True, False),         # 13.9 < 20
])
def test_each_band_of_table_9_21(period, real_part, instrument, expected):
    assert mil_h_8501a(period, real_part, instrument)[0] is expected


def test_partials():
    prob = parameter(FULL, nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-9, rtol=1e-9)


def test_vectorized_matches_scalar():
    single, triple = parameter(FULL), parameter(FULL, nn=3)
    for name in ('short_period_parameter', 'handling_margin'):
        assert_near_equal(triple.get_val(name),
                          np.full(3, single.get_val(name)[0]), 1e-12)
