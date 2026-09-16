"""G2/G3 group tests -- Tables 9.5 to 9.16 end to end (pp. 574-595).

Nothing here feeds a printed intermediate value. The group is given rotor
geometry, airframe geometry and the trim state, and has to produce Table 9.16
through eleven tables.
"""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_near_equal

from prouty.stability.forward_flight_derivatives_group import (
    COLUMN,
    CROSS_LINKS,
    EXAMPLE_DEFAULTS,
    SHARED,
    ForwardFlightDerivativesGroup,
    promoted,
)
from prouty.stability.total_derivatives_ff_comp import CONTRIBUTORS

from test_stability_ch9_g3_table916 import IN_TABLE_9_20, TABLE_9_16

#: Rows that inherit a value the book contradicts elsewhere.
INHERITED = {
    'dX_dB1': 'C9-14, da1s/dB1 transposed to -1.118',
    'dM_dB1': 'C9-14, da1s/dB1 transposed to -1.118',
    'dX_dydot': 'C9-17, the vertical stabilizer dX/dydot',
    'dR_dxdot': 'C9-17, the vertical stabilizer dY/dxdot',
}


def run(nn=1, **overrides):
    prob = om.Problem()
    prob.model.add_subsystem('g', ForwardFlightDerivativesGroup(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in overrides.items():
        prob.set_val(name, value)
    prob.run_model()
    return prob


def test_it_runs_standalone_on_the_book_defaults():
    """No inputs set: the group already holds the example helicopter."""
    prob = run()
    assert prob.get_val('dM_dq').shape == (1,)


@pytest.mark.parametrize('name', sorted(TABLE_9_16))
def test_table_9_16_end_to_end(name):
    """Geometry and trim in, Table 9.16 out, through eleven tables."""
    if name in INHERITED:
        pytest.skip(INHERITED[name])
    expected = float(TABLE_9_16[name][1])
    got = run().get_val(name)[0]
    assert abs(got - expected) <= max(0.6, 5e-2 * abs(expected)), \
        f'{name}: {got} vs {expected}'


@pytest.mark.parametrize('name, expected', IN_TABLE_9_20.items())
def test_the_table_9_20_coefficients(name, expected):
    """The ten totals the forward-flight stability analysis is built on.

    All within 2.5 % end to end, which is what a chain of eleven tables of
    two- and three-figure numbers supports.
    """
    got = run().get_val(name)[0]
    assert abs(got / expected - 1.0) < 2.5e-2, f'{name}: {got} vs {expected}'


def test_only_four_rows_fall_outside_five_per_cent():
    """And all four are rows the book already contradicts itself on."""
    prob = run()
    outside = {name for name, (_, total) in TABLE_9_16.items()
               if abs(prob.get_val(name)[0] - total)
               > max(0.6, 5e-2 * abs(total))}
    assert outside == set(INHERITED)


def test_the_airframe_reads_the_rotors():
    """Three of the five airframe tables take a rotor derivative as input.

    The horizontal stabilizer and the fuselage fly in the main rotor's
    downwash; the vertical stabilizer flies in the tail rotor's sidewash.
    Changing a rotor derivative has to move the airframe contributions.
    """
    assert len(CROSS_LINKS) == 4

    base = run()
    moved = run(dCT_sigma_dlambda_M=0.60)          # changes dZ/dzdot on the MR
    for name in ('dZ_dzdot_horiz', 'dZ_dzdot_fuse'):
        assert not np.isclose(base.get_val(name)[0], moved.get_val(name)[0]), \
            name

    tail_moved = run(dCT_sigma_dlambda_T=0.80)     # changes dY/dydot on the TR
    assert not np.isclose(base.get_val('dY_dydot_vert')[0],
                          tail_moved.get_val('dY_dydot_vert')[0])


def test_the_downwash_lag_reaches_the_table_9_20_coefficient():
    """dM/dzddot = 9 exists only because of Table 9.10's l_H/V lag.

    Set the stabilizer arm to zero and the lag vanishes, and with it the
    9 s^2 of Table 9.20's M row -- four tables away.
    """
    assert_near_equal(run().get_val('dM_dzddot')[0], 9.0, 5e-2)
    assert_near_equal(run(l_H=0.0).get_val('dM_dzddot')[0], 0.0, 1e-12)


def test_the_tags_are_not_decoration():
    """The same symbol means different things on different components."""
    assert EXAMPLE_DEFAULTS['sigma_M'][0] != EXAMPLE_DEFAULTS['sigma_T'][0]
    assert EXAMPLE_DEFAULTS['A_b_M'][0] != EXAMPLE_DEFAULTS['A_b_T'][0]
    assert EXAMPLE_DEFAULTS['A_R_H'][0] != EXAMPLE_DEFAULTS['A_R_V'][0]
    assert EXAMPLE_DEFAULTS['alpha_LO_H'][0] != EXAMPLE_DEFAULTS['alpha_LO_V'][0]


def test_d_beta_d_ydot_is_computed_four_times():
    """Tables 9.6, 9.7, 9.12 and 9.14 each produce it, and all agree on 1/V."""
    prob = run()
    values = [prob.get_val(f'd_beta_d_ydot_{tag}')[0]
              for tag in ('M', 'T', 'V', 'F')]
    for value in values:
        assert_near_equal(value, 1.0 / EXAMPLE_DEFAULTS['V'][0], 1e-12)
    assert len(set(f'd_beta_d_ydot_{t}' for t in 'MTVF')) == 4


def test_shared_quantities_take_no_tag():
    for name in SHARED:
        assert promoted(name, 'M') == name
        assert promoted(name, 'H') == name


def test_every_column_of_table_9_16_is_wired():
    """Each of the five dimensional tables promotes into the totalling."""
    prob = run()
    assert set(COLUMN.values()) == {'main', 'tail', 'horiz', 'vert', 'fuse'}
    for row, sources in CONTRIBUTORS.items():
        stem = row[:-2] if row.endswith(('_M', '_T')) else row
        for source in sources:
            assert prob.get_val(f'{stem}_{source}').shape == (1,), (row, source)


def test_it_reaches_the_characteristic_equation():
    """Rotor geometry in one end, p. 617's quartic out the other.

    The last link in the chain: build p. 618's determinant from the group's own
    M-row totals and expand it. This is the same check G0 ran at the top of the
    chapter, now with nothing hand-typed but the three non-M rows.
    """
    from prouty.stability import PolyDeterminantComp

    prob = run()
    entry = lambda c0=0.0, c1=0.0, c2=0.0: np.array([c0, c1, c2])
    matrix = np.array([[[entry(0.0, -20.0, -621.0), entry(0.0, -8.0),
                         entry(-20000.0, 3927.0)],
                        [entry(0.0, 49.0), entry(0.0, -287.0, -621.0),
                         entry(0.0, 120400.0)],
                        [entry(0.0, prob.get_val('dM_dxdot')[0]),
                         entry(0.0, prob.get_val('dM_dzdot')[0],
                               prob.get_val('dM_dzddot')[0]),
                         entry(0.0, prob.get_val('dM_dq')[0], -40000.0)]]])

    det = om.Problem()
    det.model.add_subsystem('d', PolyDeterminantComp(n=3, degree=2,
                                                     n_zero_roots=2),
                            promotes=['*'])
    det.setup()
    det['matrix_coeffs'] = matrix
    det.run_model()

    got = det['char_coeffs'][0][::-1]
    book = [1.0, 1.545, -2.618, 0.0228, 0.0949]
    # a chain of eleven tables of two- and three-figure numbers carries about
    # 2 % into the quartic; the s coefficient is the smallest and shows it most
    for value, expected in zip(got, book):
        assert abs(value - expected) <= max(0.01, 5e-2 * abs(expected)), \
            (list(np.round(got, 5)), book)


def test_totals():
    """One end-to-end gradient through the whole chain."""
    prob = run()
    data = prob.check_totals(of=['dM_dq', 'dN_dr', 'dZ_dzdot'],
                             wrt=['sigma_M', 'l_H', 'A_V'],
                             method='cs', out_stream=None)
    for key, entry in data.items():
        scale = max(1.0, np.max(np.abs(entry['J_fwd'])))
        assert entry['abs error'].forward < 1e-6 * scale, key


@pytest.mark.parametrize('nn', (1, 3))
def test_vectorized(nn):
    prob = run(nn=nn)
    for name in ('dM_dq', 'dN_dr', 'dX_dtheta0_M'):
        assert prob.get_val(name).shape == (nn,)
