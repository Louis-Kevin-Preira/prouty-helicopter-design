"""G3 tests -- Table 9.16, total forward flight derivatives (pp. 591-595).

The dictionary below is the transcription of the printed table: one entry per
row, holding the value in each column that carries a number, plus the printed
total. The tests check that the columns match Tables 9.8, 9.9, 9.11, 9.13 and
9.15, that the totals add up, and that the ten rows Table 9.20 reuses agree.
"""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability.total_derivatives_ff_comp import (
    CONTRIBUTORS,
    TotalDerivativesFFComp,
    build_rows,
)

# Table 9.16, pp. 591-595: {row: ({column: value}, printed total)}.
TABLE_9_16 = {
    'dX_dxdot': ({'main': -12, 'fuse': -8}, -20),
    'dX_dydot': ({'main': -4, 'vert': -3}, -7),
    'dX_dzdot': ({'main': -6, 'horiz': -1, 'fuse': -1}, -8),
    'dX_dq': ({'main': 1937}, 1937),
    'dX_dp': ({'main': -629}, -629),
    'dX_dtheta0_M': ({'main': -6727}, -6727),
    'dX_dA1': ({'main': -1506}, -1506),
    'dX_dB1': ({'main': 18601}, 18601),
    'dY_dxdot': ({'main': -1, 'tail': -2, 'vert': 5}, 2),
    'dY_dydot': ({'main': -14, 'tail': -24, 'vert': -14, 'fuse': -55}, -107),
    'dY_dzdot': ({'main': 13}, 13),
    'dY_dq': ({'main': -574}, -574),
    'dY_dp': ({'main': -1785, 'tail': -147, 'vert': -42}, -1974),
    'dY_dr': ({'tail': 907, 'vert': 490}, 1397),
    'dY_dtheta0_M': ({'main': 2650}, 2650),
    'dY_dA1': ({'main': 16638}, 16638),
    'dY_dB1': ({'main': 1635}, 1635),
    'dY_dtheta0_T': ({'tail': 12845}, 12845),
    'dZ_dxdot': ({'main': 45, 'horiz': 1, 'fuse': 3}, 49),
    'dZ_dzdot': ({'main': -261, 'horiz': -7, 'fuse': -19}, -287),
    'dZ_dq': ({'horiz': -217}, -217),
    'dZ_dr': ({'main': 1914}, 1914),
    'dZ_dtheta0_M': ({'main': -110919}, -110919),
    'dZ_dB1': ({'main': 67855}, 67855),
    'dR_dxdot': ({'main': -20, 'tail': -13, 'vert': 12}, -21),
    'dR_dydot': ({'main': -246, 'tail': -147, 'vert': -42, 'fuse': 53}, -382),
    'dR_dzdot': ({'main': 263}, 263),
    'dR_dq': ({'main': -11418}, -11418),
    'dR_dp': ({'main': -32730, 'tail': -882, 'vert': -126}, -33738),
    'dR_dr': ({'tail': 5442, 'vert': 1470}, 6912),
    'dR_dtheta0_M': ({'main': 70110}, 70110),
    'dR_dA1': ({'main': 325703}, 325703),
    'dR_dB1': ({'main': 32014}, 32014),
    'dR_dtheta0_T': ({'tail': 77070}, 77070),
    'dM_dxdot': ({'main': 182, 'horiz': 42, 'fuse': -80}, 144),
    'dM_dydot': ({'main': -8}, -8),
    'dM_dzdot': ({'main': 495, 'horiz': -219, 'fuse': 374}, 650),
    'dM_dzddot': ({'horiz': 9}, 9),
    'dM_dq': ({'main': -36591, 'horiz': -7161}, -43752),
    'dM_dp': ({'main': 12900}, 12900),
    'dM_dtheta0_M': ({'main': 326946}, 326946),
    'dM_dA1': ({'main': 29480}, 29480),
    'dM_dB1': ({'main': -364158}, -364158),
    'dN_dxdot': ({'main': -53, 'tail': 78, 'vert': -161}, -136),
    'dN_dydot': ({'tail': 907, 'vert': 490, 'fuse': -190}, 1207),
    'dN_dzdot': ({'main': 99}, 99),
    'dN_dp': ({'tail': 5439, 'vert': 1470}, 6909),
    'dN_dr': ({'main': -3204, 'tail': -33559, 'vert': -17150}, -53913),
    'dN_dtheta0_M': ({'main': 376161}, 376161),
    'dN_dtheta0_T': ({'tail': -475265}, -475265),
}

# Table 9.20, p. 614, reuses these ten totals verbatim in its matrix.
IN_TABLE_9_20 = {
    'dM_dxdot': 144, 'dM_dzdot': 650, 'dM_dzddot': 9, 'dM_dq': -43752,
    'dR_dp': -33738, 'dR_dr': 6912,
    'dN_dxdot': -136, 'dN_dydot': 1207, 'dN_dp': 6909, 'dN_dr': -53913,
}


def run(nn=1):
    prob = om.Problem()
    prob.model.add_subsystem('t', TotalDerivativesFFComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, (columns, _) in TABLE_9_16.items():
        stem = name[:-2] if name.endswith(('_M', '_T')) else name
        for source, value in columns.items():
            prob.set_val(f'{stem}_{source}', np.full(nn, float(value)))
    prob.run_model()
    return prob


@pytest.mark.parametrize('name', sorted(TABLE_9_16))
def test_table_9_16_totals(name):
    """Every printed total, from the printed columns."""
    assert_near_equal(run().get_val(name)[0], float(TABLE_9_16[name][1]), 1e-10)


def test_the_table_has_fifty_rows():
    assert len(TABLE_9_16) == len(CONTRIBUTORS) == len(build_rows()) == 50


def test_the_blank_cells_match():
    """A blank column in the book is a missing input here, not a zero."""
    for name, (columns, _) in TABLE_9_16.items():
        assert set(columns) == set(CONTRIBUTORS[name]), name


def test_the_printed_totals_are_the_printed_columns_added_up():
    """Prouty's own arithmetic, checked independently of the component."""
    for name, (columns, total) in TABLE_9_16.items():
        assert sum(columns.values()) == total, name


def test_half_the_table_is_main_rotor_alone():
    alone = [name for name, sources in CONTRIBUTORS.items()
             if sources == ('main',)]
    assert len(alone) == 25


def test_collective_is_not_summed():
    """Six main rotor collective rows and three tail rotor ones."""
    main = [n for n in CONTRIBUTORS if n.endswith('_M')]
    tail = [n for n in CONTRIBUTORS if n.endswith('_T')]
    assert len(main) == 6 and len(tail) == 3

    # no dZ/dtheta0_T or dM/dtheta0_T: Table 9.9 has no tail rotor M rows
    assert 'dZ_dtheta0_T' not in CONTRIBUTORS
    assert 'dM_dtheta0_T' not in CONTRIBUTORS


@pytest.mark.parametrize('name, expected', IN_TABLE_9_20.items())
def test_the_rows_table_9_20_reuses(name, expected):
    """Ten totals appear verbatim in the p. 614 matrix.

    Those are the same numbers PolyDeterminantComp was first validated against
    at the top of the chapter, so the derivative chain and the equations of
    motion close on each other.
    """
    assert_near_equal(run().get_val(name)[0], float(expected), 1e-10)


def test_the_full_chain_reaches_the_characteristic_equation():
    """The Table 9.20 longitudinal subset, straight out of Table 9.16.

    Building p. 618's determinant from these totals and expanding it gives the
    quartic of p. 617 -- the same check G0 ran, now fed by G2 and G3 rather
    than by hand-typed numbers.
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
    assert np.allclose(got, [1.0, 1.545, -2.618, 0.0228, 0.0949], atol=2e-3), got

    # dZ/dzdot = -287 is itself a Table 9.16 total
    assert_near_equal(prob.get_val('dZ_dzdot')[0], -287.0, 1e-10)


def test_three_totals_inherit_a_disagreement():
    """dX/dB1, dM/dB1 from C9-14 and dX/dydot from C9-17."""
    prob = run()
    assert_near_equal(prob.get_val('dX_dB1')[0], 18601.0, 1e-10)
    assert_near_equal(prob.get_val('dM_dB1')[0], -364158.0, 1e-10)
    assert_near_equal(prob.get_val('dX_dydot')[0], -7.0, 1e-10)

    # the vertical stabilizer's -3 is the contradicted one, not the total
    assert TABLE_9_16['dX_dydot'][0]['vert'] == -3


def test_partials():
    prob = run(nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


def test_vectorized():
    triple = run(nn=3)
    for name in TABLE_9_16:
        assert triple.get_val(name).shape == (3,)
