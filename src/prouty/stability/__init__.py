"""Chapter 9 -- Stability and Control Analysis (pp. 541-638)."""

from prouty.stability.basic_main_rotor_derivatives_ff_comp import (
    BasicMainRotorDerivativesFFComp,
)
from prouty.stability.basic_tail_rotor_derivatives_ff_comp import (
    BasicTailRotorDerivativesFFComp,
)
from prouty.stability.basic_rotor_derivatives_hover_comp import (
    BasicRotorDerivativesHoverComp,
)
from prouty.stability.characteristic_equation_group import (
    CharacteristicEquationGroup,
)
from prouty.stability.hohenemser_period_comp import HohenemserPeriodComp
from prouty.stability.control_response_hover_group import (
    ControlResponseHoverGroup,
)
from prouty.stability.control_response_hover import (
    HoverControlColumnComp,
    heaviside_step_response,
)
from prouty.stability.dutch_roll_approximations import (
    DutchRollApproxComp,
    DutchRollMatrixComp,
)
from prouty.stability.forward_flight_derivatives_group import (
    ForwardFlightDerivativesGroup,
)
from prouty.stability.fuselage_derivatives_ff_comp import (
    FuselageDerivativesFFComp,
)
from prouty.stability.fuselage_nondim_derivatives_comp import (
    FuselageNondimDerivativesComp,
)
from prouty.stability.handling_qualities_ff import (
    BoeingVertolParameterComp,
    mil_h_8501a,
)
from prouty.stability.horiz_stab_derivatives_ff_comp import (
    HorizStabDerivativesFFComp,
)
from prouty.stability.horiz_stab_nondim_derivatives_comp import (
    HorizStabNondimDerivativesComp,
)
from prouty.stability.hover_chart_slopes_comp import HoverChartSlopesComp
from prouty.stability.hover_derivatives_group import HoverDerivativesGroup
from prouty.stability.hover_lateral_matrix_comp import HoverLateralMatrixComp
from prouty.stability.hover_long_two_dof_matrix_comp import (
    HoverLongTwoDofMatrixComp,
)
from prouty.stability.hover_longitudinal_matrix_comp import (
    HoverLongitudinalMatrixComp,
)
from prouty.stability.lateral_matrix_ff_comp import LateralMatrixFFComp
from prouty.stability.lateral_stability_map_comp import (
    LateralStabilityMapComp,
    classify_lateral,
)
from prouty.stability.load_factor_margin_comp import LoadFactorMarginComp
from prouty.stability.long_matrix_ff_comp import LongMatrixFFComp
from prouty.stability.long_mode_approximations import (
    PhugoidMatrixComp,
    ShortPeriodMatrixComp,
)
from prouty.stability.long_stability_map_comp import LongStabilityMapComp
from prouty.stability.main_rotor_derivatives_ff_comp import (
    MainRotorDerivativesFFComp,
)
from prouty.stability.main_rotor_derivatives_hover_comp import (
    MainRotorDerivativesHoverComp,
)
from prouty.stability.monomial_rows_comp import MonomialRowsComp
from prouty.stability.tail_rotor_derivatives_ff_comp import (
    TailRotorDerivativesFFComp,
)
from prouty.stability.tail_rotor_derivatives_hover_comp import (
    TailRotorDerivativesHoverComp,
)
from prouty.stability.hover_stability_group import HoverStabilityGroup
from prouty.stability.matrix_column_substitution_comp import (
    MatrixColumnSubstitutionComp,
)
from prouty.stability.mil_response_requirements_comp import (
    MilResponseRequirementsComp,
    figure_9_14_violations,
)
from prouty.stability.modes import (
    describe_modes,
    format_modes,
    polynomial_roots,
    routh_tests,
)
from prouty.stability.plots import (
    classify_grid,
    plot_load_factor_margin,
    plot_lateral_stability_map,
    plot_longitudinal_stability_map,
    plot_response_box,
    plot_stability_map,
    plot_step_response,
    roots_on_grid,
)
from prouty.stability.poly_determinant_comp import PolyDeterminantComp
from prouty.stability.polynomial_discriminant_comp import (
    PolynomialDiscriminantComp,
    discriminant,
    quartic_root_character,
)
from prouty.stability.rotor_chart_derivatives_comp import (
    RotorChartDerivativesComp,
)
from prouty.stability.routh_discriminant_comp import RouthDiscriminantComp
from prouty.stability.stability_map import (
    affine_coefficients,
    constant_term_line,
    evaluate_conic,
    routh_conic,
)
from prouty.stability.total_derivatives_ff_comp import (
    TotalDerivativesFFComp,
)
from prouty.stability.total_derivatives_hover_comp import (
    TotalDerivativesHoverComp,
)
from prouty.stability.transfer_function_group import TransferFunctionGroup
from prouty.stability.vert_stab_derivatives_ff_comp import (
    VertStabDerivativesFFComp,
)
from prouty.stability.vert_stab_nondim_derivatives_comp import (
    VertStabNondimDerivativesComp,
)
from prouty.stability.yaw_mode_hover_comp import YawModeHoverComp

__all__ = [
    'BasicMainRotorDerivativesFFComp',
    'BoeingVertolParameterComp',
    'BasicRotorDerivativesHoverComp',
    'BasicTailRotorDerivativesFFComp',
    'CharacteristicEquationGroup',
    'HohenemserPeriodComp',
    'DutchRollApproxComp',
    'DutchRollMatrixComp',
    'ForwardFlightDerivativesGroup',
    'FuselageDerivativesFFComp',
    'FuselageNondimDerivativesComp',
    'HorizStabDerivativesFFComp',
    'HorizStabNondimDerivativesComp',
    'HoverChartSlopesComp',
    'ControlResponseHoverGroup',
    'HoverControlColumnComp',
    'HoverDerivativesGroup',
    'HoverLateralMatrixComp',
    'HoverLongTwoDofMatrixComp',
    'HoverLongitudinalMatrixComp',
    'HoverStabilityGroup',
    'LateralMatrixFFComp',
    'LateralStabilityMapComp',
    'LoadFactorMarginComp',
    'LongMatrixFFComp',
    'LongStabilityMapComp',
    'PhugoidMatrixComp',
    'ShortPeriodMatrixComp',
    'MainRotorDerivativesFFComp',
    'MainRotorDerivativesHoverComp',
    'MilResponseRequirementsComp',
    'MonomialRowsComp',
    'MatrixColumnSubstitutionComp',
    'PolyDeterminantComp',
    'PolynomialDiscriminantComp',
    'RotorChartDerivativesComp',
    'RouthDiscriminantComp',
    'TailRotorDerivativesFFComp',
    'TailRotorDerivativesHoverComp',
    'TotalDerivativesFFComp',
    'TotalDerivativesHoverComp',
    'TransferFunctionGroup',
    'VertStabDerivativesFFComp',
    'VertStabNondimDerivativesComp',
    'classify_grid',
    'classify_lateral',
    'affine_coefficients',
    'constant_term_line',
    'discriminant',
    'evaluate_conic',
    'routh_conic',
    'plot_lateral_stability_map',
    'plot_load_factor_margin',
    'plot_longitudinal_stability_map',
    'plot_response_box',
    'plot_stability_map',
    'plot_step_response',
    'quartic_root_character',
    'roots_on_grid',
    'figure_9_14_violations',
    'heaviside_step_response',
    'mil_h_8501a',
    'YawModeHoverComp',
    'describe_modes',
    'format_modes',
    'polynomial_roots',
    'routh_tests',
]
