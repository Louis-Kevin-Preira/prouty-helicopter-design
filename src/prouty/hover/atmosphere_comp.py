"""
AtmosphereComp -- air density and speed of sound from pressure altitude.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Appendix C, "Atmospheric Charts", Figures C.1 and C.2, p. 703-704; used as the
test conditions of step 1, p. 69.

Figure C.1 is the density ratio rho/rho_0 against pressure altitude for a range
of temperatures, with rho_0 = 0.002377 slug/ft3 printed on the chart.
Figure C.2 is the speed of sound against temperature, anchored on the chart at
1116 ft/s for a sea level standard day (59 F). Both are the standard atmosphere
plotted, so the relations are coded exactly rather than digitised:

    theta_std = 1 - 6.87535e-6 h          pressure ratio base   (troposphere)
    p/p_0     = theta_std ** 5.2559
    T         = 518.67 theta_std + dT     [deg R]
    rho/rho_0 = (p/p_0) * 518.67 / T
    V_son     = 1116.45 sqrt(T / 518.67)

    altitude, dT --> rho, V_son, density_ratio, T_air

The temperature output is named T_air, not T: step 21 uses T for rotor thrust,
which is the book's symbol and the one that must survive promotion.

Temperature is given as an offset dT from the standard day, which reaches every
constant-temperature line of Figure C.1 with a single differentiable branch.
Valid in the troposphere only, up to 36,089 ft; the charts stop at 30,000 ft.
"""

import warnings

import numpy as np
import openmdao.api as om

RHO_0 = 0.002377        # slug/ft3, printed on Figure C.1
T_0 = 518.67            # deg R, 59 F standard day
A_0 = 1116.45           # ft/s, speed of sound at T_0
LAPSE = 6.87535e-6      # 1/ft, non-dimensional lapse rate
EXP = 5.2559            # g / (lapse R), troposphere
H_TROP = 36089.0        # ft


class AtmosphereComp(om.ExplicitComponent):
    """Standard atmosphere, Figures C.1 and C.2."""

    def setup(self):
        self.add_input('altitude', val=0.0, units='ft', desc='pressure altitude')
        self.add_input('dT', val=0.0, units='degR',
                       desc='temperature offset from the standard day')

        self.add_output('rho', val=RHO_0, units='slug/ft**3', desc='air density')
        self.add_output('V_son', val=A_0, units='ft/s', desc='speed of sound')
        self.add_output('density_ratio', val=1.0, desc='rho / rho_0')
        self.add_output('T_air', val=T_0, units='degR', desc='air temperature')

        self.declare_partials(['rho', 'density_ratio', 'V_son', 'T_air'],
                              ['altitude', 'dT'])

    def _state(self, inputs):
        h, dT = inputs['altitude'], inputs['dT']
        if np.real(h) > H_TROP:
            warnings.warn(f'altitude {np.real(h):.0f} ft is above the {H_TROP:.0f} ft '
                          'troposphere limit of this model.')
        theta = 1.0 - LAPSE * h
        pressure_ratio = theta ** EXP
        T = T_0 * theta + dT
        return theta, pressure_ratio, T

    def compute(self, inputs, outputs):
        _, pressure_ratio, T = self._state(inputs)
        ratio = pressure_ratio * T_0 / T

        outputs['T_air'] = T
        outputs['density_ratio'] = ratio
        outputs['rho'] = RHO_0 * ratio
        outputs['V_son'] = A_0 * np.sqrt(T / T_0)

    def compute_partials(self, inputs, partials):
        theta, pressure_ratio, T = self._state(inputs)

        dp_dh = -EXP * LAPSE * theta ** (EXP - 1.0)
        dT_dh = -T_0 * LAPSE

        dratio_dh = T_0 * (dp_dh / T - pressure_ratio * dT_dh / T ** 2)
        dratio_ddT = -T_0 * pressure_ratio / T ** 2

        partials['T_air', 'altitude'] = dT_dh
        partials['T_air', 'dT'] = 1.0
        partials['density_ratio', 'altitude'] = dratio_dh
        partials['density_ratio', 'dT'] = dratio_ddT
        partials['rho', 'altitude'] = RHO_0 * dratio_dh
        partials['rho', 'dT'] = RHO_0 * dratio_ddT

        dV_dT = 0.5 * A_0 / np.sqrt(T * T_0)
        partials['V_son', 'altitude'] = dV_dT * dT_dh
        partials['V_son', 'dT'] = dV_dT
