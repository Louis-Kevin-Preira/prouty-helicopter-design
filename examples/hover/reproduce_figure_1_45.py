"""
End to end validation of the combined momentum and blade element method,
steps 1 to 21, against the sample calculation of Figure 1.45, p. 76.

Example helicopter main rotor, p. 669: R = 30 ft, c = 2 ft, b = 4,
V_tip = 650 ft/s, cutout 0.15 R, theta_1 = -10 deg, theta_0 = 17.5 deg,
NACA 0012, sea level standard day.
"""

import pathlib

import numpy as np
import openmdao.api as om

from prouty.hover.rotor_preprocess_group import RotorPreprocessGroup
from prouty.hover.blade_element_group import BladeElementGroup
from prouty.hover.thrust_group import ThrustGroup
from prouty.hover.torque_group import TorqueGroup
from prouty.hover.empirical_corrections_group import EmpiricalCorrectionsGroup
from prouty.hover.rotor_performance_group import RotorPerformanceGroup
from prouty.hover.plot_blade_element import plot_figure_145, case_annotations


def build(ne=10):
    nn = ne + 1
    p = om.Problem()
    m = p.model
    m.add_subsystem('pre', RotorPreprocessGroup(num_elements=ne), promotes=['*'])
    m.add_subsystem('be', BladeElementGroup(num_nodes=nn), promotes=['*'])
    m.add_subsystem('thrust', ThrustGroup(num_nodes=nn), promotes=['*'])
    m.add_subsystem('torque', TorqueGroup(num_nodes=nn), promotes=['*'])
    m.add_subsystem('empirical', EmpiricalCorrectionsGroup(), promotes=['*'])
    m.add_subsystem('performance', RotorPerformanceGroup(), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('theta_0', 17.5)
    return p


p = build()
p.final_setup()
print('--- the six groups ---')
for s in p.model._subsystems_myproc:
    if s.name != '_auto_ivc':
        print(f'  {s.name:12s} {type(s).__name__}')

p.run_model()

print('\n--- steps 1 to 21 against Figure 1.45, p. 76 ---')
rows = (('CT_no_tip_loss', 0.00741, '{:.5f}'), ('B', 0.98, '{:.4f}'),
        ('CT', 0.00719, '{:.5f}'), ('CQ0', 1.11e-4, '{:.4e}'),
        ('CQi', 4.59e-4, '{:.4e}'), ('swirl_ratio', 0.017, '{:.4f}'),
        ('DL', 7.2, '{:.2f}'), ('CT_sigma', 0.0846, '{:.4f}'),
        ('DL_CT_sigma', 0.61, '{:.3f}'), ('power_factor', 1.05, '{:.4f}'),
        ('CQ_sigma', 0.0070, '{:.5f}'), ('T', 20400.0, '{:.0f}'),
        ('power_hp', 1990.0, '{:.0f}'))
print(f"{'quantity':>16}{'computed':>14}{'book':>12}{'error %':>10}")
for name, bk, fmt in rows:
    v = p.get_val(name)[0]
    print(f'{name:>16}{fmt.format(v):>14}{bk:12.5g}{100 * (v / bk - 1):10.2f}')

print(f"\n  F.M.          = {p.get_val('FM')[0]:.3f}")
print(f"  power loading = {p.get_val('power_loading')[0]:.2f} lb/hp")
print(f"  shaft torque  = {p.get_val('Q')[0]:.0f} ft-lb")

print('\n--- thrust and power against collective ---')
print(f"{'theta_0':>8}{'T lb':>9}{'hp':>8}{'FM':>7}{'lb/hp':>8}")
for th0 in (12.5, 15.0, 17.5, 20.0, 21.0, 22.5):
    p.set_val('theta_0', th0)
    p.run_model()
    print(f"{th0:8.1f}{p.get_val('T')[0]:9.0f}{p.get_val('power_hp')[0]:8.0f}"
          f"{p.get_val('FM')[0]:7.3f}{p.get_val('power_loading')[0]:8.2f}")

print('\n--- same thrust, three tip speeds ---')
print(f"{'V_tip':>7}{'theta_0':>9}{'T lb':>9}{'hp':>8}{'FM':>7}")
for vt in (600.0, 650.0, 700.0):
    p.set_val('V_tip', vt)
    for th0 in np.linspace(14.0, 22.0, 81):        # crude match on T = 20,000
        p.set_val('theta_0', th0)
        p.run_model()
        if p.get_val('T')[0] >= 20000.0:
            break
    print(f"{vt:7.0f}{th0:9.2f}{p.get_val('T')[0]:9.0f}"
          f"{p.get_val('power_hp')[0]:8.0f}{p.get_val('FM')[0]:7.3f}")

print('\n--- total derivatives of the dimensional outputs ---')
p.set_val('V_tip', 650.0)
p.set_val('theta_0', 17.5)
p.model.add_design_var('theta_0')
p.model.add_design_var('V_tip')
p.model.add_design_var('R')
p.model.add_objective('power_hp')
p.model.add_constraint('T', equals=20000.0)
p.setup(force_alloc_complex=True)
p.set_val('theta_0', 17.5)
p.run_model()
for key, val in p.check_totals(method='fd', compact_print=True,
                               out_stream=None).items():
    fwd = val.get('J_fwd', val.get('J_rev'))
    of, wrt = key[0].split('.')[-1], key[1].split('.')[-1]
    print(f'  d({of})/d({wrt}) = {fwd.ravel()[0]:12.5e}   '
          f"(fd {val['J_fd'].ravel()[0]:12.5e})")

out = plot_figure_145(
    p, pathlib.Path(__file__).with_name('fig145_complete.png'),
    title='Example helicopter main rotor, complete method, steps 1 to 21',
    annotations=case_annotations(p), B=p.get_val('B')[0])
print('\nfigure written:', out)
