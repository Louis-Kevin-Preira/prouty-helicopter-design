"""
Validation of HoverRotorGroup in both modes.

Anchors, example helicopter, Figure 1.45 p. 76 and p. 669-670.
"""

import numpy as np
import openmdao.api as om

from prouty.hover.hover_rotor_group import HoverRotorGroup


def build(**opts):
    p = om.Problem()
    p.model.add_subsystem('rotor', HoverRotorGroup(**opts), promotes=['*'])
    p.setup(force_alloc_complex=True)
    return p


print('--- analysis mode, one line to build the whole method ---')
p = build()
p.set_val('theta_0', 17.5)
p.run_model()
print(f"  T = {p.get_val('T')[0]:.0f} lb   (book 20,400)")
print(f"  hp = {p.get_val('power_hp')[0]:.0f}      (book 1,990)")
print(f"  F.M. = {p.get_val('FM')[0]:.3f}")

print('\n--- trim mode, same case solved backwards ---')
q = build(mode='trim')
q.set_val('T_target', 19923.0)
q.run_model()
print(f"  T_target = 19,923 lb  ->  theta_0 = {q.get_val('theta_0')[0]:.4f} deg"
      '   (analysis used 17.5)')
print(f"  T = {q.get_val('T')[0]:.1f} lb   residual "
      f"{q.get_val('T')[0] - 19923.0:+.2e}")
print(f"  Newton iterations : {q.model.rotor.nonlinear_solver._iter_count}")

print('\n--- trim over a thrust sweep ---')
print(f"{'T target':>10}{'theta_0':>9}{'hp':>7}{'FM':>7}{'iters':>7}")
for target in (12000.0, 16000.0, 20000.0, 24000.0, 28000.0):
    q.set_val('T_target', target)
    q.run_model()
    print(f"{target:10.0f}{q.get_val('theta_0')[0]:9.3f}"
          f"{q.get_val('power_hp')[0]:7.0f}{q.get_val('FM')[0]:7.3f}"
          f"{q.model.rotor.nonlinear_solver._iter_count:7d}")

print('\n--- power required against altitude at constant thrust ---')
print(f"{'alt ft':>8}{'theta_0':>9}{'hp':>7}{'FM':>7}")
q.set_val('T_target', 20000.0)
for alt in (0.0, 4000.0, 8000.0, 12000.0):
    q.set_val('altitude', alt)
    q.run_model()
    print(f"{alt:8.0f}{q.get_val('theta_0')[0]:9.3f}"
          f"{q.get_val('power_hp')[0]:7.0f}{q.get_val('FM')[0]:7.3f}")
q.set_val('altitude', 0.0)

print('\n--- tail rotor, p. 670, same class ---')
t = build(mode='trim', num_elements=10)
t.set_val('R', 6.5)
t.set_val('c_root', 1.0)
t.set_val('c_tip', 1.0)
t.set_val('r_1', 6.5)
t.set_val('r_cutout', 0.975)
t.set_val('b', 3.0)
t.set_val('theta_1', -5.0)
t.set_val('V_tip', 650.0)
t.set_val('T_target', 1500.0)
t.run_model()
print(f"  sigma = {t.get_val('sigma')[0]:.4f}   (book 0.146)")
print(f"  T = 1,500 lb  ->  theta_0 = {t.get_val('theta_0')[0]:.2f} deg   "
      f"hp = {t.get_val('power_hp')[0]:.0f}   F.M. = {t.get_val('FM')[0]:.3f}")

print('\n--- trim robustness from starting points inside the valid region ---')
print('  (theta must stay positive at every station, so theta_0 > -theta_1 = 10)')
for start in (11.0, 14.0, 17.5, 24.0):
    r = build(mode='trim', theta_0_bounds=(10.5, 25.0))
    r.set_val('T_target', 20000.0)
    r.set_val('theta_0', start)
    r.run_model()
    print(f'  start {start:5.1f} deg  ->  {r.get_val("theta_0")[0]:.4f} deg   '
          f'{r.model.rotor.nonlinear_solver._iter_count} iterations')

print('\n--- total derivatives, both modes ---')
p = build()
p.model.add_design_var('theta_0')
p.model.add_objective('power_hp')
p.setup(force_alloc_complex=True)
p.set_val('theta_0', 17.5)
p.run_model()
for key, val in p.check_totals(method='fd', compact_print=True,
                               out_stream=None).items():
    fwd = val.get('J_fwd', val.get('J_rev'))
    print(f"  analysis: d(hp)/d(theta_0) = {fwd.ravel()[0]:.5e}   "
          f"(fd {val['J_fd'].ravel()[0]:.5e})")

# In trim mode check_totals does not reconverge the Newton, so the reference
# is a manual central difference over two fully retrimmed solutions.
def trimmed_hp(v):
    r = build(mode='trim')
    r.set_val('T_target', 20000.0)
    r.set_val('V_tip', v)
    r.run_model()
    return r.get_val('power_hp')[0]

q = build(mode='trim')
q.model.add_design_var('V_tip')
q.model.add_objective('power_hp')
q.setup(force_alloc_complex=True)
q.set_val('T_target', 20000.0)
q.run_model()
J = q.compute_totals(of=['power_hp'], wrt=['V_tip'], return_format='flat_dict')
manual = (trimmed_hp(651.0) - trimmed_hp(649.0)) / 2.0
print(f"  trim:     d(hp)/d(V_tip)   = "
      f"{J[('power_hp', 'V_tip')].ravel()[0]:.5e}   "
      f"(retrimmed central difference {manual:.5e})")
