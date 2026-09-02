# prouty

Prouty, *Helicopter Performance, Stability and Control*, implemented as
OpenMDAO models for preliminary rotorcraft design and optimisation.

Every component carries its page reference. Departures from the printed text,
and disagreements found inside it, are measured and recorded in
`docs/validation_forward_flight.md` rather than silently corrected.

## Installed

| package | chapter | state |
|---|---|---|
| `prouty.hover` | 1, Aerodynamics of Vertical Flight | complete |
| `prouty.airfoil` | 6, Airfoils for Rotor Blades | complete |
| `prouty.forward_flight` | 3, Aerodynamics of Forward Flight | complete |

## Chapter 3 at a glance

Two interchangeable rotor models:

* **closed form** (G1), the equations of p. 167-200, instant, assumes a mean
  drag coefficient;
* **numerical** (G2), the blade element integration of p. 208-228, which
  resolves the disc element by element and calls the Chapter 6 airfoil.

`TrimConditionsGroup`, `FixedCollectiveTrimGroup` and `WindTunnelRotorGroup`
all take `rotor='closed_form'` or `'numerical'`. The choice matters most where
the induced torque is negative — a dive, an autorotation, a rotor at positive
shaft angle in a tunnel — because C_Q is then a small difference of cancelling
terms and a mean drag coefficient has no signal left to give. On the wind
tunnel case of Table 3.5 the closed form has the rotor *extracting* energy;
G2 does not.

```python
import numpy as np, openmdao.api as om
from prouty.forward_flight import TrimConditionsGroup

# the example helicopter of Table 3.3, p. 196
ROTOR = dict(V_tip=650.0, rho=0.002377, A_b=240.0, sigma=0.084883, R=30.0,
             theta_1=np.deg2rad(-10.0), a=6.0, gamma=8.05033, B=0.97,
             x_0=0.15, cd_bar=0.0100, i_s=0.0, a1s=0.0, l_T_R=1.23,
             delta_3=np.deg2rad(-30.0))

p = om.Problem()
p.model.add_subsystem('trim', TrimConditionsGroup(mode='level'), promotes=['*'])
p.setup()
for name, value in {**ROTOR, 'mu': 0.3, 'GW': 20000.0}.items():
    p.set_val(name, value)
p.run_model()

print(p.get_val('theta_0', units='deg'), p.get_val('hp_M'))
# [16.50] [1077.34]   Table 3.3 gives 15.8 deg and 1,373 hp; see the notes
```

The group has defaults for every input, but they describe no particular
helicopter and the trim will not converge from them. Give it a rotor.


## Validation notes

| file | covers |
|---|---|
| `docs/validation_forward_flight.md` | Chapter 3, five sections, 563 lines |
| `docs/validation_airfoil.md` | Chapter 6, figure-by-figure anchors |

## Validation

The suite anchors against the book's own tables and figures, and — for the
wind tunnel case — against measurement. Known disagreements are asserted as
such, so that a change which happens to fix one is noticed rather than
absorbed.

```
pip install -e ".[dev]"
pytest -m "not slow"        # 92 tests, 30 s
pytest                      # 118 tests, about 6 min
```

The tests marked `slow` converge G2 repeatedly — chart generation, the wind
tunnel case, the collective sweep, the total derivatives. Skipping them keeps
the fast suite usable as a working loop; the full suite is for before a
commit.

## What is not done

* the shed-vorticity correction of p. 224-225 is off by default: it is correct
  outboard of r/R = 0.7 and wrong at the root, and the 1/rev assumption it
  rests on holds nowhere on the disc;
* C_H/sigma sits a factor of 1.9 below chart 3 with no cause found in nine
  measured eliminations; the resumption point is written up in the notes;
* the isolated rotor charts can be generated but the full set of 45 plates has
  not been run.

## Layout

```
src/prouty/
    hover/              Chapter 1, 32 modules
    airfoil/            Chapter 6, 13 modules
    forward_flight/     Chapter 3, 58 modules
    design/             Chapter 9, stub
    performance/        Chapter 7, stub
    stability/          Chapter 8, stub
tests/                  153 tests, 26 marked slow
docs/                   validation notes, one file per chapter
validation/             figure-reproduction scripts
scripts/                documented disagreements, kept runnable
```
