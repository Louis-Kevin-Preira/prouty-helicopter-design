# Prouty Helicopter Design

OpenMDAO implementation of the helicopter analysis and preliminary design methods
described in **R.W. Prouty, _Helicopter Performance, Stability and Control_**.

Every component carries the page numbers of the equations it implements, so the
code can be read side by side with the book. Analytic derivatives are provided
throughout, which makes the models usable inside gradient-based optimisation.

## Status

| Module | Book chapter | Pages | Status |
|---|---|---|---|
| `prouty.airfoil` | 6 — Airfoils for Rotor Blades | 426-434 | implemented, validated |
| `prouty.hover` | 1 — Aerodynamics of Hovering Flight | — | planned |
| `prouty.forward_flight` | 3 — Aerodynamics of Forward Flight | — | planned |
| `prouty.performance` | 4 — Performance Analysis | — | planned |
| `prouty.stability` | 8 — Stability and Control Analysis | — | planned |
| `prouty.design` | 10 — Preliminary Design | — | planned |

## Installation

```bash
git clone https://github.com/<user>/prouty-helicopter-design.git
cd prouty-helicopter-design
pip install -e ".[validation,dev]"
```

## Quick start

```python
import numpy as np
import openmdao.api as om
from prouty.airfoil import AirfoilHoverGroup

nn = 3
p = om.Problem()
p.model.add_subsystem('af', AirfoilHoverGroup(num_nodes=nn), promotes=['*'])
p.setup()
p.set_val('M', [0.2, 0.5, 0.7])
p.set_val('alpha', [8.0, 8.0, 8.0])
p.run_model()

print(p.get_val('cl'))   # [0.7834 0.8581 0.7048]
print(p.get_val('cd'))   # [0.0132 0.0215 0.1190]
```

## Module `prouty.airfoil`

Represents NACA 0012 lift and drag as closed-form functions of angle of attack
and Mach number, following the section *Representing Airfoil Data with
Equations* (p. 426). Two groups are provided.

### `AirfoilHoverGroup` — inputs `M`, `alpha` [deg] → `cl`, `cd`

| Component | Role | Pages |
|---|---|---|
| `LiftModelCoefsComp` | `a`, `alpha_L`, `K1`, `K2` from Mach | 427-430 |
| `LiftCoefComp` | assembles `cl` | 428, 430 |
| `DragModelCoefsComp` | `alpha_D`, `K3`, `K4`, `delta_cd_M` from Mach | 432-433 |
| `IncompDragHoverComp` | asymmetric 5-term drag series | 432 |
| `DragCoefHoverComp` | assembles `cd` | 432-433 |

### `AirfoilForwardFlightGroup` — inputs `M`, `alpha_raw` [deg] → `cl`, `cd`

Adds the two modifications required for forward flight (p. 433): an even-power
drag series, since negative angles occur on the advancing tip, and an extension
of both coefficients over the full 0-360 deg range, since inboard elements on
the retreating side operate well beyond stall.

| Component | Role | Pages |
|---|---|---|
| `AlphaWrapComp` | wraps angle onto [0,360) and [-180,180) | 214, 433 |
| `IncompDragFwdComp` | even-power 4-term drag series | 433 |
| `LiftCoefFwdComp` | 7-segment lift assembly | 433 |
| `DragCoefFwdComp` | 3-segment drag assembly | 434 |

`LiftModelCoefsComp`, `LiftCoefComp` and `DragModelCoefsComp` are shared
unchanged between the two groups: the Mach-dependent laws are identical, only
the angle handling differs.

## Design choices

**Separation of concerns.** Components split into two families: those computing
coefficients that depend on Mach only, and those assembling a coefficient as a
function of angle. This is what allows three components to be reused verbatim
between hover and forward flight, with no duplicated equation.

**Smoothing.** The book defines the model piecewise. Discontinuous branches are
joined by a cubic smoothstep so that the assembled functions are C1 and safe for
gradient-based optimisation:

- Mach break at 0.725 in the lift model: band `[0.725, 0.745]`, deliberately
  offset rather than centred, because `(M - 0.725)**0.44` has infinite slope at
  the break and the quadratic onset of the smoothstep cancels it.
- Mach break at 0.725 in the drag model: band `[0.715, 0.735]`, centred, since
  `K3` has a genuine value jump (0.00066 to 0.00035) and no singularity.
- Segment boundaries in forward flight: half-width 2 deg.

Stall onset in `cl` and drag divergence in `cd` need **no** smoothing: their
exponents `K2 = 2.05 - 0.95M` and `K4 = 2.54` both exceed 1, so the piecewise
functions are already C1 at the junction. A clip to zero is sufficient.

**Vectorisation.** All components take `num_nodes`-shaped arrays, so a full
blade-element by azimuth mesh is evaluated in a single pass.

## Validation

```bash
python validation/validate_hover_group.py   # against Fig. 6.43, p. 427
python validation/validate_fwd_group.py     # against Fig. 6.47, p. 434
```

Results are summarised in [`docs/validation_notes.md`](docs/validation_notes.md).
Headlines:

- `K1` reproduces the tabulated values of p. 429-430 exactly at M = 0.2, 0.5, 0.7.
- Segment values of p. 433-434 are reproduced exactly (`cl = ±1.15` at 45/315 deg,
  `cd = 2.05` at 90/270 deg, `cd = 0.01` at 180 deg).
- Lift antisymmetry and drag symmetry about 0 deg hold to 3e-14.
- `check_partials` and `check_totals` clean on every component and both groups.

## Known limitations of the model

These are properties of Prouty's fits, not of the implementation. They are
documented because they matter when sweeping a full (alpha, M) domain.

**Lift above M = 0.725.** `K1 = 0.0575 - 0.144(M - 0.725)**0.44` decreases with
Mach and reaches zero near M = 0.83, so the model produces no stall above that
speed and the high-Mach curves cross the lower ones. The value 0.0575 (rather
than the 0.575 printed in the constants list on p. 430) is confirmed by
continuity at the break: 0.05761 from below against 0.05750 from above. Trust
the lift model for `M <= 0.725` unrestricted, and for `alpha <= 6 deg` above it.
The `clip_K1` option on `LiftCoefComp` (default `True`) bounds `K1` at zero.

**Drag across the divergence break.** `K3` drops while the zero-angle term only
compensates at `alpha = 0`, so at `alpha = 8 deg` drag decreases from 0.119 at
M = 0.70 to 0.087 at M = 0.80. High Mach and high incidence do not coexist on a
rotor, so this region is rarely visited.

**Segment junction at 20 deg.** The high-angle data of Fig. 6.47 was measured at
low Mach (p. 433), whereas the generated branch depends strongly on Mach. The
two branches nearly meet at M = 0.1 but differ by a factor of 3.6 at M = 0.5.
The junction is only physically consistent at low Mach — which is where
inboard retreating-blade elements actually operate.

## References

R.W. Prouty, *Helicopter Performance, Stability and Control*, Krieger, 1990.
Chapter 6, "Airfoils for Rotor Blades", p. 426-434; quadrant convention p. 214;
lift coefficient bounds in the reverse flow region p. 221.
