"""Prouty's *Helicopter Performance, Stability and Control* as OpenMDAO models.

Each subpackage is one chapter of the book, implemented as differentiable
OpenMDAO components and groups with analytic partials throughout. Page and
figure references are carried in the module that uses them, so that any
equation can be traced back to its source without leaving the code.

    prouty.airfoil          Chapter 6, airfoils for rotor blades
    prouty.forward_flight   Chapter 3, aerodynamics of forward flight

Where the implementation departs from the printed text -- and it does, in
about a dozen places -- the reasoning sits in the module and the measurement
that justified it sits in docs/validation_forward_flight.md. Nothing was changed
because it looked wrong; each departure was measured against a book anchor, a
chart, or a wind tunnel result before being adopted, and the candidates that
failed are recorded alongside the ones that worked.

Three disagreements remain open and are documented rather than tuned away.
Calibrating them would mean fitting one chapter to another, which would hide
the very thing worth knowing.
"""

__version__ = '0.3.0'

__all__ = ['__version__']
