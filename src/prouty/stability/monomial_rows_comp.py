"""Base component for the dimensional derivative tables of Chapter 9.

Tables 9.2, 9.3, 9.6 to 9.9, 9.11, 9.13, 9.15 and 9.16 all have the same
shape: each entry is a sum of monomials in the non-dimensional derivatives,
the dynamic scale ``rho A_b (Omega R)^2`` and the geometric offsets. Rather
than write each table's 15 to 35 outputs and their several hundred partials by
hand, a subclass declares the table once and this class derives both.

The derivative of a monomial is taken by removing one occurrence of the factor,
never by dividing by it, so offsets that are legitimately zero -- ``y_M = 0``
for a main rotor on the centreline -- differentiate correctly.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", pp. 566-595.
"""

from collections import Counter

import numpy as np
import openmdao.api as om

#: Units of ``dFORCE_dWRT``; moments get an extra ``ft``.
DERIVATIVE_UNITS = {
    'xdot': 'lbf*s/ft', 'ydot': 'lbf*s/ft', 'zdot': 'lbf*s/ft',
    'zddot': 'lbf*s**2/ft',
    'q': 'lbf*s/rad', 'p': 'lbf*s/rad', 'r': 'lbf*s/rad',
    'theta0': 'lbf/rad', 'A1': 'lbf/rad', 'B1': 'lbf/rad',
}


def reduce_monomial(coefficient, numerator, denominator):
    """Cancel factors appearing on both sides of a monomial.

    A reference can put the same input above and below the line -- Table
    9.11's ``dX/dzddot`` is ``dX/dzdot`` times
    ``(dalpha_H/dzddot)/(dalpha_H/dzdot)``, and ``dX/dzdot`` already carries
    ``dalpha_H/dzdot``. Left uncancelled the derivative would count only one
    of the two occurrences.
    """
    common = numerator & denominator
    return coefficient, numerator - common, denominator - common


def term(coefficient, factors, over=()):
    """One monomial: ``coefficient * prod(factors) / prod(over)``."""
    return reduce_monomial(coefficient, Counter(factors), Counter(over))


def ref(coefficient, row, factors=(), over=()):
    """A row written in terms of another row, as the book writes many of them.

    ``ref(-1.0, 'dX_dxdot', ('h_M',))`` is ``-(dX/dxdot) h_M``. :func:`expand`
    resolves these into plain monomials before the component sees them, so a
    table can be declared in the nested form it is printed in.
    """
    return 'ref', coefficient, row, Counter(factors), Counter(over)


def expand(rows):
    """Resolve every :func:`ref` into monomials in the inputs alone."""
    resolved = {}

    def visit(name, stack=()):
        if name in resolved:
            return resolved[name]
        if name in stack:
            raise ValueError(f'circular reference through {name}')
        terms = []
        for entry in rows[name]:
            if entry[0] != 'ref':
                terms.append(entry)
                continue
            _, coefficient, target, numerator, denominator = entry
            for inner_c, inner_n, inner_d in visit(target, stack + (name,)):
                terms.append(reduce_monomial(coefficient * inner_c,
                                             inner_n + numerator,
                                             inner_d + denominator))
        resolved[name] = terms
        return terms

    for name in rows:
        visit(name)
    return resolved


def factors_of(terms):
    """Sorted set of input names a list of monomials depends on."""
    names = set()
    for _, numerator, denominator in terms:
        names |= set(numerator) | set(denominator)
    return sorted(names)


def evaluate(monomial, inputs):
    coefficient, numerator, denominator = monomial
    value = coefficient
    for name, power in numerator.items():
        value = value * inputs[name] ** power
    for name, power in denominator.items():
        value = value / inputs[name] ** power
    return value


def differentiate(monomial, wrt, inputs):
    """Exact derivative of one monomial, without ever dividing by ``wrt``."""
    coefficient, numerator, denominator = monomial

    if wrt in numerator:
        numerator = Counter(numerator)
        power = numerator.pop(wrt)
        if power > 1:
            numerator[wrt] = power - 1
        return evaluate((coefficient * power, numerator, denominator), inputs)

    denominator = Counter(denominator)
    power = denominator[wrt]
    denominator[wrt] = power + 1
    return evaluate((-coefficient * power, numerator, denominator), inputs)


class MonomialRowsComp(om.ExplicitComponent):
    """Outputs declared as sums of monomials in the inputs.

    Subclasses provide :meth:`rows`, and optionally ``scalar_inputs`` and
    ``input_defaults``. Output units follow the ``dFORCE_dWRT`` naming, with
    ``R``, ``M`` and ``N`` treated as moments.
    """

    #: Inputs that are one value for the whole vector, not one per node.
    scalar_inputs = ()
    #: Non-unit default values, by input name.
    input_defaults = {}
    #: Units by input name; anything absent is declared dimensionless.
    input_units = {}

    def rows(self):
        """``{output name: [term(...), ...]}`` for this table."""
        raise NotImplementedError

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)
        self._rows = expand(self.rows())

        for name in factors_of([t for terms in self._rows.values()
                                for t in terms]):
            default = self.input_defaults.get(name, 1.0)
            units = self.input_units.get(name)
            if name in self.scalar_inputs:
                self.add_input(name, val=default, units=units)
            else:
                self.add_input(name, val=np.full(nn, default), units=units)

        for name, terms in self._rows.items():
            self.add_output(name, val=np.zeros(nn), units=self.units_of(name))
            for inp in factors_of(terms):
                cols = zeros if inp in self.scalar_inputs else ar
                self.declare_partials(name, inp, rows=ar, cols=cols)

    @staticmethod
    def units_of(name):
        force, wrt = name[1], name.split('_d', 1)[1]
        units = DERIVATIVE_UNITS[wrt]
        return units.replace('lbf', 'lbf*ft') if force in 'RMN' else units

    def compute(self, inputs, outputs):
        for name, terms in self._rows.items():
            outputs[name] = sum(evaluate(monomial, inputs)
                                for monomial in terms)

    def compute_partials(self, inputs, J):
        for name, terms in self._rows.items():
            for inp in factors_of(terms):
                J[name, inp] = sum(
                    differentiate(monomial, inp, inputs)
                    for monomial in terms
                    if inp in monomial[1] or inp in monomial[2])
