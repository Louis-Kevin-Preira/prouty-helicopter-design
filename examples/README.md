# Examples

Standalone, runnable scripts — not tests. Each one reproduces a book figure or
explores a question that a `pytest` assertion doesn't fit well (a chart study,
a sensitivity sweep). Run any of them directly:

```
python3 examples/hover/reproduce_figure_1_45.py
```

Organised by the chapter/topic it exercises, mirroring `src/prouty/`:

```
examples/
    hover/            Chapter 1
    vertical/          Chapter 2
    airfoil/           Chapter 6
    forward_flight/    Chapter 3
```

Any figure a script writes is saved next to the script itself and is not
committed (`examples/**/*.png` is git-ignored) — regenerate rather than
compare stale images. The correctness checks that matter are in `tests/`;
these scripts are for looking at the result.
