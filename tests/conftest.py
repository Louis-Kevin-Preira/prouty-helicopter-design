"""Test configuration.

OpenMDAO writes a report directory for every ``Problem`` it sets up, named
after the calling script. The suite creates thousands of Problems -- many
tests build one per parametrised case -- so left on, a full run leaves
gigabytes of ``*_out`` directories beside the source. ``.gitignore`` keeps
them out of the repository but not off the disk.

The variable has to be set before ``openmdao`` is imported, which is why this
sits at the top of ``conftest.py`` rather than in a fixture: pytest imports
this module before any test module, and the test modules are what import
OpenMDAO.
"""

import os

os.environ.setdefault('OPENMDAO_REPORTS', '0')

# A handful of ``*_out`` directories still appear, holding only
# ``coloring_files``: that is the total-derivative colouring ``check_totals``
# triggers, a different mechanism from the reports and a few kilobytes rather
# than gigabytes. ``.gitignore`` already covers both.
