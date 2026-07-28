"""Augury signal service.

Owns the market pollers, the stance classifiers, the S(t) aggregation, and the
reference implementations of the LMSR and the lead-lag econometrics.

"Reference" is load-bearing: the LMSR here and the one in the C++ engine are two
independent implementations of the same equations, and the Catch2 suite asserts
they agree on the golden vectors in `schemas/testdata/`. Same for the R analytics
module. If this module's math changes, those vectors must be regenerated and the
other languages re-checked against them.
"""

__version__ = "0.2.0"
