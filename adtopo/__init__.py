"""adtopo — shared library for the ELA × SCAN functional-network topography analyses.

Installable package (see pyproject.toml) holding the configuration, the
random-effects / OLS model helpers, and small stats/FC utilities that the
numbered analysis scripts under ``code/`` import. Installing it editable
(``pip install -e .``) is what lets those scripts do ``from adtopo.config import
...`` from anywhere, replacing the previous per-script ``sys.path`` manipulation.
"""

__version__ = "0.1.0"
