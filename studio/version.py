"""version.py - single source of the STG studio version.

The version shown in the main window reads this constant. This module imports
nothing; the git hash is added by `build_tag()` in `stg_studio.py`.

Versioning:

    major 1.x.x   user-visible major change, or old data no longer usable
    minor x.1.x   new feature
    patch x.x.1   bug fix or adjustment without behaviour change

The git short hash is the reliable identity (a version bump can be
forgotten, a hash cannot), so the UI shows both.
"""

STUDIO_VERSION = "1.0.0"
