"""
Entry point shim for PyInstaller builds.

When running as an installed package, use instead:
    kazarr <command> [options]

Or import directly as a library:
    from kazar import process
"""

import os
import sys

# Determine if we are running in a PyInstaller bundle
if getattr(sys, "frozen", False):
    base_dir = sys._MEIPASS
    # Set the ECCODES_DEFINITION_PATH environment variable to the definitions directory in the bundle
    os.environ["ECCODES_DEFINITION_PATH"] = os.path.join(
        base_dir, "eccodes", "definitions"
    )

from kazarr.cli import main

if __name__ == "__main__":
    main()
