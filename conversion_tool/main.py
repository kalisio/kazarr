"""
Entry point shim for PyInstaller builds.

When running as an installed package, use instead:
    kazarr <command> [options]

Or import directly as a library:
    from kazar import process
"""

from kazarr.cli import main

if __name__ == "__main__":
    main()
