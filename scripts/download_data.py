"""Download the dataset named in configs/data.yaml into its raw-data directory.

The same as `pdeeg download --force`, kept for the instructions that name this script.
"""

from __future__ import annotations

import sys

from pdeeg.cli import main

if __name__ == "__main__":
    sys.exit(main([*sys.argv[1:], "download", "--force"]))
