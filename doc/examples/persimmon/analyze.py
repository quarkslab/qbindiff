#!/bin/env python

import os, subprocess
from pathlib import Path

IDA_PATH = Path("path_to_IDA")  # Set correct path


if __name__ == "__main__":
    cwd = Path(os.getcwd())
    for file in cwd.glob("output/*.exe"):
        subprocess.run([IDA_PATH / "idat64", "-A", "-OQuokkaAuto:TRUE", file])
