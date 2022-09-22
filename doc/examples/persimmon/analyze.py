#!/bin/env python3

import os, subprocess
import quokka
from pathlib import Path

IDA_PATH = Path("path_to_IDA")  # Set correct path if IDA is not already in PATH


if __name__ == "__main__":
    cwd = Path(os.getcwd())
    for file in cwd.glob("output/*.exe"):
        quokka.Program.from_binary(file)
