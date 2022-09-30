Here there is all the documentation of QBinDiff and some examples/tutorials.

The documentation must be built with sphinx, follow the instructions below to know how
to do it.

The examples and the tutorials can either be found in [examples](examples) or they can
be built inside the documentation.

# Build
To build the documentation sphinx is required alongside other dependencies. To install
them use pip from the root of the project.

```commandline
python -m venv venv
. venv/bin/activate
pip install .[doc]
```

Then use [make.py](make.py) to build the documentation in the preferred format. Example

```commandline
python make.py html
```

The output can be found in `build/html/`

# Examples/Tutorial
 - None for the time being
