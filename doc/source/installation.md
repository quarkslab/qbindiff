# Installation

First clone the repository

```bash
git clone https://gitlab.qb/machine_learning/qbindiff.git qbindiff
cd qbindiff
```

It is highly reccomended to install QBinDiff inside a python virtualenv

```bash
# Create the virtual environment
python -m venv venv

# Enter the virtual environment
. venv/bin/activate
```

Install QBinDiff is as easy as type

```bash
pip install .
```

## Backend installation

To use a preconfigured backend you should install the relative python module

### Quokka
Quokka can be installed with pip
```bash
pip install .[quokka]
```

For more information on how to compile and install Quokka see the official documentation
[here](https://github.com/quarkslab/quokka)

### BinExport
Follow the instructions [here](https://gitlab.qb/rmori/python-binexport)

```{warning}
There is another `python-binexport` module ([here](https://gitlab.qb/rdavid/python-binexport)) but it is not compatible with QBinDiff.

In the future the two projects will be merged into a single one
```