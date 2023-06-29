Install
=======

Pip package
-----------

You can install QBinDiff directly from pip package

..  code-block:: bash

    pip install qbindiff

Manual Installation
-------------------

If you prefer to install QBinDiff manually you first have to clone the official repository:

..  code-block:: bash
    
    git clone https://github.com/quarkslab/qbindiff qbindiff
    cd qbindiff

It is highly recommended to install QBinDiff inside a python virtual env:

..  code-block:: bash

    python -m venv venv
    . venv/bin/activate

Finally install QBinDiff via pip:

..  code-block:: bash

    pip install .


Optional Dependencies
---------------------

Apart from the required python dependencies (automatically managed by pip) there are some optional dependencies that you might want to use.
The optional dependencies are divided in several categories:

- **Backend**\: Related to the backend loader for the binary disassembly
- **Result exporter**\: About exporting the diffing result in some kind of specific file format

Backends
~~~~~~~~

In order to diff binaries, QBinDiff relies on different backends. If you want to use a specific backend, you have to install it in the same virtual env where it is QBinDiff.

The available backends are:

- `Quokka <https://github.com/quarkslab/quokka>`_, provided by Quarkslab
- `BinExport <https://github.com/google/binexport>`_ , developed by Google. Through the python bindings provided by Quarkslab with `python-binexport <https://github.com/quarkslab/python-binexport>`_
- IDA PRO (to be used directly within IDA). Through the python package `idascript <https://github.com/quarkslab/idascript>`_ also provided by Quarkslab

To install Quokka:

..  code-block:: bash

    pip install quokka-project

To install python-binexport:

..  code-block:: bash

    pip install python-binexport

To install idascript:

..  code-block:: bash

    pip install idascript

For more informations about these packages, see the official documentations:

* `Quokka <https://github.com/quarkslab/quokka>`_
* `BinExport <https://github.com/google/binexport>`_ and `python-binexport <https://github.com/quarkslab/python-binexport>`_
* `idascript <https://github.com/quarkslab/idascript>`_


Diffing result exporter
~~~~~~~~~~~~~~~~~~~~~~~

QBinDiff by default doesn't export the result of the diffing in any way, it is only possible to access it through the API.
However it optionally supports exporting the result in the common file format that `BinDiff <https://www.zynamics.com/bindiff.html>`_ uses, hence you will be able to use the BinDiff tool to visualize the output of the binary diffing.

To be able to do that it uses the python package `python-bindiff <https://github.com/quarkslab/python-bindiff>`_ provided by Quarkslab.

You can install it by using pip:

..  code-block:: bash

    pip install python-bindiff
