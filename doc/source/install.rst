Install
=======

Pip package
-----------

It can be installed directly from pip:

..  code-block:: bash

    $ pip install qbindiff

Manual Installation
-------------------

For installing it manually, one should first clone the official repository:

.. TODO Check the prerequisites for building it manually (gcc, cython, etc...)

..  code-block:: bash
    
    git clone https://github.com/quarkslab/qbindiff
    cd qbindiff

It is highly recommended to install QBinDiff inside a python virtual env:

..  code-block:: bash

    python -m venv venv
    . venv/bin/activate

Then it can be built with:

..  code-block:: bash

    pip install .


Optional Backend Dependencies
-----------------------------

The various program representation backend that can be used are to
be installed separately (in the same virtualenv than QBinDiff).

Available backends are:

- `Quokka <https://github.com/quarkslab/quokka>`_, provided by Quarkslab

..  code-block:: bash

    pip install qbindiff[quokka]

- `BinExport <https://github.com/google/binexport>`_ , developed by Google. Through the python bindings provided by Quarkslab with `python-binexport <https://github.com/quarkslab/python-binexport>`_

..  code-block:: bash

    pip install qbindiff[binexport]

- IDA Pro (to be used directly within IDA) through `idascript <https://github.com/quarkslab/idascript>`_.

..  code-block:: bash

    pip install qbindiff[ida]

You can install multiple backend loaders by using a comma to separate them. Ex:

.. code-block:: bash
    pip install qbindiff[quokka,binexport,ida]
