Install
=======

QBinDiff
--------

You can install QBinDiff by cloning the official repository : 

..  code-block:: bash
    
    git clone qbindiff_repo qbindiff
    cd qbindiff
    
[TODO:add the right address]

It is highly recommended to install QBinDiff inside a python virtual env : 


..  code-block:: bash

    python -m venv venv
    source venv/bin/activate

Install QBinDiff :

..  code-block:: bash

    pip install .
   

If you prefer to install QBinDiff directly using pip : 


..  code-block:: bash

    python -m venv venv
    source venv/bin/activate
    pip install qbindiff

Backends
--------

In order to diff binaries, QBinDiff relies on different backends. If you want to use a specific backend, you have to install it in the same virtual env than QBinDiff. 

Available backends are `Quokka <https://github.com/quarkslab/quokka>`_, provided by Quarkslab and `BinExport <https://github.com/google/binexport>`_ , developped by Google. To ease BinExport usage, especially with QBinDiff, Quarkslab has also developped a binexport python package [TODO:add link]. 

To install Quokka : 


..  code-block:: bash

    pip install quokka-project

To install BinExport 

..  code-block:: bash

    pip install python-binexport

For more informations about these packages, see the official documentations :

* `Quokka <https://quarkslab.github.io/quokka/>`_
* BinExport : [TODO:add]






.. toctree::
   :maxdepth: 2
   :caption: Contents:

