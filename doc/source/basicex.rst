Basic Example
=============

Command-line 
------------

..  code-block:: bash

    Usage: qbindiff [OPTIONS] <primary file> <secondary file>
    #[TODO:add the last version of qbindiff] 


* `-l`, `--loader` The backend loader that should be used to load the disassembly. The possible values are `binexport, quokka`. The default value is `binexport`.

BinExport is more flexible since it can be used with IDA, Ghidra and BinaryNinja but it generally yields to worst results since it doesn't export as much informations as Quokka and some features don't work with it. Quokka on the other hand usually produces better results but it's only compatible with IDA. If you want to use a different backend loader you can specify a {ref}`usage/custom_backend`

* `-f`, `--features` Specify a single feature that you want to use that will contribute to populate the similarity matrix. You can specify multiple features by using more `-f` arguments. Look at the help page to see the complete list of features. For each feature you must specify the name and optionally a weight and/or a distance that should be used, the syntax is one of the following:
  * `<name>` The default weight is 1.0 and the default distance (`-d`) is used
  * `<name>:<weight>` The weight must be a floating point value > 0. The default distance (`-d`) is used
  * `<name>:<distance>` Look at `-d` for the list of allowed distances. The default weight is 1.0
  * `<name>:<weight>:<distance>`
  
  To know more in detail how the features are combined together to generate a similarity matrix look at {ref}`features`.
* `-fopt`, `--feature-option` Set and option to a feature previously enabled. Not all the features have configurable options. To have a list of the accepted options for a particular feature look into its description

* `-n`, `--normalize` Normalize the Call Graph by removing some of the edges/nodes that should worsen the diffing result. **WARNING:** it can potentially lead to a worse matching. To know the details of the normalization step look at {ref}`normalization`

* `-d`, `--distance` Set the default distance that should be used by the features. The possible values are `canberra, correlation, cosine, euclidean, jaccard-strong`. The default one is `canberra`. To know the details of the jaccard-strong distance look here {ref}`jaccard-strong`

* `-s`, `--sparsity-ratio` Set the density of the similarity matrix. This will loose some information (hence decrease accuracy) but it will also increase the performace. `0.999` means that the 99.9% of the matrix will be filled with zeros. The default value is `0.75`

* `-sr`, `--sparse-row` If this flag is enabled the density value of the sparse similarity matrix will affect each row the matrix. That means that each row of the matrix has the defined sparsity ratio. This guarantees that there won't be rows that are completely erased and filled with zeros.

* `-t`, `--tradeoff` Tradeoff between function content (near 1.0) and call-graph topology information (near 0.0). The default value is `0.75`

* `-e`, `--epsilon` Relaxation parameter to enforce convergence on the belief propagation algorithm. For more information look at {ref}`belief-propagation`. The default value is `0.50`

* `-i`, `--maxiter` Maximum number of iteration for the belief propagation. The default value is `1000`

* `-e1`, `--executable1` Path to the primary raw executable. Must be provided if using the quokka loader, otherwise it is ignored

* `-e2`, `--executable2` Path to the secondary raw executable. Must be provided if using the quokka loader, otherwise it is ignored

* `-o`, `--output` Path to the output file where the result of the diffing is stored

* `-ff`, `--file-format` The file format of the output file. The only supported format for now is `bindiff`. For more information look at {ref}`bindiff` The default value is `bindiff`

* `--enable-cortexm` Enable the usage of the cortex-m extension when disassembling. Only relevant if using the binexport loader.

* `-v`, `--verbose` Increase verbosity. Can be supplied up to 3 times.


.. warning::
   The less sparse the similarity matrix is the better accuracy we achieve but also the slower the algorithm becomes. In general the right value must be tuned for each use case.
   Also note that after a certain threshold reducing the sparsity of the matrix **won't** yield better results.

Some examples are displayed below : 

..  code-block:: bash

    qbindiff -l quokka -e1 binary-primary.exe -e2 binary-secondary.exe \
         binary-primary.qk binary-secondary.qk \
         -f wlgk:cosine \
         -f fname:3 \
         -f dat \
         -f cst \
         -f addr:0.01 \
         -d jaccard-strong -s 0.999 -sr \
         -ff bindiff -o ./result.BinDiff -vv

..  code-block:: bash

    qbindiff binary-primary.BinExport binary-secondary.BinExport \
         -f wlgk:cosine \
         -f fname:3 \
         -f addr:0.01 \
         -s 0.7 \
         -t 0.5
         -ff bindiff -o ./result.BinDiff -vv
         
Python
------

QBinDiff cannot directly manipulate binaries : it relies on some backend files, either Quokka or BinExport. 
To export a binary to a Quokka file or a BinExport, see the corresponding documentation [TODO: add links]

For Quokka : 

..  code-block:: python

   prog = quokka.Program.from_binary('/path/to/bin',
                                  output_file='/path/to/output.quokka',                                      
                                  database_file='/path/to/db.i64')
                                  
.. warning::

   To directly export the binary to Quokka using the method *from_binary*, the IDA plugin has to be installed. See `Quokka tutorial  <https://quarkslab.github.io/quokka/tutorials/qb-crackme/01_load/>`_. 

For BinExport : [TODO]

Now that we have our exported backend files, we can start to use QBinDiff. The rest of this example uses BinExport as backend but it works the same way for Quokka.

..  code-block:: python

   import qbindiff
   from qbindiff import LoaderType 
   primary = qbindiff.Program(LoaderType.binexport, "/path/to/primary.BinExport")
   secondary = qbindiff.Program(LoaderType.binexport, "/path/to/secondary.BinExport")


At this point, we can create our differ object configuring all the **parameters** by passing them to the constructor

..  code-block:: python

    differ = qbindiff.QBinDiff(
    primary,
    secondary,
    distance="canberra",
    epsilon=0.5, 
    tradeoff=0.75, 
    normalize=False,
    sparsity_ratio=0.999,
    sparse_row=True,
    )
    
Next, we register **features** that we want to use for our task. 
[TODO:faux ! Erreur implem => voir dict de commands]

..  code-block:: python

   FEATURES_KEYS = {x.key: x for x in qbindiff.features.FEATURES}
   ENABLED_FEATURES = (
   ("wlgk", 1.0, "cosine", {"max_passes": 1}),
   ("fname", 3.0),
   ("dat", 1.0),
   ("cst", 1.0),
   ("addr", 0.01),
   )
   
   for data in ENABLED_FEATURES:
   	feature, weight = data[0], data[1]
   	distance, params = None, {}
   	if len(data) > 2:
   		distance = data[2]
   		if len(data) > 3:
   			params = data[3]
   	if feature not in FEATURES_KEYS:
   		print(f"Feature '{feature}' not recognized - ignored.")
   		continue
   	differ.register_feature_extractor(FEATURES_KEYS[feature], float(weight), distance=distance, **params)
   	
Now, our differ is ready to compute the matches.

..  code-block:: python

   result = differ.compute_matching()
   # Export the result to the BinDiff file format
   differ.export_to_bindiff('./result.BinDiff')
   # Iterate over all the matches
   for match in result:
   	print(match.primary.addr, match.secondary.addr, match.similarity, match.confidence)


Custom Backend
--------------

If you want to load the binaries with your own custom backend you can implement your own backend loader.

You have to code an implementation for all the classes that are found in `src/qbindiff/loader/backend/abstract.py`: [TODO:make a link or anchor?]
* `AbstractOperandBackend`
  * `__str__(self) -> str`
  * `@property immutable_value(self) -> int | None`
  * `@property type(self) -> int`
  * `is_immutable(self) -> bool`
* `AbstractInstructionBackend`
  * `@property addr(self) -> Addr`
  * `@property mnemonic(self) -> str`
  * `@property references(self) -> dict[ReferenceType, list[ReferenceTarget]]`
  * `@property operands(self) -> Iterator[AbstractOperandBackend]`
  * `@property groups(self) -> list[int]`
  * `@property id(self) -> int`
  * `@property comment(self) -> str`
  * `@property bytes(self) -> bytes`
* `AbstractBasicBlockBackend`
  * `@property addr(self) -> Addr`
  * `@property instructions(self) -> Iterator[AbstractInstructionBackend]`
* `AbstractFunctionBackend`
  * `@property basic_blocks(self) -> Iterator[AbstractBasicBlockBackend]`
  * `@property addr(self) -> Addr`
  * `@property graph(self) -> networkx.DiGraph`
  * `@property parents(self) -> set[Addr]`
  * `@property children(self) -> set[Addr]`
  * `@property type(self) -> FunctionType`
  * `@property name(self) -> str`
* `AbstractProgramBackend`
  * `@property functions(self) -> Iterator[AbstractFunctionBackend]`
  * `@property name(self) -> str`
  * `@property structures(self) -> list[Structure]`
  * `@property callgraph(self) -> networkx.DiGraph`
  * `@property fun_names(self) -> dict[str, Addr]`

Most of the methods are self-explanatory but if you want to know more look at the docstring in the file `src/qbindiff/loader/backend/abstract.py` [TODO:make a link or anchor?]

Once you have your own implementation of the aforementioned classes you can create your qbindiff `Program` [TODO: make a link or anchor?] instance using your backend loader like this:

..  code-block:: python
   import qbindiff
   import MyCustomProgramBackend
   
   my_custom_backend_obj = MyCustomProgramBackend('my-program.exe')
   program = qbindiff.Program.from_backend(my_custom_backend_obj)

