Export
======

As diffing two large binaries may require a significant amount of time, it may be useful to export and save QBinDiff results, to reuse them later. Several options are possible in order to do so.

BinDiff
-------

QBinDiff results can be exported to a BinDiff format. Then, the diff can be seen with the BinDiff graphic interface. 

Suppose you have initialized a differ object based on two binaries. Now, you can compute the matches and save the results.

..  code-block:: python

   matches = differ.compute_matching()
   differ.export_to_bindiff('/path/to/output.BinDiff'))
   
CSV
---

If you are not familiar with BinDiff and prefer to manipulate .csv file, as it was done by `Marcelli et al. <https://www.usenix.org/conference/usenixsecurity22/presentation/marcelli>`_, you can do this instead : 

..  code-block:: python

   matches = differ.compute_matching()
   
   f = open('/path/to/output.csv', 'w')
   writer = csv.writer(f)
   # idb_path_1 : path to the primary
   # fva_1 : function address inside the primary
   # func_name_1 : function name inside the primary
   # idb_path_2 : path to the secondary
   # fva_2 : function address inside the secondary
   # func_name_2 : function name inside the secondary
   # similarity : similarity measure of the function match
   # confidence : confidence of the match
   writer.writerow(('idb_path_1', 'fva_1', 'func_name_1', 'idb_path_2', 'fva_2', 'func_name_2', 'similarity', 'confidence'))

   for match in matches : 
   	writer.writerow((bina, hex(match.primary.addr), match.primary.name, binb, hex(match.secondary.addr), match.primary.name, match.similarity, match.confidence))
   f.close()
   


