Export
======

As diffing two large binaries may require a significant amount of time, it may be useful to export and save QBinDiff results, to reuse them later. Several options are possible in order to do so.

BinDiff
-------

QBinDiff results can be exported to a BinDiff compatible format. Then, the diff can be seen with the BinDiff graphic interface.

Suppose you have initialized a differ object based on two binaries. Now, you can compute the matches and save the results.

..  code-block:: python

    matches = differ.compute_matching()
    differ.export_to_bindiff('/path/to/output.BinDiff'))

CSV
---

If you are not familiar with BinDiff and prefer to manipulate .csv file you can do this instead:

..  code-block:: python

    import csv

    matches = differ.compute_matching()

    with open('/path/to/output.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow((
            'path_primary',
            'func_addr_primary',
            'func_name_primary',
            'path_secondary',
            'func_addr_secondary',
            'func_name_secondary',
            'similarity',
            'confidence'
        ))

        for match in matches:
            writer.writerow((
                differ.primary.name,
                hex(match.primary.addr),
                match.primary.name,
                differ.secondary.name,
                hex(match.secondary.addr),
                match.primary.name,
                match.similarity,
                match.confidence
            ))
