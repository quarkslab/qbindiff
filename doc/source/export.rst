Match Results
=============

The diffing result is 1-to-1 matching that can be saved in various file formats for
further usage and processing. Namely, it can be saved in CSV or using BinDiff
SQLite format.

BinDiff
-------

QBinDiff results can be exported to a BinDiff compatible format. Then, the diff can be seen with the BinDiff graphic interface.

Given a ``differ`` object initialized, with two binaries to diffs, the diffing and export results can be saved as follows:

..  code-block:: python

    matches = differ.compute_matching()
    differ.export_to_bindiff('/path/to/output.BinDiff'))

CSV
---

If the diff, does not represent a binary diff, or for further processing the diff
can also be saved in .csv file.

TODO: We really have to write the CSV ourselves ? There is not utility functions?

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
