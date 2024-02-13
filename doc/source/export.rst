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
    differ.export_to_bindiff('/path/to/output.BinDiff')

CSV
---

If the diff, does not represent a binary diff, or for further processing the diff
it can also be saved in .csv file.
This is the default file format as it is very lightweight and fast to generate.

It can either be obtained using the CLI option `-ff csv` or by calling the right API as follows:

..  code-block:: python

    from qbindiff.loader.types import FunctionType

    matches: Mapping = differ.compute_matching()

    # This only exports base fields (address, similarity, confidence)
    matches.to_csv("/path/to/output.csv")

    # Add extra "name" field
    matches.to_csv("/path/to/output.csv", "name")

    # Add extra "name" field and custom field
    matches.to_csv(
        "/path/to/output.csv",
        "name",
        ("is_library", lambda f: f.type == FunctionType.library)
    )
