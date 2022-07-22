import logging
import qbindiff
from collections import defaultdict
from qbindiff.loader.types import LoaderType
from qbindiff.features import FEATURES
from typing import Any

FEATURES_KEYS = {x.key: x for x in FEATURES}
ENABLED_FEATURES = (
    ("wlgk", 1.0, "cosine", {"max_passes": 1}),
    ("fname", 3.0),
    ("dat", 1.0),
    ("cst", 1.0),
    ("addr", 0.01),
)


def main():
    global result
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    primary = qbindiff.Program(
        LoaderType.quokka,
        "./output/5012647-ntoskrnl.quokka",
        "./output/5012647-ntoskrnl.exe",
    )
    secondary = qbindiff.Program(
        LoaderType.quokka,
        "./output/5013941-ntoskrnl.quokka",
        "./output/5013941-ntoskrnl.exe",
    )

    differ = qbindiff.QBinDiff(
        primary, secondary, distance="jaccard-strong", sparsity_ratio=0.999, sparse_row=True
    )

    for data in ENABLED_FEATURES:
        feature, weight = data[0], data[1]
        distance, params = None, {}
        if len(data) > 2:
            distance = data[2]
            if len(data) > 3:
                params = data[3]
        if feature not in FEATURES_KEYS:
            logging.warning(f"Feature '{feature}' not recognized - ignored.")
            continue
        differ.register_feature_extractor(
            FEATURES_KEYS[feature], float(weight), distance=distance, **params
        )

    result = differ.compute_matching()

    # Enable this to export the result to the BinDiff file format
    # ~ differ.export_to_bindiff('./result.BinDiff')

    totSame = 0  # Total number of non modified functions
    primaryUnmatched = set()  # Deleted functions
    secondaryUnmatched = set()  # New functions
    funcsModified = set()  # Modified functions

    for addr, _ in primary.items():
        primaryUnmatched.add(addr)
    for addr, _ in secondary.items():
        secondaryUnmatched.add(addr)

    for match in result:
        if match.similarity == 1:
            totSame += 1
            primaryUnmatched.remove(match.primary.addr)
            secondaryUnmatched.remove(match.secondary.addr)
        elif match.similarity < 0.3 and match.confidence < 0.7:
            # Non valid match
            pass
        else:
            primaryUnmatched.remove(match.primary.addr)
            secondaryUnmatched.remove(match.secondary.addr)
            funcsModified.add(match)

    with open("./report.txt", "w") as f:
        print("=== REPORT ===")
        print(f"\tSimilarity: {result.normalized_similarity}")
        print(f"\t{totSame} functions have not been modified")
        print(f"\t{len(funcsModified)} functions have been modified")
        print(f"\t{len(primaryUnmatched)} functions have been deleted")
        print(f"\t{len(secondaryUnmatched)} functions have been added")

        print("-- MODIFIED FUNCTIONS --")
        for m in funcsModified:
            print(
                f"\t{m.primary.name}  ->  {m.secondary.name}  similarity: {m.similarity} confidence: {m.confidence}"
            )

        print("-- DEAD FUNCTIONS --")
        for f in primaryUnmatched:
            print(f"\t{primary[f].name}")

        print("-- NEW FUNCTIONS --")
        for f in secondaryUnmatched:
            print(f"\t{secondary[f].name}")


if __name__ == "__main__":
    main()
