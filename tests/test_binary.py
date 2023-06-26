import json
from pathlib import Path

from qbindiff import Program, QBinDiff, Distance
from qbindiff.features import WeisfeilerLehman, Constant, Address

BASE_TEST_PATH = Path("tests/data")


class TestQuokka:
    """Regression tests for Quokka"""

    base_path = BASE_TEST_PATH / "binaries"
    # We need the address heuristic with a very low weight to discern between nearly
    # identical functions
    features = [
        (WeisfeilerLehman, 1.0, Distance.cosine),
        (Constant, 1.0, Distance.canberra),
        (Address, 0.01, Distance.canberra),
    ]

    def path(self, p: str) -> Path:
        return self.base_path / p

    def test_x86_binary(self):
        primary = Program.from_quokka(
            self.path("test-bin.quokka"), self.path("test-bin")
        )
        secondary = Program.from_quokka(
            self.path("test-bin2.quokka"), self.path("test-bin2")
        )
        differ = QBinDiff(
            primary,
            secondary,
            sparsity_ratio=0.75,
            tradeoff=0.75,
            epsilon=0.5,
            distance=Distance.canberra,
        )

        for f, w, d in self.features:
            differ.register_feature_extractor(f, w, distance=d)

        differ.process()
        mapping = differ.compute_matching()
        output = {(match.primary.addr, match.secondary.addr) for match in mapping}

        with open(self.path("test-bin-quokka.results")) as fp:
            expected = json.load(fp)

        assert not (output ^ set(tuple(e) for e in expected))
