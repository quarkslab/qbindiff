import unittest, click, networkx, json, scipy, logging
import scipy.io
from pathlib import Path
from dataclasses import dataclass

import qbindiff
from qbindiff.features import FEATURES
from qbindiff.loader import LoaderType

BASE_TEST_PATH = Path("tests/data")


@dataclass
class Unit:
    primary: str
    secondary: str
    result: str
    loader: LoaderType = None
    similarity: str = None
    primary_exe: str = None
    secondary_exe: str = None


class load_sim_matrix:
    def __init__(self, path: Path):
        self._path = path

    def __call__(self, sim_matrix, *_, **__):
        sparse_sim_matrix = scipy.io.mmread(self._path)
        sim_matrix[:] = sparse_sim_matrix.toarray()


class BinaryTest(unittest.TestCase):
    """Regression Test for binaries"""

    def setUp(self):
        self.base_path = BASE_TEST_PATH / "binaries"
        self.units = [
            Unit(
                "test-bin.BinExport",
                "test-bin2.BinExport",
                "test-bin.results",
                loader=LoaderType.binexport,
            ),
        ]

        # Add QBinExport test cases
        try:
            import qbinexport

            self.units.append(
                Unit(
                    "test-bin.QBinExport",
                    "test-bin2.QBinExport",
                    "test-bin-qbinexport.results",
                    loader=LoaderType.qbinexport,
                    primary_exe="test-bin",
                    secondary_exe="test-bin2",
                )
            )
        except ModuleNotFoundError:
            pass

        self.features = []
        FEATURES_KEYS = {x.key: x for x in FEATURES}
        for feature in set(tuple(x.key for x in FEATURES)):
            # Ignore non-deterministic heuristics
            if feature == "Gcom":
                continue
            weight = 1.0
            self.features.append((FEATURES_KEYS[feature], float(weight)))

    def path(self, p):
        return self.base_path / p

    def basic_test(self, unit: Unit):
        if unit.loader == LoaderType.qbinexport:
            p = qbindiff.Program(
                self.path(unit.primary), unit.loader, self.path(unit.primary_exe)
            )
            s = qbindiff.Program(
                self.path(unit.secondary), unit.loader, self.path(unit.secondary_exe)
            )
        else:
            p = qbindiff.Program(self.path(unit.primary), unit.loader)
            s = qbindiff.Program(self.path(unit.secondary), unit.loader)
        differ = qbindiff.QBinDiff(
            p, s, sparsity_ratio=0.75, tradeoff=0.75, epsilon=0.5
        )

        for f, w in self.features:
            differ.register_feature_extractor(f, w)

        differ.process()
        mapping = differ.compute_matching()
        output = {(match.primary.addr, match.secondary.addr) for match in mapping}

        with open(self.path(unit.result)) as fp:
            expected = json.load(fp)

        self.assertFalse(output ^ set(tuple(e) for e in expected))

    def test_binaries(self):
        for unit in self.units:
            with self.subTest(unit=unit):
                self.basic_test(unit)


class GraphSimTest(unittest.TestCase):
    """Regression tests for generic graphs with a custom supplied similarity matrix"""

    def setUp(self):
        self.base_path = BASE_TEST_PATH / "graphs_sim"
        self.units = (
            Unit(
                "simple-graph.1",
                "simple-graph.2",
                "simple-graph.output",
                similarity="simple-graph.similarity",
            ),
            Unit(
                "partial-match.1",
                "partial-match.2",
                "partial-match.output",
                similarity="partial-match.similarity",
            ),
        )

    def path(self, p):
        return self.base_path / p

    def basic_test(self, unit: Unit):
        graph1 = networkx.read_gml(self.path(unit.primary))
        graph2 = networkx.read_gml(self.path(unit.secondary))
        differ = qbindiff.DiGraphDiffer(
            graph1, graph2, sparsity_ratio=0, tradeoff=0.75, epsilon=0.5
        )

        # Provide custom similarity matrix
        differ.register_pass(load_sim_matrix(self.path(unit.similarity)))

        mapping = differ.compute_matching()
        output = {(match.primary, match.secondary) for match in mapping}

        with open(self.path(unit.result)) as fp:
            expected = json.load(fp)

        self.assertFalse(output ^ set(tuple(e) for e in expected))

    def test_sim_graphs(self):
        for unit in self.units:
            with self.subTest(unit=unit):
                self.basic_test(unit)


class GraphTest(unittest.TestCase):
    """Regression tests for generic graphs"""

    def setUp(self):
        self.base_path = BASE_TEST_PATH / "graphs_no_sim"
        self.units = (
            Unit(
                "simple-graph.1",
                "simple-graph.2",
                "simple-graph.output",
            ),
        )

    def path(self, p):
        return self.base_path / p

    def basic_test(self, unit: Unit):
        graph1 = networkx.read_gml(self.path(unit.primary))
        graph2 = networkx.read_gml(self.path(unit.secondary))
        differ = qbindiff.DiGraphDiffer(
            graph1, graph2, sparsity_ratio=0, tradeoff=0, epsilon=0.5
        )

        mapping = differ.compute_matching()
        output = {(match.primary, match.secondary) for match in mapping}

        with open(self.path(unit.result)) as fp:
            expected = json.load(fp)

        self.assertFalse(output ^ set(tuple(e) for e in expected))

    def test_no_sim_graphs(self):
        for unit in self.units:
            with self.subTest(unit=unit):
                self.basic_test(unit)


def suite():
    s = unittest.TestSuite()
    s.addTest(BinaryTest("test_binaries"))
    s.addTest(GraphSimTest("test_sim_graphs"))
    s.addTest(GraphTest("test_no_sim_graphs"))
    return s


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], max_content_width=300)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-v", "--verbose", count=True, help="Enable a higher level of verbosity")
def main(verbose):
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite())
    if not result.wasSuccessful():
        exit(1)


if __name__ == "__main__":
    main()
