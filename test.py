import unittest, click, networkx, json, scipy
import numpy as np
import qbindiff
from pathlib import Path
from qbindiff.features import FEATURES


class BinaryTest(unittest.TestCase):
    """Regression Test for binaries"""

    def setUp(self):
        self.base_path = "test/binaries/"
        self.units = (
            ("test-bin.BinExport", "test-bin2.BinExport", "test-bin.results"),
        )

        self.features = []
        FEATURES_KEYS = {x.key: x for x in FEATURES}
        for feature in set(tuple(x.key for x in FEATURES)):
            # Ignore non-deterministic heuristics
            if feature == "Gcom":
                continue
            weight = 1.0
            self.features.append((FEATURES_KEYS[feature], float(weight)))

    def basic_test(self, primary, secondary, results):
        p = qbindiff.Program(Path(self.base_path + primary))
        s = qbindiff.Program(Path(self.base_path + secondary))
        differ = qbindiff.QBinDiff(
            p, s, sparsity_ratio=0.75, tradeoff=0.75, epsilon=0.5
        )

        for f, w in self.features:
            differ.register_feature_extractor(f, w)

        differ.process()
        mapping = differ.compute_matching()
        output = [[match.primary.addr, match.secondary.addr] for match in mapping]

        with open(self.base_path + results) as fp:
            expected = json.load(fp)

        self.assertEqual(sorted(output), sorted(expected))

    def test_binaries(self):
        for f1, f2, results in self.units:
            with self.subTest(f1=f1, f2=f2, results=results):
                self.basic_test(f1, f2, results)


class GraphSimTest(unittest.TestCase):
    """Regression tests for generic graphs with a custom supplied similarity matrix"""

    def setUp(self):
        self.base_path = "test/graphs_sim/"
        self.units = (
            (
                "simple-graph.1",
                "simple-graph.2",
                "simple-graph.similarity",
                "simple-graph.output",
            ),
            (
                "partial-match.1",
                "partial-match.2",
                "partial-match.similarity",
                "partial-match.output",
            ),
        )

    def basic_test(self, g1, g2, sim, result):
        graph1 = networkx.read_gml(self.base_path + g1)
        graph2 = networkx.read_gml(self.base_path + g2)
        differ = qbindiff.DiGraphDiffer(
            graph1, graph2, sparsity_ratio=0, tradeoff=0.75, epsilon=0.5
        )

        # Provide custom similarity matrix
        sparse_sim_matrix = scipy.io.mmread(self.base_path + sim)
        differ.sim_matrix = sparse_sim_matrix.toarray()

        mapping = differ.compute_matching()
        output = [[match.primary, match.secondary] for match in mapping]

        with open(self.base_path + result) as fp:
            expected = json.load(fp)

        self.assertEqual(sorted(output), sorted(expected))

    def test_sim_graphs(self):
        for g1, g2, sim, res in self.units:
            with self.subTest(g1=g1, g2=g2, sim=sim, res=res):
                self.basic_test(g1, g2, sim, res)


class GraphTest(unittest.TestCase):
    """Regression tests for generic graphs"""

    def setUp(self):
        self.base_path = "test/graphs_no_sim/"
        self.units = (
            (
                "simple-graph.1",
                "simple-graph.2",
                "simple-graph.output",
            ),
        )

    def basic_test(self, g1, g2, result):
        graph1 = networkx.read_gml(self.base_path + g1)
        graph2 = networkx.read_gml(self.base_path + g2)
        differ = qbindiff.DiGraphDiffer(
            graph1, graph2, sparsity_ratio=0, tradeoff=0, epsilon=0.5
        )

        mapping = differ.compute_matching()
        output = [[match.primary, match.secondary] for match in mapping]

        with open(self.base_path + result) as fp:
            expected = json.load(fp)

        self.assertEqual(sorted(output), sorted(expected))

    def test_no_sim_graphs(self):
        for g1, g2, res in self.units:
            with self.subTest(g1=g1, g2=g2, res=res):
                self.basic_test(g1, g2, res)


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
