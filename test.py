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
        differ = qbindiff.QBinDiff(p, s)

        for f, w in self.features:
            differ.register_feature_extractor(f, w)

        differ.compute_similarity()

        matcher = qbindiff.Matcher(
            differ.sim_matrix, differ.primary_adj_matrix, differ.secondary_adj_matrix
        )
        matcher._compute_sparse_sim_matrix(0.75)
        matcher._compute_squares_matrix()

        belief = qbindiff.matcher.belief_propagation.BeliefQAP(
            matcher.sparse_sim_matrix, matcher.squares_matrix, 0.75, 0.5
        )

        g = belief.compute(1000)
        for k in g:
            pass

        s = matcher.sparse_sim_matrix.copy()
        s.data[:] = belief.best_marginals.data
        mapping = matcher.refine(belief.current_mapping, s)

        with open(self.base_path + results) as fp:
            expected = json.load(fp)

        output = list(map(list, zip(mapping[0], mapping[1])))
        self.assertEqual(output, expected)

    def test_binaries(self):
        for f1, f2, results in self.units:
            with self.subTest(f1=f1, f2=f2, results=results):
                self.basic_test(f1, f2, results)


class GraphTest(unittest.TestCase):
    """Regression tests for generic graphs"""

    def setUp(self):
        self.base_path = "test/graphs/"
        self.units = (
            (
                "simple-graph.1",
                "simple-graph.2",
                "simple-graph.similarity",
                "simple-graph.output",
            ),
        )

    def basic_test(self, g1, g2, sim, result):
        graph1 = networkx.read_gml(self.base_path + g1)
        graph2 = networkx.read_gml(self.base_path + g2)
        differ = qbindiff.Differ(graph1, graph2, None)

        sparse_sim_matrix = scipy.io.mmread(self.base_path + sim)

        differ.sim_matrix = sparse_sim_matrix.toarray()

        graph1_map = {}
        graph2_map = {}
        for i, node in enumerate(graph1.nodes):
            graph1_map[node] = i
        for i, node in enumerate(graph2.nodes):
            graph2_map[node] = i

        graph1_N = len(graph1.nodes)
        graph1_adj = np.zeros((graph1_N, graph1_N), dtype=bool)
        for x, y in graph1.edges:
            graph1_adj[graph1_map[x]][graph1_map[y]] = True
        differ.primary_adj_matrix = graph1_adj

        graph2_N = len(graph2.nodes)
        graph2_adj = np.zeros((graph2_N, graph2_N), dtype=bool)
        for x, y in graph2.edges:
            graph2_adj[graph2_map[x]][graph2_map[y]] = True
        differ.secondary_adj_matrix = graph2_adj

        matcher = qbindiff.Matcher(
            differ.sim_matrix, differ.primary_adj_matrix, differ.secondary_adj_matrix
        )
        matcher._compute_sparse_sim_matrix(0.75)
        matcher._compute_squares_matrix()

        belief = qbindiff.matcher.belief_propagation.BeliefQAP(
            matcher.sparse_sim_matrix, matcher.squares_matrix, 0.75, 0.5
        )

        g = belief.compute(1000)
        for k in g:
            pass

        s = matcher.sparse_sim_matrix.copy()
        s.data[:] = belief.best_marginals.data
        mapping = matcher.refine(belief.current_mapping, s)

        with open(self.base_path + result) as fp:
            expected = json.load(fp)

        output = list(map(list, zip(mapping[0], mapping[1])))
        self.assertEqual(output, expected)

    def test_graphs(self):
        for g1, g2, sim, res in self.units:
            with self.subTest(g1=g1, g2=g2, sim=sim, res=res):
                self.basic_test(g1, g2, sim, res)


def suite():
    s = unittest.TestSuite()
    s.addTest(BinaryTest("test_binaries"))
    s.addTest(GraphTest("test_graphs"))
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
