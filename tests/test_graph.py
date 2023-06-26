import pytest, networkx, json
import scipy.io
from pathlib import Path

import qbindiff


BASE_TEST_PATH = Path("tests/data")


class TestGraph:
    """Regression tests for generic graphs"""

    base_path = BASE_TEST_PATH / "graphs_no_sim"

    def path(self, p: str) -> Path:
        return self.base_path / p

    def test_no_sim_graphs(self):
        graph1 = networkx.read_gml(self.path("simple-graph.1"))
        graph2 = networkx.read_gml(self.path("simple-graph.2"))
        differ = qbindiff.DiGraphDiffer(graph1, graph2, sparsity_ratio=0, tradeoff=0, epsilon=0.5)

        mapping = differ.compute_matching()
        output = {(match.primary, match.secondary) for match in mapping}

        with open(self.path("simple-graph.output")) as fp:
            expected = json.load(fp)

        assert not (output ^ set(tuple(e) for e in expected))


class load_sim_matrix:
    def __init__(self, path: Path):
        self._path = path

    def __call__(self, sim_matrix, *_, **__):
        sparse_sim_matrix = scipy.io.mmread(self._path)
        sim_matrix[:] = sparse_sim_matrix.toarray()


@pytest.mark.parametrize(
    "primary,secondary,similarity,expected",
    [
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
    ],
)
class TestGraphSim:
    """Regression tests for generic graphs with a custom supplied similarity matrix"""

    base_path = BASE_TEST_PATH / "graphs_sim"

    def path(self, p: str) -> Path:
        return self.base_path / p

    def test_sim_graphs(self, primary: str, secondary: str, similarity: str, expected: str):
        graph1 = networkx.read_gml(self.path(primary))
        graph2 = networkx.read_gml(self.path(secondary))
        differ = qbindiff.DiGraphDiffer(
            graph1, graph2, sparsity_ratio=0, tradeoff=0.75, epsilon=0.5
        )

        # Provide custom similarity matrix
        differ.register_prepass(load_sim_matrix(self.path(similarity)))

        mapping = differ.compute_matching()
        output = {(match.primary, match.secondary) for match in mapping}

        with open(self.path(expected)) as fp:
            expected = json.load(fp)

        assert not (output ^ set(tuple(e) for e in expected))
