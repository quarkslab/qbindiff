import pytest
from networkx import DiGraph

from qbindiff.features import CyclomaticComplexity
from qbindiff.features.extractor import FeatureCollector
from qbindiff.types import Addr


class FakeFunction:
    def __init__(self, flowgraph: DiGraph):
        self.flowgraph = flowgraph

    @property
    def edges(self) -> list[tuple[Addr, Addr]]:
        return list(self.flowgraph.edges)


# noinspection PyTypeChecker
class TestCyclomaticComplexity:
    def test_simple_linear_graph_complexity_of_1(self):
        """
        Graph :
            A -> B -> C
        """
        flowgraph = DiGraph()
        flowgraph.add_edges_from([
            ("A", "B"),
            ("B", "C"),
        ])

        function = FakeFunction(flowgraph)
        feature_collector = FeatureCollector()
        cc = CyclomaticComplexity()
        cc.visit_function(None, function, feature_collector)
        result = feature_collector.get(CyclomaticComplexity.key)

        assert result == 1


    def test_simple_if_clause_complexity_of_2(self):
        """
        Graph :
               -> B ->
             /        \
            A          D
             \        /
               -> C ->
        """
        flowgraph = DiGraph()
        flowgraph.add_edges_from([
            ("A", "B"),
            ("A", "C"),
            ("B", "D"),
            ("C", "D"),
        ])

        function = FakeFunction(flowgraph)
        feature_collector = FeatureCollector()
        cc = CyclomaticComplexity()
        cc.visit_function(None, function, feature_collector)
        result = feature_collector.get(CyclomaticComplexity.key)

        assert result == 2

    def test_simple_if_clause_two_exits_complexity_of_2(self):
        """
        Graph :
               -> B
             /
            A
             \
               -> C
        """
        flowgraph = DiGraph()
        flowgraph.add_edges_from([
            ("A", "B"),
            ("A", "C")
        ])

        function = FakeFunction(flowgraph)
        feature_collector = FeatureCollector()
        cc = CyclomaticComplexity()
        cc.visit_function(None, function, feature_collector)
        result = feature_collector.get(CyclomaticComplexity.key)

        assert result == 2

    def test_simple_if_clause_two_exits_two_components_complexity_of_4(self):
        """
        Graph :
               -> B
             /
            A
             \
               -> C

               -> F
             /
            E
             \
               -> G
        """
        flowgraph = DiGraph()
        flowgraph.add_edges_from([
            ("A", "B"),
            ("A", "C"),
            ("E", "F"),
            ("E", "G")
        ])

        function = FakeFunction(flowgraph)
        feature_collector = FeatureCollector()
        cc = CyclomaticComplexity()
        cc.visit_function(None, function, feature_collector)
        result = feature_collector.get(CyclomaticComplexity.key)

        assert result == 4


    def test_two_serial_choices_complexity_of_3(self):
        """
        Graph :
               -> B ->
             /        \
            A          D -> E  ->  G
             \        /      \    /
               -> C ->        -> F
        """
        flowgraph = DiGraph()
        flowgraph.add_edges_from([
            ("A", "B"),
            ("A", "C"),
            ("B", "D"),
            ("C", "D"),
            ("D", "E"),
            ("E", "F"),
            ("E", "G"),
            ("F", "G"),
        ])

        function = FakeFunction(flowgraph)
        feature_collector = FeatureCollector()
        cc = CyclomaticComplexity()
        cc.visit_function(None, function, feature_collector)
        result = feature_collector.get(CyclomaticComplexity.key)

        assert result == 3


    def test_simple_loop_complexity_of_2(self):
        """
            A -> B -> C
                 ^    |
                 |____|
        """
        flowgraph = DiGraph()
        flowgraph.add_edges_from([
            ("A", "B"),
            ("B", "C"),
            ("C", "B"),
        ])

        function = FakeFunction(flowgraph)
        feature_collector = FeatureCollector()
        cc = CyclomaticComplexity()
        cc.visit_function(None, function, feature_collector)
        result = feature_collector.get(CyclomaticComplexity.key)

        assert result == 2

    def test_simple_loop_two_components_complexity_of_4(self):
        """
            A -> B -> C
                 ^    |
                 |____|

            D -> E -> F
                 ^    |
                 |____|
        """
        flowgraph = DiGraph()
        flowgraph.add_edges_from([
            ("A", "B"),
            ("B", "C"),
            ("C", "B"),
            ("D", "E"),
            ("E", "F"),
            ("F", "E")
        ])

        function = FakeFunction(flowgraph)
        feature_collector = FeatureCollector()
        cc = CyclomaticComplexity()
        cc.visit_function(None, function, feature_collector)
        result = feature_collector.get(CyclomaticComplexity.key)

        assert result == 4

    def test_simple_loop_in_one_component_and_multiple_exits_in_another_component_complexity_of_4(self):
        """
            A -> B -> C
                 ^    |
                 |____|

               -> E
             /
            D
             \
               -> F
        """
        flowgraph = DiGraph()
        flowgraph.add_edges_from([
            ("A", "B"),
            ("B", "C"),
            ("C", "B"),
            ("D", "E"),
            ("D", "F")
        ])

        function = FakeFunction(flowgraph)
        feature_collector = FeatureCollector()
        cc = CyclomaticComplexity()
        cc.visit_function(None, function, feature_collector)
        result = feature_collector.get(CyclomaticComplexity.key)

        assert result == 4

    def test_empty_graph_complexity_of_0(self):
        flowgraph = DiGraph()

        function = FakeFunction(flowgraph)
        feature_collector = FeatureCollector()
        cc = CyclomaticComplexity()
        cc.visit_function(None, function, feature_collector)
        result = feature_collector.get(CyclomaticComplexity.key)

        assert result == 0


    def test_simple_two_components_complexity_of_2(self):
        """
            A -> B
            C -> D
        """
        flowgraph = DiGraph()
        flowgraph.add_edges_from([
            ("A", "B"),
            ("C", "D"),
        ])

        function = FakeFunction(flowgraph)
        feature_collector = FeatureCollector()
        cc = CyclomaticComplexity()
        cc.visit_function(None, function, feature_collector)
        result = feature_collector.get(CyclomaticComplexity.key)

        assert result == 2

    def test_simple_self_loop_complexity_of_2(self):
        """
            A -> B -> C
                    / |
                   |__|
        """
        flowgraph = DiGraph()
        flowgraph.add_edges_from([
            ("A", "B"),
            ("B", "C"),
            ("C", "C"),
        ])

        function = FakeFunction(flowgraph)
        feature_collector = FeatureCollector()
        cc = CyclomaticComplexity()
        cc.visit_function(None, function, feature_collector)
        result = feature_collector.get(CyclomaticComplexity.key)

        assert result == 2


    @pytest.mark.parametrize(
        "edges, expected",
        [
            # Linear graph
            ([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")], 1),

            # Simple if/else
            ([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")], 2),

            # simple loop
            ([("A", "B"), ("B", "C"), ("C", "B"), ("C", "D")], 2),

            # four linear components
            ([("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")], 4),
        ],
    )
    def test_cyclomatic_complexity(self, edges, expected):
        flowgraph = DiGraph()
        flowgraph.add_edges_from(edges)

        function = FakeFunction(flowgraph)
        feature_collector = FeatureCollector()
        cc = CyclomaticComplexity()
        cc.visit_function(None, function, feature_collector)
        result = feature_collector.get(CyclomaticComplexity.key)

        assert result == expected
