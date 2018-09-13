from __future__ import absolute_import
import logging

from qbindiff.differ.preprocessing import Preprocessor
from qbindiff.differ.postprocessing import Postprocessor
from qbindiff.belief.belief_propagation import BeliefMWM, BeliefNAQP

# Import for types
from qbindiff.types import Tuple, Set, Ratio, BeliefMatching, FinalMatching
from qbindiff.loader.program import Program
from qbindiff.features.visitor import FeatureExtractor


class QBinDiff:

    name = "QBinDiff"

    def __init__(self, primary: Program, secondary: Program):
        self.primary = primary
        self.secondary = secondary
        self.primary_features = None
        self.secondary_features = None

        self.sim_matrix = None
        self.square_matrix = None

        self.matchindex = dict()
        self.objective = .0
        self._sim_index = dict()

    def initialize(self, features: FeatureExtractor=[], distance: str ="cosine", sim_ratio: Ratio=.9, sq_ratio: Ratio=.6) ->  bool:
        """
        Initialize the diffing by extracting the features in the programs, computing
        the call graph as needed by the belief propagation and by applying the threshold
        to produce the distance matrix.
        :return: None
        """
        preprocessor = Preprocessor(self.primary, self.secondary)
        distance = self._check_distance(distance)
        sim_matrix, affinity1, affinity2 = preprocessor.extract_features(features, distance)
        self.primary_features = preprocessor.primary_features
        self.secondary_features = preprocessor.secondary_features

        if preprocessor.check_matrix(sim_matrix):
            sim_matrix, square_matrix = preprocessor.filter_matrices(sim_matrix, affinity1, affinity2, sim_ratio, sq_ratio)
            self.sim_matrix = sim_matrix
            self.square_matrix = square_matrix
            size = self.sim_matrix.shape[0] * self.sim_matrix.shape[1]
            logging.debug("[+] preprocessing sparseness : %f (%d/%d)" %(self.sim_matrix.size/size, self.sim_matrix.size, size))
            return True
        return False

    def compute(self, tradeoff: Ratio=0.5, maxiter: int=100,) -> None:
        """
        Run the belief propagation algorithm. This method hangs until the computation is done.
        The resulting matching is then converted into a binary-based format
        :return: None
        """
        if tradeoff == 0:
            logging.info("[+] switching to Maximum Weight Matching (tradeoff is 0)")
            belief = BeliefMWM(weights=self.sim_matrix)
        else:
            belief = BeliefNAQP(weights=self.sim_matrix, squares=self.square_matrix, tradeoff=tradeoff)

        for it in belief.compute_matching(maxiter=maxiter):
            yield it

        self._convert_matching(belief.matching, belief.objective[-1])

        if isinstance(belief, BeliefNAQP):
            logging.debug("[+] squares number : %d" % belief.numsquares)
        unmatched1, unmatched2 = self._get_unmatched()
        logging.debug("[+] unmatched functions before refinement | p1 : %d/%d, p2 : %d/%d"
                      %(len(unmatched1),len(self.primary), len(unmatched2), len(self.secondary)))

    def refine(self) -> None:
        """
        Postprocessing pass that tries to make small or excluded functions to match against
        each other to refine results and obtaining a better match.
        :return: None
        """
        postprocessor = Postprocessor(self.primary, self.secondary, self.primary_features, self.secondary_features, self.matchindex)
        postprocessor.match_relatives()
        postprocessor.match_lonely()
        self.objective += postprocessor.add_objective
        unmatched1, unmatched2 = self._get_unmatched()
        logging.debug("[+] unmatched functions after refinement | p1 : %d/%d, p2 : %d/%d"
                      %(len(unmatched1),len(self.primary), len(unmatched2), len(self.secondary)))

    def _check_distance(self, distance:str) -> str:
        distance = 'cosine'  # TODO: Elie
        return distance
        '''
        Si auto:
            Si features de dimension 1:
                euclidean
            Si variance nulle:
                cosine
            sinon:
                correlation
        sinon:
            ne fait rien
        '''

    def _convert_matching(self, belief_matching:BeliefMatching, belief_objective:float) -> None:
        """
        Converts index matching of Belief propagation into matching of addresses.
        :return: None
        """
        idx1, idx2, weights = zip(*belief_matching)
        adds1 = self.primary_features.index[list(idx1)]
        adds2 = self.secondary_features.index[list(idx2)]
        self.matchindex = dict(zip(adds1, adds2))
        self.objective = belief_objective
        self._sim_index = dict(zip(adds1, weights))

    def _get_unmatched(self) -> Tuple[Set, Set]:
        """
        Extract the set of adddresses of yet unmatched functions.
        :return: set of yet unmatched function in primary, set of yet unmatched function in secondary
        """
        matched1, matched2 = set(self.matchindex), set(self.matchindex.values())
        unmatched1 = set(self.primary).difference(self.matchindex)
        unmatched2 = set(self.secondary).difference(self.matchindex.values())
        return unmatched1, unmatched2

    def _format_matching(self) -> FinalMatching:
        """
        Convert mathindex into the final matching format.
        :return: dict containing the final matching
        """
        matching = [{"primary": adds1, "secondary": adds2, "similarity": float(self._sim_index.get(adds1, 1.))} for adds1, adds2 in self.matchindex.items()]
        unmatched1, unmatched2 = self._get_unmatched()
        matching += [{"primary": adds1, "secondary": None, "similarity": 0.} for adds1 in unmatched1]
        matching += [{"primary": None, "secondary": adds2, "similarity": 0.} for adds2 in unmatched2]
        return {"score": self.objective, "matching": matching}

    @property
    def matching(self) -> FinalMatching:
        """
        Returns the matching or None if it has not been computed yet.
        :return: final matching
        """
        return self._format_matching()

