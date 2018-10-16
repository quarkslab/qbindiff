from __future__ import absolute_import
import logging

from qbindiff.differ.preprocessing import Preprocessor
from qbindiff.belief.belief_propagation import BeliefMWM, BeliefNAQP

# Import for types
from qbindiff.types import Tuple, Set, Ratio, BeliefMatching, FinalMatching, Addr
from qbindiff.loader.program import Program
from qbindiff.features.visitor import FeatureExtractor
from typing import List, Optional
from qbindiff.differ.matching import Matching

class QBinDiff:

    name = "QBinDiff"

    def __init__(self, primary: Program, secondary: Program):
        self.primary = primary
        self.secondary = secondary
        self.primary_features = None
        self.secondary_features = None
        self._matching = None  # final matching filled after computation

        self.sim_matrix = None
        self.square_matrix = None

        self.matchindex = dict()
        self.objective = .0
        self._sim_index = dict()

    def initialize(self, features: List[FeatureExtractor]=[], distance: str ="cosine", sim_threshold: Ratio=.9, sq_threshold: Ratio=.6) -> bool:
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
            sim_matrix, square_matrix = preprocessor.filter_matrices(sim_matrix, affinity1, affinity2, sim_threshold, sq_threshold)
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
        logging.debug("[+] unmatched functions before refinement | p1 : %d/%d, p2 : %d/%d"
                      % (self._matching.nb_unmatched_primary, len(self.primary),
                         self._matching.nb_unmatched_secondary, len(self.secondary)))

    def refine(self) -> None:
        """
        Postprocessing pass that tries to make small or excluded functions to match against
        each other to refine results and obtaining a better match.
        :return: None
        """
        self._match_relatives()
        self._match_lonely()
        logging.debug("[+] unmatched functions after refinement | p1 : %d/%d, p2 : %d/%d"
                      % (self._matching.nb_unmatched_primary, len(self.primary),
                         self._matching.nb_unmatched_secondary, len(self.secondary)))

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

    def _convert_matching(self, belief_matching: BeliefMatching, belief_objective: float) -> None:
        """
        Converts index matching of Belief propagation into matching of addresses.
        :return: None
        """
        self._matching = Matching(set(self.primary.keys()), set(self.secondary.keys()))
        for idx1, idx2, sim in belief_matching:
            addr1 = self.primary_features.index[idx1]  # TODO: check
            addr2 = self.secondary_features.index[idx2]
            self._matching.add_match(addr1, addr2, sim)
        self._matching.similarity = belief_objective

# ================ POST PROCESSOR ====================

    def _match_relatives(self)-> None:
        """
        Matches primary unmatched functions according to their neighborhoods
        Matched if same parents and same children according to the current matchindex as well as same feature_vector
        Lonely functions are recorded for future matchindex
        """
        unmatched = set(self.primary.keys()).difference(self._matching.primary_address_matched)
        self.lonely = []
        for addr in unmatched:
            candidates = self._get_candidates(addr)
            if candidates is None:
                self.lonely.append(addr)
            else:
                for candidate in candidates:
                    if self._compare_function(addr, candidate):
                        self._matching.similarity += 1. + len(self.primary[addr].parents) + len(self.primary[addr].children)
                        self._matching.add_match(addr, candidate, 1.0)
                        break

    def _match_lonely(self) -> None:
        """
        Matches secondary lonely unmatched functions to primary ones
        Matched if same feature_vector
        """
        unmatched = set(self.secondary.keys()).difference(self._matching.secondary_address_matched)
        lone_candidates = list(filter(lambda addr: self.secondary[addr].is_alone(), unmatched))
        for addr in self.lonely:
            for candidate in lone_candidates:
                if self._compare_function(addr, candidate, lone=True):
                    self._matching.similarity += 1.
                    self._matching.add_match(addr, candidate, 1.0)
                    lone_candidates.remove(candidate)
                    break

    def _get_candidates(self, addr: Addr) -> Optional[Set[Addr]]:
        """
        Extracts the candidate set of the function "address"
        Intersects the children sets of all secondary functions matched with parents of "address" in primay
        Do the same for parents of functions coupled with children of "address"
        Return the union of both sets
        """
        if self.primary[addr].is_alone():
            return None
        candidates = set()

        for x in self.primary[addr].parents:  # get parents of the unmatch function in primary
            if self._matching.is_match_primary(x):  # if the parent is matched
                for parentmatch in self._matching.match_primary(x).addr_secondary:  # retrieve parent's match addr
                    candidates.update(self.secondary[parentmatch].children)   # in secondary and get its child

        for x in self.primary[addr].children:  # get parents of the unmatch function in primary
            if self._matching.is_match_primary(x):  # if the parent is matched
                for childrenmatch in self._matching.match_primary(x).addr_secondary:  # retrieve parent's match addr
                    candidates.update(self.secondary[childrenmatch].parents)   # in secondary and get its child
        return candidates

    def _compare_function(self, addr1: Addr, addr2: Addr, lone: bool=False) -> bool:
        """
        True if adds1 and adds2 have the same parents and the same children according
        to the current matchindex as well as same feature_vector
        """
        if (self.primary_features.loc[addr1] != self.secondary_features.loc[addr2]).any():
            return False
        if lone:
            return True

        par_primary = {self._matching.match_primary(x).addr_secondary for x in self.primary[addr1].parents if self._matching.is_match_primary(x)}
        par_secondary = {self._matching.match_primary(x).addr_secondary for x in self.primary[addr1].children if self._matching.is_match_primary(x)}
        if self.secondary[addr2].parents != par_primary or self.secondary[addr2].children != par_secondary:
            return False
        return True

# =========================================================

    @property
    def matching(self) -> Matching:
        """
        Returns the matching or None if it has not been computed yet.
        :return: final matching
        """
        return self._matching

