from __future__ import absolute_import
import logging

from qbindiff.features.visitor import ProgramVisitor
from qbindiff.features.visitor import FeatureExtractor
from qbindiff.differ.preprocessing import load_features, build_weight_matrix, build_callgraphs
from qbindiff.differ.postprocessing import convert_matching, match_relatives, match_lonely, format_final_matching
from qbindiff.belief.belief_propagation import BeliefMWM, BeliefNAQP
from qbindiff.types import FinalMatching, Generator, Optional
from qbindiff.loader.program import Program


class QBinDiff:

    name = "QBinDiff"

    def __init__(self, primary: Program, secondary: Program, distance: str="auto", threshold: float=0.0, sparsity: float=0.25, maxiter: int=100, tradeoff: float=0.5):
        self.primary = primary
        self.secondary = secondary
        self.visitor = ProgramVisitor()

        # parameters
        self.distance = distance
        self.threshold = threshold
        self.maxiter = maxiter
        self.tradeoff = tradeoff
        self.sparsity = sparsity

        # final values
        self._matching = None

        # temporary values of computation
        self.features1, self.features2 = None, None
        self.adds1, self.adds2 = None, None
        self.weight_matrix = None
        self.callgraph1, self.callgraph2 = None, None

    def register_feature(self, ft: FeatureExtractor) -> None:
        """
        Call the visitor method to add the feature.
        """
        self.visitor.register_feature(ft)

    def initialize(self) -> bool:
        """
        Initialize the diffing by extracting the features in the programs, computing
        the call graph as needed by the belief propagation and by applying the threshold
        to produce the distance matrix.
        :return: None
        """
        # Preprocessing to extract features and filters functions
        logging.info("[+] extracting features")
        self.features1, self.features2 = load_features(self.primary, self.secondary, self.visitor)
        self.check_distance_function()
        self.adds1, self.adds2, self.weight_matrix = build_weight_matrix(self.features1, self.features2, self.distance, self.threshold, self.sparsity)
        if self.weight_matrix is None:
            logging.warning("Incompatibilty between distance and features (nan returned)")
            return False
        if self.weight_matrix.shape[0] == 0:  # check the weight matrix size
            logging.warning("No possible function match: empty weight matrix (you can retry lowering the threshold)")
            return False
        self.callgraph1, self.callgraph2 = build_callgraphs(self.primary, self.secondary, self.adds1, self.adds2)
        return True

    def check_distance_function(self):
        self.distance = 'cosine'  # TODO: Elie
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

    def run(self, match_refine: bool=True) -> None:
        """
        Run the belief propagation algorithm. This method hangs until the computation is done.
        The resulting matching is then put in the self.matching attribute.
        :param match_refine: bool on whether or not trying to match small unmatched functions
        :return: None
        """
        for _ in self.run_iter(match_refine=match_refine):
            pass

    def run_iter(self, match_refine: bool=True) -> Generator[int, None, None]:
        """
        Main run functions. Initialize the belief propagation algorithm with the different
        parameters initialized by ``initialize`` and computes the matching. The ith iteration
        is yielded each time the belief propagation compputes one. Then perform the refinement
        pass if activated.
        :param match_refine: bool on whether or not trying to match small unmatched functions
        :return: Generator of belief iterations
        """
        # Performing the matching

        if self.tradeoff == 1:
            belief = BeliefMWM(self.weight_matrix)
        else:
            belief = BeliefNAQP(self.weight_matrix, self.callgraph1, self.callgraph2, self.tradeoff)
        for it in belief.compute_matching(self.maxiter):  # push back yield from when IDA will be python3
            yield it

        if self.tradeoff != 1:
            logging.info("[+] squares number : %d" % belief.numsquares)

        matching = belief.matching
        score = belief.objective[-1]
        self._matching, similarity = convert_matching(self.adds1, self.adds2, matching)

        # print stats about belief matching
        unmatched_p1 = set(self.adds1) - set(self._matching.keys())
        unmatched_p2 = set(self.adds2) - set(self._matching.values())
        logging.debug("belief unmatched functions: p1:%d, p2:%d" % (len(unmatched_p1), len(unmatched_p2)))

        if match_refine:
            self.refine_matching(score)

        min_fun_nb = min(len(self.primary), len(self.secondary))
        match_len = len(self._matching)
        self._matching = format_final_matching(self.primary, self.secondary, self._matching, similarity, score)
        logging.info("final unmatched p1:%d, p2:%d" % (len(self.primary)-match_len, len(self.secondary)-match_len))
        logging.info("final matched functions: %d / %d" % (match_len, min_fun_nb))

    def refine_matching(self, score) -> None:
        """
        Postprocessing pass that tries to make small or excluded functions to match against
        each other to refine results and obtaining a better match.
        :return: None
        """
        len_tmp = len(self._matching)
        matching, lonely = match_relatives(self.primary, self.secondary, self.features1, self.features2, self._matching, score)
        self._matching = match_lonely(self.secondary, self.features1, self.features2, matching, lonely, score)
        logging.info("[+] %d new matches found by refinement" % (len(self._matching) - len_tmp))

    @property
    def matching(self) -> Optional[FinalMatching]:
        """
        Returns the matching or None if it has not been computed yet.
        :return: final matching
        """
        if not self._matching:
            logging.error("[-] matching not computed")
            return None
        else:
            return self._matching
