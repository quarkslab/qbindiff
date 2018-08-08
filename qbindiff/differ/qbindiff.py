import logging

from qbindiff.features.visitor import ProgramVisitor
from qbindiff.features.visitor import FeatureExtractor
from qbindiff.differ.preprocessing import load_features, build_weight_matrix, build_callgraphs
from qbindiff.differ.postprocessing import convert_matching, match_relatives, match_lonely, format_matching
from qbindiff.belief.belief_propagation import BeliefNAQP
from qbindiff.types import FinalMatching, Generator, Optional
from qbindiff.loader.program import Program


class QBinDiff:

    name = "QBinDiff"

    def __init__(self, primary: Program, secondary: Program, distance: str="correlation",
                                                    threshold: float=0.5, maxiter: int=100, alpha: int=1, beta: int=2):
        super().__init__()
        self.primary = primary
        self.secondary = secondary
        self.visitor = ProgramVisitor()

        # parameters
        self.distance = distance
        self.threshold = threshold
        self.maxiter = maxiter
        self.alpha = alpha
        self.beta = beta

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

    def initialize(self) -> None:
        """
        Initialize the diffing by extracting the features in the programs, computing
        the call graph as needed by the belief propagation and by applying the threshold
        to produce the distance matrix.
        :return: None
        """
        # Preprocessing to extract features and filters functions
        logging.info("[+] extracting features")
        self.features1, self.features2 = load_features(self.primary, self.secondary, self.visitor)
        self.adds1, self.adds2, self.weight_matrix = build_weight_matrix(
                                                        self.features1, self.features2, self.distance, self.threshold)
        self.callgraph1, self.callgraph2 = build_callgraphs(self.primary, self.secondary, self.adds1, self.adds2)

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
        belief = BeliefNAQP(self.weight_matrix, self.callgraph1, self.callgraph2, self.alpha, self.beta)
        yield from belief.compute_matching(self.maxiter)

        logging.info("[+] squares number : %d" % belief.numsquares)
        matching = belief.matching  # TODO: See what to do of intermediate matching
        self._matching = convert_matching(self.adds1, self.adds2, matching)

        if match_refine:
            self.refine_matching()

        min_fun_nb = min(len(self.primary), len(self.secondary))
        logging.info("[+] matched functions : %d / %d" % (len(self._matching), min_fun_nb))
        self._matching = format_matching(self.adds1, self.adds2, self._matching)

    def refine_matching(self) -> None:
        """
        Postprocessing pass that tries to make small or excluded functions to match against
        each other to refine results and obtaining a better match.
        :return: None
        """
        # Postprocessing to refine results
        logging.info("[+] match refinement")

        tmp_nb_match = len(self._matching)
        matching, lonely = match_relatives(self.primary, self.secondary, self.features1, self.features2, self._matching)
        self._matching = match_lonely(self.secondary, self.features1, self.features2, matching, lonely)
        logging.info("[+] %d new matches found by refinement" % (len(self._matching) - tmp_nb_match))

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
