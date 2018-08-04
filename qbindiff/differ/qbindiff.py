import logging

from qbindiff.features.visitor import ProgramVisitor
from qbindiff.features.visitor import FeatureExtractor
from qbindiff.differ.preprocessing import load_features, build_weight_matrix, build_callgraphs
from qbindiff.differ.postprocessing import convert_matching, match_relatives, match_lonely, format_matching
from qbindiff.belief.belief_propagation import BeliefNAQP
from qbindiff.types import FinalMatching, Generator, Optional
from json import dump as json_dump
from pathlib import Path


class QBinDiff:

    name = "QBinDiff"

    def __init__(self, primary, secondary):
        super().__init__()
        self.primary = primary
        self.secondary = secondary
        self.visitor = ProgramVisitor()

        # parameters
        self.distance = "correlation"
        self.threshold = 0.5
        self.maxiter = 50
        self.alpha = 1
        self.beta = 2

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
        # Preprocessing to extract features and filters functions
        logging.info("[+] extracting features")
        self.features1, self.features2 = load_features(self.primary, self.secondary, self.visitor)
        self.adds1, self.adds2, self.weight_matrix = build_weight_matrix(
                                                        self.features1, self.features2, self.distance, self.threshold)
        self.callgraph1, self.callgraph2 = build_callgraphs(self.primary, self.secondary, self.adds1, self.adds2)

    def run(self, match_refine=True) -> None:
        for _ in self.run_iter(match_refine=match_refine):
            pass

    def run_iter(self, match_refine=True) -> Generator[int, None, None]:
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
        # Postprocessing to refine results
        logging.info("[+] match refinement")

        tmp_nb_match = len(self._matching)
        matching, lonely = match_relatives(self.primary, self.secondary, self.features1, self.features2, self._matching)
        self._matching = match_lonely(self.secondary, self.features1, self.features2, matching, lonely)
        logging.info("[+] %d new matches found by refinement" % (len(self._matching) - tmp_nb_match))

    @property
    def matching(self) -> Optional[FinalMatching]:
        if not self._matching:
            logging.error("[-] matching not computed")
            return None
        else:
            return self._matching

    def save_matching(self, output: Path) -> None:
        with open(str(output), 'w') as outputfile:
            json_dump(self.matching, outputfile)
        logging.info("[+] matching successfully saved to: %s" % output)
