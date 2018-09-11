from __future__ import absolute_import
import logging

from qbindiff.differ.preprocessing import PreProcessor
from qbindiff.differ.postprocessing import PostProcessor
from qbindiff.belief.belief_propagation import BeliefMWM, BeliefNAQP

# Import for types
from qbindiff.types import Ratio, FinalMatching
from qbindiff.loader.program import Program
from qbindiff.features.visitor import FeatureExtractor


class QBinDiff:

    name = "QBinDiff"

    def __init__(self, primary: Program, secondary: Program):
        # init values
        self.data = PreProcessor(primary=primary, secondary=secondary)

    def initialize(self, features: FeatureExtractor=[], distance: str ="cosine", sim_ratio: Ratio=.9, sq_ratio: Ratio=.6):
        """
        Initialize the diffing by extracting the features in the programs, computing
        the call graph as needed by the belief propagation and by applying the threshold
        to produce the distance matrix.
        :return: None
        """
        distance = self._check_distance(distance)
        self.data.process(features=features, distance=distance, sim_ratio=sim_ratio, sq_ratio=sq_ratio)

    def compute(self, tradeoff: Ratio=0.5, maxiter: int=100,) -> None:

        if tradeoff == 0:
            logging.info("[+] switching to Maximum Weight Matching (tradeoff is 0)")
            belief = BeliefMWM(weights=self.data.sim_matrix)
        else:
            belief = BeliefNAQP(weights=self.data.sim_matrix, squares=self.data.square_matrix, tradeoff=tradeoff)

        for it in belief.compute_matching(maxiter=maxiter):
            yield it

        self.data.belief_results = (belief.matching, belief.objective[-1])

        #if isinstance(belief, BeliefNAQP):
        #    logging.info("[+] squares number : %d" % belief.numsquares)

    def refine(self):

        self.data = PostProcessor(self.data)
        self.data.process()

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

    @property
    def matching(self) -> FinalMatching:
        """
        Returns the matching or None if it has not been computed yet.
        :return: final matching
        """
        return self.data.matching

