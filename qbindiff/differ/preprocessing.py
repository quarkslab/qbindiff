from __future__ import absolute_import
import logging

import numpy as np
from pandas import DataFrame
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from qbindiff.features.visitor import ProgramVisitor

# Import for types
from qbindiff.loader.program import Program
from qbindiff.features.visitor import FeatureExtractor
from qbindiff.types import Tuple, AddrIndex, Vector, Matrix, Anchors, Ratio


class PreProcessor:
    """docstring for Preprocessor"""
    def __init__(self, primary: Program, secondary: Program) -> None:
        self.primary = primary
        self.secondary = secondary

    def process(self, features: FeatureExtractor=[], distance: str ="cosine", sim_ratio: Ratio=.9, sq_ratio: Ratio=.6) -> None:

        affinity1, affinity2 = self._prepare_matrices(features=features, distance=distance)
        if self._check_matrix():
            self._build_matrices(affinity1=affinity1, affinity2=affinity2, sim_ratio=sim_ratio, sq_ratio=sq_ratio)

    def _prepare_matrices(self, features: FeatureExtractor=[], distance: str ="cosine") -> Tuple[Matrix, Matrix]:
        self._load_features(features)
        self._process_features()
        anchors = self._set_anchors()
        self._compute_sim_matrix(distance)
        affinity1, affinity2 = self._build_affinity(anchors)
        self._apply_anchors(anchors)
        return affinity1, affinity2

    def _build_matrices(self, affinity1: Matrix, affinity2: Matrix, sim_ratio: Ratio, sq_ratio: Ratio) -> None:
        mask = self._compute_sim_mask(sim_ratio)
        mask[:] = self._compute_mask(affinity1, affinity2, mask, sq_ratio)
        sim_matrix_data = self.sim_matrix[mask]
        self.sim_matrix = csr_matrix(mask, dtype=np.float32)
        self.sim_matrix.data = sim_matrix_data

    def _load_features(self, features: FeatureExtractor) -> None:

        def _build_visitor(features: FeatureExtractor) -> ProgramVisitor:
            visitor = ProgramVisitor()
            if not isinstance(features, list):
                visitor.register_feature(features())
            else:
                for feature in features:
                    visitor.register_feature(feature())
            return visitor

        def _build_feature_idx(primary_features: dict, secondary_features: dict) -> dict:
            """
            Builds a set a all existing features in both programs
            """
            features_names = set()
            for p_features in [primary_features, secondary_features]:
                for fun in p_features.values():
                    features_names.update(fun.keys())
            return dict(zip(features_names, range(len(features_names))))

        def _vectorize_features(program_features: dict, features_idx: dict, dtype=np.float32) -> DataFrame:
            """
            Converts function features into vector forms (DataFrame)
            """
            features = np.zeros((len(program_features), len(features_idx)), dtype=dtype)
            for funid, pfeatures in enumerate(program_features.values()):
                if pfeatures:  # if the function have features (otherwise array cells are already zeroes
                    opid, count = zip(*((features_idx[opc], count) for opc, count in pfeatures.items() if opc in features_idx))
                    features[funid, opid] = count
            features = DataFrame(features, index=program_features.keys())
            return features

        visitor = _build_visitor(features)

        primary_features, secondary_features = (visitor.visit_program(p) for p in (self.primary, self.secondary))

        features_idx = _build_feature_idx(primary_features, secondary_features)

        self.primary_features = _vectorize_features(primary_features, features_idx)
        self.secondary_features = _vectorize_features(secondary_features, features_idx)

    def _process_features(self) -> None:
        opcsum = self.primary_features.sum(0) + self.secondary_features.sum(0)
        opcsum[opcsum==0] = 1
        self.primary_features /= opcsum  # feature ponderation via total
        self.secondary_features /= opcsum  # number of appearance per features

    def _set_anchors(self) -> Anchors:
        try:
            imports2 = {fun.name: addr for addr, fun in self.secondary.items() if fun.is_import()}
            anchors = ((addr, imports2[fun.name]) for addr, fun in self.primary.items() if fun.name in imports2)
            anchors = zip(*anchors)
            return [list(anchor) for anchor in anchors]
        except NotImplementedError:
            return [None, None]

    def _compute_sim_matrix(self, distance: str) -> None:
        self.sim_matrix = cdist(self.primary_features, self.secondary_features, distance).astype(np.float32)
        self.sim_matrix[np.isnan(self.sim_matrix)] = 0
        self.sim_matrix /= self.sim_matrix.max()
        np.negative(self.sim_matrix, out=self.sim_matrix)
        self.sim_matrix += 1

    def _build_affinity(self, anchors: Anchors) -> Tuple[Matrix, Matrix]:
        """
        Builds affinity matrix based on the call-graph of functions selected for the matchings (subgraph)
        Converts addrress -> index
        """
        def _build_affinity_(program: Program, addrs: AddrIndex, anchors: Anchors=None) -> Matrix:
            n = len(addrs)
            affinity = np.zeros((n, n), bool)
            addrindex = dict(zip(addrs, range(n)))
            edges = ((addrindex[addr], [addrindex[caddr] for caddr in fun.children if caddr in addrindex]) for addr, fun in program.items()) # if full graph
            #edges = [(index, [addrindex[naddr] for naddr in program[addr].children if naddr in addrs]) for addr, index in addrindex.items()] # if subgraph
            for idx, idy in edges:
                affinity[idx, idy] = True
            np.fill_diagonal(affinity, False)
            if anchors is not None:
                anchors[:] = [addrindex[addr] for addr in anchors]
                affinity[anchors] = False
                affinity[:,anchors] = False
            return affinity

        affinity1 = _build_affinity_(self.primary, self.primary_features.index, anchors[0])
        affinity2 = _build_affinity_(self.secondary, self.secondary_features.index, anchors[1])
        return affinity1, affinity2

    def _apply_anchors(self, anchors: Anchors) -> None:
        if anchors[0] is not None:
            idx1, idx2 = anchors
            self.sim_matrix[idx1] = 0
            self.sim_matrix[:,idx2] = 0
            self.sim_matrix[idx1, idx2] = 1

    def _build_matrices(self, affinity1: Matrix, affinity2: Matrix, sim_ratio: Ratio=.7, sq_ratio: Ratio=.6) -> None:
        self.sim_matrix, self.square_matrix = self.build_matrices(self.sim_matrix, affinity1=affinity1, affinity2=affinity2, sim_ratio=sim_ratio, sq_ratio=sq_ratio)

    @staticmethod
    def build_matrices(sim_matrix:Matrix, affinity1: Matrix, affinity2: Matrix, sim_ratio: Ratio=.7, sq_ratio: Ratio=.6) -> Tuple[csr_matrix, csr_matrix]:

        def _compute_sim_mask(sim_matrix, sim_ratio: Ratio=.7) -> Matrix:
            sim_ratio = int(sim_ratio * sim_matrix.size)
            threshold = np.partition(sim_matrix.reshape(-1), sim_ratio)[sim_ratio]
            sim_mask = sim_matrix >= threshold
            return sim_mask

        def _compute_mask(affinity1: Matrix, affinity2: Matrix, mask: Matrix, sqratio: Ratio=.6) -> Matrix:

            def _build_squares(affinity1: Matrix, affinity2: Matrix, shape: int) -> csr_matrix:
                row, col = _kronecker(affinity1, affinity2)
                data = np.ones_like(row, dtype=bool)
                squares = csr_matrix((data, (row, col)), shape=2*(shape,), copy=False)
                squares += squares.T
                return squares

            def _kronecker(affinity1: Matrix, affinity2: Matrix) -> Tuple[Vector, Vector]:
                row1, col1 = map(np.uint32, affinity1.nonzero())
                row2, col2 = map(np.uint32, affinity2.nonzero())
                row = np.add.outer(row1 * affinity2.shape[0], row2)
                col = np.add.outer(col1 * affinity2.shape[0], col2)
                row, col = row.reshape(-1), col.reshape(-1)
                return row, col

            def _compute_sq_mask(squares: csr_matrix, sim_mask: Matrix, sqratio: Ratio=.6) -> None:
                sim_mask = sim_mask.reshape(-1)
                sq_mask = squares[sim_mask].sum(0, dtype=np.float32).getA1()
                tot_squares = squares.sum(1, dtype=np.float32).getA1()
                tot_squares[tot_squares==0] = 1
                sq_mask /= tot_squares
                sq_mask = sq_mask  >= sqratio
                sim_mask |= sq_mask

            def _mask_squares(squares: csr_matrix, mask: Matrix) -> csr_matrix:
                mask = mask.reshape(-1)
                squares = squares[mask]
                squares = squares[:,mask]
                return squares

            #from time import time
            #t = time()
            squares = _build_squares(affinity1, affinity2, mask.size)
            #print("build from list: %f"%(time()-t))
            #t = time()
            _compute_sq_mask(squares, mask, sqratio)
            #print("square mask: %f"%(time()-t))
            #t = time()
            square_matrix = _mask_squares(squares, mask)
            #print("build from mask: %f"%(time()-t))
            return square_matrix

        mask = _compute_sim_mask(sim_matrix, sim_ratio)
        square_matrix = _compute_mask(affinity1, affinity2, mask, sq_ratio)
        sim_matrix_data = sim_matrix[mask]
        sim_matrix = csr_matrix(mask, dtype=np.float32)
        sim_matrix.data = sim_matrix_data
        return sim_matrix, square_matrix

    def _check_matrix(self) -> bool:
        if np.isnan(self.sim_matrix).any():
            logging.warning("Incompatibilty between distance and features (nan returned)")
            return False
        if self.sim_matrix.shape[0] == 0:  # check the weight matrix size
            logging.warning("No possible function match: empty weight matrix (you can retry lowering the threshold)")
            return False
        return True

