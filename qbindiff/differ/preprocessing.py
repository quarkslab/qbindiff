from __future__ import absolute_import
import logging
import numpy as np
from pandas import DataFrame
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

# Import for types
from qbindiff.loader.program import Program
from qbindiff.features.visitor import ProgramVisitor
from qbindiff.types import Tuple, AddrIndex, CallGraph, Matrix, Anchors


def load_features(program1: Program, program2: Program, visitor: ProgramVisitor) -> Tuple[DataFrame, DataFrame]:
    program_features1, program_features2 = (visitor.visit_program(p) for p in (program1, program2))

    features_idx = _build_feature_idx(program_features1, program_features2)

    features1 = _vectorize_features(program_features1, features_idx)
    features2 = _vectorize_features(program_features2, features_idx)
    return features1, features2


def _build_feature_idx(program_features1: dict, program_features2: dict) -> dict:
    """
    Builds a set a all existing features in both programs
    """
    features_names = set()
    for p_features in [program_features1, program_features2]:
        for fun in p_features.values():
            features_names.update(fun.keys())
    return dict(zip(features_names, range(len(features_names))))


def _vectorize_features(program_features: dict, features_idx: dict) -> DataFrame:
    """
    Converts function features into vector forms (DataFrame)
    """
    features = np.zeros((len(program_features), len(features_idx)), np.float32)  # Check floating size wrt. program size
    for funid, pfeatures in enumerate(program_features.values()):
        if pfeatures:  # if the function have features (otherwise array cells are already zeroes
            opid, count = zip(*((features_idx[opc], count) for opc, count in pfeatures.items() if opc in features_idx))
            features[funid, opid] = count
    features = DataFrame(features, index=program_features.keys())
    return features

def extract_anchors(program1: Program, program2: Program) -> Anchors:
    try:
        import1 = {add: fun.name for add, fun in program1.items() if fun.is_import()}
        import2 = {fun.name: add for add, fun in program2.items() if fun.is_import()}
        anchors = zip(*((add, import2[name]) for add, name in import1.items()))
        return list(anchors)
    except NotImplementedError:
        return None


def build_weight_matrix(features1: DataFrame, features2: DataFrame, distance: str="cosine", threshold: float=0.0, sparsity: float=0.25, anchors: Anchors=None) -> Tuple[AddrIndex, AddrIndex, csr_matrix]:
    """
    Processes features, then builds the weight matrix and applies the specified threshold
    Recall : the weights are to be MAXIMISED so they should computed according to a SIMILARITY measure (not a distance)
    """
    features1, features2 = _process_features(features1, features2)
    weight_matrix = cdist(features1, features2, distance)            # Compute distance
    if np.isnan(weight_matrix).any():
        return None, None, None
    adds1 = features1.index
    adds2 = features2.index
    weight_matrix = _process_weight_matrix(weight_matrix, threshold, sparsity)
    if anchors:
        weight_matrix = _apply_anchors(weight_matrix, anchors, adds1, adds2)
    rowmask = weight_matrix.any(1)                                   # Keep vertex with at least
    colmask = weight_matrix.any(0)                                   # one possible matching
    adds1 = adds1[rowmask]
    adds2 = adds2[colmask]
    weight_matrix = csr_matrix(weight_matrix[np.ix_(rowmask, colmask)])
    logging.debug("Number of functions to be matched after preprocessing step: %d/%d"%weight_matrix.shape +" (before: %d/%d)"%(len(features1), len(features2)))
    return adds1, adds2, weight_matrix


def _process_features(features1: DataFrame, features2: DataFrame) -> Tuple[DataFrame, DataFrame]:
    features1 = features1[features1.any(1)]
    features2 = features2[features2.any(1)]
    opcsum = features1.sum(0) + features2.sum(0)
    opcsum[opcsum==0] = 1
    features1 /= opcsum  # feature ponderation via total
    features2 /= opcsum  # number of appearance per features
    return features1, features2

def _process_weight_matrix(weight_matrix: Matrix, threshold: float=0.0, sparsity: float=0.2) -> Matrix:
    weight_matrix /=  weight_matrix.max()                            # Normalization
    weight_matrix = 1 - weight_matrix                                # Distance to similarity
    threshmask = weight_matrix >= threshold                          # apply global threshold
    nsparse = int((1-sparsity) * weight_matrix.shape[1])
    minrows = np.partition(weight_matrix, nsparse)[:,nsparse, None]  # Set row threshold
    threshmask &= weight_matrix > minrows                            # Apply threshold
    weight_matrix *= threshmask
    _compute_sparsity(threshmask)
    return weight_matrix


def _apply_anchors(weight_matrix: Matrix, anchors: Anchors, adds1: AddrIndex, adds2: AddrIndex) -> Matrix:
    idx1, idx2 = anchors
    idx1 = adds1.get_indexer(idx1)
    idx2 = adds2.get_indexer(idx2)
    weight_matrix[idx1] = 0
    weight_matrix[:,idx2] = 0
    weight_matrix[idx1, idx2] = 1
    return weight_matrix


def build_callgraphs(program1: Program, program2: Program, adds1: AddrIndex, adds2: AddrIndex) -> Tuple[CallGraph, CallGraph]:
    """
    Builds call-graph of functions selected for the matchings (subgraph)
    Converts address -> index
    """
    def _build_callgraph(program, adds):
        addindex = dict(zip(adds, range(len(adds))))
        return [list({addindex[nadd] for nadd in program[funadd].children if nadd in adds}) for funadd in adds]
    callgraph_p1 = _build_callgraph(program1, adds1)
    callgraph_p2 = _build_callgraph(program2, adds2)
    return callgraph_p1, callgraph_p2


def _compute_sparsity(mask: DataFrame) -> None:
    nnz = np.count_nonzero(mask)
    size = np.prod(mask.shape)
    sparse = 100 * nnz / size
    logging.debug("[+] items number : %d/%d (sparsity: %.2f%%)" % (nnz, size, sparse))
