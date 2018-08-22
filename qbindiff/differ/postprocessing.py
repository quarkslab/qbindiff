from __future__ import absolute_import
import logging
from functools import reduce

# Imports for types
from qbindiff.types import Tuple, List, Optional, Set
from qbindiff.types import AddrIndex, BeliefMatching, Matching, DataFrame, Addr, FinalMatching
from qbindiff.loader.program import Program


def convert_matching(adds1: AddrIndex, adds2: AddrIndex, matching: BeliefMatching) -> Tuple[Matching, dict]:
    """
    Converts index matching of Belief propagation into matching of addresses.
    """
    matchidx1, matchidx2, weights = zip(*matching)
    matchadds1 = adds1[list(matchidx1)]
    matchadds2 = adds2[list(matchidx2)]
    return dict(zip(matchadds1, matchadds2)), dict(zip(matchadds1, weights))


def match_relatives(program1: Program, program2: Program, features1: DataFrame, features2: DataFrame, matching: Matching, score: float) -> Tuple[Matching, List[Addr]]:
    """
    Matches primary unmatched functions according to their neighborhoods
    Matched if same parents and same children according to the current matching as well as same feature_vector
    Lonely functions are recorded for future matching
    """
    unmatched = set(program1.keys()).difference(matching)
    lonely = []
    for address in unmatched:
        candidates = get_candidates(address, program1, program2, matching)
        if candidates is None:
            lonely.append(address)
            continue
        for candidate in candidates:
            if compare_function(address, candidate, program1, program2, features1, features2, matching):
                score += 1. + len(program1[address].parents) + len(program1[address].children)
                matching.update({address: candidate})
                break
    return matching, lonely


def match_lonely(program2: Program, features1: DataFrame, features2: DataFrame, matching: Matching, lonely: List[Addr], score: float) -> Matching:
    """
    Matches secondary lonely unmatched functions to primary ones
    Matched if same feature_vector
    """
    unmatched = set(program2.keys()).difference(matching.values())
    lone_candidates = [x for x in unmatched if program2[x].is_alone()]
    for address in lonely:
        for candidate in iter(lone_candidates):
            if (features1.loc[address] == features2.loc[candidate]).all():
                score += 1.
                matching.update({address: candidate})
                lone_candidates.remove(candidate)
                break
    return matching


def get_candidates(address: int, program1: Program, program2: Program, matching: Matching) -> Optional[Set[Addr]]:
    """
    Extracts the candidate set of the function "address"
    Intersects the children sets of all secondary functions matched with parents of "address" in primay
    Do the same for parents of functions coupled with children of "address"
    Return the union of both sets
    """
    if not(program1[address].parents or program1[address].children):
        return None
    candidates = set()
    if program1[address].parents:
        parentcandidates = [program2[f_p2].children for f_p2 in map(matching.get, program1[address].parents) if f_p2]
        candidates.update(reduce(set.intersection, parentcandidates, set()))

    if program1[address].children:
        childrencandidates = [program2[f_p2].parents for f_p2 in map(matching.get, program1[address].children) if f_p2]
        candidates.update(reduce(set.intersection, childrencandidates, set()))
    return candidates


def compare_function(add1: Addr, add2: Addr,  program1: Program, program2: Program, features1: DataFrame, features2: DataFrame, matching: Matching) -> bool:
    """
    True if adds1 and adds2 have the same parents and the same children according
    to the current matching as well as same feature_vector
    """
    if (features1.loc[add1] != features2.loc[add2]).any():
        return False
    # RMQ: Is None expected in the set (cc. elie)?
    if program2[add2].parents != {matching.get(x) for x in program1[add1].parents}:
        return False
    if program2[add2].children != {matching.get(x) for x in program1[add1].children}:  # idem
        return False
    return True


def format_final_matching(primary: Program, secondary: Program, matching: Matching, similarity: dict, score: float) -> FinalMatching:
    matching_ = [{"primary": adds1, "secondary": adds2, "similarity": similarity.get(adds1, 1.)} for adds1, adds2 in matching.items()]
    unmatched1 = set(primary.keys()).difference(matching.keys())
    unmatched2 = set(secondary.keys()).difference(matching.values())
    matching_ += [{"primary": adds1, "secondary": None, "similarity": 0.} for adds1 in unmatched1]
    matching_ += [{"primary": None, "secondary": adds2, "similarity": 0.} for adds2 in unmatched2]
    output = {"score": score, "matching": matching_}
    return output