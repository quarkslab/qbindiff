# coding: utf-8

from functools import reduce
from operator import itemgetter

#Imports for type
from pandas import DataFrame, Index
from qbindiff.loader.program import Program
from typing import Dict, Tuple, List, Optional, Set
Addr = int
Idx = int
AddrIndex = Index  # panda index of addresses
Matching = Dict[Addr, Addr]
BeliefMatching = Dict[Idx, Idx]  # FIXME: Elie (mauvais type cf. belief)
FinalMatching = List[Tuple[Optional[Addr], Optional[Addr]]]


def convert_matching(adds1:AddrIndex, adds2:AddrIndex, matching:BeliefMatching) -> Matching:
    '''
    Converts index matching of Belief propagation into matching of addresses.
    '''
    match1, match2 = map(list, zip(*filter(itemgetter(1), matching)))
    madds1 = adds1[match1]
    madds2 = adds2[match2]
    return dict(zip(madds1, madds2))

def match_relatives(program1:Program, program2:Program, features1:DataFrame, features2:DataFrame, matching:Matching) -> Tuple[Matching, List[Addr]]:
    '''
    Matches primary unmatched functions according to their neighborhoods
    Matched if same parents and same children according to the current matching as well as same feature_vector
    Lonely functions are recorded for future matching
    '''
    unmatched = set(program1.keys()).difference(matching)
    lonely = []
    for address in unmatched:
        candidates = get_candidates(address, program1, program2, matching)
        if candidates is None:
            lonely.append(address)
            continue
        for candidate in candidates:
            if compare_function(address, candidate, program1, program2, features1, features2, matching):
                matching.update({address: candidate})
                break
    return matching, lonely

def match_lonely(program2, features1:DataFrame, features2:DataFrame, matching:Matching, lonely:List[Addr])-> Matching:
    '''
    Matches secondary lonely unmatched functions to primary ones
    Matched if same feature_vector
    '''
    unmatched = set(program2.keys()).difference(matching.values())
    lone_candidates = list(filter(program2.is_lonely, unmatched))
    for address in lonely:
        for candidate in iter(lone_candidates):
            if (features1.loc[address] == features2.loc[candidate]).all():
                matching.update({address: candidate})
                lone_candidates.remove(candidate)
                break
    return matching

def get_candidates(address:int, program1:Program, program2:Program, matching:Matching) -> Optional[Set[Addr]]:
    '''
    Extracts the candidate set of the function "address"
    Intersects the children sets of all secondary functions matched with parents of "address" in primay
    Do the same for parents of functions coupled with children of "address"
    Return the union of both sets
    '''
    if not(program1[address].parents or program1[address].children):
        return None
    candidates = set()
    if program1[address].parents:
        candidates.update(reduce(set.intersection, map(program2.get_children, map(matching.get, program1[address].parents))))
    if program1[address].children:
        candidates.update(reduce(set.intersection, map(program2.get_parents, map(matching.get, program1[address].children))))
    return candidates

def compare_function(add1:Addr, add2:Addr,  program1:Program, program2:Program, features1:DataFrame, features2:DataFrame, matching:Matching) -> bool:
    '''
    True if adds1 and adds2 have the same parents and the same children according to the current matching as well as same feature_vector
    '''
    if (features1.loc[add1] != features2.loc[add2]).any():
        return False
    if program2[add2].parents != {matching.get(x) for x in program1[add1].parents}:  # RMQ: Is None expected in the set (cc. elie)?
        return False
    if program2[add2].children != {matching.get(x) for x in program1[add1].children}:  # idem
        return False
    return True

def format_matching(adds1: AddrIndex, adds2: AddrIndex, matching: Matching) -> FinalMatching:
    unmatched_p1 = set(adds1) - set(matching.keys())
    unmatched_p2 = set(adds2) - set(matching.values())
    matching = list(matching.items())
    matching += [(addr1, None) for addr1 in unmatched_p1]
    matching += [(None, addr2) for addr2 in unmatched_p2]
    return matching
