# Copyright 2023 Quarkslab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collection of passes (pre and post) that can be optionally used.
"""

from __future__ import annotations
import logging
from collections import defaultdict
import hashlib

from qbindiff.features import BytesHash
from qbindiff.loader.types import InstructionGroup

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qbindiff.loader import Program
    from qbindiff.types import SimMatrix, FeatureValue, Addr, Idx, Function
    from qbindiff.features.extractor import FeatureCollector


def match_same_hash_functions(
    sim_matrix: SimMatrix,
    primary: Program,
    secondary: Program,
    primary_mapping: dict[Addr, Idx],
    secondary_mapping: dict[Addr, Idx],
    primary_features: dict[Addr, FeatureCollector],
    secondary_features: dict[Addr, FeatureCollector],
) -> None:
    """
    This function is used as an optional postpass. It empties the similarity matrix right after the
    FeatureExtraction step so that functions with the same hash are directly matched and other
    match candidate on the same row are no longer valid. If the hash matches multiple functions then
    all of them will be set as 1 in the similarity matrix.

    :param sim_matrix: The similarity matrix of between the primary and secondary, of
                       type :py:class:`qbindiff.types:SimMatrix`
    :param primary: The primary binary of type :py:class:`qbindiff.loader.Program`
    :param secondary: The secondary binary of type :py:class:`qbindiff.loader.Program`
    :param primary_mapping: Mapping between the primary function addresses and their corresponding index
    :param secondary_mapping: Mapping between the secondary function addresses and their corresponding index
    :param primary_features: Mapping between function addresses and the associated FeatureCollector
                             object for the primary program
    :param secondary_features: Mapping between function addresses and the associated FeatureCollector
                               object for the secondary program
    """

    # Map of hashes (in this case ByteHash stores features values as float, not as dict) to a set
    # of addresses whose functions are associated to that hash. For ease of use the set of addresses
    # is split in two: (primary_functions, secondary_functions)
    hash_map: dict[FeatureValue, tuple[set[Addr], set[Addr]]] = defaultdict(lambda: (set(), set()))

    # Store the hash of the primary functions
    for f1_addr in primary.keys():
        # If no features for this function skip it
        if (p_feature := primary_features.get(f1_addr)) is None:
            continue

        if (byte_hash_value := p_feature.get(BytesHash.key)) is not None:
            hash_map[byte_hash_value][0].add(f1_addr)

    # Store the hash of the secondary functions
    for f2_addr in secondary.keys():
        # If no features for this function skip it
        if (s_feature := secondary_features.get(f2_addr)) is None:
            continue

        if (byte_hash_value := s_feature.get(BytesHash.key)) is not None:
            hash_map[byte_hash_value][1].add(f2_addr)

    # No hash means no feature registered
    if len(hash_map) == 0:
        logging.warning("To use this post-pass, please use BytesHash as a feature")
        return

    # If the hashes of two functions are equals, function are identical.
    # They should be matched.
    # Other match candidate on the same row are no longer valid.
    for primary_addr_set, secondary_addr_set in hash_map.values():
        # At least a match is required, otherwise ignore it
        if len(primary_addr_set) == 0 or len(secondary_addr_set) == 0:
            continue

        # Create the indexes for fast numpy access
        primary_indexes = tuple(map(primary_mapping.get, primary_addr_set))
        secondary_indexes = tuple(map(secondary_mapping.get, secondary_addr_set))

        # Initialize the rows and cols to zero
        sim_matrix[primary_indexes, :] = 0
        sim_matrix[:, secondary_indexes] = 0

        # Set to 1 only the matches
        sim_matrix[primary_indexes, secondary_indexes] = 1


def match_custom_functions(
    sim_matrix: SimMatrix,
    primary: Program,
    secondary: Program,
    primary_mapping: dict[Addr, Idx],
    secondary_mapping: dict[Addr, Idx],
    *,
    custom_anchors: list[tuple[Addr, Addr]],
) -> None:
    """
    Custom Anchoring pass. It enforces the matching between functions using the user
    supplied anchors.
    Determining these anchors can be done by a deeper look at the binaries.

    :param sim_matrix: The similarity matrix of between the primary and secondary, of
                       type :py:class:`qbindiff.types:SimMatrix`
    :param primary: The primary binary of type :py:class:`qbindiff.loader.Program`
    :param secondary: The secondary binary of type :py:class:`qbindiff.loader.Program`
    :param primary_mapping: Mapping between the primary function addresses and their corresponding index
    :param secondary_mapping: Mapping between the secondary function addresses and their corresponding index
    :param custom_anchors: List of tuples where each tuple represent an anchor between
                           two functions (ex: [(addr1, addr2), (addr3, addr4)])
    """

    for addr1, addr2 in custom_anchors:
        if addr1 in primary_mapping and addr2 in secondary_mapping:
            sim_matrix[primary_mapping[addr1], :] = 0
            sim_matrix[:, secondary_mapping[addr2]] = 0
            sim_matrix[primary_mapping[addr1], secondary_mapping[addr2]] = 1
        else:
            logging.warning(f"Addresses are out of bounds: ({addr1:#x}, {addr2:#x})")


def compute_flirt_signature(function: Function) -> int:
    """Recreate FLIRT/FunctionID like hashes. Basically FLIRT hashes are hashs
    of the function bytes without the "address relative" instructions.
    Since we don't have access to the real hash because of the backend, try
    to mimic this feature on our own.
    """

    def is_relative(instr):
        # More checks may come in futur version
        return InstructionGroup.GRP_BRANCH_RELATIVE in instr.groups

    data = b""
    for bb_addr, bb in function.items():
        for instr in bb.instructions:
            # Check if instruction is address relative
            if is_relative(instr):
                data += b"\x00" * len(instr.bytes)
            else:
                data += instr.bytes

    # Return the MD5 value of the masked
    return int(hashlib.md5(data).hexdigest(), 16)


def match_same_flirt_hash(
    sim_matrix: SimMatrix,
    primary: Program,
    secondary: Program,
    primary_mapping: dict[Addr, Idx],
    secondary_mapping: dict[Addr, Idx],
    *args,
    **kwargs,
) -> None:
    """
    FLIRT hash anchoring pass. It enforces the matching between functions that share the same hash.
    If multiple functions shares the same hash, then all of them are set with a similarity of 1.
    This works both as a PrePass and as a PostPass.

    :param sim_matrix: The similarity matrix of between the primary and secondary, of
                       type :py:class:`qbindiff.types:SimMatrix`
    :param primary: The primary binary of type :py:class:`qbindiff.loader.Program`
    :param secondary: The secondary binary of type :py:class:`qbindiff.loader.Program`
    :param primary_mapping: Mapping between the primary function addresses and their corresponding index
    :param secondary_mapping: Mapping between the secondary function addresses and their corresponding index
    """

    matched = 0
    # First compute all the hash from the primary
    hashmap_primary = defaultdict(set)
    for addr, function in primary.items():
        hashmap_primary[compute_flirt_signature(function)].add(addr)

    # First compute all the hash from the primary
    hashmap_secondary = defaultdict(set)
    for addr, function in secondary.items():
        hashmap_secondary[compute_flirt_signature(function)].add(addr)

    # Now try to match with the functions from the secondary
    for h in hashmap_primary.keys() & hashmap_secondary.keys():
        # First zero-out the rows and cols
        for addr in hashmap_primary[h]:
            sim_matrix[primary_mapping[addr], :] = 0
        for addr in hashmap_secondary[h]:
            sim_matrix[:, secondary_mapping[addr]] = 0

        # FLIRT signature matches, we can anchor this match
        for addr1 in hashmap_primary[h]:
            for addr2 in hashmap_secondary[h]:
                sim_matrix[primary_mapping[addr1], secondary_mapping[addr2]] = 1

        matched += len(hashmap_primary[h]) * len(hashmap_secondary[h])

    logging.info(f"{matched} functions were anchored during pass 'match_same_flirt_hash'")
