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

"""BinDiff file format
"""

from collections import defaultdict
from collections.abc import Generator
from functools import lru_cache

# third-party imports
from bindiff import BindiffFile

# local imports
# from qbindiff import __version__
from qbindiff.loader import Program, Function, BasicBlock
from qbindiff.types import Addr

# from qbindiff import Mapping


@lru_cache
def primes() -> list[int]:
    """
    The primes up to 1'000'000 using the segmented sieve algorithm
    """

    n = 1000000
    S = 30000
    nsqrt = round(n**0.5)

    is_prime = [True] * (nsqrt + 1)
    low_primes = []
    start_indices = []
    for i in range(3, nsqrt + 1, 2):
        if is_prime[i]:
            low_primes.append(i)
            start_indices.append(i * i // 2)
            for j in range(i * i, nsqrt + 1, 2 * i):
                is_prime[j] = False

    primes = [2]
    high = (n - 1) // 2
    for low in range(0, high + 1, S):
        block = [True] * S
        for i in range(len(low_primes)):
            p = low_primes[i]
            idx = start_indices[i]
            while idx < S:
                block[idx] = False
                idx += p
            start_indices[i] = idx - S
        if low == 0:
            block[0] = False
        i = 0
        while i < S and low + i <= high:
            if block[i]:
                primes.append((low + i) * 2 + 1)
            i += 1
    return primes


def _compute_bb_prime_product(basic_block: BasicBlock) -> int:
    """
    Calculate the prime product value of the basic block.

    :param basic_block: Basic block on which to evaluate.
    :return: prime product
    """

    tot = 1
    for instruction in basic_block:
        tot *= primes()[instruction.id]
    return tot


def compute_basic_block_match(
    primary_func: Function, secondary_func: Function
) -> Generator[tuple[Addr, Addr]]:
    """
    Matches the basic blocks between the two functions

    :param primary_func: function in the primary
    :param secondary_func: function in the secondary
    :return: matches between basic blocks of the two functions
    """

    primary_hash = defaultdict(list)  # {prime product -> addrs}
    secondary_hash = defaultdict(list)  # {prime product -> addrs}

    # compute hashes on primary
    for bb_addr, basic_block in primary_func.items():
        pp = _compute_bb_prime_product(basic_block)
        primary_hash[pp].append(bb_addr)

    # compute hashes on secondary
    for bb_addr, basic_block in secondary_func.items():
        pp = _compute_bb_prime_product(basic_block)
        secondary_hash[pp].append(bb_addr)

    matches = primary_hash.keys() & secondary_hash.keys()
    for h in matches:
        yield from zip(primary_hash[h], secondary_hash[h])


def compute_instruction_match(
    primary_bb: BasicBlock, secondary_bb: BasicBlock
) -> Generator[tuple[Addr, Addr]]:
    """
    Matches the instructions between the two basic blocks

    :param primary_bb: basic block in the primary
    :param secondary_bb: basic block in the secondary
    :return: matches between instructions of the two basic blocks
    """

    primary_instr = defaultdict(list)
    secondary_instr = defaultdict(list)
    for instr in primary_bb:
        primary_instr[instr.bytes].append(instr.addr)
    for instr in secondary_bb:
        secondary_instr[instr.bytes].append(instr.addr)
    common = primary_instr.keys() & secondary_instr.keys()
    for k in common:
        yield from zip(primary_instr[k], secondary_instr[k])


def export_to_bindiff(
    filename: str, primary: Program, secondary: Program, mapping: "Mapping"
) -> None:
    """
    Exports diffing results inside the BinDiff format

    :param filename: Name of the output diffing file
    :param primary: primary program
    :param secondary: secondary program
    :param mapping: diffing mapping between the two programs
    """
    from qbindiff import __version__  # import the version here to avoid circular definition

    def count_items(program: Program) -> tuple[int, int, int, int]:
        fp, flib, bbs, inst = 0, 0, 0, 0
        for f_addr, f in program.items():
            fp += int(not (f.is_import()))
            flib += int(f.is_import())
            bbs += len(f)
            inst += sum(len(x) for x in f)
        return fp, flib, bbs, inst

    binfile = BindiffFile.create(
        filename,
        primary.exec_path,
        secondary.exec_path,
        f"Qbindiff {__version__}",
        "",
        mapping.normalized_similarity,
        0.0,
    )

    for m in mapping:  # iterate all the matchs
        with m.primary, m.secondary:  # Do not unload basic blocks
            # Add the function match
            faddr1, faddr2 = m.primary.addr, m.secondary.addr

            # Compute the basic block match (bindiff style) and add it in database
            same_bb_count = 0
            bb_matches = compute_basic_block_match(m.primary, m.secondary)
            for addr1, addr2 in bb_matches:
                bb1, bb2 = m.primary[addr1], m.secondary[addr2]
                same_bb_count += 1
                entry_id = binfile.add_basic_block_match(faddr1, faddr2, addr1, addr2)

                # Compute the instruction match (bindiff style) and add it in database
                for instr_addr1, instr_addr2 in compute_instruction_match(bb1, bb2):
                    binfile.add_instruction_match(entry_id, instr_addr1, instr_addr2)

            # Add the function match here to provide the same_bb_count
            binfile.add_function_match(
                faddr1,
                faddr2,
                m.primary.name,
                m.secondary.name,
                float(m.similarity),
                float(m.confidence),
                same_bb_count,
            )

    # Update file infos about primary
    f, lib, bbs, insts = count_items(primary)
    binfile.update_file_infos(1, f, lib, bbs, insts)
    # Update file infos about secondary
    f, lib, bbs, insts = count_items(secondary)
    binfile.update_file_infos(2, f, lib, bbs, insts)

    # binfile.commit()
