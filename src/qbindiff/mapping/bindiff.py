import sqlite3
import hashlib
import datetime
from collections.abc import Generator
from collections import defaultdict
from functools import cached_property
from typing import List, Tuple

from qbindiff.loader import Program, Function, BasicBlock
from qbindiff.mapping.mapping import Mapping
from qbindiff.types import Addr, Match


class BinDiffFormat:
    """
    Helper class to export the diffing result to BinDiff file format
    """

    def __init__(
        self, filename: str, primary: Program, secondary: Program, mapping: Mapping
    ):
        # Create a new file
        open(filename, "w").close()
        #: Connection to the database
        self.db = sqlite3.connect(filename)
        self.db.row_factory = sqlite3.Row
        #: Primary program
        self.primary = primary
        #: Secondary program
        self.secondary = secondary
        #: Diff mapping
        self.mapping = mapping

        # Properties loaded at run time
        self._primary_instructions = 0
        self._primary_basic_blocks = 0
        self._primary_functions = 0
        self._primary_lib_functions = 0
        self._secondary_instructions = 0
        self._secondary_basic_blocks = 0
        self._secondary_functions = 0
        self._secondary_lib_functions = 0

        self.init_database()

    def __del__(self) -> None:
        self.db.close()

    @property
    def version(self) -> str:
        return "QBinDiff 0.2"

    def init_database(self) -> None:
        """
        Initialize the database by creating all the tables
        
        :return: None
        """

        conn = self.db.cursor()
        conn.execute(
            """
            CREATE TABLE file (
                id INTEGER PRIMARY KEY,
                filename TEXT,
                exefilename TEXT,
                hash CHARACTER(40),
                functions INT,
                libfunctions INT,
                calls INT,
                basicblocks INT,
                libbasicblocks INT,
                edges INT,
                libedges INT,
                instructions INT,
                libinstructions INT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE metadata (
                version TEXT,
                file1 INTEGER,
                file2 INTEGER,
                description TEXT,
                created DATE,
                modified DATE,
                similarity DOUBLE PRECISION,
                confidence DOUBLE PRECISION,
                FOREIGN KEY(file1) REFERENCES file(id),
                FOREIGN KEY(file2) REFERENCES file(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE functionalgorithm (id SMALLINT PRIMARY KEY, name TEXT)
            """
        )
        conn.execute(
            """
            CREATE TABLE function (
                id INTEGER PRIMARY KEY,
                address1 BIGINT,
                name1 TEXT,
                address2 BIGINT,
                name2 TEXT,
                similarity DOUBLE PRECISION,
                confidence DOUBLE PRECISION,
                flags INTEGER,
                algorithm SMALLINT,
                evaluate BOOLEAN,
                commentsported BOOLEAN,
                basicblocks INTEGER,
                edges INTEGER,
                instructions INTEGER,
                UNIQUE(address1, address2),
                FOREIGN KEY(algorithm) REFERENCES functionalgorithm(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE basicblockalgorithm (id INTEGER PRIMARY KEY, name TEXT)
            """
        )
        conn.execute(
            """
            CREATE TABLE basicblock (
                id INTEGER,
                functionid INT,
                address1 BIGINT,
                address2 BIGINT,
                algorithm SMALLINT,
                evaluate BOOLEAN,
                PRIMARY KEY(id),
                FOREIGN KEY(functionid) REFERENCES function(id),
                FOREIGN KEY(algorithm) REFERENCES basicblockalgorithm(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE instruction (
                basicblockid INT,
                address1 BIGINT,
                address2 BIGINT,
                FOREIGN KEY(basicblockid) REFERENCES basicblock(id)
            )
            """
        )

        self.db.commit()

        conn.execute(
            """
            INSERT INTO basicblockalgorithm(name) VALUES ("basicBlock: edges prime product")
            """
        )

        self.db.commit()

    @cached_property
    def primes(self) -> List[int]:
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

    def _prime_product(self, basic_block: BasicBlock) -> Tuple[int, int]:
        """
        Calculate the prime product value of the basic block.
        The function will return a tuple where the first element is the prime product
        and the second is the number of instructions

        :param basic_block: Basic block to consider
        :return: prime product, number of instructions
        """

        tot = 1
        count = 0
        for instruction in basic_block:
            count += 1
            tot *= self.primes[instruction.id]
        return (tot, count)

    def _basic_block_match(
        self, primary_func: Function, secondary_func: Function
    ) -> Generator[Tuple[Addr, Addr]]:
        """
        Matches the basic blocks between the two functions

        :param primary_func: function in the primary
        :param secondary_func: function in the secondary
        :return: matches between basic blocks of the two functions
        """

        primary_hash = defaultdict(list)  # {prime product -> addrs}
        secondary_hash = defaultdict(list)  # {prime product -> addrs}
        for bb_addr, basic_block in primary_func.items():
            self._primary_basic_blocks += 1
            pp, instr_tot = self._prime_product(basic_block)
            self._primary_instructions += instr_tot
            primary_hash[pp].append(bb_addr)
        for bb_addr, basic_block in secondary_func.items():
            self._secondary_basic_blocks += 1
            pp, instr_tot = self._prime_product(basic_block)
            self._secondary_instructions += instr_tot
            secondary_hash[pp].append(bb_addr)

        matches = primary_hash.keys() & secondary_hash.keys()
        for h in matches:
            yield from zip(primary_hash[h], secondary_hash[h])

    def _instruction_match(
        self, primary_bb: BasicBlock, secondary_bb: BasicBlock
    ) -> Generator[Tuple[Addr, Addr]]:
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

    def save_file(self, program: Program) -> None:
        """
        Save the file `program` in database

        :param program: program to save
        :return: None
        """

        conn = self.db.cursor()

        params = {"filename": program.name, "hash": None}
        if program.exec_path:
            with open(program.exec_path, "rb") as f:
                params["hash"] = hashlib.sha256(f.read()).hexdigest()

        conn.execute(
            """
            INSERT INTO file
                (filename, exefilename, hash)
            VALUES
                (:filename, :filename, :hash)
            """,
            params,
        )

    def save_match(self, match: Match) -> None:
        """
        Save in the BinDiff file the specified match (aka a pair of function)

        :param match: match to save
        :return: None
        """

        conn = self.db.cursor()

        # Function

        params = {
            "address1": match.primary.addr,
            "address2": match.secondary.addr,
            "name1": match.primary.name,
            "name2": match.secondary.name,
            "similarity": float(match.similarity),
            "confidence": float(match.confidence),
        }
        conn.execute(
            """
            INSERT INTO function
                (address1, address2, name1, name2, similarity, confidence, basicblocks)
            VALUES
                (:address1, :address2, :name1, :name2, :similarity, :confidence, 0)
            """,
            params,
        )

        # Basic Block & Instruction

        same_bb_count = 0
        bb_matches = self._basic_block_match(match.primary, match.secondary)
        for addr1, addr2 in bb_matches:
            same_bb_count += 1
            params = {
                "function_address1": match.primary.addr,
                "function_address2": match.secondary.addr,
                "address1": addr1,
                "address2": addr2,
                "algorithm": "1",
            }

            conn.execute(
                """
                INSERT INTO basicblock
                    (functionid, address1, address2, algorithm)
                VALUES
                    ((SELECT id FROM function WHERE address1=:function_address1 AND address2=:function_address2), :address1, :address2, :algorithm)
                """,
                params,
            )

            basicblock_id = conn.lastrowid
            for instr_addr1, instr_addr2 in self._instruction_match(
                match.primary[addr1], match.secondary[addr2]
            ):
                params = {
                    "address1": instr_addr1,
                    "address2": instr_addr2,
                    "basicblockid": basicblock_id,
                }
                conn.execute(
                    """
                    INSERT INTO instruction
                        (basicblockid, address1, address2)
                    VALUES
                        (:basicblockid, :address1, :address2)
                    """,
                    params,
                )

        # Set number of equal basic blocks
        conn.execute(
            """
            UPDATE function
            SET basicblocks = :basicblocks
            WHERE id = (SELECT id FROM function WHERE address1=:function_address1 AND address2=:function_address2)
            """,
            {
                "basicblocks": same_bb_count,
                "function_address1": match.primary.addr,
                "function_address2": match.secondary.addr,
            },
        )

    def save(self) -> None:
        """
        Save the entire diffing result in the file

        :return: None
        """
        
        self.save_file(self.primary)
        self.save_file(self.secondary)

        conn = self.db.cursor()

        # Metadata

        params = {
            "version": self.version,
            "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "similarity": self.mapping.normalized_similarity,
        }
        conn.execute(
            """
            INSERT INTO metadata
                (version, file1, file2, created, similarity)
            VALUES
                (:version, 1, 2, :created, :similarity)
            """,
            params,
        )

        # Functions & Basic Block & Instruction

        for match in self.mapping:
            if match.primary.is_import():
                self._primary_lib_functions += 1
            else:
                self._primary_functions += 1
            if match.secondary.is_import():
                self._secondary_lib_functions += 1
            else:
                self._secondary_functions += 1

            with match.primary, match.secondary:  # Do not unload basic blocks
                self.save_match(match)
            self.db.commit()

        conn.execute(
            """
            UPDATE file
            SET functions = :functions, libfunctions = :libfunctions, basicblocks = :basicblocks, instructions = :instructions
            WHERE id = 1
            """,
            {
                "functions": self._primary_functions,
                "libfunctions": self._primary_lib_functions,
                "basicblocks": self._primary_basic_blocks,
                "instructions": self._primary_instructions,
            },
        )
        conn.execute(
            """
            UPDATE file
            SET functions = :functions, libfunctions = :libfunctions, basicblocks = :basicblocks, instructions = :instructions
            WHERE id = 2
            """,
            {
                "functions": self._secondary_functions,
                "libfunctions": self._secondary_lib_functions,
                "basicblocks": self._secondary_basic_blocks,
                "instructions": self._secondary_instructions,
            },
        )
        self.db.commit()
