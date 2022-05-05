import sqlite3, hashlib, datetime
from collections import defaultdict

from qbindiff.loader import Program
from qbindiff.mapping.mapping import Mapping


class BinDiffFormat:
    """Helper class to export the diffing result to BinDiff file format"""

    def __init__(
        self, filename: str, primary: Program, secondary: Program, mapping: Mapping
    ):
        # Create a new file
        open(filename, "w").close()

        self.db = sqlite3.connect(filename)
        self.primary = primary
        self.secondary = secondary
        self.mapping = mapping

        self.init_database()

    def __del__(self):
        self.db.close()

    @property
    def version(self):
        return "QBinDiff 0.2"

    def init_database(self):
        """Initialize the database by creating all the tables"""

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
                id INT,
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
                PRIMARY KEY(id),
                FOREIGN KEY(algorithm) REFERENCES functionalgorithm(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE basicblockalgorithm (id SMALLINT PRIMARY KEY, name TEXT)
            """
        )
        conn.execute(
            """
            CREATE TABLE basicblock (
                id INT,
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

    def save_file(self, program: Program) -> None:
        """Save the file `program` in database"""

        conn = self.db.cursor()

        params = defaultdict(lambda: None)
        params["filename"] = program.name
        if program.exec_path:
            with open(program.exec_path, "rb") as f:
                params["hash"] = hashlib.sha256(f.read()).hexdigest()

        import_func = 0
        library_func = 0
        tot_func = 0
        tot_bb = 0
        tot_instr = 0
        for addr, func in program.items():
            if func.is_import():
                import_func += 1
            else:
                tot_func += 1
            if func.is_library():
                library_func += 1

            for bb_addr, bb in func.items():
                tot_bb += 1
                for instr in bb:
                    tot_instr += 1
        params["functions"] = tot_func
        params["libfunctions"] = import_func
        params["basicblocks"] = tot_bb
        params["instructions"] = tot_instr

        conn.execute(
            """
            INSERT INTO file
                (filename, exefilename, hash, functions, libfunctions, basicblocks, instructions)
            VALUES
                (:filename, :filename, :hash, :functions, :libfunctions, :basicblocks, :instructions)
            """,
            params,
        )

    def save(self):
        """Save the entire diffing result in the file"""

        self.save_file(self.primary)
        self.save_file(self.secondary)

        conn = self.db.cursor()

        # Metadata

        params = {
            "version": self.version,
            "created": datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
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

        # Functions

        idx = 1
        for match in self.mapping:
            params = {}
            params["id"] = idx
            params["address1"] = match.primary.addr
            params["address2"] = match.secondary.addr
            params["name1"] = match.primary.name
            params["name2"] = match.secondary.name
            params["similarity"] = float(match.similarity)
            params["confidence"] = 1

            conn.execute(
                """
                INSERT INTO function
                    (id, address1, address2, name1, name2, similarity, confidence)
                VALUES
                    (:id, :address1, :address2, :name1, :name2, :similarity, :confidence)
                """,
                params,
            )

            idx += 1

        self.db.commit()
