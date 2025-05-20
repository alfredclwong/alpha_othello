import sqlite3
from abc import ABC, abstractmethod
from typing import Optional


class Database(ABC):
    conn: sqlite3.Connection

    @abstractmethod
    def get_topk_completions(self, k: int) -> list[int]:
        """Get the completion ids of the top k scoring completions."""
        pass

    @abstractmethod
    def store_llm(self, llm_name: str) -> int:
        pass

    @abstractmethod
    def store_completion(self, completion: str, llm_id: int, prompt_id: int) -> int:
        pass

    @abstractmethod
    def store_score(self, score: int, completion_id: int) -> int:
        pass

    @abstractmethod
    def store_prompt(self, prompt: str, inspiration_ids: list[int]) -> int:
        pass

    @abstractmethod
    def get_llm(self, llm_id: int) -> Optional[str]:
        pass

    @abstractmethod
    def get_completion(self, completion_id: int) -> Optional[str]:
        pass

    @abstractmethod
    def get_score(self, completion_id: int) -> Optional[int]:
        pass

    @abstractmethod
    def get_prompt(self, prompt_id: int) -> Optional[str]:
        pass


class SQLiteDatabase(Database):
    def __init__(self, db_path: str = "alpha_othello.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def __del__(self):
        self.conn.close()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                inspiration_ids TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS completion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                completion TEXT NOT NULL,
                llm_id TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                FOREIGN KEY(llm_id) REFERENCES llm(id),
                FOREIGN KEY(prompt_id) REFERENCES prompt(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS score (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                score INTEGER NOT NULL,
                completion_id TEXT NOT NULL,
                FOREIGN KEY(completion_id) REFERENCES completion(id)
            )
        """)
        self.conn.commit()

    def get_topk_completions(self, k: int) -> list[int]:
        cursor = self.conn.cursor()
        query = """
            SELECT completion_id
            FROM score
            LEFT JOIN completion ON score.completion_id = completion.id
            ORDER BY score DESC
            LIMIT ?
        """
        cursor.execute(query, (k,))
        results = cursor.fetchall()
        return [row[0] for row in results]

    def store_llm(self, llm_name: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO llm (name) VALUES (?)",
            (llm_name,),
        )
        self.conn.commit()
        llm_id = cursor.lastrowid
        if llm_id is None:
            raise ValueError("Failed to store LLM.")
        return llm_id
    
    def store_completion(self, completion: str, llm_id: int, prompt_id: int) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO completion (completion, llm_id, prompt_id) VALUES (?, ?, ?)",
            (completion, llm_id, prompt_id),
        )
        self.conn.commit()
        completion_id = cursor.lastrowid
        if completion_id is None:
            raise ValueError("Failed to store completion.")
        return completion_id
    
    def store_score(self, score: int, completion_id: int) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO score (score, completion_id) VALUES (?, ?)",
            (score, completion_id),
        )
        self.conn.commit()
        score_id = cursor.lastrowid
        if score_id is None:
            raise ValueError("Failed to store score.")
        return score_id
    
    def store_prompt(self, prompt: str, inspiration_ids: list[int]) -> int:
        cursor = self.conn.cursor()
        inspiration_ids_str = ",".join(map(str, inspiration_ids))
        cursor.execute(
            "INSERT INTO prompt (prompt, inspiration_ids) VALUES (?, ?)",
            (prompt, inspiration_ids_str),
        )
        self.conn.commit()
        prompt_id = cursor.lastrowid
        if prompt_id is None:
            raise ValueError("Failed to store prompt.")
        return prompt_id
    
    def get_llm(self, llm_id: int) -> Optional[str]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT name FROM llm WHERE id = ?",
            (llm_id,),
        )
        result = cursor.fetchone()
        if result is None:
            return None
        return result[0]
    
    def get_completion(self, completion_id: int) -> Optional[str]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT completion FROM completion WHERE id = ?",
            (completion_id,),
        )
        result = cursor.fetchone()
        if result is None:
            return None
        return result[0]
    
    def get_score(self, completion_id: int) -> Optional[int]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT score FROM score WHERE completion_id = ?",
            (completion_id,),
        )
        result = cursor.fetchone()
        if result is None:
            return None
        return result[0]
    
    def get_prompt(self, prompt_id: int) -> Optional[str]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT prompt FROM prompt WHERE id = ?",
            (prompt_id,),
        )
        result = cursor.fetchone()
        if result is None:
            return None
        return result[0]
