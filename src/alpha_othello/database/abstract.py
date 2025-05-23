from abc import ABC, abstractmethod
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from alpha_othello.database.base import Base


class AbstractDatabase(ABC):
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self):
        return self.Session()

    def close(self):
        self.engine.dispose()

    @abstractmethod
    def get_topk_completion_ids(self, k: int) -> list[int]:
        pass

    @abstractmethod
    def get_completion(self, completion_id: int) -> str:
        pass

    @abstractmethod
    def get_inspirations(self, completion_id: int) -> list[int]:
        pass

    @abstractmethod
    def get_score(self, completion_id: int) -> int:
        pass

    @abstractmethod
    def get_reasoning(self, completion_id: int) -> str:
        pass

    @abstractmethod
    def store_completion(
        self, completion: str, reasoning: Optional[str], inspiration_ids: list[int]
    ) -> int:
        pass

    @abstractmethod
    def store_score(self, score: int, completion_id: int) -> int:
        pass
