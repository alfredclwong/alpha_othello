from typing import Optional

from alpha_othello.database.abstract import AbstractDatabase
from alpha_othello.database.base import Completion, Inspiration, Prompt, Score


class Database(AbstractDatabase):
    def __init__(self, db_url: str):
        super().__init__(db_url)

    def get_topk_completion_ids(self, k: int) -> list[int]:
        session = self.get_session()
        top_completions = (
            session.query(Completion)
            .join(Score)
            .group_by(Completion.id)
            .order_by(Score.score.desc())
            .limit(k)
            .all()
        )
        session.close()
        completion_ids = [getattr(completion, "id") for completion in top_completions]
        return completion_ids

    def get_lastp_completion_ids(self, p: int) -> list[int]:
        session = self.get_session()
        last_completions = (
            session.query(Completion)
            .order_by(Completion.id.desc())
            .limit(p)
            .all()
        )
        session.close()
        completion_ids = [getattr(completion, "id") for completion in last_completions]
        return completion_ids

    def get_all_completion_ids(self) -> list[int]:
        session = self.get_session()
        all_completions = session.query(Completion).all()
        session.close()
        completion_ids = [getattr(completion, "id") for completion in all_completions]
        return completion_ids

    def get_completion_count(self) -> int:
        session = self.get_session()
        count = session.query(Completion).count()
        session.close()
        return count

    def get_completion(self, completion_id: int) -> str:
        session = self.get_session()
        completion = (
            session.query(Completion).filter(Completion.id == completion_id).first()
        )
        session.close()
        return getattr(completion, "completion")

    def get_inspirations(self, completion_id: int) -> list[int]:
        session = self.get_session()
        inspirations = (
            session.query(Inspiration)
            .filter(Inspiration.completion_id == completion_id)
            .all()
        )
        session.close()
        inspiration_ids = [
            getattr(inspiration, "inspired_by_id") for inspiration in inspirations
        ]
        return inspiration_ids

    def get_scores(self, completion_id: int) -> dict[str, float]:
        session = self.get_session()
        scores = session.query(Score).filter(Score.completion_id == completion_id).all()
        session.close()
        score_dict = {
            getattr(score, "name"): getattr(score, "score") for score in scores
        }
        return score_dict

    def get_reasoning(self, completion_id: int) -> str:
        session = self.get_session()
        completion = (
            session.query(Completion).filter(Completion.id == completion_id).first()
        )
        session.close()
        return getattr(completion, "reasoning")

    def get_prompt(self, completion_id: int) -> dict[str, Optional[str]]:
        session = self.get_session()
        completion = (
            session.query(Completion).filter(Completion.id == completion_id).first()
        )
        session.close()
        return {
            "prompt": getattr(completion, "prompt", None),
            "variation": getattr(completion, "variation", None),
        }

    def store_completion(
        self, completion: str, reasoning: Optional[str], inspiration_ids: list[int]
    ) -> int:
        session = self.get_session()
        completion_obj = Completion(completion=completion, reasoning=reasoning)
        session.add(completion_obj)
        session.commit()
        completion_id = getattr(completion_obj, "id")
        for inspiration_id in inspiration_ids:
            inspiration_obj = Inspiration(
                completion_id=completion_id, inspired_by_id=inspiration_id
            )
            session.add(inspiration_obj)
        session.commit()
        session.close()
        return completion_id

    def store_score(self, score: float, name: str, completion_id: int) -> int:
        session = self.get_session()
        score_obj = Score(score=score, name=name, completion_id=completion_id)
        session.add(score_obj)
        session.commit()
        score_id = getattr(score_obj, "id")
        session.close()
        return score_id

    def store_prompt(
        self, prompt: Optional[str], variation: Optional[str], completion_id: int
    ) -> int:
        session = self.get_session()
        prompt_obj = Prompt(
            prompt=prompt, variation=variation, completion_id=completion_id
        )
        session.add(prompt_obj)
        session.commit()
        prompt_id = getattr(prompt_obj, "id")
        session.close()
        return prompt_id
