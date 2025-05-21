from sqlalchemy import Column, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class LLM(Base):
    __tablename__ = "llm"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    completions = relationship("Completion", back_populates="llm")


class Prompt(Base):
    __tablename__ = "prompt"
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt = Column(Text, nullable=False)
    inspiration_ids = Column(String)  # store as comma-separated string
    completions = relationship("Completion", back_populates="prompt")


class Completion(Base):
    __tablename__ = "completion"
    id = Column(Integer, primary_key=True, autoincrement=True)
    completion = Column(Text, nullable=False)
    llm_id = Column(Integer, ForeignKey("llm.id"), nullable=False)
    prompt_id = Column(Integer, ForeignKey("prompt.id"), nullable=False)
    llm = relationship("LLM", back_populates="completions")
    prompt = relationship("Prompt", back_populates="completions")
    scores = relationship("Score", back_populates="completion")


class Score(Base):
    __tablename__ = "score"
    id = Column(Integer, primary_key=True, autoincrement=True)
    score = Column(Integer, nullable=False)
    completion_id = Column(Integer, ForeignKey("completion.id"), nullable=False)
    completion = relationship("Completion", back_populates="scores")


class SQLiteDatabase:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # If LLM or Prompt tables are empty, add a null entries
        session = self.get_session()
        llm_count = session.query(LLM).count()
        prompt_count = session.query(Prompt).count()
        if llm_count == 0:
            null_llm = LLM(name="null")
            session.add(null_llm)
        if prompt_count == 0:
            null_prompt = Prompt(prompt="null", inspiration_ids="")
            session.add(null_prompt)
        session.commit()

    def get_session(self):
        return self.Session()

    def close(self):
        self.engine.dispose()

    def get_topk_completions(self, k: int) -> list[int]:
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

    def store_llm(self, name: str) -> int:
        session = self.get_session()
        # If the LLM already exists, don't add it again
        llm = session.query(LLM).filter(LLM.name == name).first()
        if not llm:
            llm = LLM(name=name)
            session.add(llm)
            session.commit()
        llm_id = getattr(llm, "id")
        session.close()
        return llm_id

    def store_prompt(self, prompt: str, inspiration_ids: list[int]) -> int:
        session = self.get_session()
        inspiration_ids_str = ",".join(map(str, inspiration_ids))
        prompt_obj = Prompt(prompt=prompt, inspiration_ids=inspiration_ids_str)
        session.add(prompt_obj)
        session.commit()
        prompt_id = getattr(prompt_obj, "id")
        session.close()
        return prompt_id

    def store_completion(self, completion: str, llm_id: int, prompt_id: int) -> int:
        session = self.get_session()
        completion_obj = Completion(
            completion=completion, llm_id=llm_id, prompt_id=prompt_id
        )
        session.add(completion_obj)
        session.commit()
        completion_id = getattr(completion_obj, "id")
        session.close()
        return completion_id

    def store_score(self, score: int, completion_id: int) -> int:
        session = self.get_session()
        score_obj = Score(score=score, completion_id=completion_id)
        session.add(score_obj)
        session.commit()
        score_id = getattr(score_obj, "id")
        session.close()
        return score_id

    def get_llm(self, llm_id: int) -> str:
        session = self.get_session()
        llm = session.query(LLM).filter(LLM.id == llm_id).first()
        session.close()
        return getattr(llm, "name")

    def get_prompt(self, prompt_id: int) -> str:
        session = self.get_session()
        prompt = session.query(Prompt).filter(Prompt.id == prompt_id).first()
        session.close()
        return getattr(prompt, "prompt")

    def get_completion(self, completion_id: int) -> str:
        session = self.get_session()
        completion = (
            session.query(Completion).filter(Completion.id == completion_id).first()
        )
        session.close()
        return getattr(completion, "completion")

    def get_score(self, score_id: int) -> int:
        session = self.get_session()
        score = session.query(Score).filter(Score.id == score_id).first()
        session.close()
        return getattr(score, "score")

    def get_all_llms(self) -> list[str]:
        session = self.get_session()
        llms = session.query(LLM).all()
        session.close()
        return [getattr(llm, "name") for llm in llms]
