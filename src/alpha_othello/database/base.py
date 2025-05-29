from sqlalchemy import Column, ForeignKey, Integer, Text, Float
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Completion(Base):
    __tablename__ = "completion"
    id = Column(Integer, primary_key=True, autoincrement=True)
    completion = Column(Text, nullable=False)
    reasoning = Column(Text, nullable=True)


class Inspiration(Base):
    """
    Represents inspiration relationships between completions.
    Each completion can be inspired by multiple other completions.
    Each completion should only be inspired by completions with a lower id, but this is not enforced.
    This is a many-to-many relationship.
    """

    __tablename__ = "inspiration"
    id = Column(Integer, primary_key=True, autoincrement=True)
    completion_id = Column(Integer, ForeignKey("completion.id"), nullable=False)
    inspired_by_id = Column(Integer, ForeignKey("completion.id"), nullable=False)
    completion = relationship(
        "Completion", foreign_keys=[completion_id], backref="inspirations"
    )
    inspired_by = relationship("Completion", foreign_keys=[inspired_by_id])


class Score(Base):
    __tablename__ = "score"
    id = Column(Integer, primary_key=True, autoincrement=True)
    score = Column(Float, nullable=False)
    completion_id = Column(Integer, ForeignKey("completion.id"), nullable=False)
    completion = relationship("Completion", backref="scores")
