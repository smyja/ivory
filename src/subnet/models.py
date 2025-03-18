from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    Enum as SQLAlchemyEnum,
    types,
    Sequence,
    Float,
    ForeignKey,
)
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from enum import Enum
from datetime import datetime

Base = declarative_base()
id_seq = Sequence("dataset_id_seq")
category_id_seq = Sequence("category_id_seq")
subcluster_id_seq = Sequence("subcluster_id_seq")
question_id_seq = Sequence("question_id_seq")


class DownloadStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Question(BaseModel):
    question: str
    answer: str


class QuestionList(BaseModel):
    questions: List[Question]


class ClusteredQuestion(Question):
    cluster: int
    cluster_title: str
    category: str


class DatasetRequest(BaseModel):
    dataset_name: str
    split: Optional[str] = "train"
    num_rows: Optional[int] = None
    config: Optional[str] = "default"
    subset: Optional[str] = None


class Category(Base):
    __tablename__ = "categories"

    id = Column(
        Integer,
        category_id_seq,
        server_default=category_id_seq.next_value(),
        primary_key=True,
    )
    name = Column(String, nullable=False)
    total_rows = Column(Integer, nullable=False)
    percentage = Column(Float, nullable=False)

    # Relationship to subclusters
    subclusters = relationship("Subcluster", back_populates="category")


class Subcluster(Base):
    __tablename__ = "subclusters"

    id = Column(
        Integer,
        subcluster_id_seq,
        server_default=subcluster_id_seq.next_value(),
        primary_key=True,
    )
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    title = Column(String, nullable=False)
    row_count = Column(Integer, nullable=False)
    percentage = Column(Float, nullable=False)

    # Relationships
    category = relationship("Category", back_populates="subclusters")
    questions = relationship("QuestionCluster", back_populates="subcluster")


class QuestionDB(Base):
    __tablename__ = "questions"

    id = Column(
        Integer,
        question_id_seq,
        server_default=question_id_seq.next_value(),
        primary_key=True,
    )
    question = Column(String, nullable=False)
    answer = Column(String, nullable=False)

    # Relationship to clusters
    clusters = relationship("QuestionCluster", back_populates="question")


class QuestionCluster(Base):
    __tablename__ = "question_clusters"

    question_id = Column(Integer, ForeignKey("questions.id"), primary_key=True)
    subcluster_id = Column(Integer, ForeignKey("subclusters.id"), primary_key=True)
    membership_score = Column(Float, nullable=False)

    # Relationships
    subcluster = relationship("Subcluster", back_populates="questions")
    question = relationship("QuestionDB", back_populates="clusters")


class DatasetMetadata(Base):
    __tablename__ = "dataset_metadata"

    id = Column(Integer, id_seq, server_default=id_seq.next_value(), primary_key=True)

    name = Column(String, index=True)
    subset = Column(String, nullable=True)
    split = Column(String, nullable=True)
    download_date = Column(types.TIMESTAMP, server_default=func.now())
    is_clustered = Column(Boolean, default=False)
    status = Column(SQLAlchemyEnum(DownloadStatus), default=DownloadStatus.PENDING)


class CategoryResponse(BaseModel):
    id: int
    name: str
    total_rows: int
    percentage: float
    model_config = ConfigDict(from_attributes=True)


class SubclusterResponse(BaseModel):
    id: int
    category_id: int
    title: str
    row_count: int
    percentage: float
    model_config = ConfigDict(from_attributes=True)


class DatasetMetadataResponse(BaseModel):
    id: int
    name: str
    subset: Optional[str]
    split: Optional[str]
    download_date: datetime
    is_clustered: bool
    status: DownloadStatus

    model_config = ConfigDict(from_attributes=True)


# Configure Pydantic to work with SQLAlchemy models
class SQLModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Config:
    arbitrary_types_allowed = True
