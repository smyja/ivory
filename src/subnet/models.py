from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    Enum as SQLAlchemyEnum,
    types,
    Float,
    ForeignKey,
    DateTime,
    Sequence,
)
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from enum import Enum
from datetime import datetime

Base = declarative_base()

# Define sequences
dataset_id_seq = Sequence("dataset_metadata_id_seq")
category_id_seq = Sequence("categories_id_seq")
subcluster_id_seq = Sequence("subclusters_id_seq")
text_id_seq = Sequence("texts_id_seq")
clustering_history_id_seq = Sequence("clustering_history_id_seq")


class DownloadStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Text(BaseModel):
    text: str


class TextList(BaseModel):
    texts: List[Text]


class ClusteredText(Text):
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
    dataset_id = Column(Integer, ForeignKey("dataset_metadata.id"), nullable=False)
    name = Column(String, nullable=False)
    total_rows = Column(Integer, nullable=False)
    percentage = Column(Float, nullable=False)
    version = Column(Integer, nullable=False, default=1)

    # Relationships
    dataset = relationship("DatasetMetadata", back_populates="categories")
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
    version = Column(Integer, nullable=False, default=1)

    # Relationships
    category = relationship("Category", back_populates="subclusters")
    texts = relationship("TextCluster", back_populates="subcluster")


class TextDB(Base):
    __tablename__ = "texts"

    id = Column(
        Integer, text_id_seq, server_default=text_id_seq.next_value(), primary_key=True
    )
    text = Column(String, nullable=False)

    # Relationship to clusters
    clusters = relationship("TextCluster", back_populates="text")


class TextCluster(Base):
    __tablename__ = "text_clusters"

    text_id = Column(Integer, ForeignKey("texts.id"), primary_key=True)
    subcluster_id = Column(Integer, ForeignKey("subclusters.id"), primary_key=True)
    membership_score = Column(Float, nullable=False)

    # Relationships
    subcluster = relationship("Subcluster", back_populates="texts")
    text = relationship("TextDB", back_populates="clusters")


class DatasetMetadata(Base):
    __tablename__ = "dataset_metadata"

    id = Column(
        Integer,
        dataset_id_seq,
        server_default=dataset_id_seq.next_value(),
        primary_key=True,
    )
    name = Column(String, index=True)
    subset = Column(String, nullable=True)
    split = Column(String, nullable=True)
    download_date = Column(DateTime, default=datetime.utcnow)
    is_clustered = Column(Boolean, default=False)
    status = Column(SQLAlchemyEnum(DownloadStatus), default=DownloadStatus.PENDING)
    clustering_status = Column(String, nullable=True, default=None)

    # Add relationships
    categories = relationship("Category", back_populates="dataset")
    clustering_attempts = relationship("ClusteringHistory", back_populates="dataset")


class ClusteringHistory(Base):
    __tablename__ = "clustering_history"

    id = Column(
        Integer,
        clustering_history_id_seq,
        server_default=clustering_history_id_seq.next_value(),
        primary_key=True,
    )
    dataset_id = Column(Integer, ForeignKey("dataset_metadata.id"))
    clustering_status = Column(String, nullable=False)
    titling_status = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=False), nullable=False)
    completed_at = Column(DateTime(timezone=False))
    error_message = Column(String)
    clustering_version = Column(Integer, nullable=False, default=1)

    # Relationship with DatasetMetadata
    dataset = relationship("DatasetMetadata", back_populates="clustering_attempts")


# Pydantic models for responses
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


class DatasetStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"


class ClusteringStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class DatasetResponse(BaseModel):
    id: int
    name: str
    subset: Optional[str] = None
    split: Optional[str] = None
    status: str
    download_date: datetime
    is_clustered: bool
    clustering_status: Optional[str] = "not_started"

    class Config:
        from_attributes = True
