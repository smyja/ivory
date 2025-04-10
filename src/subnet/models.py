import uuid
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    ForeignKey,
    MetaData,
    Sequence,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

Base = declarative_base()

# --- Sequences ---
dataset_id_seq = Sequence("dataset_metadata_id_seq")
text_id_seq = Sequence("texts_id_seq")
category_id_seq = Sequence("categories_id_seq")
l1_cluster_id_seq = Sequence("level1_clusters_id_seq")
assignment_id_seq = Sequence("text_assignments_id_seq")
history_id_seq = Sequence("clustering_history_id_seq")  # Keep history seq

# --- Core Tables ---


class DatasetMetadata(Base):
    __tablename__ = "dataset_metadata"

    id = Column(
        Integer,
        dataset_id_seq,
        server_default=dataset_id_seq.next_value(),
        primary_key=True,
    )
    name = Column(String, nullable=False, unique=True)
    description = Column(String)
    source = Column(String)  # e.g., 'url', 'upload', 'huggingface'
    identifier = Column(String)  # e.g., URL or original filename
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(
        String, default="pending"
    )  # pending, downloading, downloaded, failed, verifying, verified
    verification_status = Column(
        String, default="pending"
    )  # pending, checking, valid, invalid
    clustering_status = Column(
        String, default="pending"
    )  # pending, processing, completed, failed
    is_clustered = Column(
        Boolean, default=False
    )  # Flag for successful clustering completion
    total_rows = Column(Integer)
    error_message = Column(String)  # For download/verification errors
    file_path = Column(String)  # Path where the dataset is stored locally

    # New fields for Hugging Face integration
    hf_dataset_name = Column(
        String
    )  # The actual dataset name on HF (e.g., "databricks/databricks-dolly-15k")
    hf_config = Column(String)  # Dataset configuration name
    hf_split = Column(String)  # Dataset split (train, test, validation)
    hf_revision = Column(String)  # Dataset version/revision

    # Schema information for flexible datasets
    dataset_schema = Column(String)  # JSON string representing dataset schema
    field_mappings = Column(
        String
    )  # JSON mapping of HF fields to our system (e.g., {"text_field": "content"})

    # Relationships
    texts = relationship("TextDB", back_populates="dataset")
    categories = relationship("Category", back_populates="dataset")
    clustering_history = relationship("ClusteringHistory", back_populates="dataset")


class TextDB(Base):
    __tablename__ = "texts"

    id = Column(
        Integer, text_id_seq, server_default=text_id_seq.next_value(), primary_key=True
    )
    dataset_id = Column(
        Integer, ForeignKey("dataset_metadata.id"), nullable=False, index=True
    )
    text = Column(String, nullable=False)

    # Unique constraint per dataset
    __table_args__ = (
        UniqueConstraint("dataset_id", "text", name="uq_text_per_dataset"),
    )

    # Relationships
    dataset = relationship("DatasetMetadata", back_populates="texts")
    assignment = relationship(
        "TextAssignment", back_populates="text", uselist=False
    )  # One-to-one


# --- Clustering Result Tables ---


class Category(Base):
    __tablename__ = "categories"

    id = Column(
        Integer,
        category_id_seq,
        server_default=category_id_seq.next_value(),
        primary_key=True,
    )
    dataset_id = Column(
        Integer, ForeignKey("dataset_metadata.id"), nullable=False, index=True
    )
    version = Column(Integer, nullable=False, index=True)  # Clustering version
    l2_cluster_id = Column(
        Integer, nullable=False
    )  # The ID assigned by L2 HDBSCAN (-1 for noise/unclustered)
    name = Column(String, nullable=False)  # Generated category name
    # total_l1_clusters = Column(Integer) # Count of L1 clusters belonging to this category

    # Relationships
    dataset = relationship("DatasetMetadata", back_populates="categories")
    level1_clusters = relationship("Level1Cluster", back_populates="category")

    # Unique constraint per dataset and version
    __table_args__ = (
        UniqueConstraint(
            "dataset_id", "version", "l2_cluster_id", name="uq_category_per_version"
        ),
        Index("ix_category_dataset_version", "dataset_id", "version"),
    )


class Level1Cluster(Base):
    __tablename__ = "level1_clusters"

    id = Column(
        Integer,
        l1_cluster_id_seq,
        server_default=l1_cluster_id_seq.next_value(),
        primary_key=True,
    )
    category_id = Column(
        Integer, ForeignKey("categories.id"), nullable=False, index=True
    )  # Link to parent Category
    version = Column(
        Integer, nullable=False, index=True
    )  # Clustering version (denormalized for easier querying)
    l1_cluster_id = Column(
        Integer, nullable=False
    )  # The ID assigned by L1 HDBSCAN (-1 for noise)
    title = Column(String, nullable=False)  # Generated title for this L1 cluster
    # total_texts = Column(Integer) # Count of texts directly assigned here

    # Relationships
    category = relationship("Category", back_populates="level1_clusters")
    text_assignments = relationship("TextAssignment", back_populates="level1_cluster")

    # Unique constraint per category and version
    __table_args__ = (
        UniqueConstraint(
            "category_id",
            "version",
            "l1_cluster_id",
            name="uq_l1_cluster_per_category_version",
        ),
        Index("ix_l1_cluster_category_version", "category_id", "version"),
    )


class TextAssignment(Base):
    __tablename__ = "text_assignments"

    id = Column(
        Integer,
        assignment_id_seq,
        server_default=assignment_id_seq.next_value(),
        primary_key=True,
    )
    text_id = Column(
        Integer, ForeignKey("texts.id"), nullable=False, index=True
    )  # Each text can be assigned to different clusters in different versions
    version = Column(Integer, nullable=False, index=True)  # Clustering version
    level1_cluster_id = Column(
        Integer, ForeignKey("level1_clusters.id"), nullable=False, index=True
    )  # Assigned L1 Cluster DB ID
    l1_probability = Column(Float, nullable=False)  # Probability from L1 clustering
    l2_probability = Column(
        Float, nullable=False
    )  # Probability from L2 clustering (based on title)

    # Relationships
    text = relationship("TextDB", back_populates="assignment")
    level1_cluster = relationship("Level1Cluster", back_populates="text_assignments")

    # Unique constraint on text_id and version combination
    __table_args__ = (
        UniqueConstraint("text_id", "version", name="uq_text_assignment_per_version"),
        Index("ix_assignment_version_l1", "version", "level1_cluster_id"),
    )


# --- Other Tables (Mostly Unchanged) ---


class DownloadStatus(Base):
    __tablename__ = "download_status"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_name = Column(String, nullable=False)
    status = Column(String, default="downloading")  # downloading, completed, failed
    progress = Column(Float, default=0.0)
    error_message = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# --- Subcluster Tables ---

# Define sequences for new tables
subcluster_id_seq = Sequence("subclusters_id_seq")
text_cluster_id_seq = Sequence("text_clusters_id_seq")


class Subcluster(Base):
    __tablename__ = "subclusters"

    id = Column(
        Integer,
        subcluster_id_seq,
        server_default=subcluster_id_seq.next_value(),
        primary_key=True,
    )
    category_id = Column(
        Integer, ForeignKey("categories.id"), nullable=False, index=True
    )
    title = Column(String, nullable=False)
    row_count = Column(Integer, nullable=False)
    percentage = Column(Float, nullable=False)
    version = Column(Integer, nullable=False, index=True)

    # Relationships
    category = relationship("Category")
    text_clusters = relationship("TextCluster", back_populates="subcluster")

    # Indexes
    __table_args__ = (
        Index("ix_subcluster_category_version", "category_id", "version"),
    )


class TextCluster(Base):
    __tablename__ = "text_clusters"

    id = Column(
        Integer,
        text_cluster_id_seq,
        server_default=text_cluster_id_seq.next_value(),
        primary_key=True,
    )
    text_id = Column(Integer, ForeignKey("texts.id"), nullable=False, index=True)
    subcluster_id = Column(
        Integer, ForeignKey("subclusters.id"), nullable=False, index=True
    )
    membership_score = Column(Float, nullable=False)

    # Relationships
    subcluster = relationship("Subcluster", back_populates="text_clusters")

    # Unique constraint - a text can only be in one subcluster
    __table_args__ = (
        UniqueConstraint("text_id", "subcluster_id", name="uq_text_subcluster"),
        Index("ix_text_clusters_subcluster", "subcluster_id"),
    )


class ClusteringHistory(Base):
    __tablename__ = "clustering_history"

    id = Column(
        Integer,
        history_id_seq,
        server_default=history_id_seq.next_value(),
        primary_key=True,
    )
    dataset_id = Column(
        Integer, ForeignKey("dataset_metadata.id"), nullable=False, index=True
    )
    clustering_version = Column(
        Integer, nullable=False, index=True
    )  # Explicit version number
    clustering_status = Column(
        String, nullable=False
    )  # e.g., started, embedding, clustering_l1, titling_l1, clustering_l2, naming_l2, saving, completed, failed
    # titling_status = Column(String, nullable=False) # Maybe combine into clustering_status
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(String, nullable=True)
    details = Column(
        String, nullable=True
    )  # Store metadata like counts, parameters used? JSON?

    # Relationships
    dataset = relationship("DatasetMetadata", back_populates="clustering_history")

    # Unique constraint per dataset and version
    __table_args__ = (
        UniqueConstraint(
            "dataset_id", "clustering_version", name="uq_clustering_history_version"
        ),
        Index("ix_history_dataset_version", "dataset_id", "clustering_version"),
    )


# --- Pydantic Models for API ---


class DatasetRequest(BaseModel):
    name: str  # Display name for the dataset
    description: Optional[str] = None
    source: str  # 'huggingface', 'url', or 'upload'
    identifier: str  # URL, filename, or HF dataset ID

    # Hugging Face specific parameters
    hf_dataset_name: Optional[str] = (
        None  # Full dataset name (e.g., "databricks/databricks-dolly-15k")
    )
    hf_config: Optional[str] = None  # Dataset configuration
    hf_split: Optional[str] = None  # Dataset split (train, test, validation)
    hf_revision: Optional[str] = (
        None  # Dataset version/commit hash (for reproducibility)
    )
    hf_token: Optional[str] = None  # HF token for private datasets

    # Column selection & Field mapping
    selected_columns: Optional[List[str]] = None  # List of columns user wants to import
    text_field: Optional[str] = None  # Which field contains the text for clustering
    label_field: Optional[str] = (
        None  # Which field contains predefined labels/categories
    )

    # Other options
    limit_rows: Optional[int] = None  # Limit number of rows to download (0 = all)


class TextDBResponse(BaseModel):
    id: int
    text: str

    class Config:
        from_attributes = True


class Level1ClusterResponse(BaseModel):
    id: int  # DB ID
    l1_cluster_id: int  # HDBSCAN L1 ID
    title: str
    # total_texts: Optional[int] = None
    texts: List[TextDBResponse] = []  # Include assigned texts? Might be too large.
    text_count: int = 0  # Add this field to show total count of texts

    class Config:
        from_attributes = True


class CategoryResponse(BaseModel):
    id: int  # DB ID
    l2_cluster_id: int  # HDBSCAN L2 ID
    name: str
    # total_l1_clusters: Optional[int] = None
    level1_clusters: List[Level1ClusterResponse] = []  # Include L1 clusters?
    category_text_count: int = 0  # Add total texts for this category

    class Config:
        from_attributes = True


# Response model for the dataset list endpoint
class DatasetMetadataResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    source: Optional[str] = None
    identifier: Optional[str] = None
    created_at: datetime
    status: str
    verification_status: str
    clustering_status: str
    is_clustered: bool
    total_rows: Optional[int] = None
    error_message: Optional[str] = None
    latest_version: Optional[int] = None  # Add latest successful version

    class Config:
        from_attributes = True


# Response model for individual dataset retrieval (potentially including clusters)
class DatasetDetailResponse(DatasetMetadataResponse):
    categories: Optional[List["CategoryResponse"]] = None  # Use forward reference
    dataset_total_texts: Optional[int] = None  # Add total texts for the dataset version

    class Config:
        from_attributes = True


# Response model for text assignments (maybe useful for specific queries)
class TextAssignmentResponse(BaseModel):
    text_id: int
    text: str  # Denormalize for convenience?
    version: int
    l1_cluster_id: int  # L1 DB ID
    l1_title: str  # Denormalize
    l1_prob: float
    category_id: int  # Category DB ID
    category_name: str  # Denormalize
    l2_prob: float

    class Config:
        from_attributes = True


# Model for clustering request (if needed separately)
class ClusterRequest(BaseModel):
    dataset_id: int
    # Add parameters if clustering can be configured via API


# Add DownloadStatusEnum
class DownloadStatusEnum(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# Add DownloadStatusUpdate Pydantic model
class DownloadStatusUpdate(BaseModel):
    status: DownloadStatusEnum

    class Config:
        from_attributes = True


# Enum for Clustering Status
class ClusteringStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
