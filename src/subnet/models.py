from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from sqlalchemy import Column, Integer, String, Boolean, Enum as SQLAlchemyEnum, types, Sequence
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from enum import Enum
from datetime import datetime
Base = declarative_base()
id_seq = Sequence('dataset_id_seq')

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
    split: Optional[str] = None
    num_rows: Optional[int] = None
    config: Optional[str] = "default"
    subset: Optional[str] = None

class DatasetMetadata(Base):
    __tablename__ = 'dataset_metadata'

    id = Column(Integer, id_seq, server_default=id_seq.next_value(), primary_key=True)

    name = Column(String, index=True)
    subset = Column(String, nullable=True)
    split = Column(String, nullable=True)
    download_date = Column(types.TIMESTAMP, server_default=func.now()) 
    is_clustered = Column(Boolean, default=False)
    status = Column(SQLAlchemyEnum(DownloadStatus), default=DownloadStatus.PENDING)

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