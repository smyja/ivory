from pydantic import BaseModel
from typing import List

class Question(BaseModel):
    question: str
    answer: str

class QuestionList(BaseModel):
    questions: List[Question]

class ClusteredQuestion(Question):
    cluster: int
    cluster_title: str