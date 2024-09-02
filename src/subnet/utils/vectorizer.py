from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import numpy as np

def vectorize_questions(questions: List[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(stop_words='english')
    return vectorizer.fit_transform(questions).toarray()