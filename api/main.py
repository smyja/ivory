from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan
import numpy as np
import logging
from dotenv import load_dotenv
import httpx
import json

# router = APIRouter()
load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class Question(BaseModel):
    question: str
    answer: str

class QuestionList(BaseModel):
    questions: List[Question]

@app.post("/cluster")
async def cluster_questions(question_list: QuestionList):
    try:
        # Extract questions
        questions = [q.question for q in question_list.questions]
        
        logger.info(f"Received {len(questions)} questions for clustering")

        # Vectorize the questions
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(questions)
        
        logger.info(f"Vectorized questions. Shape: {X.shape}")

        # Perform clustering with HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=0.5)
        cluster_labels = clusterer.fit_predict(X.toarray())
        
        logger.info(f"Clustering completed. Cluster labels: {cluster_labels}")

        # Generate cluster titles
        feature_names = vectorizer.get_feature_names_out()
        cluster_titles = {}
        for label in set(cluster_labels):
            if label != -1:  # -1 is the noise label in HDBSCAN
                cluster_docs = np.array(questions)[cluster_labels == label]
                cluster_vector = vectorizer.transform(cluster_docs).sum(axis=0)
                top_indices = cluster_vector.argsort()[0, -5:].tolist()  # Convert to list and flatten
                cluster_words = [str(feature_names[idx]) for idx in top_indices]  # Ensure strings
                cluster_titles[label] = " ".join(cluster_words[::-1])
        
        # Add a title for noise points
        cluster_titles[-1] = "Unclustered"
        
        logger.info(f"Generated cluster titles: {cluster_titles}")

        # Prepare results
        results = []
        for q, label in zip(question_list.questions, cluster_labels):
            results.append({
                "question": q.question,
                "answer": q.answer,
                "cluster": int(label),
                "cluster_title": cluster_titles[label]
            })
        
        logger.info("Successfully prepared results")
        
        return results
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)