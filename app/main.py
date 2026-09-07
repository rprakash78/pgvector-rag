from fastapi import FastAPI
from pydantic import BaseModel

from .rag import ask

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Endpoint to ask a question and retrieve an answer based on relevant document chunks.
    
    Args:
        request (QuestionRequest): The request body containing the question and optional top_k parameter.
    
    Returns:
        dict: A dictionary containing the generated answer and sources of relevant document chunks.
    """
    response = ask(request.question, request.top_k)
    return response
