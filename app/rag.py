
from multiprocessing import context
import os

from groq import Groq
from .db import get_connection
from .embeddings import generate_embedding

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_context(rows):
    documents = []
    for row in rows:
        content = row[1]
        documents.append(content)
    return "\n\n".join(documents)

def generate_answer(question: str, context: str):
    """
    Generate an answer to the given question based on the provided context using the Groq API.
    
    Args:
        question (str): The input question to generate an answer for.
        context (str): The context to use for generating the answer.
    
    Returns:
        str: The generated answer.
    """
    prompt = f"""
            Use only the context below.

            If  the answer is unavailable,
            say:
            "I don't know based on the available information."

            Context:
            {context}

            Question:
            {question}
            """
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def retrieve(question: str, top_k: int = 5):
    """
    Retrieve the most relevant document chunks based on the input question.
    
    Args:
        question (str): The input question to retrieve relevant chunks for.
        top_k (int): The number of top relevant chunks to retrieve. Default is 5.
    
    Returns:
        List[Tuple]: A list of tuples containing the retrieved document chunks and their metadata.
    """
    question_embedding = generate_embedding(question)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id, content, source, chunk_index
                FROM document_chunks
                ORDER BY embedding <-> %s
                LIMIT %s
                """,
                (question_embedding, top_k)
            )
            results = cur.fetchall()

    return results


def ask(question: str, top_k: int = 3):
    """
    Retrieve relevant document chunks based on the input question and generate an answer.
    
    Args:
        question (str): The input question to retrieve relevant chunks and generate an answer for.
        top_k (int): The number of top relevant chunks to retrieve. Default is 3.

    Returns:
        str: The generated answer.
    """
    results = retrieve(question, top_k)
    context = build_context(results)
    answer = generate_answer(question, context)

    return {
        "answer": answer,
        "sources": [
            {
                "source": result[2],
                "chunk_index": result[3],
                "content": result[1]
            }
            for result in results
        ]
    }

if __name__ == "__main__":
    question = "What is the time complexity of binary search?"
    results = retrieve(question)
    for result in results:
        print(f"Document ID: {result[0]}, Content: {result[1]}, Source: {result[2]}, Chunk Index: {result[3]}")
