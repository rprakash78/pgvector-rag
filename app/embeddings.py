from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def generate_embedding(text: str):
    """
    Generate an embedding for the given text using the SentenceTransformer model.
    
    Args:
        text (str): The input text to generate an embedding for.
    """

    embedding = model.encode(text)
    return embedding