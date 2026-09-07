import os
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_connection():
    conn = psycopg.connect(DATABASE_URL)
    try:
        register_vector(conn)
    except Exception:
        conn.close()
        raise
    return conn
