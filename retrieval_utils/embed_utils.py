import os
from typing import List, Dict, Any
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Use OpenAI Python SDK for embedding generation

def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        input=chunks,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", 5433),
        user=os.getenv("PGUSER", "postgres"), 
        password=os.getenv("PGPASSWORD", "password"),
        dbname=os.getenv("PGDATABASE", "pdfrag")
    )

def upsert_chunks_with_embeddings(chunks: List[Dict[str, Any]], embeddings: List[List[float]], source: str):
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Drop existing table and recreate with correct schema
    cur.execute('DROP TABLE IF EXISTS documents;')
    cur.execute('''
        CREATE TABLE documents (
            id SERIAL PRIMARY KEY,
            source TEXT,
            content_type TEXT,
            content TEXT,
            embedding VECTOR(1536),
            metadata JSONB
        );
    ''')
    
    for chunk, emb in zip(chunks, embeddings):
        # Convert metadata to JSON string
        metadata_json = json.dumps(chunk.get('metadata', {}))
        
        cur.execute(
            """
            INSERT INTO documents (source, content_type, content, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                source,
                chunk['type'],
                chunk['content'],
                emb,
                metadata_json
            )
        )
    
    conn.commit()
    cur.close()
    conn.close() 