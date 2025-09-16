import os
import sys
from typing import List, Tuple, Dict
import psycopg2
from dotenv import load_dotenv
from .retriever import get_retriever
from openai import OpenAI

# Add utils to path to use the configured LLM
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from utils.llm_utils import get_llm
except ImportError:
    # Fallback if utils not available
    from langchain_litellm import ChatLiteLLM
    def get_llm():
        return ChatLiteLLM(model=os.getenv("MODEL_NAME", "gpt-4o"), temperature=0.1)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Use OpenAI Python SDK for embedding generation (available in embed_utils.py)

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", 5433),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "password"),
        dbname=os.getenv("PGDATABASE", "pdfrag")
    )

def get_top_k_chunks(query: str, k: int = 5) -> List[Tuple[str, Dict[str, float]]]:
    """
    Get top k most relevant chunks using custom reranking.
    Returns chunks with their detailed scoring information.
    """
    # Get retriever
    retriever = get_retriever(k=k)
    
    # Get reranked documents
    docs = retriever.invoke(query)
    
    # Convert to expected format with detailed scoring
    return [
        (
            doc.page_content,
            {
                "cosine_similarity": doc.metadata.get("cosine_similarity", 0.0),
                "word_overlap": doc.metadata.get("word_overlap", 0.0),
                "keyword_density": doc.metadata.get("keyword_density", 0.0),
                "combined_score": doc.metadata.get("combined_score", 0.0)
            }
        )
        for doc in docs
    ]

def generate_answer(query: str, context_chunks: List[str], history_text: str = "") -> str:
    """Generate an answer using the configured LLM with context and conversation history."""
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are an expert assistant. Use the following context and conversation history to answer the user's question. If the answer is not in the context, say you don't know.

Conversation history:
{history_text}

Context:
{context}

Question: {query}
Answer:
"""
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content.strip() if hasattr(response, 'content') else str(response) 
