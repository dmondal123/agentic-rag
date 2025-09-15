from typing import List, Dict, Any
import psycopg2
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import json

load_dotenv()

class CustomReranker:
    """A simple custom reranker that combines multiple similarity metrics."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _get_word_overlap(self, query: str, doc: str) -> float:
        """Calculate word overlap similarity."""
        query_words = set(self._preprocess_text(query).split())
        doc_words = set(self._preprocess_text(doc).split())
        
        if not query_words or not doc_words:
            return 0.0
            
        intersection = len(query_words.intersection(doc_words))
        union = len(query_words.union(doc_words))
        return intersection / union if union > 0 else 0.0
    
    def _get_keyword_density(self, query: str, doc: str) -> float:
        """Calculate keyword density score."""
        query_words = self._preprocess_text(query).split()
        doc_words = self._preprocess_text(doc).split()
        
        if not query_words or not doc_words:
            return 0.0
            
        # Count occurrences of query words in document
        doc_counter = Counter(doc_words)
        total_matches = sum(doc_counter[word] for word in query_words)
        
        # Normalize by document length
        return total_matches / len(doc_words)
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank documents using multiple similarity metrics."""
        # Get embeddings for query and documents
        query_embedding = self.embeddings.embed_query(query)
        doc_embeddings = self.embeddings.embed_documents([doc.page_content for doc in documents])
        
        # Calculate different similarity scores
        for i, doc in enumerate(documents):
            # Cosine similarity
            cosine_sim = cosine_similarity([query_embedding], [doc_embeddings[i]])[0][0]
            
            # Word overlap
            word_overlap = self._get_word_overlap(query, doc.page_content)
            
            # Keyword density
            keyword_density = self._get_keyword_density(query, doc.page_content)
            
            # Combine scores (you can adjust weights)
            combined_score = (
                0.5 * cosine_sim +  # Semantic similarity
                0.3 * word_overlap +  # Exact word matches
                0.2 * keyword_density  # Keyword importance
            )
            
            # Store all scores in metadata
            doc.metadata.update({
                "cosine_similarity": float(cosine_sim),
                "word_overlap": float(word_overlap),
                "keyword_density": float(keyword_density),
                "combined_score": float(combined_score)
            })
        
        # Sort by combined score
        reranked_docs = sorted(documents, key=lambda x: x.metadata["combined_score"], reverse=True)
        return reranked_docs[:top_k]

class PGVectorRetriever(BaseRetriever):
    """Custom retriever that uses pgvector for similarity search with custom reranking."""
    
    k: int = 5
    embeddings: OpenAIEmbeddings = None
    reranker: CustomReranker = None
    
    def __init__(self, k: int = 5):
        super().__init__(k=k)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.reranker = CustomReranker()
        
    def get_db_connection(self):
        return psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            port=os.getenv("PGPORT", 5433),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "password"),
            dbname=os.getenv("PGDATABASE", "pdfrag")
        )

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Convert embedding to string format for pgvector
        vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Get initial results using vector similarity
        conn = self.get_db_connection()
        cur = conn.cursor()
        
        # Get more results than needed for reranking
        cur.execute(f'''
            SELECT id, content, metadata, embedding <#> %s::vector AS distance
            FROM documents
            ORDER BY distance ASC
            LIMIT %s;
        ''', (vector_str, self.k * 2))  # Get 2x more results for reranking
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        # Convert to LangChain Documents
        documents = []
        for doc_id, content, metadata, distance in results:
            # Parse metadata if it's a string
            if isinstance(metadata, str):
                try:
                    metadata_dict = json.loads(metadata)
                except:
                    metadata_dict = {}
            else:
                metadata_dict = metadata or {}
            
            # Add distance to metadata
            metadata_dict["distance"] = distance
            metadata_dict["id"] = str(doc_id)  # Convert database id (int) to string
            
            documents.append(Document(
                page_content=content,
                metadata=metadata_dict
            ))
        
        # Rerank using custom reranker
        reranked_docs = self.reranker.rerank(query, documents, top_k=self.k)
        
        return reranked_docs

def get_retriever(k: int = 5) -> PGVectorRetriever:
    """Create a retriever with custom reranking."""
    return PGVectorRetriever(k=k) 