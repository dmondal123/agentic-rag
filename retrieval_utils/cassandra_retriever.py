"""
Optimized Cassandra retriever for existing embeddings table with SAI vector search.
Designed to work with the pre-existing schema without requiring changes.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uuid
import json
from collections import Counter
import re

from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import (
    RoundRobinPolicy, 
    RetryPolicy, 
    ExponentialReconnectionPolicy,
    ConsistencyLevel
)
from cassandra.query import SimpleStatement, PreparedStatement
from cassandra.concurrent import execute_concurrent_with_args

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from dotenv import load_dotenv

load_dotenv()


class CassandraCustomReranker:
    """Multi-metric reranker optimized for Cassandra SAI results."""
    
    def __init__(self):
        # Determine embedding model based on dimensions
        self.embedding_dim = 3072  # From your schema
        
        # Initialize embeddings - assuming text-embedding-3-large for 3072 dims
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",  # 3072 dimensions
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing for word overlap calculations."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text
    
    def _get_word_overlap(self, query: str, doc: str) -> float:
        """Calculate Jaccard similarity for word overlap."""
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
            
        doc_counter = Counter(doc_words)
        total_matches = sum(doc_counter[word] for word in query_words)
        
        return total_matches / len(doc_words) if doc_words else 0.0
    
    def _get_filter_tag_boost(self, query: str, doc: Document) -> float:
        """Boost score based on filter tag relevance."""
        filter_tags = doc.metadata.get('filter_tags', [])
        if not filter_tags:
            return 0.0
        
        query_lower = query.lower()
        tag_matches = sum(1 for tag in filter_tags if tag.lower() in query_lower)
        
        return tag_matches / len(filter_tags) if filter_tags else 0.0
    
    async def _calculate_doc_similarity(self, query: str, doc: Document, 
                                      doc_embedding: Optional[List[float]], 
                                      query_embedding: List[float]) -> Dict[str, float]:
        """Calculate all similarity metrics for a document."""
        
        # Calculate cosine similarity if embeddings are available
        if doc_embedding:
            cosine_sim = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        else:
            # Fallback: use SAI distance if available in metadata
            sai_distance = doc.metadata.get('sai_distance', 1.0)
            cosine_sim = 1.0 / (1.0 + sai_distance)  # Convert distance to similarity
        
        # Calculate other metrics concurrently
        word_overlap_task = asyncio.create_task(
            asyncio.to_thread(self._get_word_overlap, query, doc.page_content)
        )
        keyword_density_task = asyncio.create_task(
            asyncio.to_thread(self._get_keyword_density, query, doc.page_content)
        )
        filter_boost_task = asyncio.create_task(
            asyncio.to_thread(self._get_filter_tag_boost, query, doc)
        )
        
        word_overlap, keyword_density, filter_boost = await asyncio.gather(
            word_overlap_task, keyword_density_task, filter_boost_task
        )
        
        # Combined score with SAI-optimized weights
        combined_score = (
            0.45 * cosine_sim +         # Primary: SAI cosine similarity
            0.25 * word_overlap +       # Secondary: exact word matches
            0.15 * keyword_density +    # Tertiary: keyword importance
            0.15 * filter_boost         # Quaternary: filter tag relevance
        )
        
        return {
            "cosine_similarity": float(cosine_sim),
            "word_overlap": float(word_overlap),
            "keyword_density": float(keyword_density),
            "filter_tag_boost": float(filter_boost),
            "combined_score": float(combined_score)
        }
    
    async def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank documents using multi-metric scoring."""
        if not documents:
            return []
        
        # Get query embedding for cosine similarity calculation
        query_embedding = await asyncio.to_thread(self.embeddings.embed_query, query)
        
        # Calculate similarity scores for all documents concurrently
        similarity_tasks = [
            self._calculate_doc_similarity(
                query, doc, 
                doc.metadata.get('embedding'),  # May be None
                query_embedding
            )
            for doc in documents
        ]
        
        similarity_results = await asyncio.gather(*similarity_tasks)
        
        # Update document metadata with similarity scores
        for doc, scores in zip(documents, similarity_results):
            doc.metadata.update(scores)
        
        # Sort by combined score and return top_k
        reranked_docs = sorted(documents, key=lambda x: x.metadata["combined_score"], reverse=True)
        return reranked_docs[:top_k]


class CassandraVectorRetriever(BaseRetriever):
    """Optimized Cassandra vector retriever using existing embeddings table with SAI."""
    
    def __init__(self, 
                 keyspace: str = 'vectordb',
                 hosts: List[str] = ['127.0.0.1'], 
                 port: int = 9042,
                 username: str = None,
                 password: str = None,
                 k: int = 5):
        
        super().__init__(k=k)
        self.keyspace = keyspace
        self.k = k
        
        # Set up authentication if provided
        auth_provider = None
        if username and password:
            auth_provider = PlainTextAuthProvider(username=username, password=password)
        elif os.getenv('CASSANDRA_USERNAME') and os.getenv('CASSANDRA_PASSWORD'):
            auth_provider = PlainTextAuthProvider(
                username=os.getenv('CASSANDRA_USERNAME'),
                password=os.getenv('CASSANDRA_PASSWORD')
            )
        
        # Create execution profile for optimized vector queries
        vector_profile = ExecutionProfile(
            consistency_level=ConsistencyLevel.ONE,  # Fast reads for vector search
            request_timeout=30,                      # Longer timeout for vector ops
            row_factory=None                         # Use default row factory
        )
        
        # Initialize cluster with optimized settings
        self.cluster = Cluster(
            contact_points=hosts,
            port=port,
            auth_provider=auth_provider,
            execution_profiles={EXEC_PROFILE_DEFAULT: vector_profile},
            # Performance optimizations
            compression=True,
            protocol_version=4,
            load_balancing_policy=RoundRobinPolicy(),
            reconnection_policy=ExponentialReconnectionPolicy(base_delay=1, max_delay=60),
            max_connections_per_host=15,
            max_requests_per_connection=150
        )
        
        # Connect to keyspace
        self.session = self.cluster.connect(keyspace)
        
        # Initialize reranker
        self.reranker = CassandraCustomReranker()
        
        # Prepare frequently used statements
        self._prepare_statements()
        
        print(f"✅ Cassandra Vector Retriever initialized")
        print(f"   Keyspace: {keyspace}")
        print(f"   Hosts: {hosts}")
        print(f"   Default k: {k}")
    
    def _prepare_statements(self):
        """Prepare commonly used CQL statements for better performance."""
        
        # Basic SAI vector search
        self.sai_search_stmt = self.session.prepare("""
            SELECT id, file_id, client_id, content, metadata, filter_tags, created_at
            FROM embeddings 
            ORDER BY embedding ANN OF ? 
            LIMIT ?
        """)
        
        # Client-filtered SAI search
        self.client_filtered_search_stmt = self.session.prepare("""
            SELECT id, file_id, client_id, content, metadata, filter_tags, created_at
            FROM embeddings 
            WHERE client_id = ?
            ORDER BY embedding ANN OF ? 
            LIMIT ?
        """)
        
        # File-filtered SAI search
        self.file_filtered_search_stmt = self.session.prepare("""
            SELECT id, file_id, client_id, content, metadata, filter_tags, created_at
            FROM embeddings 
            WHERE file_id = ?
            ORDER BY embedding ANN OF ? 
            LIMIT ?
        """)
        
        # Combined client + file filter
        self.client_file_filtered_search_stmt = self.session.prepare("""
            SELECT id, file_id, client_id, content, metadata, filter_tags, created_at
            FROM embeddings 
            WHERE client_id = ? AND file_id = ?
            ORDER BY embedding ANN OF ? 
            LIMIT ?
        """)
    
    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Synchronous wrapper for async retrieval."""
        try:
            loop = asyncio.get_running_loop()
            return asyncio.run_coroutine_threadsafe(
                self._async_get_relevant_documents(query, **kwargs), loop
            ).result()
        except RuntimeError:
            return asyncio.run(self._async_get_relevant_documents(query, **kwargs))
    
    async def _async_get_relevant_documents(self, 
                                          query: str,
                                          client_id: Optional[str] = None,
                                          file_id: Optional[str] = None,
                                          filter_tags: Optional[List[str]] = None) -> List[Document]:
        """Async retrieval with optional filtering."""
        
        # Generate query embedding (assuming same model as ingestion)
        query_embedding = await asyncio.to_thread(self.reranker.embeddings.embed_query, query)
        
        # Get more results than needed for reranking (SAI is pretty good, so 2x should suffice)
        search_limit = self.k * 2
        
        # Choose appropriate prepared statement based on filters
        if client_id and file_id:
            future = self.session.execute_async(
                self.client_file_filtered_search_stmt,
                [client_id, file_id, query_embedding, search_limit]
            )
        elif client_id:
            future = self.session.execute_async(
                self.client_filtered_search_stmt,
                [client_id, query_embedding, search_limit]
            )
        elif file_id:
            future = self.session.execute_async(
                self.file_filtered_search_stmt,
                [file_id, query_embedding, search_limit]
            )
        else:
            future = self.session.execute_async(
                self.sai_search_stmt,
                [query_embedding, search_limit]
            )
        
        try:
            rows = future.result()
        except Exception as e:
            print(f"❌ Cassandra query failed: {e}")
            return []
        
        # Convert rows to LangChain Documents
        documents = []
        for row in rows:
            # Parse metadata
            metadata_dict = {}
            if row.metadata:
                if isinstance(row.metadata, dict):
                    metadata_dict = row.metadata
                elif isinstance(row.metadata, str):
                    try:
                        metadata_dict = json.loads(row.metadata)
                    except:
                        metadata_dict = {}
            
            # Add additional metadata
            metadata_dict.update({
                "id": str(row.id),
                "file_id": row.file_id,
                "client_id": row.client_id,
                "filter_tags": row.filter_tags if row.filter_tags else [],
                "created_at": row.created_at.isoformat() if row.created_at else None
            })
            
            documents.append(Document(
                page_content=row.content,
                metadata=metadata_dict
            ))
        
        # Apply post-filtering by filter_tags if specified
        if filter_tags:
            documents = [
                doc for doc in documents
                if any(tag in doc.metadata.get('filter_tags', []) for tag in filter_tags)
            ]
        
        # Apply multi-metric reranking
        reranked_docs = await self.reranker.rerank(query, documents, top_k=self.k)
        
        return reranked_docs
    
    def retrieve_with_filters(self, 
                            query: str,
                            client_id: Optional[str] = None,
                            file_id: Optional[str] = None,
                            filter_tags: Optional[List[str]] = None) -> List[Document]:
        """Public method for filtered retrieval."""
        return self._get_relevant_documents(
            query=query,
            client_id=client_id,
            file_id=file_id,
            filter_tags=filter_tags
        )
    
    def close(self):
        """Clean up Cassandra connection."""
        if hasattr(self, 'cluster') and self.cluster:
            self.cluster.shutdown()


def get_cassandra_retriever(keyspace: str = 'vectordb',
                           hosts: List[str] = ['127.0.0.1'],
                           port: int = 9042,
                           k: int = 5) -> CassandraVectorRetriever:
    """Factory function to create a Cassandra retriever."""
    return CassandraVectorRetriever(
        keyspace=keyspace,
        hosts=hosts, 
        port=port,
        k=k
    )


# Async version for high-concurrency scenarios
class AsyncCassandraVectorRetriever:
    """Async-first Cassandra vector retriever for high-concurrency workloads."""
    
    def __init__(self, **kwargs):
        # Same initialization as sync version
        self.sync_retriever = CassandraVectorRetriever(**kwargs)
    
    async def retrieve(self, 
                      query: str,
                      client_id: Optional[str] = None,
                      file_id: Optional[str] = None,
                      filter_tags: Optional[List[str]] = None) -> List[Document]:
        """Async retrieval method."""
        return await self.sync_retriever._async_get_relevant_documents(
            query=query,
            client_id=client_id,
            file_id=file_id,
            filter_tags=filter_tags
        )
    
    async def batch_retrieve(self, queries: List[Dict[str, Any]]) -> List[List[Document]]:
        """Batch retrieval for multiple queries concurrently."""
        tasks = [
            self.retrieve(**query_params) 
            for query_params in queries
        ]
        return await asyncio.gather(*tasks)
