"""Simplified Execution Agent for PostgreSQL vectordb retrieval only."""

import os
import sys
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .base_agent import ContextObject
from utils.logging_utils import setup_logger, log_agent_step, log_error, log_warning
from retrieval_utils import get_top_k_chunks, generate_answer

class ExecutionOutput(BaseModel):
    """Output structure for Execution Agent."""
    fused_context: str = Field(..., description="Synthesized answer using retrieved information")
    sources: List[Dict[str, Any]] = Field(..., description="Retrieved chunks with relevance scores")

class ExecutionAgent(Runnable):
    """Simplified execution agent that only does PostgreSQL vectordb retrieval."""
    
    def __init__(self):
        self.logger = setup_logger("execution_agent")
        
        # Test PostgreSQL connection using retrieval utils
        try:
            # Test the connection by trying to get a small number of chunks
            test_chunks = get_top_k_chunks("test", k=1)
            log_agent_step(self.logger, "Execution", "Successfully verified PostgreSQL connection via retrieval_utils")
        except Exception as e:
            error_msg = f"Failed to connect to PostgreSQL via retrieval_utils: {e}"
            log_error(self.logger, "PostgreSQL Setup", e)
            raise RuntimeError(f"ExecutionAgent requires PostgreSQL connection. {error_msg}") from e
    
    def _build_conversation_history(self, context: ContextObject) -> str:
        """Build conversation history from context for better synthesis."""
        history_parts = []
        
        if hasattr(context, 'user_context') and context.user_context:
            short_term_memory = context.user_context.get('short_term_memory', {})
            
            # Add conversation summary
            summary = short_term_memory.get('summary', '')
            if summary:
                history_parts.append(f"Previous conversation: {summary}")
            
            # Add current topic
            current_topic = short_term_memory.get('current_topic', '')
            if current_topic and current_topic != "none":
                history_parts.append(f"Current topic: {current_topic}")
        
        return "\n".join(history_parts) if history_parts else ""
    
    def _get_query_for_retrieval(self, context: ContextObject) -> str:
        """Extract the best query for retrieval from context."""
        # Try to get enhanced query first
        if hasattr(context, 'query_understanding') and context.query_understanding:
            enhanced_query = context.query_understanding.get('enhanced_query', {})
            rewritten_query = enhanced_query.get('rewritten_query', '')
            if rewritten_query:
                log_agent_step(self.logger, "Query Selection", f"Using enhanced query: {rewritten_query}")
                return rewritten_query
        
        # Fall back to original user query
        original_query = context.user_query if hasattr(context, 'user_query') else ""
        if original_query:
            log_agent_step(self.logger, "Query Selection", f"Using original query: {original_query}")
            return original_query
        
        # Last resort
        log_warning(self.logger, "No valid query found, using default")
        return "information request"
    
    def invoke(self, context: ContextObject, config=None) -> ContextObject:
        """Simplified execution: just do PostgreSQL retrieval and synthesis."""
        try:
            # Get the query for retrieval
            query = self._get_query_for_retrieval(context)
            
            log_agent_step(self.logger, "Vectordb Retrieval", f"Retrieving from PostgreSQL for query: {query}")
            
            # Retrieve chunks using retrieval_utils (this includes reranking and scoring)
            chunks_with_scores = get_top_k_chunks(query, k=5)
            
            if not chunks_with_scores:
                log_warning(self.logger, "No results found from PostgreSQL retrieval")
                context.execution = {
                    "fused_context": "I don't have enough information in my knowledge base to answer your question.",
                    "sources": []
                }
                context.current_stage = "execution_complete"
                return context
            
            # Extract content and build sources info
            content_chunks = []
            sources_info = []
            
            for i, (content, scores) in enumerate(chunks_with_scores):
                content_chunks.append(content)
                sources_info.append({
                    "source_id": f"vectordb_{i}",
                    "tool": "vectordb",
                    "sub_query": query,
                    "content_snippet": content[:200] + "..." if len(content) > 200 else content,
                    "relevance_score": scores.get("combined_score", 0.0),
                    "cosine_similarity": scores.get("cosine_similarity", 0.0),
                    "word_overlap": scores.get("word_overlap", 0.0),
                    "keyword_density": scores.get("keyword_density", 0.0)
                })
            
            log_agent_step(self.logger, "Vectordb Retrieval", f"Retrieved {len(content_chunks)} chunks")
            
            # Build conversation history
            history_text = self._build_conversation_history(context)
            
            # Use generate_answer from retrieval_utils for synthesis
            log_agent_step(self.logger, "Answer Generation", "Generating answer using retrieval_utils")
            synthesized_response = generate_answer(query, content_chunks, history_text)
            
            # Set execution results
            context.execution = {
                "fused_context": synthesized_response,
                "sources": sources_info
            }
            
            context.current_stage = "execution_complete"
            log_agent_step(self.logger, "Execution Complete", "Successfully completed PostgreSQL retrieval and synthesis")
            
            return context
            
        except Exception as e:
            log_error(self.logger, "Execution Error", e)
            context.error_occurred = True
            context.error_message = f"Execution Agent error: {str(e)}"
            return context