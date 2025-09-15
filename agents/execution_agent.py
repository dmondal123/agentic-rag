"""Simplified Execution Agent for async retrieval and synthesis."""

import json
import os
import sys
import asyncio
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .base_agent import ContextObject, get_llm
from utils.context_utils import format_context_json
from utils.memory_utils import get_relevant_memory_context
from utils.logging_utils import setup_logger, log_agent_step, log_error, log_warning
from retrieval_utils import get_async_retriever, AsyncPGVectorRetriever

class ExecutionOutput(BaseModel):
    """Output structure for Execution Agent."""
    fused_context: str = Field(..., description="Synthesized information from all sources")
    sources: List[Dict[str, Any]] = Field(..., description="Source and query mapping for citations")

class ToolResult(BaseModel):
    """Tool result structure for document retrieval."""
    tool_name: str
    sub_query: str
    content: str
    source_id: str
    relevance_score: float = Field(default=0.0, description="Similarity score for retrieved content")

SYSTEM_PROMPT = """You are an Execution Agent responsible for executing the planned tools and synthesizing results.

Your tasks:
1. Execute each tool in the plan with appropriate sub-queries
2. Collect all retrieved data snippets
3. Synthesize information into coherent response
4. Embed citations for every piece of information
5. Map sources to their origins

For this implementation, you'll work with the execution plan and simulate tool execution.
Provide a comprehensive synthesis with proper citations in the format [Source: source_id].
"""

class ExecutionAgent(Runnable):
    """Simplified async retrieval agent for executing tools and synthesizing results."""
    
    def __init__(self):
        self.logger = setup_logger("execution_agent")
        self.llm = get_llm()
        
        # Initialize async PostgreSQL/pgvector retriever (required)
        try:
            self.async_retriever = get_async_retriever(k=10)
            log_agent_step(self.logger, "Execution", "Successfully initialized async PostgreSQL retriever")
        except Exception as e:
            error_msg = f"Failed to initialize PostgreSQL retriever: {e}"
            log_error(self.logger, "PostgreSQL Setup", e)
            raise RuntimeError(f"ExecutionAgent requires PostgreSQL connection. {error_msg}") from e
        
        self.prompt = PromptTemplate(
            input_variables=["context_object", "tool_results"],
            template=SYSTEM_PROMPT + """

FULL CONTEXT OBJECT INPUT:
{context_object}

TOOL RETRIEVAL RESULTS:
{tool_results}

Synthesize the tool results into a comprehensive response that:
1. Directly addresses the user's original query from the context
2. Integrates information from all sources
3. Includes proper citations for each piece of information  
4. Maintains coherent flow and logical structure
5. Uses the enhanced query information for better synthesis

Response format:
{{
    "fused_context": "Complete synthesized response with citations [Source: source_id]",
    "sources": [
        {{
            "source_id": "unique_identifier",
            "tool": "tool_name",
            "sub_query": "query_used",
            "content_snippet": "relevant_content",
            "metadata": "additional_info"
        }}
    ]
}}
"""
        )
        
        self.parser = JsonOutputParser(pydantic_object=ExecutionOutput)
        self.chain = self.prompt | self.llm | self.parser
    
    async def _async_retrieval(self, query: str, context: str, tool_name: str, top_k: int = 3) -> List[ToolResult]:
        """Simple async document retrieval using PostgreSQL/pgvector database."""
        
        # Use async retrieval with custom reranking
        documents = await self.async_retriever.retrieve(query)
        
        # Convert LangChain documents to ToolResult format
        results = []
        for i, doc in enumerate(documents[:top_k]):
            # Ensure source_id is always a string (database id might be int)
            doc_id = doc.metadata.get("id", f"{tool_name}_{i}")
            source_id = str(doc_id) if doc_id is not None else f"{tool_name}_{i}"
            
            result = ToolResult(
                tool_name=tool_name,
                sub_query=query,
                content=doc.page_content,
                source_id=f"async_{source_id}",
                relevance_score=doc.metadata.get("combined_score", 0.8)  # From custom reranker
            )
            results.append(result)
        
        log_agent_step(self.logger, "Async Retrieval", f"Retrieved {len(results)} documents for {tool_name}")
        return results
    
    def _get_relevant_context_from_short_term_memory(self, user_context: Dict[str, Any], enhanced_query: Dict[str, Any]) -> str:
        """Build context for retrieval from short_term_memory and enhanced query."""
        context_parts = []
        
        # Add enhanced query context
        if enhanced_query:
            context_parts.append(f"Enhanced query: {enhanced_query.get('rewritten_query', '')}")
            
            # Add expansion terms
            expansion_terms = enhanced_query.get('expansion_terms', [])
            if expansion_terms:
                context_parts.append(f"Related terms: {', '.join(expansion_terms)}")
        
        # Add short-term memory context
        short_term_memory = user_context.get('short_term_memory', {})
        
        # Add current topic
        current_topic = short_term_memory.get('current_topic', '')
        if current_topic and current_topic != "none":
            context_parts.append(f"Current topic: {current_topic}")
        
        # Add conversation summary
        summary = short_term_memory.get('summary', '')
        if summary:
            context_parts.append(f"Conversation summary: {summary}")
        
        return " | ".join(context_parts) if context_parts else "No additional context available"
    
    async def _execute_tools_async(self, plan: List[Dict[str, Any]], enhanced_query: Dict[str, Any], user_context: Dict[str, Any]) -> List[ToolResult]:
        """Execute tools using async PostgreSQL retrieval."""
        all_results = []
        
        # Build context from user memory
        context_string = self._get_relevant_context_from_short_term_memory(user_context, enhanced_query)
        
        # Execute each tool in the plan
        for tool_step in plan:
            tool_name = tool_step.get("tool", "unknown_tool")
            sub_queries = tool_step.get("sub_queries", [tool_step.get("query", "")])
            
            log_agent_step(self.logger, "Tool Execution", f"Executing {tool_name} with {len(sub_queries)} queries")
            
            # Create tasks for concurrent query processing
            query_tasks = []
            for query in sub_queries:
                if query:  # Skip empty queries
                    task = self._async_retrieval(
                        query=query,
                        context=context_string,
                        tool_name=tool_name,
                        top_k=3
                    )
                    query_tasks.append(task)
            
            # Execute all queries for this tool concurrently
            if query_tasks:
                tool_results = await asyncio.gather(*query_tasks)
                # Flatten the results
                for result_list in tool_results:
                    all_results.extend(result_list)
        
        log_agent_step(self.logger, "Tool Execution", f"Total results collected: {len(all_results)}")
        return all_results
    
    def invoke(self, context: ContextObject, config=None) -> ContextObject:
        """Process the context through async execution."""
        try:
            # Check if planning was successful and action is PROCEED_TO_EXECUTE
            if not context.planning:
                context.error_occurred = True
                context.error_message = "Execution Agent: No planning results available"
                return context
                
            if context.planning.get("action") != "PROCEED_TO_EXECUTE":
                context.error_occurred = True
                context.error_message = "Execution Agent: Action is not PROCEED_TO_EXECUTE"
                return context
                
            plan = context.planning.get("plan", [])
            if not plan:
                context.error_occurred = True
                context.error_message = "Execution Agent: No execution plan available"
                return context
                
            enhanced_query = context.query_understanding.get("enhanced_query", {})
            
            # Execute tools using async retrieval
            import asyncio
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    tool_results = loop.run_until_complete(
                        self._execute_tools_async(plan, enhanced_query, context.user_context)
                    )
                finally:
                    loop.close()
                    
                # Convert tool results to dict format for prompt
                tool_results_dict = [result.model_dump() for result in tool_results]
                
                # Invoke the synthesis chain with full context
                result = self.chain.invoke({
                    "context_object": json.dumps(context.model_dump(), indent=2, default=str),
                    "tool_results": json.dumps(tool_results_dict, indent=2, default=str)
                })
                
                context.execution = result
                    
            except Exception as execution_error:
                log_error(self.logger, "Async execution", execution_error)
                context.error_occurred = True
                context.error_message = f"Execution failed: {str(execution_error)}"
                return context
            
            context.current_stage = "execution_complete"
            return context
            
        except Exception as e:
            context.error_occurred = True
            context.error_message = f"Execution Agent error: {str(e)}"
            return context