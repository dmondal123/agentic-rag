"""Execution Agent for tool execution, data collection, and synthesis with HyDE and Sparse Context Selection."""

import json
import os
import sys
import asyncio
import numpy as np
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
from retrieval_utils import get_retriever, get_top_k_chunks, PGVectorRetriever, get_async_retriever

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

    embedding: Optional[List[float]] = None

# Simple retrieval system without HyDE

# New classes for Sparse Context Selection
class EncodedDocument(BaseModel):
    """Enhanced document representation with embedding and attention metadata."""
    content: str
    source_id: str
    tool_name: str
    embedding: Optional[List[float]] = None
    relevance_score: float = 0.0
    attention_weight: float = 1.0
    processing_priority: str = "medium"

class ControlTokens(BaseModel):
    """Control tokens for guiding sparse attention selection."""
    high_priority_docs: List[int] = Field(default_factory=list)
    medium_priority_docs: List[int] = Field(default_factory=list) 
    skip_docs: List[int] = Field(default_factory=list)
    attention_strategy: str = "focused"
    max_contexts: int = 4

class ParallelDocumentEncoder:
    """Encode retrieved documents in parallel using OpenAI embeddings."""
    
    def __init__(self):
        self.logger = setup_logger("document_encoder")
        self.batch_size = 8  # Process 8 documents simultaneously
        self.embedding_model = "text-embedding-3-small"  # OpenAI embedding model
        
        # Initialize OpenAI client
        try:
            from openai import AsyncOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                log_warning(self.logger, "OpenAI Setup", "OPENAI_API_KEY not found. Embeddings will be disabled.")
                self.openai_client = None
            else:
                self.openai_client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            log_warning(self.logger, "OpenAI Setup", "OpenAI package not found. Install with: pip install openai")
            self.openai_client = None
        except Exception as e:
            log_warning(self.logger, "OpenAI Setup", f"Could not initialize OpenAI client: {e}")
            self.openai_client = None
        
    async def encode_documents_parallel(self, tool_results: List[ToolResult]) -> List[EncodedDocument]:
        """Encode all documents in parallel batches using OpenAI embeddings."""
        
        # If no OpenAI client, return basic encoded docs without embeddings
        if not self.openai_client:
            return [EncodedDocument(
                content=result.content,
                source_id=result.source_id,
                tool_name=result.tool_name,
                relevance_score=result.relevance_score,
                embedding=None  # No embeddings available
            ) for result in tool_results]
        
        # Group documents into batches for parallel processing
        document_batches = [tool_results[i:i+self.batch_size] 
                           for i in range(0, len(tool_results), self.batch_size)]
        
        # Process all batches concurrently
        encoding_tasks = [
            self._encode_batch_with_openai(batch) for batch in document_batches
        ]
        
        batch_results = await asyncio.gather(*encoding_tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions
        encoded_docs = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                print(f"Warning: Batch encoding failed: {batch_result}")
                continue
            encoded_docs.extend(batch_result)
            
        return encoded_docs
    
    async def _encode_batch_with_openai(self, batch: List[ToolResult]) -> List[EncodedDocument]:
        """Encode a batch of documents using OpenAI embeddings API."""
        contents = [result.content for result in batch]
        
        try:
            # Generate embeddings using OpenAI API
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=contents,
                encoding_format="float"
            )
            
            # Extract embeddings from response
            embeddings = []
            for embedding_data in response.data:
                embeddings.append(embedding_data.embedding)
                
        except Exception as e:
            print(f"Warning: OpenAI embedding generation failed: {e}")
            embeddings = [None] * len(contents)
        
        # Create encoded documents
        encoded_docs = []
        for i, result in enumerate(batch):
            encoded_doc = EncodedDocument(
                content=result.content,
                source_id=result.source_id,
                tool_name=result.tool_name,
                embedding=embeddings[i] if i < len(embeddings) and embeddings[i] is not None else None,
                relevance_score=result.relevance_score
            )
            encoded_docs.append(encoded_doc)
            
        return encoded_docs

class ControlTokenGenerator:
    """Generate control tokens to guide sparse attention selection."""
    
    def __init__(self):
        self.token_llm = get_llm()  # Use fast model for control token generation
        
    async def generate_control_tokens(self, enhanced_query: Dict[str, Any], encoded_documents: List[EncodedDocument]) -> ControlTokens:
        """Generate control tokens based on query intent and document relevance."""
        
        # Create document summaries for token generation
        doc_summaries = []
        for i, doc in enumerate(encoded_documents):
            summary = f"Doc {i}: {doc.content[:150]}... (relevance: {doc.relevance_score:.3f})"
            doc_summaries.append(summary)
        
        control_prompt = f"""You are an AI assistant that analyzes documents and selects the most relevant ones for a query.

Query: {enhanced_query.get('rewritten_query', 'No enhanced query')}

Available documents ({len(encoded_documents)} total):
{chr(10).join(doc_summaries)}

Analyze the documents and select the most relevant ones. Consider relevance scores and content overlap with the query.

You MUST respond with ONLY valid JSON in this exact format:
{{
    "high_priority_docs": [0, 1],
    "medium_priority_docs": [2], 
    "skip_docs": [3, 4],
    "attention_strategy": "focused",
    "max_contexts": 4
}}

Rules:
- high_priority_docs: Document indices of most relevant documents (up to 3)
- medium_priority_docs: Document indices for supporting information (up to 2)
- skip_docs: Document indices to ignore (low relevance or redundant)
- attention_strategy: Always use "focused"
- max_contexts: Total documents to use (typically 3-5)
- Use only document indices from 0 to {len(encoded_documents)-1}

Respond with ONLY the JSON object, no other text."""
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.token_llm.invoke([{"role": "user", "content": control_prompt}])
            )
            
            # Clean and parse JSON response
            response_content = response.content.strip()
            
            # Remove any markdown formatting if present
            if response_content.startswith('```'):
                lines = response_content.split('\n')
                # Find the first line that starts with {
                json_start = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('{'):
                        json_start = i
                        break
                # Find the last line that ends with }
                json_end = len(lines)
                for i in range(len(lines)-1, -1, -1):
                    if lines[i].strip().endswith('}'):
                        json_end = i + 1
                        break
                response_content = '\n'.join(lines[json_start:json_end])
            
            # Parse JSON response
            try:
                control_data = json.loads(response_content)
            except json.JSONDecodeError as json_error:
                print(f"JSON parsing failed: {json_error}")
                print(f"Response content: {response_content[:200]}...")
                raise json_error
                
            # Validate document indices
            max_index = len(encoded_documents) - 1
            valid_control_data = {}
            
            for key in ['high_priority_docs', 'medium_priority_docs', 'skip_docs']:
                if key in control_data:
                    # Filter out invalid indices
                    valid_indices = [idx for idx in control_data[key] 
                                   if isinstance(idx, int) and 0 <= idx <= max_index]
                    valid_control_data[key] = valid_indices
                else:
                    valid_control_data[key] = []
            
            valid_control_data['attention_strategy'] = control_data.get('attention_strategy', 'focused')
            valid_control_data['max_contexts'] = min(control_data.get('max_contexts', 4), len(encoded_documents))
            
            return ControlTokens(**valid_control_data)
            
        except Exception as e:
            print(f"Warning: Control token generation failed: {e}")
            # Enhanced fallback: select top documents by relevance score
            sorted_docs = sorted(enumerate(encoded_documents), 
                               key=lambda x: x[1].relevance_score, reverse=True)
            
            # Smart fallback selection
            total_docs = len(encoded_documents)
            high_count = min(2, total_docs)
            medium_count = min(2, max(0, total_docs - high_count))
            
            return ControlTokens(
                high_priority_docs=[i for i, _ in sorted_docs[:high_count]],
                medium_priority_docs=[i for i, _ in sorted_docs[high_count:high_count + medium_count]],
                skip_docs=[i for i, _ in sorted_docs[high_count + medium_count:]],
                attention_strategy="focused",
                max_contexts=min(4, total_docs)
            )

class SparseContextSelector:
    """Select optimal contexts based on control tokens and relevance scores."""
    
    def select_contexts(self, encoded_documents: List[EncodedDocument], control_tokens: ControlTokens) -> List[EncodedDocument]:
        """Apply sparse selection based on control tokens."""
        
        selected_contexts = []
        
        # High priority documents (full attention)
        for doc_idx in control_tokens.high_priority_docs:
            if doc_idx < len(encoded_documents):
                doc = encoded_documents[doc_idx].copy()
                doc.attention_weight = 1.0
                doc.processing_priority = "high"
                selected_contexts.append(doc)
        
        # Medium priority documents (reduced attention) 
        for doc_idx in control_tokens.medium_priority_docs:
            if doc_idx < len(encoded_documents):
                doc = encoded_documents[doc_idx].copy()
                doc.attention_weight = 0.6
                doc.processing_priority = "medium"
                selected_contexts.append(doc)
        
        # Limit total contexts to prevent latency
        max_contexts = control_tokens.max_contexts
        selected_contexts = selected_contexts[:max_contexts]
        
        # Sort by attention weight for processing order
        selected_contexts.sort(key=lambda x: x.attention_weight, reverse=True)
        
        return selected_contexts

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
    """Simple retrieval agent for executing tools and synthesizing results with optional Sparse Context Selection."""
    
    def __init__(self):
        self.logger = setup_logger("execution_agent")
        self.llm = get_llm()
        
        # Initialize PostgreSQL/pgvector retriever (required)
        try:
            self.retriever = get_retriever(k=10)  # Get more results for better selection
            log_agent_step(self.logger, "Execution", "Successfully initialized PostgreSQL retriever")
        except Exception as e:
            error_msg = f"Failed to initialize PostgreSQL retriever: {e}"
            log_error(self.logger, "PostgreSQL Setup", e)
            raise RuntimeError(f"ExecutionAgent requires PostgreSQL connection. {error_msg}") from e
        
        # Initialize Sparse Context Selection components (optional)
        self.use_sparse_context = os.getenv("ENABLE_SPARSE_CONTEXT", "false").lower() == "true"
        if self.use_sparse_context:
            self.document_encoder = ParallelDocumentEncoder()
            self.context_selector = SparseContextSelector()
            self.control_token_generator = ControlTokenGenerator()
            log_agent_step(self.logger, "Execution", "Sparse Context Selection enabled")
        
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
    
    
    def _simple_retrieval(self, query: str, context: str, tool_name: str, top_k: int = 3) -> List[ToolResult]:
        """Simple document retrieval using PostgreSQL/pgvector database."""
        
        # Use direct retrieval with custom reranking
        documents = self.retriever.invoke(query)
        
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
                source_id=source_id,
                relevance_score=doc.metadata.get("combined_score", 0.8)  # From custom reranker
            )
            results.append(result)
        
        log_agent_step(self.logger, "Simple Retrieval", f"Retrieved {len(results)} documents for {tool_name}")
        return results
    
    
    def _direct_pgvector_retrieval(self, query: str, context: str, tool_name: str, top_k: int = 3) -> List[ToolResult]:
        """Direct PostgreSQL/pgvector retrieval without HyDE (faster alternative)."""
        
        # Use get_top_k_chunks directly for faster retrieval
        chunks_with_scores = get_top_k_chunks(query, k=top_k)
        
        results = []
        for i, (content, scores) in enumerate(chunks_with_scores):
            result = ToolResult(
                tool_name=tool_name,
                sub_query=query,
                content=content,
                source_id=f"{tool_name}_direct_{i}",
                relevance_score=scores.get("combined_score", 0.8)
            )
            results.append(result)
        
        log_agent_step(self.logger, "Direct PGVector Retrieval", f"Retrieved {len(results)} documents for {tool_name}")
        return results

    async def _direct_pgvector_retrieval_async(self, query: str, context: str, tool_name: str, top_k: int = 3) -> List[ToolResult]:
        """Async direct PostgreSQL/pgvector retrieval without HyDE (high-performance alternative)."""
        
        # Use async get_top_k_chunks for concurrent retrieval
        chunks_with_scores = await get_top_k_chunks_async(query, k=top_k)
        
        results = []
        for i, (content, scores) in enumerate(chunks_with_scores):
            result = ToolResult(
                tool_name=tool_name,
                sub_query=query,
                content=content,
                source_id=f"{tool_name}_direct_async_{i}",
                relevance_score=scores.get("combined_score", 0.8)
            )
            results.append(result)
        
        log_agent_step(self.logger, "Async Direct PGVector Retrieval", f"Retrieved {len(results)} documents for {tool_name}")
        return results

    async def _simple_retrieval_async(self, query: str, context: str, tool_name: str, top_k: int = 3) -> List[ToolResult]:
        """Async simple document retrieval using PostgreSQL/pgvector database."""
        
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
        
        log_agent_step(self.logger, "Async Simple Retrieval", f"Retrieved {len(results)} documents for {tool_name}")
        return results
    
    def _get_relevant_context_from_short_term_memory(self, user_context: Dict[str, Any], enhanced_query: Dict[str, Any]) -> str:
        """Build context for retrieval from short_term_memory and enhanced query."""
        context_parts = []
        
        # Add enhanced query context
        if enhanced_query:
            context_parts.append(f"Enhanced query: {enhanced_query.get('rewritten_query', '')}")
            if enhanced_query.get('query_variations'):
                context_parts.append(f"Query variations: {', '.join(enhanced_query['query_variations'])}")
            if enhanced_query.get('expansion_terms'):
                context_parts.append(f"Related terms: {', '.join(enhanced_query['expansion_terms'])}")
        
        # Add filtered conversation context from short_term_memory
        short_term_memory = user_context.get("short_term_memory", {})
        if short_term_memory.get("summary"):
            context_parts.append(f"Conversation context: {short_term_memory['summary']}")
        if short_term_memory.get("current_topic") and short_term_memory["current_topic"] != "none":
            context_parts.append(f"Current topic: {short_term_memory['current_topic']}")
        
        return ". ".join(context_parts)
    
    def _execute_tools_with_hyde(self, plan: List[Dict[str, Any]], enhanced_query: Dict[str, Any], user_context: Dict[str, Any]) -> List[ToolResult]:
        """Execute tools using HyDE for enhanced document retrieval."""
        all_results = []
        
        # Build context from short_term_memory and enhanced query
        context = self._get_relevant_context_from_short_term_memory(user_context, enhanced_query)
        
        for i, step in enumerate(plan):
            tool_name = step.get("tool", f"knowledge_base_{i+1}")
            sub_query = step.get("sub_query", enhanced_query.get("rewritten_query", "default query"))
            
            # Choose retrieval method based on configuration
            if self.use_hyde_retrieval:
                # Use HyDE-enhanced retrieval for better semantic understanding
                results = self._hyde_retrieval(sub_query, context, tool_name)
            else:
                # Use direct PostgreSQL retrieval for faster response
                results = self._direct_pgvector_retrieval(sub_query, context, tool_name)
            
            all_results.extend(results)
        
        return all_results
    
    async def _execute_tools_with_sparse_context(self, context: ContextObject, plan: List[Dict[str, Any]], enhanced_query: Dict[str, Any], user_context: Dict[str, Any]) -> ExecutionOutput:
        """Execute tools with Sparse Context Selection for enhanced performance."""
        
        # Step 1: Execute HyDE retrieval (existing method)
        tool_results = self._execute_tools_with_hyde(plan, enhanced_query, user_context)
        
        if not tool_results:
            return ExecutionOutput(
                fused_context="No relevant documents found for the query.",
                sources=[]
            )
        
        # Step 2: Parallel document encoding
        encoded_documents = await self.document_encoder.encode_documents_parallel(tool_results)
        
        # Step 3: Generate control tokens for context selection  
        control_tokens = await self.control_token_generator.generate_control_tokens(enhanced_query, encoded_documents)
        
        # Step 4: Select sparse contexts based on control tokens
        selected_contexts = self.context_selector.select_contexts(encoded_documents, control_tokens)
        
        # Step 5: Enhanced synthesis with sparse contexts
        return await self._synthesize_with_sparse_context(context, selected_contexts, enhanced_query)
    
    async def _synthesize_with_sparse_context(self, context: ContextObject, sparse_contexts: List[EncodedDocument], enhanced_query: Dict[str, Any]) -> ExecutionOutput:
        """Synthesize response using sparse contexts with weighted attention."""
        
        if not sparse_contexts:
            return ExecutionOutput(
                fused_context="No contexts selected for synthesis.",
                sources=[]
            )
        
        # Build weighted contexts for synthesis
        weighted_contexts = []
        sources = []
        
        for i, ctx in enumerate(sparse_contexts):
            weight = ctx.attention_weight
            priority = ctx.processing_priority
            
            # Format context with attention indicators
            formatted_context = f"""
            [Priority: {priority.upper()}] [Attention Weight: {weight:.1f}]
            Source: {ctx.source_id} | Tool: {ctx.tool_name}
            Content: {ctx.content}
            Relevance Score: {ctx.relevance_score:.3f}
            """
            weighted_contexts.append(formatted_context)
            
            # Build sources for response
            sources.append({
                "source_id": ctx.source_id,
                "tool": ctx.tool_name,
                "sub_query": f"Enhanced query with {priority} priority",
                "content_snippet": ctx.content[:200] + "..." if len(ctx.content) > 200 else ctx.content,
                "metadata": {
                    "attention_weight": weight,
                    "processing_priority": priority,
                    "relevance_score": ctx.relevance_score
                }
            })
        
        # Enhanced synthesis using the chain with full context object
        try:
            # Format sparse context results as tool results for consistency
            sparse_tool_results = []
            for ctx in sparse_contexts:
                sparse_tool_results.append({
                    "tool_name": ctx.tool_name,
                    "sub_query": f"Sparse context with {ctx.processing_priority} priority",
                    "content": ctx.content,
                    "source_id": ctx.source_id,
                    "relevance_score": ctx.relevance_score,
                    "attention_weight": ctx.attention_weight,
                    "processing_priority": ctx.processing_priority
                })
            
            # Use the chain for consistency with legacy mode
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.chain.invoke({
                    "context_object": json.dumps(context.model_dump(), indent=2, default=str),
                    "tool_results": json.dumps(sparse_tool_results, indent=2, default=str)
                })
            )
            
            return ExecutionOutput(
                fused_context=result["fused_context"],
                sources=result.get("sources", sources)  # Use generated sources or fallback
            )
            
        except Exception as e:
            print(f"Error in sparse context synthesis: {e}")
            # Fallback to basic synthesis
            basic_content = f"Error in synthesis: {str(e)}. Retrieved {len(sparse_contexts)} relevant contexts."
            return ExecutionOutput(
                fused_context=basic_content,
                sources=sources
            )
    
    def invoke(self, context: ContextObject, config=None) -> ContextObject:
        """Process the context through execution with optional Sparse Context Selection."""
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
            
            # Choose execution method based on feature flag
            if self.use_sparse_context:
                # Use new Sparse Context Selection approach
                import asyncio
                try:
                    # Run async method in sync context
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            self._execute_tools_with_sparse_context(context, plan, enhanced_query, context.user_context)
                        )
                    finally:
                        loop.close()
                        
                    # Convert ExecutionOutput to dict for context
                    context.execution = {
                        "fused_context": result.fused_context,
                        "sources": result.sources
                    }
                    
                except Exception as sparse_error:
                    print(f"Sparse context execution failed: {sparse_error}")
                    # Fallback to original method
                    result = self._execute_legacy_synthesis(context, plan, enhanced_query, context.user_context)
                    context.execution = result
                    
            else:
                # Use simple retrieval method (standard mode)
                result = self._execute_legacy_synthesis(context, plan, enhanced_query, context.user_context)
                context.execution = result
            
            context.current_stage = "execution_complete"
            return context
            
        except Exception as e:
            context.error_occurred = True
            context.error_message = f"Execution Agent error: {str(e)}"
            return context
    
    def _execute_legacy_synthesis(self, context: ContextObject, plan: List[Dict[str, Any]], enhanced_query: Dict[str, Any], user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using simple retrieval for backward compatibility."""
        
        # Execute tools using simple retrieval
        tool_results = self._execute_tools_simple(plan, enhanced_query, user_context)
        
        # Convert tool results to dict format for prompt
        tool_results_dict = [result.model_dump() for result in tool_results]
        
        # Invoke the synthesis chain with full context
        result = self.chain.invoke({
            "context_object": json.dumps(context.model_dump(), indent=2, default=str),
            "tool_results": json.dumps(tool_results_dict, indent=2, default=str)
        })
        
        return result
    
    def _execute_tools_simple(self, plan: List[Dict[str, Any]], enhanced_query: Dict[str, Any], user_context: Dict[str, Any]) -> List[ToolResult]:
        """Execute tools using simple PostgreSQL retrieval."""
        all_results = []
        
        # Build context from user memory
        context_string = self._get_relevant_context_from_short_term_memory(user_context, enhanced_query)
        
        # Execute each tool in the plan
        for tool_step in plan:
            tool_name = tool_step.get("tool", "unknown_tool")
            sub_queries = tool_step.get("sub_queries", [tool_step.get("query", "")])
            
            log_agent_step(self.logger, "Tool Execution", f"Executing {tool_name} with {len(sub_queries)} queries")
            
            for query in sub_queries:
                if query:  # Skip empty queries
                    # Use simple retrieval
                    results = self._simple_retrieval(
                        query=query,
                        context=context_string,
                        tool_name=tool_name,
                        top_k=3
                    )
                    all_results.extend(results)
        
        log_agent_step(self.logger, "Tool Execution", f"Total results collected: {len(all_results)}")
        return all_results
