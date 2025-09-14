"""Execution Agent for tool execution, data collection, and synthesis with HyDE."""

import json
import os
import numpy as np
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import dspy

from .base_agent import ContextObject, get_llm

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

class DocumentChunk(BaseModel):
    """Represents a document chunk for retrieval."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

# DSPy Signature for HyDE - generates hypothetical documents
class HyDEGeneration(dspy.Signature):
    """Generate a hypothetical document that would answer the given query."""
    query: str = dspy.InputField(desc="The search query or question")
    context: str = dspy.InputField(desc="Additional context about the domain or topic")
    hypothetical_document: str = dspy.OutputField(desc="A detailed hypothetical document that would contain the answer to the query")

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
    """HyDE-powered agent for executing tools and synthesizing results."""
    
    def __init__(self):
        self.llm = get_llm()
        
        # Initialize DSPy for HyDE
        model_name = os.getenv("MODEL_NAME", "gpt-4o")
        api_key = os.getenv("OPENAI_API_KEY")
        
        try:
            lm = dspy.LM(model=model_name, api_key=api_key, temperature=0.1)
            dspy.configure(lm=lm)
        except:
            try:
                lm = dspy.LM(f"openai/{model_name}", api_key=api_key)
                dspy.configure(lm=lm)
            except:
                dspy.configure()
        
        # Initialize HyDE module
        self.hyde_generator = dspy.ChainOfThought(HyDEGeneration)
        
        # Initialize mock document store (in production, this would be your vector DB)
        self.document_store = self._initialize_mock_documents()
        
        self.prompt = PromptTemplate(
            input_variables=["user_query", "enhanced_query", "execution_plan", "tool_results"],
            template=SYSTEM_PROMPT + """

Original User Query: {user_query}
Enhanced Query: {enhanced_query}
Execution Plan: {execution_plan}
Tool Results: {tool_results}

Synthesize the tool results into a comprehensive response that:
1. Directly addresses the user's query
2. Integrates information from all sources
3. Includes proper citations for each piece of information
4. Maintains coherent flow and logical structure

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
    
    def _initialize_mock_documents(self) -> List[DocumentChunk]:
        """Initialize a mock document store for demonstration."""
        # In production, this would load from your vector database
        documents = [
            DocumentChunk(
                id="doc_1",
                content="React is a JavaScript library for building user interfaces. It uses a virtual DOM which provides excellent performance for large applications. React's component-based architecture allows for better code reusability and maintainability. The latest versions include features like Concurrent Mode and Suspense.",
                metadata={"source": "react_docs", "type": "technical_documentation"}
            ),
            DocumentChunk(
                id="doc_2", 
                content="Vue.js is a progressive JavaScript framework that is incrementally adoptable. It offers excellent performance through its reactive data system and virtual DOM implementation. Vue provides simpler state management and has a gentler learning curve compared to React. It's particularly good for rapid prototyping.",
                metadata={"source": "vue_docs", "type": "technical_documentation"}
            ),
            DocumentChunk(
                id="doc_3",
                content="Performance comparison between React and Vue shows that both frameworks offer excellent performance for most applications. React excels in large, complex applications with its fiber architecture, while Vue.js provides better out-of-the-box performance for smaller to medium applications. Bundle size differences are minimal in modern versions.",
                metadata={"source": "performance_study", "type": "research_paper"}
            ),
            DocumentChunk(
                id="doc_4",
                content="Large-scale React applications benefit from features like code splitting, lazy loading, and React.memo for optimization. The ecosystem includes robust tools like Redux for state management and Next.js for server-side rendering. Enterprise adoption is high due to Facebook's backing.",
                metadata={"source": "enterprise_guide", "type": "best_practices"}
            ),
            DocumentChunk(
                id="doc_5",
                content="Vue.js in enterprise applications has grown significantly. The Vue 3 Composition API provides better TypeScript support and code organization for large codebases. Nuxt.js serves as the full-stack framework equivalent to Next.js for React.",
                metadata={"source": "vue_enterprise", "type": "case_study"}
            )
        ]
        return documents
    
    def _simple_embedding_similarity(self, text1: str, text2: str) -> float:
        """Simple similarity based on word overlap - in production use proper embeddings."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    def _hyde_retrieval(self, query: str, context: str, tool_name: str, top_k: int = 3) -> List[ToolResult]:
        """Perform HyDE-enhanced document retrieval."""
        
        # Step 1: Generate hypothetical document using HyDE
        hyde_result = self.hyde_generator(
            query=query,
            context=context
        )
        hypothetical_doc = hyde_result.hypothetical_document
        
        # Step 2: Find documents similar to the hypothetical document
        # In production: embed hypothetical_doc and search vector database
        similarities = []
        for doc in self.document_store:
            similarity = self._simple_embedding_similarity(hypothetical_doc, doc.content)
            similarities.append((doc, similarity))
        
        # Step 3: Sort by similarity and return top-k results
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_documents = similarities[:top_k]
        
        # Step 4: Convert to tool results
        results = []
        for i, (doc, score) in enumerate(top_documents):
            result = ToolResult(
                tool_name=tool_name,
                sub_query=query,
                content=doc.content,
                source_id=doc.id,
                relevance_score=score
            )
            results.append(result)
        
        return results
    
    def _get_relevant_context_from_short_term_memory(self, user_context: Dict[str, Any], enhanced_query: Dict[str, Any]) -> str:
        """Build context for HyDE from short_term_memory and enhanced query."""
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
            
            # Use HyDE for retrieval
            hyde_results = self._hyde_retrieval(sub_query, context, tool_name)
            all_results.extend(hyde_results)
        
        return all_results
    
    def invoke(self, context: ContextObject, config=None) -> ContextObject:
        """Process the context through execution."""
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
            
            # Execute tools using HyDE for enhanced retrieval
            tool_results = self._execute_tools_with_hyde(plan, enhanced_query, context.user_context)
            
            # Convert tool results to dict format for prompt
            tool_results_dict = [result.dict() for result in tool_results]
            
            # Invoke the synthesis chain
            result = self.chain.invoke({
                "user_query": context.user_query,
                "enhanced_query": json.dumps(enhanced_query),
                "execution_plan": json.dumps(plan),
                "tool_results": json.dumps(tool_results_dict)
            })
            
            # Update context with results
            context.execution = result
            context.current_stage = "execution_complete"
            
            return context
            
        except Exception as e:
            context.error_occurred = True
            context.error_message = f"Execution Agent error: {str(e)}"
            return context
