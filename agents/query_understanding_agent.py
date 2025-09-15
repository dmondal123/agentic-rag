"""Query Understanding Agent using LlamaIndex HyDE for query enhancement."""

import json
import sys
import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# LlamaIndex imports for HyDE query transformation
try:
    from llama_index.core.indices.query.query_transform import HyDEQueryTransform
    from llama_index.core.schema import QueryBundle
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .base_agent import ContextObject, get_llm
from utils.context_utils import format_context_json, extract_conversation_history
from utils.memory_utils import (
    get_short_term_memory, 
    detect_topic_change, 
    manage_conversation_length, 
    update_short_term_memory
)
from utils.logging_utils import setup_logger, log_agent_step, log_error, log_warning

class QueryUnderstandingOutput(BaseModel):
    """Output structure for Query Understanding Agent."""
    intent: str = Field(..., description="Intent classification: abcd_1, abcd_2, abcd_3, abcd_4, abcd_5, or UNKNOWN")
    confidence: float = Field(..., description="Classification confidence between 0 and 1", ge=0, le=1)
    is_ambiguous: bool = Field(..., description="Whether the query is ambiguous")
    ambiguity_reason: Optional[str] = Field(default=None, description="Reason for ambiguity if applicable")
    clarification_question: Optional[str] = Field(default=None, description="Question to clarify ambiguity")
    enhanced_query: Optional[Dict[str, Any]] = Field(default=None, description="Enhanced query information")

class EnhancedQuery(BaseModel):
    """Enhanced query structure."""
    original_query: str
    rewritten_query: str
    query_variations: List[str]
    expansion_terms: List[str]

# Prompt templates for LLM-based query understanding
INTENT_CLASSIFICATION_PROMPT = """
You are an AI assistant that classifies user query intents. Analyze the query and context to determine the intent category.

Context (JSON format):
{context_json}

User Query: {user_query}

Intent Categories:
- abcd_1: Factual information requests
- abcd_2: Analytical or comparison requests  
- abcd_3: Procedural or how-to questions
- abcd_4: Opinion or recommendation requests
- abcd_5: Complex multi-step queries
- UNKNOWN: Unclear or unclassifiable queries

Respond with ONLY valid JSON:
{{
    "intent": "category_name",
    "confidence": 0.0-1.0
}}
"""

AMBIGUITY_DETECTION_PROMPT = """
You are an AI assistant that detects query ambiguity. Analyze if the query needs clarification.

Context (JSON format):
{context_json}

User Query: {user_query}

Check if this query is:
1. A follow-up answer to a previous clarification request
2. Missing critical information
3. Has multiple possible interpretations
4. Contains unclear references

If this appears to be answering a previous clarification question, mark as NOT ambiguous.

Respond with ONLY valid JSON:
{{
    "is_ambiguous": true/false,
    "ambiguity_reason": "reason or empty string",
    "clarification_question": "question or empty string"
}}
"""


class QueryUnderstandingAgent(Runnable):
    """LlamaIndex HyDE-powered agent for understanding and enhancing user queries."""
    
    def __init__(self):
        self.logger = setup_logger("query_understanding_agent")
        self.llm = get_llm()
        
        # Initialize LlamaIndex HyDE for query enhancement
        if LLAMAINDEX_AVAILABLE:
            try:
                self.hyde_transform = HyDEQueryTransform(include_original=True)
                log_agent_step(self.logger, "QueryUnderstanding", "Initialized LlamaIndex HyDE transform")
            except Exception as e:
                log_warning(self.logger, "HyDE Setup", f"Failed to initialize HyDE: {e}")
                self.hyde_transform = None
        else:
            log_warning(self.logger, "LlamaIndex", "LlamaIndex not available, HyDE disabled")
            self.hyde_transform = None
        
        # Initialize LangChain chains for intent and ambiguity detection
        self.intent_chain = self._create_intent_chain()
        self.ambiguity_chain = self._create_ambiguity_chain()
        
        # Configuration for conversation management
        self.max_recent_turns = 6  # Keep last 6 turns in recent_turns
        self.max_summary_words = 150  # Summary word limit
    
    def _create_intent_chain(self):
        """Create LangChain chain for intent classification."""
        intent_prompt = PromptTemplate(
            input_variables=["context_json", "user_query"],
            template=INTENT_CLASSIFICATION_PROMPT
        )
        intent_parser = JsonOutputParser()
        return intent_prompt | self.llm | intent_parser
    
    def _create_ambiguity_chain(self):
        """Create LangChain chain for ambiguity detection."""
        ambiguity_prompt = PromptTemplate(
            input_variables=["context_json", "user_query"],
            template=AMBIGUITY_DETECTION_PROMPT
        )
        ambiguity_parser = JsonOutputParser()
        return ambiguity_prompt | self.llm | ambiguity_parser
    
    def _enhance_query_with_hyde(self, query: str, context_json: str) -> Dict[str, Any]:
        """Enhance query using LlamaIndex HyDE."""
        if not self.hyde_transform:
            # Fallback to simple query enhancement without HyDE
            return {
                "original_query": query,
                "rewritten_query": query,
                "query_variations": [query],
                "expansion_terms": []
            }
        
        try:
            # Use HyDE to generate hypothetical document and enhance query
            query_bundle = QueryBundle(query_str=query)
            enhanced_bundle = self.hyde_transform(query_bundle)
            
            # Extract the hypothetical document (first embedding string)
            hypothetical_doc = enhanced_bundle.embedding_strs[0] if enhanced_bundle.embedding_strs else ""
            
            # Create enhanced query by combining original query with hypothetical content
            enhanced_query = f"{query} {hypothetical_doc}"
            
            # Generate query variations based on HyDE output
            variations = [query, enhanced_query]
            if hypothetical_doc:
                variations.append(hypothetical_doc)
            
            return {
                "original_query": query,
                "rewritten_query": enhanced_query,
                "query_variations": variations[:3],  # Limit to 3 variations
                "expansion_terms": []  # Could extract key terms from hypothetical_doc
            }
            
        except Exception as e:
            log_error(self.logger, "HyDE Enhancement", e)
            # Fallback to original query
            return {
                "original_query": query,
                "rewritten_query": query,
                "query_variations": [query],
                "expansion_terms": []
            }
    
    def _detect_topic_change(self, query: str, intent: str, short_term_memory: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified topic change detection without DSPy."""
        previous_topic = short_term_memory.get("current_topic", "none")
        
        # Simple topic detection based on intent change and query keywords
        if previous_topic == "none":
            return {
                "changed": True,
                "new_topic": f"{intent}_topic",
                "confidence": 1.0
            }
        
        # Basic keyword-based topic change detection
        topic_keywords = {
            "abcd_1": ["facts", "information", "what", "define", "explain"],
            "abcd_2": ["compare", "analyze", "difference", "vs", "versus"],
            "abcd_3": ["how", "steps", "procedure", "guide", "tutorial"],
            "abcd_4": ["recommend", "suggest", "opinion", "best", "choose"],
            "abcd_5": ["complex", "multi-step", "comprehensive", "detailed"]
        }
        
        query_lower = query.lower()
        current_keywords = topic_keywords.get(intent, [])
        
        # If query contains keywords from a different intent category, it's a topic change
        for other_intent, keywords in topic_keywords.items():
            if other_intent != intent and any(keyword in query_lower for keyword in keywords):
                return {
                    "changed": True,
                    "new_topic": f"{other_intent}_topic",
                    "confidence": 0.8
                }
        
        # No significant topic change detected
        return {
            "changed": False,
            "new_topic": previous_topic,
            "confidence": 0.9
        }
    
    def _manage_conversation_length(self, conversation_history: List[Dict], current_topic: str) -> Dict[str, Any]:
        """Simplified conversation length management."""
        if len(conversation_history) <= self.max_recent_turns:
            # Short conversation - use all turns as recent_turns
            return {
                "summary": "",
                "recent_turns": conversation_history
            }
        
        # Long conversation - keep recent ones and create simple summary
        recent_turns = conversation_history[-self.max_recent_turns:]
        older_turns = conversation_history[:-self.max_recent_turns]
        
        # Create simple summary from older turns
        summary_parts = []
        for turn in older_turns[:3]:  # Sample first few turns
            if turn.get("user"):
                summary_parts.append(f"User asked about: {turn['user'][:50]}...")
        
        summary = f"Earlier conversation on {current_topic}: " + "; ".join(summary_parts)
        
        return {
            "summary": summary[:self.max_summary_words],
            "recent_turns": recent_turns
        }
    
    def _update_short_term_memory(self, user_context: Dict[str, Any], topic_change: Dict[str, Any], conversation_history: List[Dict]) -> None:
        """Update short_term_memory based on topic changes and conversation length."""
        short_term_memory = get_short_term_memory(user_context)
        
        if topic_change["changed"] and topic_change["confidence"] >= 0.8:  # Fixed threshold
            # Topic changed - reset context
            short_term_memory["summary"] = ""
            short_term_memory["recent_turns"] = conversation_history[-2:] if conversation_history else []  # Keep just the topic change context
            short_term_memory["current_topic"] = topic_change["new_topic"]
        else:
            # Same topic - manage conversation length
            current_topic = short_term_memory.get("current_topic", "general")
            
            # Always check conversation length, even for same topic
            if len(conversation_history) > self.max_recent_turns:
                conversation_data = self._manage_conversation_length(conversation_history, current_topic)
                short_term_memory["summary"] = conversation_data["summary"]
                short_term_memory["recent_turns"] = conversation_data["recent_turns"]
            else:
                # Short conversation - use all recent turns
                short_term_memory["recent_turns"] = conversation_history
                # Don't overwrite existing summary unless it's empty
                if not short_term_memory.get("summary"):
                    short_term_memory["summary"] = ""
            
            # Update topic if it was "none" (first interaction)
            if short_term_memory.get("current_topic") == "none":
                short_term_memory["current_topic"] = topic_change["new_topic"]
    
    def invoke(self, context: ContextObject, config=None) -> ContextObject:
        """Process the context through LlamaIndex HyDE-powered query understanding."""
        try:
            log_agent_step(self.logger, "QueryUnderstanding", "Starting query processing with LlamaIndex HyDE")
            
            # Extract conversation history using utils
            raw_conversation_history = extract_conversation_history(context.user_context)
            user_query = context.user_query
            
            # Initialize/get short_term_memory using utils
            short_term_memory = get_short_term_memory(context.user_context)
            
            # Prepare full context JSON for LLM calls
            full_context_json = format_context_json(context)
            
            # Step 1: Classify intent using LangChain
            intent_result = self.intent_chain.invoke({
                "context_json": full_context_json,
                "user_query": user_query
            })
            
            # Step 2: Detect topic changes using simplified method
            topic_change = self._detect_topic_change(
                query=user_query,
                intent=intent_result["intent"],
                short_term_memory=short_term_memory
            )
            
            # Step 3: Update short_term_memory using simplified method
            self._update_short_term_memory(
                user_context=context.user_context,
                topic_change=topic_change,
                conversation_history=raw_conversation_history
            )
            
            # Step 4: Check for ambiguity using LangChain
            ambiguity_result = self.ambiguity_chain.invoke({
                "context_json": full_context_json,
                "user_query": user_query
            })
            
            # Step 5: Enhance query using LlamaIndex HyDE if not ambiguous
            enhanced_query = None
            if not ambiguity_result["is_ambiguous"]:
                enhanced_query = self._enhance_query_with_hyde(user_query, full_context_json)
                
            # Construct the result
            result = {
                "intent": intent_result["intent"],
                "confidence": float(intent_result["confidence"]),
                "is_ambiguous": ambiguity_result["is_ambiguous"],
                "ambiguity_reason": ambiguity_result.get("ambiguity_reason") if ambiguity_result["is_ambiguous"] else None,
                "clarification_question": ambiguity_result.get("clarification_question") if ambiguity_result["is_ambiguous"] else None,
                "enhanced_query": enhanced_query,
                "topic_change_detected": topic_change["changed"],
                "current_topic": short_term_memory["current_topic"]
            }
            
            # Update context with results
            context.query_understanding = result
            context.current_stage = "query_understanding_complete"
            
            log_agent_step(self.logger, "QueryUnderstanding", f"Completed processing - Intent: {result['intent']}, HyDE: {enhanced_query is not None}")
            return context
            
        except Exception as e:
            # Handle errors gracefully with proper logging
            log_error(self.logger, "LlamaIndex Query Understanding Agent", e)
            context.error_occurred = True
            context.error_message = f"Query Understanding Agent error: {str(e)}"
            return context
