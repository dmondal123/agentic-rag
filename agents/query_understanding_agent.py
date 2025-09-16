"""Unified Query Understanding Agent with single LLM call for all tasks."""

import json
import sys
import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# No longer using LlamaIndex HyDE - using unified LLM approach

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

# Single comprehensive system prompt for query understanding
UNIFIED_QUERY_UNDERSTANDING_PROMPT = """
You are a Query Understanding Agent responsible for analyzing user queries in a single comprehensive response.

Context (JSON format):
{context_json}

User Query: {user_query}

Analyze the query and provide a complete understanding including:

1. INTENT CLASSIFICATION - Classify into one of:
   - information_retrieval: Factual information and knowledge base queries
   - analysis_comparison: Data analysis, calculations and comparisons
   - procedural_guidance: Step-by-step instructions and how-to guides
   - recommendation: Opinions, suggestions and recommendations
   - complex_query: Multi-step queries requiring multiple tools/steps
   - UNKNOWN: Unclear or unclassifiable queries

2. AMBIGUITY DETECTION - Check if the query needs clarification by examining:
   - If this is a follow-up answer to a previous clarification (mark as NOT ambiguous)
   - Missing critical information
   - Multiple possible interpretations
   - Unclear references

3. QUERY ENHANCEMENT - If NOT ambiguous, enhance the query by:
   - Rewriting it more clearly and directly
   - Creating 2-3 alternative phrasings
   - Identifying key expansion terms and related concepts
   - Using conversation context to resolve pronouns and references

IMPORTANT: Use the full context to understand follow-up queries and conversation flow.

Respond with ONLY valid JSON:
{{
    "intent": "category_name",
    "confidence": 0.0-1.0,
    "is_ambiguous": true/false,
    "ambiguity_reason": "reason or empty string",
    "clarification_question": "question or empty string",
    "enhanced_query": {{
        "original_query": "the original user query",
        "rewritten_query": "clearer, more direct version",
        "query_variations": ["variation1", "variation2", "variation3"],
        "expansion_terms": ["term1", "term2", "term3"]
    }} or null
}}
"""


class QueryUnderstandingAgent(Runnable):
    """Unified query understanding agent with single LLM call for all tasks."""
    
    def __init__(self):
        self.logger = setup_logger("query_understanding_agent")
        self.llm = get_llm()
        
        # Initialize single unified chain for all query understanding tasks
        self.unified_chain = self._create_unified_chain()
        
        # Configuration for conversation management
        self.max_recent_turns = 6  # Keep last 6 turns in recent_turns
        self.max_summary_words = 150  # Summary word limit
        
        log_agent_step(self.logger, "QueryUnderstanding", "Initialized unified query understanding agent")
    
    def _create_unified_chain(self):
        """Create single LangChain chain for all query understanding tasks."""
        unified_prompt = PromptTemplate(
            input_variables=["context_json", "user_query"],
            template=UNIFIED_QUERY_UNDERSTANDING_PROMPT
        )
        unified_parser = JsonOutputParser()
        return unified_prompt | self.llm | unified_parser
    
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
            "specific_fact_retrieval": ["what", "who", "when", "where", "why", "define", "explain", "tell me about"],
            "document_summary": ["summarize", "overview", "brief", "main points", "key points", "synopsis"],
            "comparative_analysis": ["compare", "contrast", "difference", "similarities", "versus", "vs"],
            "procedural_inquiry": ["how to", "steps", "procedure", "guide", "tutorial", "instructions"],
            "general_inquiry": ["tell me more", "elaborate", "details", "explain further", "background"],
            "chitchat": ["hello", "hi", "how are you", "thanks", "bye", "nice", "good"],
            "unknown": ["unclear", "confused", "not sure", "what do you mean", "clarify"]
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
        """Process the context through unified query understanding with single LLM call."""
        try:
            log_agent_step(self.logger, "QueryUnderstanding", "Starting unified query processing")
            
            # Extract conversation history using utils
            raw_conversation_history = extract_conversation_history(context.user_context)
            user_query = context.user_query
            
            # Initialize/get short_term_memory using utils
            short_term_memory = get_short_term_memory(context.user_context)
            
            # Prepare full context JSON for the unified LLM call
            full_context_json = format_context_json(context)
            
            # Single LLM call for all query understanding tasks
            unified_result = self.unified_chain.invoke({
                "context_json": full_context_json,
                "user_query": user_query
            })
            
            # Step 2: Detect topic changes using simplified method
            topic_change = self._detect_topic_change(
                query=user_query,
                intent=unified_result["intent"],
                short_term_memory=short_term_memory
            )
            
            # Step 3: Update short_term_memory using simplified method
            self._update_short_term_memory(
                user_context=context.user_context,
                topic_change=topic_change,
                conversation_history=raw_conversation_history
            )
            
            # Construct the final result using unified response
            result = {
                "intent": unified_result["intent"],
                "confidence": float(unified_result["confidence"]),
                "is_ambiguous": unified_result["is_ambiguous"],
                "ambiguity_reason": unified_result.get("ambiguity_reason") if unified_result["is_ambiguous"] else None,
                "clarification_question": unified_result.get("clarification_question") if unified_result["is_ambiguous"] else None,
                "enhanced_query": unified_result.get("enhanced_query"),  # Already structured by LLM
                "topic_change_detected": topic_change["changed"],
                "current_topic": short_term_memory["current_topic"]
            }
            
            # Update context with results
            context.query_understanding = result
            context.current_stage = "query_understanding_complete"
            
            has_enhancement = result["enhanced_query"] is not None
            log_agent_step(self.logger, "QueryUnderstanding", f"Completed unified processing - Intent: {result['intent']}, Enhanced: {has_enhancement}")
            return context
            
        except Exception as e:
            # Handle errors gracefully with proper logging
            log_error(self.logger, "Unified Query Understanding Agent", e)
            context.error_occurred = True
            context.error_message = f"Query Understanding Agent error: {str(e)}"
            return context
