"""Query Understanding Agent for intent classification, ambiguity detection, and query enhancement."""

import json
import sys
import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import dspy

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
from utils.logging_utils import setup_logger, log_agent_step, log_error

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

# DSPy Signatures for structured query understanding
class IntentClassification(dspy.Signature):
    """Classify user query intent into one of the predefined categories using full context."""
    user_query: str = dspy.InputField(desc="The user's query to classify")
    conversation_history: str = dspy.InputField(desc="Full context object JSON containing conversation history, user context, and previous agent results")
    intent: str = dspy.OutputField(desc="Intent category: abcd_1, abcd_2, abcd_3, abcd_4, abcd_5, or UNKNOWN")
    confidence: float = dspy.OutputField(desc="Classification confidence between 0 and 1")

class AmbiguityDetection(dspy.Signature):
    """Detect if a query is ambiguous and needs clarification using full context."""
    user_query: str = dspy.InputField(desc="The user's query to analyze")
    conversation_history: str = dspy.InputField(desc="Full context object JSON containing conversation history, user context, and previous agent results - check if this is a follow-up answer")
    is_ambiguous: bool = dspy.OutputField(desc="Whether the query is ambiguous - should be False if this is answering a previous clarification")
    ambiguity_reason: str = dspy.OutputField(desc="Reason for ambiguity if applicable, empty if not ambiguous")
    clarification_question: str = dspy.OutputField(desc="Question to clarify ambiguity, empty if not ambiguous")

class QueryEnhancement(dspy.Signature):
    """Enhance a clear query by rewriting, creating variations, and expanding terms using full context."""
    user_query: str = dspy.InputField(desc="The original user query - may be combined with previous context for follow-up answers")
    conversation_history: str = dspy.InputField(desc="Full context object JSON containing conversation history, user context, and previous agent results - use this to understand follow-up context")
    rewritten_query: str = dspy.OutputField(desc="Direct, keyword-based rewritten query that combines current query with previous context if this is a follow-up")
    query_variations: str = dspy.OutputField(desc="Comma-separated list of query variations")
    expansion_terms: str = dspy.OutputField(desc="Comma-separated list of related terms and synonyms")

# DSPy Signatures for topic change detection and conversation management
class TopicChangeDetection(dspy.Signature):
    """Detect if the current query represents a significant topic change from recent conversation."""
    current_query: str = dspy.InputField(desc="The current user query")
    current_intent: str = dspy.InputField(desc="Current query intent classification")
    previous_topic: str = dspy.InputField(desc="Previous conversation topic or 'none' if first interaction")
    recent_context: str = dspy.InputField(desc="Summary of recent conversation turns")
    topic_changed: bool = dspy.OutputField(desc="Whether a significant topic change occurred")
    new_topic: str = dspy.OutputField(desc="Brief description of the new topic (e.g., 'financial_analysis', 'cooking', 'programming')")
    confidence: float = dspy.OutputField(desc="Confidence level of topic change detection (0-1)")

class ConversationSummarization(dspy.Signature):
    """Summarize a conversation to preserve key context while reducing length."""
    conversation_turns: str = dspy.InputField(desc="Full conversation history to summarize")
    current_topic: str = dspy.InputField(desc="Current conversation topic")
    max_summary_length: int = dspy.InputField(desc="Maximum length for summary in words")
    summary: str = dspy.OutputField(desc="Concise summary preserving key context and decisions")
    key_entities: str = dspy.OutputField(desc="Comma-separated list of important entities mentioned")

SYSTEM_PROMPT = """You are a Query Understanding Agent responsible for analyzing user queries and preparing them for downstream processing.

CRITICAL: You will receive a full context object JSON that contains conversation history, user context, and previous agent results. 
Use this context to understand if the current query is:
1. A standalone new query
2. A follow-up answer to a previous clarification request
3. Part of an ongoing conversation

Your tasks are:
1. Intent Classification: Classify the intent into one of: abcd_1, abcd_2, abcd_3, abcd_4, abcd_5, or UNKNOWN
   - abcd_1: Factual information requests
   - abcd_2: Analytical or comparison requests  
   - abcd_3: Procedural or how-to questions
   - abcd_4: Opinion or recommendation requests
   - abcd_5: Complex multi-step queries
   - UNKNOWN: Unclear or unclassifiable queries

2. Ambiguity Detection: 
   - Check conversation_history for previous clarification requests
   - If current query answers a previous clarification, mark as NOT ambiguous
   - Otherwise examine the query for vague terms, missing context, multiple interpretations

3. Query Enhancement: For clear queries or follow-up answers:
   - Combine current query with previous context if this is a follow-up
   - Resolve pronouns using conversation history
   - Rewrite into direct, keyword-based query
   - Generate query variations
   - Provide expansion terms (synonyms, related concepts)

IMPORTANT: Look for patterns like:
- Previous query needed clarification about "aspects" or "what specifically"
- Current query provides aspects/details (e.g., "performance, syntax, learning curve")
- This should be treated as a complete, non-ambiguous query by combining contexts

Return your response in the exact JSON format specified, with all required fields.
"""

class QueryUnderstandingAgent(Runnable):
    """DSPy-powered agent for understanding and enhancing user queries."""
    
    def __init__(self):
        # Initialize DSPy with the LLM configuration from base_agent
        import os
        
        # Use the same model configuration as base_agent
        model_name = os.getenv("MODEL_NAME", "gpt-4o")
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Configure DSPy with the correct v3.0 API
        try:
            # Try the new API first
            lm = dspy.LM(model=model_name, api_key=api_key, temperature=0.1)
            dspy.configure(lm=lm)
        except:
            try:
                # Try alternative configuration
                lm = dspy.LM(f"openai/{model_name}", api_key=api_key)
                dspy.configure(lm=lm)
            except:
                # Final fallback - use default LM configuration
                dspy.configure()
        
        # Initialize DSPy modules with Chain of Thought reasoning
        self.intent_classifier = dspy.ChainOfThought(IntentClassification)
        self.ambiguity_detector = dspy.ChainOfThought(AmbiguityDetection)
        self.query_enhancer = dspy.ChainOfThought(QueryEnhancement)
        
        # Initialize conversation management modules
        self.topic_detector = dspy.ChainOfThought(TopicChangeDetection)
        self.conversation_summarizer = dspy.ChainOfThought(ConversationSummarization)
        
        # Initialize logger
        self.logger = setup_logger("query_understanding_agent")
        
        # Configuration for conversation management
        self.max_recent_turns = 6  # Keep last 6 turns in recent_turns
        self.max_summary_words = 150  # Summary word limit
        self.topic_change_threshold = 0.8  # Confidence threshold for topic changes (increased for less sensitivity)
    
    
    def _detect_topic_change(self, query: str, intent: str, short_term_memory: Dict[str, Any]) -> Dict[str, Any]:
        """Detect if the current query represents a topic change."""
        previous_topic = short_term_memory.get("current_topic", "none")
        recent_context = short_term_memory.get("summary", "") + " " + str(short_term_memory.get("recent_turns", []))
        
        # Use DSPy to detect topic changes
        topic_result = self.topic_detector(
            current_query=query,
            current_intent=intent,
            previous_topic=previous_topic,
            recent_context=recent_context
        )
        
        return {
            "changed": topic_result.topic_changed,
            "new_topic": topic_result.new_topic,
            "confidence": float(topic_result.confidence)
        }
    
    def _manage_conversation_length(self, conversation_history: List[Dict], current_topic: str) -> Dict[str, Any]:
        """Manage conversation length by summarizing if too long."""
        if len(conversation_history) <= self.max_recent_turns:
            # Short conversation - use all turns as recent_turns
            return {
                "summary": "",
                "recent_turns": conversation_history
            }
        
        # Long conversation - summarize older turns, keep recent ones
        recent_turns = conversation_history[-self.max_recent_turns:]
        older_turns = conversation_history[:-self.max_recent_turns]
        
        # Convert older turns to string for summarization
        older_conversation = "\n".join([
            f"{turn.get('role', 'unknown')}: {turn.get('content', '')}"
            for turn in older_turns
        ])
        
        # Summarize older conversation
        summary_result = self.conversation_summarizer(
            conversation_turns=older_conversation,
            current_topic=current_topic,
            max_summary_length=self.max_summary_words
        )
        
        return {
            "summary": summary_result.summary,
            "recent_turns": recent_turns
        }
    
    def _update_short_term_memory(self, user_context: Dict[str, Any], topic_change: Dict[str, Any], conversation_history: List[Dict]) -> None:
        """Update short_term_memory based on topic changes and conversation length."""
        short_term_memory = self._get_short_term_memory(user_context)
        
        if topic_change["changed"] and topic_change["confidence"] >= self.topic_change_threshold:
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
        """Process the context through DSPy-powered query understanding with conversation management."""
        try:
            log_agent_step(self.logger, "QueryUnderstanding", "Starting query processing")
            
            # Extract conversation history using utils
            raw_conversation_history = extract_conversation_history(context.user_context)
            user_query = context.user_query
            
            # Initialize/get short_term_memory using utils
            short_term_memory = get_short_term_memory(context.user_context)
            
            # Step 1: Classify intent using full context object
            full_context_json = format_context_json(context)
            intent_result = self.intent_classifier(
                user_query=user_query,
                conversation_history=full_context_json
            )
            
            # Step 2: Detect topic changes using utils
            topic_change = detect_topic_change(
                query=user_query,
                intent=intent_result.intent,
                short_term_memory=short_term_memory
            )
            
            # Step 3: Update short_term_memory using utils
            update_short_term_memory(
                user_context=context.user_context,
                topic_change=topic_change,
                conversation_history=raw_conversation_history
            )
            
            # Step 4: Check for ambiguity using full context object
            ambiguity_result = self.ambiguity_detector(
                user_query=user_query,
                conversation_history=full_context_json
            )
            
            # Step 5: Enhance query if not ambiguous - DSPy handles all context automatically
            enhanced_query = None
            if not ambiguity_result.is_ambiguous:
                enhancement_result = self.query_enhancer(
                    user_query=user_query,
                    conversation_history=full_context_json
                )
                
                # Parse comma-separated strings into lists
                query_variations = [v.strip() for v in enhancement_result.query_variations.split(',') if v.strip()]
                expansion_terms = [t.strip() for t in enhancement_result.expansion_terms.split(',') if t.strip()]
                
                enhanced_query = {
                    "original_query": user_query,
                    "rewritten_query": enhancement_result.rewritten_query,
                    "query_variations": query_variations,
                    "expansion_terms": expansion_terms
                }
                
            # Construct the result - DSPy handles all logic automatically
            result = {
                "intent": intent_result.intent,
                "confidence": float(intent_result.confidence),
                "is_ambiguous": ambiguity_result.is_ambiguous,
                "ambiguity_reason": ambiguity_result.ambiguity_reason if ambiguity_result.is_ambiguous else None,
                "clarification_question": ambiguity_result.clarification_question if ambiguity_result.is_ambiguous else None,
                "enhanced_query": enhanced_query,
                "topic_change_detected": topic_change["changed"],
                "current_topic": short_term_memory["current_topic"]
            }
            
            # Update context with results
            context.query_understanding = result
            context.current_stage = "query_understanding_complete"
            
            return context
            
        except Exception as e:
            # Handle errors gracefully with proper logging
            log_error(self.logger, "DSPy Query Understanding Agent", e)
            context.error_occurred = True
            context.error_message = f"DSPy Query Understanding Agent error: {str(e)}"
            return context
