"""Query Understanding Agent for intent classification, ambiguity detection, and query enhancement."""

import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import dspy

from .base_agent import ContextObject, get_llm

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
    """Classify user query intent into one of the predefined categories."""
    user_query: str = dspy.InputField(desc="The user's query to classify")
    conversation_history: str = dspy.InputField(desc="Previous conversation context")
    intent: str = dspy.OutputField(desc="Intent category: abcd_1, abcd_2, abcd_3, abcd_4, abcd_5, or UNKNOWN")
    confidence: float = dspy.OutputField(desc="Classification confidence between 0 and 1")

class AmbiguityDetection(dspy.Signature):
    """Detect if a query is ambiguous and needs clarification."""
    user_query: str = dspy.InputField(desc="The user's query to analyze")
    conversation_history: str = dspy.InputField(desc="Previous conversation context")
    is_ambiguous: bool = dspy.OutputField(desc="Whether the query is ambiguous")
    ambiguity_reason: str = dspy.OutputField(desc="Reason for ambiguity if applicable, empty if not ambiguous")
    clarification_question: str = dspy.OutputField(desc="Question to clarify ambiguity, empty if not ambiguous")

class QueryEnhancement(dspy.Signature):
    """Enhance a clear query by rewriting, creating variations, and expanding terms."""
    user_query: str = dspy.InputField(desc="The original user query")
    conversation_history: str = dspy.InputField(desc="Previous conversation context")
    rewritten_query: str = dspy.OutputField(desc="Direct, keyword-based rewritten query")
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

Your tasks are:
1. Intent Classification: Classify the intent into one of: abcd_1, abcd_2, abcd_3, abcd_4, abcd_5, or UNKNOWN
   - abcd_1: Factual information requests
   - abcd_2: Analytical or comparison requests  
   - abcd_3: Procedural or how-to questions
   - abcd_4: Opinion or recommendation requests
   - abcd_5: Complex multi-step queries
   - UNKNOWN: Unclear or unclassifiable queries

2. Ambiguity Detection: Examine the query for:
   - Vague terms or concepts
   - Unresolved entities or pronouns
   - Missing context or information
   - Multiple possible interpretations

3. Query Enhancement: If the query is clear:
   - Resolve pronouns using conversation history
   - Rewrite into direct, keyword-based query
   - Generate query variations
   - Provide expansion terms (synonyms, related concepts)

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
        
        # Configuration for conversation management
        self.max_recent_turns = 6  # Keep last 6 turns in recent_turns
        self.max_summary_words = 150  # Summary word limit
        self.topic_change_threshold = 0.8  # Confidence threshold for topic changes (increased for less sensitivity)
    
    def _get_short_term_memory(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get or initialize short_term_memory structure."""
        if "short_term_memory" not in user_context:
            user_context["short_term_memory"] = {
                "summary": "",
                "recent_turns": [],
                "current_topic": "none"
            }
        return user_context["short_term_memory"]
    
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
            if short_term_memory["current_topic"] == "none":
                short_term_memory["current_topic"] = topic_change["new_topic"]
    
    def _get_relevant_context(self, user_context: Dict[str, Any]) -> str:
        """Get relevant conversation context from short_term_memory."""
        short_term_memory = self._get_short_term_memory(user_context)
        
        context_parts = []
        
        # Add summary if available
        if short_term_memory.get("summary"):
            context_parts.append(f"Previous conversation summary: {short_term_memory['summary']}")
        
        # Add recent turns
        recent_turns = short_term_memory.get("recent_turns", [])
        if recent_turns:
            turns_str = "\n".join([
                f"{turn.get('role', 'unknown')}: {turn.get('content', '')}"
                for turn in recent_turns
            ])
            context_parts.append(f"Recent conversation:\n{turns_str}")
        
        return "\n\n".join(context_parts) if context_parts else "No previous conversation"
    
    def invoke(self, context: ContextObject, config=None) -> ContextObject:
        """Process the context through DSPy-powered query understanding with conversation management."""
        try:
            # Extract conversation history and initialize short_term_memory
            raw_conversation_history = context.user_context.get("conversation_history", [])
            user_query = context.user_query
            
            # Ensure conversation_history is a list
            if isinstance(raw_conversation_history, str):
                raw_conversation_history = []
            
            # Initialize/get short_term_memory
            short_term_memory = self._get_short_term_memory(context.user_context)
            
            # Step 1: Classify intent (using raw context for now)
            temp_context = str(raw_conversation_history) if raw_conversation_history else "No previous conversation"
            intent_result = self.intent_classifier(
                user_query=user_query,
                conversation_history=temp_context
            )
            
            # Step 2: Detect topic changes
            topic_change = self._detect_topic_change(
                query=user_query,
                intent=intent_result.intent,
                short_term_memory=short_term_memory
            )
            
            # Step 3: Update short_term_memory based on topic changes and conversation length
            self._update_short_term_memory(
                user_context=context.user_context,
                topic_change=topic_change,
                conversation_history=raw_conversation_history
            )
            
            # Step 4: Get relevant context from short_term_memory
            relevant_context = self._get_relevant_context(context.user_context)
            
            # Step 5: Check for ambiguity using filtered context
            ambiguity_result = self.ambiguity_detector(
                user_query=user_query,
                conversation_history=relevant_context
            )
            
            # Step 6: Enhance query if not ambiguous, using filtered context
            enhanced_query = None
            if not ambiguity_result.is_ambiguous:
                enhancement_result = self.query_enhancer(
                    user_query=user_query,
                    conversation_history=relevant_context
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
            
            # Construct the result in the expected format
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
            context.error_occurred = True
            context.error_message = f"DSPy Query Understanding Agent error: {str(e)}"
            return context
