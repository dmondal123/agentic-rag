"""Memory management utilities for conversation context and short-term memory."""

from typing import Dict, Any, List
from .context_utils import extract_conversation_history

def get_short_term_memory(user_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get or initialize short_term_memory from user context.
    
    Args:
        user_context: User context dictionary
        
    Returns:
        dict: Short-term memory with default values if not present
    """
    short_term_memory = user_context.get("short_term_memory", {})
    
    # Ensure all required keys exist with defaults
    defaults = {
        "session_length": 0,
        "last_query": None,
        "summary": "",
        "recent_turns": [],
        "current_topic": "none"
    }
    
    for key, default_value in defaults.items():
        if key not in short_term_memory:
            short_term_memory[key] = default_value
    
    return short_term_memory

def detect_topic_change(query: str, intent: str, short_term_memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect if current query represents a topic change.
    
    Args:
        query: Current user query
        intent: Classified intent
        short_term_memory: Current short-term memory
        
    Returns:
        dict: Topic change information
    """
    current_topic = short_term_memory.get("current_topic", "none")
    
    # Simple topic detection based on intent and query keywords
    if current_topic == "none":
        return {
            "changed": True,
            "new_topic": f"{intent}_topic",
            "reason": "First interaction"
        }
    
    # For now, basic topic change detection
    # This could be enhanced with more sophisticated NLP
    topic_indicators = {
        "abcd_1": ["facts", "information", "what", "define"],
        "abcd_2": ["compare", "analyze", "difference", "vs"],
        "abcd_3": ["how", "steps", "procedure", "guide"],
        "abcd_4": ["recommend", "suggest", "opinion", "best"],
        "abcd_5": ["complex", "multi-step", "comprehensive"]
    }
    
    query_lower = query.lower()
    detected_topics = []
    
    for topic, keywords in topic_indicators.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_topics.append(topic)
    
    # Determine if topic changed
    if detected_topics and detected_topics[0] != intent:
        return {
            "changed": True,
            "new_topic": f"{detected_topics[0]}_topic",
            "reason": f"Topic shift detected from {current_topic} to {detected_topics[0]}"
        }
    
    return {
        "changed": False,
        "new_topic": current_topic,
        "reason": "No topic change detected"
    }

def manage_conversation_length(conversation_history: List[Dict], current_topic: str) -> Dict[str, Any]:
    """
    Manage conversation history to prevent excessive length.
    
    Args:
        conversation_history: List of conversation entries
        current_topic: Current conversation topic
        
    Returns:
        dict: Summary and recent turns for memory management
    """
    max_recent_turns = 5
    max_total_turns = 15
    
    if len(conversation_history) <= max_recent_turns:
        return {
            "summary": "",
            "recent_turns": conversation_history,
            "needs_summarization": False
        }
    
    # If we have many turns, keep recent ones and summarize the rest
    if len(conversation_history) > max_total_turns:
        old_turns = conversation_history[:-max_recent_turns]
        recent_turns = conversation_history[-max_recent_turns:]
        
        # Simple summarization - could be enhanced with LLM
        summary_points = []
        for turn in old_turns[:3]:  # Sample first few turns for summary
            if turn.get("user"):
                summary_points.append(f"User asked about: {turn['user'][:100]}")
        
        summary = f"Previous discussion on {current_topic}: " + "; ".join(summary_points)
        
        return {
            "summary": summary,
            "recent_turns": recent_turns,
            "needs_summarization": True
        }
    
    return {
        "summary": "",
        "recent_turns": conversation_history,
        "needs_summarization": False
    }

def update_short_term_memory(user_context: Dict[str, Any], topic_change: Dict[str, Any], 
                           conversation_history: List[Dict]) -> None:
    """
    Update short-term memory in user context based on current state.
    
    Args:
        user_context: User context to update
        topic_change: Topic change information
        conversation_history: Current conversation history
    """
    if "short_term_memory" not in user_context:
        user_context["short_term_memory"] = {}
    
    short_term_memory = user_context["short_term_memory"]
    
    # Update session length
    short_term_memory["session_length"] = len(conversation_history)
    
    # Update last query
    if conversation_history:
        short_term_memory["last_query"] = conversation_history[-1].get("user")
    
    # Handle conversation length management
    memory_info = manage_conversation_length(conversation_history, 
                                           short_term_memory.get("current_topic", "none"))
    
    short_term_memory["recent_turns"] = memory_info["recent_turns"]
    
    # Update summary if needed
    if memory_info["needs_summarization"]:
        short_term_memory["summary"] = memory_info["summary"]
    elif not short_term_memory.get("summary"):
        short_term_memory["summary"] = ""
    
    # Update topic if it changed or was "none"
    if topic_change["changed"] or short_term_memory.get("current_topic") == "none":
        short_term_memory["current_topic"] = topic_change["new_topic"]

def get_relevant_memory_context(user_context: Dict[str, Any], enhanced_query: Dict[str, Any]) -> str:
    """
    Extract relevant context from short-term memory for query processing.
    
    Args:
        user_context: User context containing short-term memory
        enhanced_query: Enhanced query information for context selection
        
    Returns:
        str: Formatted relevant context
    """
    short_term_memory = get_short_term_memory(user_context)
    
    context_parts = []
    
    # Add summary if available
    summary = short_term_memory.get("summary", "")
    if summary and summary.strip():
        context_parts.append(f"Conversation Summary: {summary}")
    
    # Add recent turns
    recent_turns = short_term_memory.get("recent_turns", [])
    if recent_turns:
        turns_text = []
        for turn in recent_turns[-3:]:  # Last 3 turns for context
            if turn.get("user"):
                turns_text.append(f"User: {turn['user']}")
            if turn.get("assistant"):
                turns_text.append(f"Assistant: {turn['assistant'][:100]}...")
        
        if turns_text:
            context_parts.append("Recent Context:\n" + "\n".join(turns_text))
    
    # Add current topic information
    current_topic = short_term_memory.get("current_topic", "none")
    if current_topic != "none":
        context_parts.append(f"Current Topic: {current_topic}")
    
    return "\n\n".join(context_parts) if context_parts else "No previous context available"
