"""Input and output validation utilities."""

from typing import Dict, Any, List, Optional, Tuple, Union
import json

def validate_input_data(input_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate input data for the orchestration pipeline.
    
    Args:
        input_data: Input data to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(input_data, dict):
        return False, "Input data must be a dictionary"
    
    # Check required fields
    if "user_query" not in input_data:
        return False, "Missing required field: user_query"
    
    user_query = input_data["user_query"]
    if not isinstance(user_query, str):
        return False, "user_query must be a string"
    
    if not user_query.strip():
        return False, "user_query cannot be empty"
    
    # Validate user_context if present
    if "user_context" in input_data:
        user_context = input_data["user_context"]
        if not isinstance(user_context, dict):
            return False, "user_context must be a dictionary"
        
        # Validate conversation_history if present
        if "conversation_history" in user_context:
            conv_history = user_context["conversation_history"]
            if not isinstance(conv_history, list):
                return False, "conversation_history must be a list"
            
            # Validate each conversation entry
            for i, entry in enumerate(conv_history):
                if not isinstance(entry, dict):
                    return False, f"conversation_history[{i}] must be a dictionary"
    
    return True, None

def validate_agent_output(agent_name: str, output_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate output from an agent.
    
    Args:
        agent_name: Name of the agent that produced the output
        output_data: Output data to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(output_data, dict):
        return False, f"{agent_name} output must be a dictionary"
    
    # Agent-specific validations
    if agent_name.lower() == "query_understanding":
        return _validate_query_understanding_output(output_data)
    elif agent_name.lower() == "planning":
        return _validate_planning_output(output_data)
    elif agent_name.lower() == "execution":
        return _validate_execution_output(output_data)
    
    return True, None

def _validate_query_understanding_output(output: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate query understanding agent output."""
    required_fields = ["intent", "confidence", "is_ambiguous"]
    
    for field in required_fields:
        if field not in output:
            return False, f"Missing required field in query understanding output: {field}"
    
    # Validate field types
    if not isinstance(output["intent"], str):
        return False, "intent must be a string"
    
    if not isinstance(output["confidence"], (int, float)):
        return False, "confidence must be a number"
    
    if not isinstance(output["is_ambiguous"], bool):
        return False, "is_ambiguous must be a boolean"
    
    # Validate confidence range
    confidence = output["confidence"]
    if not (0 <= confidence <= 1):
        return False, "confidence must be between 0 and 1"
    
    # If ambiguous, check for clarification question
    if output["is_ambiguous"]:
        if "clarification_question" not in output or not output["clarification_question"]:
            return False, "clarification_question required when query is ambiguous"
    
    return True, None

def _validate_planning_output(output: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate planning agent output."""
    if "action" not in output:
        return False, "Missing required field in planning output: action"
    
    action = output["action"]
    valid_actions = ["EXECUTE", "CLARIFY"]
    
    if action not in valid_actions:
        return False, f"Invalid action: {action}. Must be one of {valid_actions}"
    
    if action == "EXECUTE":
        if "plan" not in output:
            return False, "plan required when action is EXECUTE"
        
        plan = output["plan"]
        if not isinstance(plan, list):
            return False, "plan must be a list"
        
        if not plan:
            return False, "plan cannot be empty when action is EXECUTE"
        
        # Validate each plan step
        for i, step in enumerate(plan):
            if not isinstance(step, dict):
                return False, f"plan[{i}] must be a dictionary"
            
            required_step_fields = ["tool", "sub_query", "expected_output"]
            for field in required_step_fields:
                if field not in step:
                    return False, f"Missing required field in plan[{i}]: {field}"
    
    elif action == "CLARIFY":
        if "message_to_user" not in output:
            return False, "message_to_user required when action is CLARIFY"
    
    return True, None

def _validate_execution_output(output: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate execution agent output."""
    if "fused_context" not in output:
        return False, "Missing required field in execution output: fused_context"
    
    if not isinstance(output["fused_context"], str):
        return False, "fused_context must be a string"
    
    if "sources" in output:
        sources = output["sources"]
        if not isinstance(sources, list):
            return False, "sources must be a list"
        
        # Validate each source
        for i, source in enumerate(sources):
            if not isinstance(source, dict):
                return False, f"sources[{i}] must be a dictionary"
    
    return True, None

def validate_json_string(json_str: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate and parse JSON string.
    
    Args:
        json_str: JSON string to validate
        
    Returns:
        tuple: (is_valid, error_message, parsed_data)
    """
    try:
        data = json.loads(json_str)
        return True, None, data
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}", None

def validate_context_structure(context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate the overall context structure.
    
    Args:
        context: Context dictionary to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check for required top-level fields
    if "user_query" not in context:
        return False, "Missing user_query in context"
    
    if "user_context" not in context:
        return False, "Missing user_context in context"
    
    # Validate short_term_memory structure if present
    user_context = context["user_context"]
    if "short_term_memory" in user_context:
        stm = user_context["short_term_memory"]
        if not isinstance(stm, dict):
            return False, "short_term_memory must be a dictionary"
        
        expected_stm_fields = ["session_length", "current_topic"]
        for field in expected_stm_fields:
            if field not in stm:
                return False, f"Missing field in short_term_memory: {field}"
    
    return True, None

def sanitize_string_input(input_str: str, max_length: int = 1000) -> str:
    """
    Sanitize string input by trimming and removing potentially harmful content.
    
    Args:
        input_str: String to sanitize
        max_length: Maximum allowed length
        
    Returns:
        str: Sanitized string
    """
    if not isinstance(input_str, str):
        return ""
    
    # Strip whitespace
    sanitized = input_str.strip()
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    # Remove null bytes and other control characters
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\t\n\r')
    
    return sanitized

def validate_query_length(query: str, min_length: int = 1, max_length: int = 5000) -> Tuple[bool, Optional[str]]:
    """
    Validate query length constraints.
    
    Args:
        query: Query string to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(query, str):
        return False, "Query must be a string"
    
    query_length = len(query.strip())
    
    if query_length < min_length:
        return False, f"Query too short. Minimum length: {min_length}"
    
    if query_length > max_length:
        return False, f"Query too long. Maximum length: {max_length}"
    
    return True, None
