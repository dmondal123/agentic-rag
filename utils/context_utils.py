"""Context object utilities for formatting and validation."""

import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

def format_context_json(context_object: BaseModel, indent: int = 2) -> str:
    """
    Format a context object as JSON string for LLM consumption.
    
    Args:
        context_object: Pydantic model to serialize
        indent: JSON indentation level
        
    Returns:
        str: Formatted JSON string
    """
    try:
        return json.dumps(context_object.model_dump(), indent=indent, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to serialize context: {str(e)}"}, indent=indent)

def safe_model_dump(model: BaseModel) -> Dict[str, Any]:
    """
    Safely dump a Pydantic model to dictionary.
    
    Args:
        model: Pydantic model instance
        
    Returns:
        dict: Model data as dictionary
    """
    try:
        return model.model_dump()
    except Exception as e:
        return {"error": f"Failed to dump model: {str(e)}"}

def extract_conversation_history(user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract and validate conversation history from user context.
    
    Args:
        user_context: User context dictionary
        
    Returns:
        list: Conversation history as list of dictionaries
    """
    conversation_history = user_context.get("conversation_history", [])
    
    # Ensure it's a list
    if isinstance(conversation_history, str):
        try:
            conversation_history = json.loads(conversation_history)
        except json.JSONDecodeError:
            return []
    
    if not isinstance(conversation_history, list):
        return []
    
    return conversation_history

def validate_context_object(context_obj: BaseModel) -> tuple[bool, Optional[str]]:
    """
    Validate that a context object has required fields.
    
    Args:
        context_obj: Context object to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Check if it has the basic required attributes
        if not hasattr(context_obj, 'user_query'):
            return False, "Missing user_query field"
        
        if not hasattr(context_obj, 'user_context'):
            return False, "Missing user_context field"
        
        if not context_obj.user_query or not context_obj.user_query.strip():
            return False, "Empty user_query"
        
        return True, None
    
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def merge_context_updates(original_context: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely merge updates into original context.
    
    Args:
        original_context: Original context dictionary
        updates: Updates to merge
        
    Returns:
        dict: Merged context
    """
    merged = original_context.copy()
    
    for key, value in updates.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_context_updates(merged[key], value)
        else:
            merged[key] = value
    
    return merged

def sanitize_for_logging(data: Dict[str, Any], max_length: int = 200) -> Dict[str, Any]:
    """
    Sanitize context data for safe logging by truncating long strings.
    
    Args:
        data: Data to sanitize
        max_length: Maximum string length before truncation
        
    Returns:
        dict: Sanitized data
    """
    if isinstance(data, dict):
        return {k: sanitize_for_logging(v, max_length) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_logging(item, max_length) for item in data]
    elif isinstance(data, str) and len(data) > max_length:
        return data[:max_length] + "..."
    else:
        return data
