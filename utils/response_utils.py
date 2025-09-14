"""Response formatting and status management utilities."""

from typing import Dict, Any, Optional
from .context_utils import safe_model_dump

def format_error_response(context_obj, error_message: Optional[str] = None) -> Dict[str, Any]:
    """
    Format error response consistently across agents.
    
    Args:
        context_obj: Context object (can be ContextObject or dict)
        error_message: Override error message
        
    Returns:
        dict: Formatted error response
    """
    if hasattr(context_obj, 'error_message'):
        error_msg = error_message or context_obj.error_message
    else:
        error_msg = error_message or "Unknown error occurred"
    
    # Handle both ContextObject and dict inputs
    if hasattr(context_obj, 'model_dump'):
        context_data = safe_model_dump(context_obj)
    else:
        context_data = context_obj
    
    return {
        "status": "error",
        "error": error_msg,
        "context_object": context_data
    }

def format_clarification_response(context_obj, message: Optional[str] = None) -> Dict[str, Any]:
    """
    Format clarification response consistently.
    
    Args:
        context_obj: Context object containing clarification info
        message: Override clarification message
        
    Returns:
        dict: Formatted clarification response
    """
    # Handle both ContextObject and dict inputs
    if hasattr(context_obj, 'model_dump'):
        context_data = safe_model_dump(context_obj)
        planning_result = context_obj.planning if hasattr(context_obj, 'planning') else {}
    else:
        context_data = context_obj
        planning_result = context_obj.get('planning', {})
    
    clarification_message = message or planning_result.get("message_to_user", "Clarification needed")
    
    return {
        "status": "clarification_needed",
        "message": clarification_message,
        "context_object": context_data
    }

def format_completion_response(context_obj, fused_context: Optional[str] = None, 
                             sources: Optional[list] = None) -> Dict[str, Any]:
    """
    Format successful completion response.
    
    Args:
        context_obj: Context object containing execution results
        fused_context: Override fused context
        sources: Override sources list
        
    Returns:
        dict: Formatted completion response
    """
    # Handle both ContextObject and dict inputs
    if hasattr(context_obj, 'model_dump'):
        context_data = safe_model_dump(context_obj)
        execution_result = context_obj.execution if hasattr(context_obj, 'execution') else {}
    else:
        context_data = context_obj
        execution_result = context_obj.get('execution', {})
    
    response_fused_context = fused_context or execution_result.get("fused_context", "")
    response_sources = sources or execution_result.get("sources", [])
    
    return {
        "status": "completed",
        "fused_context": response_fused_context,
        "sources": response_sources,
        "context_object": context_data
    }

def check_for_errors(context_obj) -> bool:
    """
    Check if context object indicates an error occurred.
    
    Args:
        context_obj: Context object to check
        
    Returns:
        bool: True if error occurred
    """
    if hasattr(context_obj, 'error_occurred'):
        return context_obj.error_occurred
    elif isinstance(context_obj, dict):
        return context_obj.get('error_occurred', False)
    
    return False

def should_clarify(context_obj) -> bool:
    """
    Check if planning agent returned CLARIFY action.
    
    Args:
        context_obj: Context object to check
        
    Returns:
        bool: True if clarification is needed
    """
    if hasattr(context_obj, 'planning'):
        planning_result = context_obj.planning
    elif isinstance(context_obj, dict):
        planning_result = context_obj.get('planning')
    else:
        return False
    
    return planning_result and planning_result.get("action") == "CLARIFY"

def extract_status_from_response(response: Dict[str, Any]) -> str:
    """
    Extract status from a response dictionary.
    
    Args:
        response: Response dictionary
        
    Returns:
        str: Status string
    """
    return response.get("status", "unknown")

def is_successful_response(response: Dict[str, Any]) -> bool:
    """
    Check if response indicates success.
    
    Args:
        response: Response dictionary
        
    Returns:
        bool: True if response is successful
    """
    status = extract_status_from_response(response)
    return status == "completed"

def get_response_message(response: Dict[str, Any]) -> str:
    """
    Get the main message from a response.
    
    Args:
        response: Response dictionary
        
    Returns:
        str: Response message
    """
    if "fused_context" in response:
        return response["fused_context"]
    elif "message" in response:
        return response["message"]
    elif "error" in response:
        return f"Error: {response['error']}"
    else:
        return "No response message available"

def format_agent_step_result(agent_name: str, input_data: Any, output_data: Any, 
                           error: Optional[Exception] = None) -> Dict[str, Any]:
    """
    Format the result of an agent step for logging and monitoring.
    
    Args:
        agent_name: Name of the agent
        input_data: Input data to the agent
        output_data: Output data from the agent
        error: Exception if one occurred
        
    Returns:
        dict: Formatted step result
    """
    result = {
        "agent": agent_name,
        "timestamp": None,  # Could be added with datetime
        "success": error is None
    }
    
    if error:
        result["error"] = str(error)
        result["error_type"] = type(error).__name__
    else:
        result["output_available"] = output_data is not None
        
        # Add specific output information based on agent type
        if isinstance(output_data, dict):
            if "intent" in output_data:
                result["intent"] = output_data["intent"]
            if "action" in output_data:
                result["action"] = output_data["action"]
            if "fused_context" in output_data:
                result["response_length"] = len(output_data["fused_context"])
    
    return result
