"""Orchestration Agent for managing the multi-agent RAG pipeline."""

from typing import Dict, Any, Optional
from langchain_core.runnables import Runnable, RunnableSequence, RunnableBranch, RunnableLambda

from .base_agent import ContextObject
from .query_understanding_agent import QueryUnderstandingAgent
from .planning_agent import PlanningAgent
from .execution_agent import ExecutionAgent

def _initialize_context(input_data: Dict[str, Any]) -> ContextObject:
    """Initialize context object from input data."""
    return ContextObject(
        user_query=input_data["user_query"],
        user_context=input_data.get("user_context", {}),
        current_stage="initialization"
    )

def _check_for_errors(context: ContextObject) -> bool:
    """Check if context has errors."""
    return context.error_occurred

def _format_error_response(context: ContextObject) -> Dict[str, Any]:
    """Format error response consistently."""
    return {
        "status": "error",
        "error": context.error_message,
        "context_object": context.dict()
    }

def _should_clarify(context: ContextObject) -> bool:
    """Check if planning agent returned CLARIFY action."""
    planning_result = context.planning
    return planning_result and planning_result.get("action") == "CLARIFY"

def _format_clarification_response(context: ContextObject) -> Dict[str, Any]:
    """Format clarification response."""
    return {
        "status": "clarification_needed",
        "message": context.planning.get("message_to_user"),
        "context_object": context.dict()
    }

def _format_completion_response(context: ContextObject) -> Dict[str, Any]:
    """Format successful completion response."""
    return {
        "status": "completed",
        "fused_context": context.execution.get("fused_context"),
        "sources": context.execution.get("sources"),
        "context_object": context.dict()
    }

class OrchestrationAgent(Runnable):
    """Main orchestration agent using RunnableBranch for conditional flow."""
    
    def __init__(self):
        self.query_understanding_agent = QueryUnderstandingAgent()
        self.planning_agent = PlanningAgent()
        self.execution_agent = ExecutionAgent()
        
        # Create the orchestration chain using RunnableBranch
        self.chain = (
            RunnableLambda(_initialize_context)
            | self.query_understanding_agent
            | RunnableBranch(
                # Branch 1: If error occurred during query understanding
                (
                    lambda context: _check_for_errors(context),
                    RunnableLambda(_format_error_response)
                ),
                # Branch 2: Continue to planning agent
                self.planning_agent
                | RunnableBranch(
                    # Sub-branch 1: If error occurred during planning
                    (
                        lambda context: _check_for_errors(context),
                        RunnableLambda(_format_error_response)
                    ),
                    # Sub-branch 2: If clarification is needed
                    (
                        lambda context: _should_clarify(context),
                        RunnableLambda(_format_clarification_response)
                    ),
                    # Sub-branch 3: Proceed to execution (default)
                    self.execution_agent
                    | RunnableBranch(
                        # Final error check
                        (
                            lambda context: _check_for_errors(context),
                            RunnableLambda(_format_error_response)
                        ),
                        # Success response
                        RunnableLambda(_format_completion_response)
                    )
                )
            )
        )
    
    def invoke(self, input_data: Dict[str, Any], config=None) -> Dict[str, Any]:
        """
        Orchestrate the multi-agent pipeline using RunnableBranch.
        
        Args:
            input_data: Dictionary containing 'user_query' and 'user_context'
            
        Returns:
            Dictionary with final results or clarification request
        """
        try:
            result = self.chain.invoke(input_data, config)
            return result
        except Exception as e:
            return {
                "status": "error",
                "error": f"Orchestration error: {str(e)}",
                "context_object": None
            }

# Convenience function for easier usage
def create_orchestrated_chain() -> OrchestrationAgent:
    """Create and return the orchestrated multi-agent chain."""
    return OrchestrationAgent()
