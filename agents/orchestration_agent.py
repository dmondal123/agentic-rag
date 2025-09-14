"""Orchestration Agent for managing the multi-agent RAG pipeline."""

import sys
import os
from typing import Dict, Any, Optional
from langchain_core.runnables import Runnable, RunnableSequence, RunnableBranch, RunnableLambda

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .base_agent import ContextObject
from .query_understanding_agent import QueryUnderstandingAgent
from .planning_agent import PlanningAgent
from .execution_agent import ExecutionAgent
from utils.response_utils import (
    format_error_response,
    format_clarification_response,
    format_completion_response,
    check_for_errors,
    should_clarify
)
from utils.validation_utils import validate_input_data
from utils.logging_utils import setup_logger, log_agent_step, log_error

def _initialize_context(input_data: Dict[str, Any]) -> ContextObject:
    """Initialize context object from input data."""
    return ContextObject(
        user_query=input_data["user_query"],
        user_context=input_data.get("user_context", {}),
        current_stage="initialization"
    )

class OrchestrationAgent(Runnable):
    """Main orchestration agent using RunnableBranch for conditional flow."""
    
    def __init__(self):
        self.logger = setup_logger("orchestration_agent")
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
                    check_for_errors,
                    RunnableLambda(format_error_response)
                ),
                # Branch 2: Continue to planning agent
                self.planning_agent
                | RunnableBranch(
                    # Sub-branch 1: If error occurred during planning
                    (
                        check_for_errors,
                        RunnableLambda(format_error_response)
                    ),
                    # Sub-branch 2: If clarification is needed
                    (
                        should_clarify,
                        RunnableLambda(format_clarification_response)
                    ),
                    # Sub-branch 3: Proceed to execution (default)
                    self.execution_agent
                    | RunnableBranch(
                        # Final error check
                        (
                            check_for_errors,
                            RunnableLambda(format_error_response)
                        ),
                        # Success response
                        RunnableLambda(format_completion_response)
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
            # Validate input data
            is_valid, error_message = validate_input_data(input_data)
            if not is_valid:
                log_error(self.logger, "Input validation", ValueError(error_message))
                return format_error_response(
                    {"error_message": f"Invalid input: {error_message}"}, 
                    error_message=f"Invalid input: {error_message}"
                )
            
            log_agent_step(self.logger, "Orchestration", "Starting pipeline execution")
            result = self.chain.invoke(input_data, config)
            log_agent_step(self.logger, "Orchestration", "Pipeline execution completed")
            return result
            
        except Exception as e:
            log_error(self.logger, "Orchestration pipeline", e)
            return format_error_response(
                {"error_message": str(e)}, 
                error_message=f"Orchestration error: {str(e)}"
            )

# Convenience function for easier usage
def create_orchestrated_chain() -> OrchestrationAgent:
    """Create and return the orchestrated multi-agent chain."""
    return OrchestrationAgent()
