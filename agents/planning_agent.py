"""Planning Agent for tool selection and execution plan generation."""

import json
import sys
import os
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .base_agent import ContextObject, get_llm
from utils.context_utils import format_context_json
from utils.logging_utils import setup_logger, log_agent_step, log_error

class PlanningOutput(BaseModel):
    """Output structure for Planning Agent."""
    action: str = Field(..., description="Action to take: CLARIFY or PROCEED_TO_EXECUTE")
    plan: Optional[List[Dict[str, Any]]] = Field(default=None, description="Step-by-step execution plan")
    message_to_user: Optional[str] = Field(default=None, description="Message to user if clarification needed")

SYSTEM_PROMPT = """You are a Planning Agent responsible for creating execution plans and handling ambiguous queries.

Available Tools:
- tool1: Document search and retrieval
- tool2: Knowledge base querying
- tool3: Web search and information gathering
- tool4: Data analysis and computation

Your tasks:
1. Check if the query is ambiguous (from query understanding results)
2. If ambiguous: return CLARIFY action with message_to_user
3. If clear: select appropriate tools based on the rewritten query keywords
4. Create a detailed step-by-step execution plan

Tool Selection Guidelines:
- tool1: Use for specific document or file searches, internal knowledge
- tool2: Use for structured knowledge base queries, factual lookups  
- tool3: Use for current information, web-based research, external data
- tool4: Use for calculations, data processing, analytical tasks

Return your response in the exact JSON format specified.
"""

class PlanningAgent(Runnable):
    """Agent for planning query execution strategy."""
    
    def __init__(self):
        self.logger = setup_logger("planning_agent")
        self.llm = get_llm()
        self.prompt = PromptTemplate(
            input_variables=["context_object"],
            template=SYSTEM_PROMPT + """

FULL CONTEXT OBJECT INPUT:
{context_object}

Based on the query understanding results within the context object, create an appropriate response:

If is_ambiguous is true:
- Set action to "CLARIFY"  
- Use the EXACT clarification_question from query understanding as message_to_user
- Set plan to null

If is_ambiguous is false:
- Set action to "PROCEED_TO_EXECUTE"
- Create a detailed execution plan with selected tools
- Set message_to_user to null

IMPORTANT: If the query is ambiguous, you MUST use the exact "clarification_question" text from the query understanding results. Do not generate your own clarification message.

Response format:
{{
    "action": "CLARIFY or PROCEED_TO_EXECUTE",
    "plan": [
        {{
            "step": 1,
            "tool": "tool1/tool2/tool3/tool4", 
            "description": "What this step accomplishes",
            "sub_query": "Specific query for this tool",
            "expected_output": "What we expect to get"
        }}
    ] or null,
    "message_to_user": "string or null"
}}
"""
        )
        
        self.parser = JsonOutputParser(pydantic_object=PlanningOutput)
        self.chain = self.prompt | self.llm | self.parser
    
    def invoke(self, context: ContextObject, config=None) -> ContextObject:
        """Process the context through planning."""
        try:
            log_agent_step(self.logger, "Planning", "Starting plan generation")
            
            # Check if query understanding was successful
            if not context.query_understanding:
                error_msg = "Planning Agent: No query understanding results available"
                log_error(self.logger, "Planning Agent", ValueError(error_msg))
                context.error_occurred = True
                context.error_message = error_msg
                return context
            
            # Invoke the chain with full context object using utils
            result = self.chain.invoke({
                "context_object": format_context_json(context)
            })
            
            # Update context with results
            context.planning = result
            context.current_stage = "planning_complete"
            
            log_agent_step(self.logger, "Planning", "Plan generation completed")
            return context
            
        except Exception as e:
            log_error(self.logger, "Planning Agent", e)
            context.error_occurred = True
            context.error_message = f"Planning Agent error: {str(e)}"
            return context
