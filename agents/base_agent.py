"""Base agent class and common utilities for the multi-agent RAG system."""

import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_litellm import ChatLiteLLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ContextObject(BaseModel):
    """Stateful context object passed between agents."""
    
    # Initial inputs
    user_query: str = Field(..., description="Original user query")
    user_context: Dict[str, Any] = Field(default_factory=dict, description="User context including conversation_history, short_term_memory, user_feedback")
    
    # Query Understanding Agent outputs
    query_understanding: Optional[Dict[str, Any]] = Field(default=None, description="Output from query understanding agent")
    
    # Planning Agent outputs
    planning: Optional[Dict[str, Any]] = Field(default=None, description="Output from planning agent")
    
    # Execution Agent outputs
    execution: Optional[Dict[str, Any]] = Field(default=None, description="Output from execution agent")
    
    # Metadata
    current_stage: str = Field(default="initialization", description="Current stage in the pipeline")
    error_occurred: bool = Field(default=False, description="Flag indicating if an error occurred")
    error_message: Optional[str] = Field(default=None, description="Error message if any")

def get_llm():
    """Get the configured LLM instance."""
    model_name = os.getenv("MODEL_NAME", "gpt-4o")
    return ChatLiteLLM(model=model_name, temperature=0.1)
