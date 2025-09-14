"""LLM configuration and management utilities."""

import os
from langchain_litellm import ChatLiteLLM
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

def get_llm(model_name: Optional[str] = None, temperature: float = 0.1) -> ChatLiteLLM:
    """
    Get a configured LLM instance.
    
    Args:
        model_name: Override the default model name
        temperature: Model temperature setting
        
    Returns:
        ChatLiteLLM: Configured LLM instance
    """
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "gpt-4o")
    
    return ChatLiteLLM(model=model_name, temperature=temperature)

def get_fast_llm(temperature: float = 0.1) -> ChatLiteLLM:
    """
    Get a fast LLM instance for control tokens and quick operations.
    
    Args:
        temperature: Model temperature setting
        
    Returns:
        ChatLiteLLM: Fast LLM instance (typically GPT-4o-mini or similar)
    """
    fast_model = os.getenv("FAST_MODEL_NAME", "gpt-4o-mini")
    return ChatLiteLLM(model=fast_model, temperature=temperature)

def check_openai_key() -> bool:
    """
    Check if OpenAI API key is configured.
    
    Returns:
        bool: True if API key is available
    """
    return bool(os.getenv("OPENAI_API_KEY"))

def get_model_config() -> dict:
    """
    Get current model configuration.
    
    Returns:
        dict: Model configuration details
    """
    return {
        "primary_model": os.getenv("MODEL_NAME", "gpt-4o"),
        "fast_model": os.getenv("FAST_MODEL_NAME", "gpt-4o-mini"),
        "openai_key_configured": check_openai_key()
    }
