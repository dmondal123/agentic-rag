"""Logging utilities to replace print statements with proper logging."""

import logging
import sys
from typing import Any, Dict, Optional
from enum import Enum
from .context_utils import sanitize_for_logging

class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

def setup_logger(name: str = "agentic_rag", level: str = "INFO", 
                log_to_file: bool = False, log_file: str = "agentic_rag.log") -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_file: Log file name
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_agent_step(logger: logging.Logger, agent_name: str, step: str, 
                  data: Optional[Dict[str, Any]] = None, level: LogLevel = LogLevel.INFO):
    """
    Log an agent processing step.
    
    Args:
        logger: Logger instance
        agent_name: Name of the agent
        step: Description of the step
        data: Optional data to log (will be sanitized)
        level: Log level
    """
    message = f"[{agent_name}] {step}"
    
    if data:
        sanitized_data = sanitize_for_logging(data, max_length=100)
        message += f" - Data: {sanitized_data}"
    
    getattr(logger, level.value.lower())(message)

def log_error(logger: logging.Logger, context: str, error: Exception, 
              additional_data: Optional[Dict[str, Any]] = None):
    """
    Log an error with context.
    
    Args:
        logger: Logger instance
        context: Context where error occurred
        error: The exception
        additional_data: Additional data for debugging
    """
    message = f"Error in {context}: {str(error)}"
    
    if additional_data:
        sanitized_data = sanitize_for_logging(additional_data, max_length=150)
        message += f" - Additional data: {sanitized_data}"
    
    logger.error(message, exc_info=True)

def log_agent_input(logger: logging.Logger, agent_name: str, input_data: Dict[str, Any]):
    """
    Log agent input data in a structured format.
    
    Args:
        logger: Logger instance
        agent_name: Name of the agent
        input_data: Input data to log
    """
    sanitized_input = sanitize_for_logging(input_data, max_length=200)
    message = f"[{agent_name}] Input: {sanitized_input}"
    logger.debug(message)

def log_agent_output(logger: logging.Logger, agent_name: str, output_data: Dict[str, Any]):
    """
    Log agent output data in a structured format.
    
    Args:
        logger: Logger instance
        agent_name: Name of the agent
        output_data: Output data to log
    """
    sanitized_output = sanitize_for_logging(output_data, max_length=200)
    message = f"[{agent_name}] Output: {sanitized_output}"
    logger.debug(message)

def log_pipeline_start(logger: logging.Logger, user_query: str, user_context_size: int):
    """
    Log the start of a pipeline execution.
    
    Args:
        logger: Logger instance
        user_query: User query being processed
        user_context_size: Size of user context
    """
    message = f"Pipeline started - Query: '{user_query[:100]}...' - Context size: {user_context_size}"
    logger.info(message)

def log_pipeline_end(logger: logging.Logger, status: str, duration_ms: Optional[float] = None):
    """
    Log the end of a pipeline execution.
    
    Args:
        logger: Logger instance
        status: Final status of the pipeline
        duration_ms: Execution duration in milliseconds
    """
    message = f"Pipeline completed - Status: {status}"
    if duration_ms:
        message += f" - Duration: {duration_ms:.2f}ms"
    logger.info(message)

def log_memory_operation(logger: logging.Logger, operation: str, details: Dict[str, Any]):
    """
    Log memory operations like context updates, topic changes, etc.
    
    Args:
        logger: Logger instance
        operation: Type of memory operation
        details: Operation details
    """
    sanitized_details = sanitize_for_logging(details, max_length=100)
    message = f"Memory operation: {operation} - {sanitized_details}"
    logger.debug(message)

def log_warning(logger: logging.Logger, context: str, message: str, 
               additional_data: Optional[Dict[str, Any]] = None):
    """
    Log a warning with context.
    
    Args:
        logger: Logger instance
        context: Context where warning occurred
        message: Warning message
        additional_data: Additional data for context
    """
    warning_message = f"Warning in {context}: {message}"
    
    if additional_data:
        sanitized_data = sanitize_for_logging(additional_data, max_length=100)
        warning_message += f" - Data: {sanitized_data}"
    
    logger.warning(warning_message)

# Global logger instance for convenience
_default_logger = None

def get_default_logger() -> logging.Logger:
    """Get the default logger instance."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger()
    return _default_logger
