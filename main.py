"""Main entry point for the Multi-Agent RAG System."""

import json
from typing import Dict, Any
from agents.orchestration_agent import create_orchestrated_chain

def run_multi_agent_rag(user_query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run the complete multi-agent RAG pipeline.
    
    Args:
        user_query: The user's query string
        user_context: Dictionary containing conversation_history, short_term_memory, user_feedback
        
    Returns:
        Dictionary with results, clarification request, or error information
    """
    if user_context is None:
        user_context = {
            "conversation_history": [],
            "short_term_memory": {},
            "user_feedback": []
        }
    
    # Create the orchestrated chain
    orchestrator = create_orchestrated_chain()
    
    # Prepare input
    input_data = {
        "user_query": user_query,
        "user_context": user_context
    }
    
    # Execute the pipeline
    result = orchestrator.invoke(input_data)
    
    return result

def main():
    """Example usage of the multi-agent RAG system."""
    
    print("🤖 Multi-Agent RAG System Demo")
    print("=" * 50)
    
    # Example 1: Clear query
    print("\n📝 Example 1: Clear Query")
    query1 = "What are the main benefits of renewable energy?"
    context1 = {
        "conversation_history": [],
        "short_term_memory": {},
        "user_feedback": []
    }
    
    result1 = run_multi_agent_rag(query1, context1)
    print(f"Query: {query1}")
    print(f"Status: {result1['status']}")
    
    if result1['status'] == 'completed':
        print("Response:")
        print(result1['fused_context'])
        print("\nSources:")
        for source in result1['sources']:
            print(f"- {source['source_id']}: {source['tool']} - {source['sub_query']}")
    elif result1['status'] == 'clarification_needed':
        print(f"Clarification needed: {result1['message']}")
    else:
        print(f"Error: {result1.get('error', 'Unknown error')}")
    
    print("\n" + "="*50)
    
    # Example 2: Ambiguous query
    print("\n📝 Example 2: Potentially Ambiguous Query")
    query2 = "How does it work?"
    context2 = {
        "conversation_history": [
            {"user": "Tell me about solar panels", "assistant": "Solar panels convert sunlight into electricity..."}
        ],
        "short_term_memory": {"last_topic": "solar panels"},
        "user_feedback": []
    }
    
    result2 = run_multi_agent_rag(query2, context2)
    print(f"Query: {query2}")
    print(f"Status: {result2['status']}")
    
    if result2['status'] == 'completed':
        print("Response:")
        print(result2['fused_context'])
    elif result2['status'] == 'clarification_needed':
        print(f"Clarification needed: {result2['message']}")
    else:
        print(f"Error: {result2.get('error', 'Unknown error')}")
    
    print("\n" + "="*50)
    print("✅ Demo completed!")

if __name__ == "__main__":
    main()
