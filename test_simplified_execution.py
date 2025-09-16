#!/usr/bin/env python3
"""
Test script for the simplified execution agent that only uses vectordb retrieval.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def test_simplified_execution_agent():
    """Test the simplified execution agent that only does PostgreSQL vectordb retrieval."""
    try:
        from agents.execution_agent import ExecutionAgent
        from agents.base_agent import ContextObject
        
        print("Testing simplified execution agent...")
        
        # Test agent initialization (this will test PostgreSQL connection)
        print("1. Testing agent initialization...")
        agent = ExecutionAgent()
        print("✓ Agent initialized successfully with PostgreSQL connection")
        
        # Create a simple context object for testing (no complex planning needed!)
        print("2. Creating test context...")
        context = ContextObject(
            user_query="What is machine learning?",
            user_context={
                "short_term_memory": {
                    "current_topic": "AI concepts",
                    "summary": "User asking about AI and ML topics"
                }
            },
            query_understanding={
                "enhanced_query": {
                    "rewritten_query": "What is machine learning and how does it work?",
                    "expansion_terms": ["artificial intelligence", "algorithms", "data science"]
                }
            }
        )
        print("✓ Test context created (simple, no complex planning required)")
        
        # Test the execution
        print("3. Testing vectordb retrieval and synthesis...")
        result_context = agent.invoke(context)
        
        if result_context.error_occurred:
            print(f"✗ Execution failed: {result_context.error_message}")
            return False
        
        # Check if execution results exist
        if hasattr(result_context, 'execution') and result_context.execution:
            execution_result = result_context.execution
            print("✓ Vectordb retrieval and synthesis completed successfully")
            
            # Check if it has the expected structure
            if 'fused_context' in execution_result and 'sources' in execution_result:
                print("✓ Execution result has correct structure")
                print(f"   - Fused context length: {len(execution_result['fused_context'])}")
                print(f"   - Number of sources: {len(execution_result['sources'])}")
                
                # Show detailed source info
                for i, source in enumerate(execution_result['sources']):
                    print(f"   - Source {i}: tool={source.get('tool', 'N/A')}, relevance={source.get('relevance_score', 0.0):.3f}")
                
                # Show a snippet of the result
                snippet = execution_result['fused_context'][:300] + "..." if len(execution_result['fused_context']) > 300 else execution_result['fused_context']
                print(f"   - Result snippet: {snippet}")
                
                return True
            else:
                print("✗ Execution result missing expected fields")
                return False
        else:
            print("✗ No execution results found")
            return False
            
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING SIMPLIFIED EXECUTION AGENT")
    print("(Only vectordb retrieval using retrieval_utils)")
    print("=" * 60)
    
    success = test_simplified_execution_agent()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ ALL TESTS PASSED - Execution agent now properly uses only vectordb!")
    else:
        print("✗ TESTS FAILED - Check the errors above")
    print("=" * 60)
