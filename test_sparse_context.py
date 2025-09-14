"""Test script for the enhanced Execution Agent with Sparse Context Selection."""

import os
import json
from agents.base_agent import ContextObject
from agents.execution_agent import ExecutionAgent

def test_sparse_context_execution():
    """Test the Sparse Context Selection integration."""
    
    print("🚀 Testing Sparse Context Selection Integration")
    print("=" * 60)
    
    # Enable sparse context feature
    os.environ["ENABLE_SPARSE_CONTEXT"] = "true"
    os.environ["MODEL_NAME"] = "gpt-4o"
    
    # Initialize execution agent
    execution_agent = ExecutionAgent()
    print(f"✅ Sparse Context Enabled: {execution_agent.use_sparse_context}")
    
    # Create test context
    test_context = ContextObject(
        user_query="What are the main differences between React and Vue.js?",
        user_context={
            "conversation_history": [],
            "short_term_memory": {
                "summary": "",
                "recent_turns": [],
                "current_topic": "frontend_frameworks"
            }
        }
    )
    
    # Add query understanding results
    test_context.query_understanding = {
        "intent": "abcd_2",  # Comparison request
        "confidence": 0.9,
        "is_ambiguous": False,
        "enhanced_query": {
            "original_query": "What are the main differences between React and Vue.js?",
            "rewritten_query": "Compare React JavaScript library with Vue.js framework features performance",
            "query_variations": ["React vs Vue comparison", "React Vue differences"],
            "expansion_terms": ["JavaScript", "framework", "library", "performance", "components"]
        }
    }
    
    # Add planning results
    test_context.planning = {
        "action": "PROCEED_TO_EXECUTE",
        "plan": [
            {
                "step": 1,
                "tool": "knowledge_base_1",
                "description": "Search for React information",
                "sub_query": "React JavaScript library features components virtual DOM",
                "expected_output": "React technical information"
            },
            {
                "step": 2,
                "tool": "knowledge_base_2", 
                "description": "Search for Vue.js information",
                "sub_query": "Vue.js framework features reactivity template syntax",
                "expected_output": "Vue.js technical information"
            },
            {
                "step": 3,
                "tool": "knowledge_base_3",
                "description": "Search for comparison information", 
                "sub_query": "React Vue.js performance comparison enterprise adoption",
                "expected_output": "Comparison analysis"
            }
        ]
    }
    
    print("\n📋 Test Configuration:")
    print(f"   Query: {test_context.user_query}")
    print(f"   Intent: {test_context.query_understanding['intent']}")
    print(f"   Plan Steps: {len(test_context.planning['plan'])}")
    
    print("\n⚡ Executing with Sparse Context Selection...")
    
    try:
        # Execute the agent
        result_context = execution_agent.invoke(test_context)
        
        if result_context.error_occurred:
            print(f"❌ Error: {result_context.error_message}")
            return False
            
        print("\n✅ Execution completed successfully!")
        
        # Display results
        execution_result = result_context.execution
        
        print(f"\n📊 Results Summary:")
        print(f"   Sources Retrieved: {len(execution_result.get('sources', []))}")
        print(f"   Response Length: {len(execution_result.get('fused_context', ''))}")
        
        print(f"\n📄 Generated Response:")
        print("-" * 40)
        print(execution_result.get('fused_context', 'No response generated'))
        
        print(f"\n📚 Sources Used:")
        for i, source in enumerate(execution_result.get('sources', [])):
            priority = source.get('metadata', {}).get('processing_priority', 'unknown')
            weight = source.get('metadata', {}).get('attention_weight', 0.0)
            relevance = source.get('metadata', {}).get('relevance_score', 0.0)
            
            print(f"   {i+1}. {source.get('source_id', 'unknown')} "
                  f"[Priority: {priority.upper()}, Weight: {weight:.1f}, Relevance: {relevance:.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

def test_legacy_mode():
    """Test the legacy mode (without sparse context)."""
    
    print("\n\n🔄 Testing Legacy Mode (No Sparse Context)")
    print("=" * 60)
    
    # Disable sparse context feature
    os.environ["ENABLE_SPARSE_CONTEXT"] = "false"
    
    # Initialize execution agent
    execution_agent = ExecutionAgent()
    print(f"✅ Sparse Context Enabled: {execution_agent.use_sparse_context}")
    
    # Use same test context as above
    test_context = ContextObject(
        user_query="What are the benefits of using React?",
        user_context={"conversation_history": [], "short_term_memory": {}}
    )
    
    test_context.query_understanding = {
        "intent": "abcd_1",
        "confidence": 0.8,
        "is_ambiguous": False,
        "enhanced_query": {
            "original_query": "What are the benefits of using React?",
            "rewritten_query": "React JavaScript library benefits advantages features",
            "query_variations": ["React advantages", "Why use React"],
            "expansion_terms": ["JavaScript", "components", "virtual DOM"]
        }
    }
    
    test_context.planning = {
        "action": "PROCEED_TO_EXECUTE",
        "plan": [
            {
                "step": 1,
                "tool": "knowledge_base_1",
                "description": "Search for React benefits",
                "sub_query": "React benefits advantages performance",
                "expected_output": "React advantages information"
            }
        ]
    }
    
    try:
        result_context = execution_agent.invoke(test_context)
        
        if result_context.error_occurred:
            print(f"❌ Error: {result_context.error_message}")
            return False
            
        print("✅ Legacy mode execution completed!")
        print(f"   Response generated: {len(result_context.execution.get('fused_context', '')) > 0}")
        
        return True
        
    except Exception as e:
        print(f"❌ Legacy test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Sparse Context Selection Integration Test")
    print("Testing the enhanced Execution Agent capabilities")
    print("=" * 70)
    
    # Test sparse context mode
    sparse_success = test_sparse_context_execution()
    
    # Test legacy mode  
    legacy_success = test_legacy_mode()
    
    print("\n" + "=" * 70)
    print("📈 Test Results Summary:")
    print(f"   Sparse Context Mode: {'✅ PASSED' if sparse_success else '❌ FAILED'}")
    print(f"   Legacy Mode: {'✅ PASSED' if legacy_success else '❌ FAILED'}")
    
    if sparse_success and legacy_success:
        print("\n🎉 All tests passed! Sparse Context Selection is successfully integrated.")
        print("\nNext steps:")
        print("1. Set ENABLE_SPARSE_CONTEXT=true in your .env file")
        print("2. Ensure you have openai>=1.0.0 installed: pip install openai>=1.0.0")
        print("3. Test with your actual queries and monitor performance improvements")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
