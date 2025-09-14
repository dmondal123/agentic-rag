"""Test script for the Multi-Agent RAG System."""

import json
import os
from typing import Dict, Any
from agents import create_orchestrated_chain

def test_agent_outputs():
    """Test each agent's output structure compliance."""
    
    print("🧪 Testing Multi-Agent RAG System")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            "name": "Clear Factual Query",
            "query": "What are the main advantages of solar energy?",
            "context": {
                "conversation_history": [],
                "short_term_memory": {},
                "user_feedback": []
            },
            "expected_intent": "abcd_1"  # Factual information
        },
        {
            "name": "Ambiguous Query", 
            "query": "How does it work?",
            "context": {
                "conversation_history": [],
                "short_term_memory": {},
                "user_feedback": []
            },
            "expected_ambiguous": True
        },
        {
            "name": "Query with Context",
            "query": "What are the efficiency rates?",
            "context": {
                "conversation_history": [
                    {"user": "Tell me about solar panels", "assistant": "Solar panels convert sunlight..."}
                ],
                "short_term_memory": {"last_topic": "solar panels"},
                "user_feedback": []
            },
            "expected_intent": "abcd_1"
        },
        {
            "name": "Analytical Query",
            "query": "Compare wind energy versus solar energy efficiency.",
            "context": {
                "conversation_history": [],
                "short_term_memory": {},
                "user_feedback": []
            },
            "expected_intent": "abcd_2"  # Analytical/comparison
        }
    ]
    
    # Create orchestrator
    orchestrator = create_orchestrated_chain()
    
    # Run tests
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔬 Test {i}: {test_case['name']}")
        print(f"Query: '{test_case['query']}'")
        print("-" * 40)
        
        try:
            # Execute pipeline
            result = orchestrator.invoke({
                "user_query": test_case["query"],
                "user_context": test_case["context"]
            })
            
            print(f"✅ Status: {result['status']}")
            
            if result['status'] == 'completed':
                print("📊 Query Understanding Results:")
                context_obj = result['context_object']
                query_understanding = context_obj.get('query_understanding', {})
                
                print(f"  - Intent: {query_understanding.get('intent', 'N/A')}")
                print(f"  - Confidence: {query_understanding.get('confidence', 'N/A')}")
                print(f"  - Is Ambiguous: {query_understanding.get('is_ambiguous', 'N/A')}")
                
                if query_understanding.get('enhanced_query'):
                    enhanced = query_understanding['enhanced_query']
                    print(f"  - Rewritten Query: {enhanced.get('rewritten_query', 'N/A')}")
                    print(f"  - Query Variations: {len(enhanced.get('query_variations', []))}")
                    print(f"  - Expansion Terms: {len(enhanced.get('expansion_terms', []))}")
                
                print("\n📋 Planning Results:")
                planning = context_obj.get('planning', {})
                print(f"  - Action: {planning.get('action', 'N/A')}")
                if planning.get('plan'):
                    print(f"  - Plan Steps: {len(planning['plan'])}")
                    for step in planning['plan']:
                        print(f"    • {step.get('tool', 'N/A')}: {step.get('description', 'N/A')}")
                
                print("\n⚡ Execution Results:")
                execution = context_obj.get('execution', {})
                if execution:
                    fused_context = execution.get('fused_context', '')
                    sources = execution.get('sources', [])
                    print(f"  - Response Length: {len(fused_context)} characters")
                    print(f"  - Number of Sources: {len(sources)}")
                    print(f"  - Response Preview: {fused_context[:100]}...")
                
            elif result['status'] == 'clarification_needed':
                print(f"❓ Clarification Message: {result['message']}")
                
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
        
        print("\n" + "=" * 60)

def test_output_structure_compliance():
    """Test that outputs match the specified structures exactly."""
    
    print("\n🔍 Testing Output Structure Compliance")
    print("=" * 60)
    
    orchestrator = create_orchestrated_chain()
    
    # Test with a simple query
    result = orchestrator.invoke({
        "user_query": "What is renewable energy?",
        "user_context": {
            "conversation_history": [],
            "short_term_memory": {},
            "user_feedback": []
        }
    })
    
    print("🔬 Checking Query Understanding Output Structure:")
    context_obj = result.get('context_object', {})
    query_understanding = context_obj.get('query_understanding', {})
    
    required_fields = ['intent', 'confidence', 'is_ambiguous']
    for field in required_fields:
        if field in query_understanding:
            print(f"  ✅ {field}: {query_understanding[field]}")
        else:
            print(f"  ❌ Missing required field: {field}")
    
    optional_fields = ['ambiguity_reason', 'clarification_question', 'enhanced_query']
    for field in optional_fields:
        if field in query_understanding:
            print(f"  ✅ {field}: Present")
        else:
            print(f"  ⚪ {field}: Not present (optional)")
    
    print("\n🔬 Checking Planning Output Structure:")
    planning = context_obj.get('planning', {})
    
    required_planning_fields = ['action']
    for field in required_planning_fields:
        if field in planning:
            print(f"  ✅ {field}: {planning[field]}")
        else:
            print(f"  ❌ Missing required field: {field}")
    
    print("\n🔬 Checking Execution Output Structure:")
    execution = context_obj.get('execution', {})
    
    if execution:
        required_execution_fields = ['fused_context', 'sources']
        for field in required_execution_fields:
            if field in execution:
                print(f"  ✅ {field}: Present")
            else:
                print(f"  ❌ Missing required field: {field}")
    else:
        print("  ⚪ Execution not performed (query may have been ambiguous)")

def main():
    """Run all tests."""
    
    # Check if environment is set up
    if not os.getenv("MODEL_NAME"):
        print("⚠️  Warning: MODEL_NAME environment variable not set. Using default 'gpt-4o'")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. You'll need to set this for actual LLM calls.")
        print("For testing structure only, this is okay.\n")
    
    # Run tests
    test_agent_outputs()
    test_output_structure_compliance()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()
