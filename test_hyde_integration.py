"""Test script for HyDE integration with the unified query understanding agent."""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from agents.query_understanding_agent import QueryUnderstandingAgent
from agents.base_agent import ContextObject

def test_hyde_integration():
    """Test HyDE integration with various query types."""
    
    # Test queries that should benefit from HyDE
    test_queries = [
        {
            "query": "What did Paul Graham do after RISD?",
            "description": "Temporal/biographical query - HyDE should help"
        },
        {
            "query": "How to configure database connections in PostgreSQL?",
            "description": "Technical how-to query - HyDE should generate helpful doc"
        },
        {
            "query": "What are the benefits of RAG systems?", 
            "description": "Conceptual query - HyDE should provide comprehensive context"
        },
        {
            "query": "Compare vector databases vs traditional databases",
            "description": "Comparison query - HyDE should generate balanced perspective"
        }
    ]
    
    print("🧪 Testing HyDE Integration with Query Understanding Agent")
    print("=" * 60)
    
    # Initialize the query understanding agent
    try:
        agent = QueryUnderstandingAgent()
        print(f"✅ Agent initialized successfully")
        print(f"   HyDE enabled: {agent.use_hyde}")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Test each query
    for i, test_case in enumerate(test_queries, 1):
        print(f"🔍 Test {i}: {test_case['description']}")
        print(f"   Query: \"{test_case['query']}\"")
        
        # Create context object
        context = ContextObject(
            user_query=test_case['query'],
            user_context={
                "user_id": "test_user",
                "conversation_history": [],
                "short_term_memory": {
                    "current_topic": "none",
                    "summary": "",
                    "recent_turns": []
                }
            }
        )
        
        try:
            # Process the query
            result_context = agent.invoke(context)
            
            if result_context.error_occurred:
                print(f"   ❌ Error: {result_context.error_message}")
                continue
                
            query_result = result_context.query_understanding
            enhanced_query = query_result.get("enhanced_query")
            
            print(f"   ✅ Intent: {query_result['intent']}")
            print(f"   📊 Confidence: {query_result['confidence']:.2f}")
            print(f"   🤔 Ambiguous: {query_result['is_ambiguous']}")
            
            if enhanced_query:
                print(f"   📝 Rewritten: \"{enhanced_query['rewritten_query'][:80]}...\"")
                print(f"   🔍 Expansion Terms: {enhanced_query.get('expansion_terms', [])}")
                
                # Check HyDE output
                hyde_doc = enhanced_query.get('hypothetical_document')
                if hyde_doc:
                    print(f"   🚀 HyDE Document Generated: {len(hyde_doc)} chars")
                    print(f"      Preview: \"{hyde_doc[:120]}...\"")
                else:
                    print(f"   ⚠️  No HyDE document generated")
            else:
                print(f"   ⚠️  No enhanced query generated")
                
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 60)
    
    print("🏁 HyDE Integration Testing Complete!")

def test_hyde_failure_cases():
    """Test HyDE with queries that might not benefit or could mislead."""
    
    failure_test_queries = [
        {
            "query": "What is Bel?",
            "description": "Ambiguous query - might mislead without context"
        },
        {
            "query": "Hello",
            "description": "Simple greeting - shouldn't need HyDE"
        },
        {
            "query": "asdfghjkl random text",
            "description": "Nonsensical query - HyDE might struggle"
        }
    ]
    
    print("\n🚨 Testing HyDE Failure/Edge Cases")
    print("=" * 60)
    
    agent = QueryUnderstandingAgent()
    
    for i, test_case in enumerate(failure_test_queries, 1):
        print(f"⚠️  Test {i}: {test_case['description']}")
        print(f"   Query: \"{test_case['query']}\"")
        
        context = ContextObject(
            user_query=test_case['query'],
            user_context={
                "user_id": "test_user", 
                "conversation_history": [],
                "short_term_memory": {"current_topic": "none", "summary": "", "recent_turns": []}
            }
        )
        
        try:
            result_context = agent.invoke(context)
            query_result = result_context.query_understanding
            enhanced_query = query_result.get("enhanced_query")
            
            if enhanced_query:
                hyde_doc = enhanced_query.get('hypothetical_document')
                if hyde_doc:
                    print(f"   🤖 HyDE Generated: \"{hyde_doc[:100]}...\"")
                    print(f"   📊 Quality Assessment: Check if this makes sense for the query")
                else:
                    print(f"   ✅ No HyDE document (appropriate for this query)")
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
        
        print("-" * 60)

if __name__ == "__main__":
    print("🔧 Checking environment variables...")
    
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("   Please set them in your .env file")
        sys.exit(1)
    
    print("✅ Environment variables configured")
    print()
    
    # Run main tests
    test_hyde_integration()
    
    # Run failure case tests
    test_hyde_failure_cases()
