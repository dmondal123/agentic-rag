"""
Test script for the Cassandra vector retriever.
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from retrieval_utils.cassandra_retriever import get_cassandra_retriever, AsyncCassandraVectorRetriever

def test_basic_retrieval():
    """Test basic vector retrieval without filters."""
    
    print("🔍 Testing Basic Cassandra Vector Retrieval")
    print("=" * 50)
    
    # Initialize retriever
    retriever = get_cassandra_retriever(
        keyspace=os.getenv('CASSANDRA_KEYSPACE', 'vectordb'),
        hosts=[os.getenv('CASSANDRA_HOST', '127.0.0.1')],
        port=int(os.getenv('CASSANDRA_PORT', '9042')),
        k=5
    )
    
    # Test queries
    test_queries = [
        "What are the benefits of machine learning?",
        "How to configure database connections?",
        "Explain quantum computing concepts",
    ]
    
    try:
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: {query}")
            
            # Retrieve documents
            results = retriever.retrieve_with_filters(query=query)
            
            print(f"   Found {len(results)} documents")
            
            for j, doc in enumerate(results):
                print(f"   📄 Doc {j+1}:")
                print(f"      Content: {doc.page_content[:100]}...")
                print(f"      File ID: {doc.metadata.get('file_id')}")
                print(f"      Client ID: {doc.metadata.get('client_id')}")
                print(f"      Score: {doc.metadata.get('combined_score', 0):.3f}")
                print(f"      Filter Tags: {doc.metadata.get('filter_tags', [])}")
                
    except Exception as e:
        print(f"❌ Error during retrieval: {e}")
        
    finally:
        retriever.close()

def test_filtered_retrieval():
    """Test retrieval with various filters."""
    
    print("\n🎯 Testing Filtered Cassandra Vector Retrieval")
    print("=" * 50)
    
    retriever = get_cassandra_retriever(
        keyspace=os.getenv('CASSANDRA_KEYSPACE', 'vectordb'),
        hosts=[os.getenv('CASSANDRA_HOST', '127.0.0.1')],
        port=int(os.getenv('CASSANDRA_PORT', '9042')),
        k=3
    )
    
    test_cases = [
        {
            "description": "Filter by Client ID",
            "params": {
                "query": "machine learning algorithms",
                "client_id": "client-123"
            }
        },
        {
            "description": "Filter by File ID", 
            "params": {
                "query": "database optimization",
                "file_id": "file-456"
            }
        },
        {
            "description": "Filter by Client + File",
            "params": {
                "query": "API documentation",
                "client_id": "client-123",
                "file_id": "file-789"
            }
        },
        {
            "description": "Filter by Tags",
            "params": {
                "query": "security best practices",
                "filter_tags": ["security", "documentation"]
            }
        }
    ]
    
    try:
        for case in test_cases:
            print(f"\n🔎 {case['description']}")
            print(f"   Query: {case['params']['query']}")
            
            # Show filters
            filters = {k: v for k, v in case['params'].items() if k != 'query'}
            if filters:
                print(f"   Filters: {filters}")
            
            results = retriever.retrieve_with_filters(**case['params'])
            
            print(f"   ✅ Found {len(results)} documents")
            
            for j, doc in enumerate(results):
                print(f"      📄 Doc {j+1}: {doc.page_content[:80]}...")
                print(f"         Score: {doc.metadata.get('combined_score', 0):.3f}")
                
    except Exception as e:
        print(f"❌ Error during filtered retrieval: {e}")
        
    finally:
        retriever.close()

async def test_async_retrieval():
    """Test async retrieval for high-concurrency scenarios."""
    
    print("\n⚡ Testing Async Cassandra Vector Retrieval")
    print("=" * 50)
    
    # Initialize async retriever
    async_retriever = AsyncCassandraVectorRetriever(
        keyspace=os.getenv('CASSANDRA_KEYSPACE', 'vectordb'),
        hosts=[os.getenv('CASSANDRA_HOST', '127.0.0.1')],
        port=int(os.getenv('CASSANDRA_PORT', '9042')),
        k=3
    )
    
    # Test concurrent queries
    concurrent_queries = [
        {"query": "machine learning models", "client_id": "client-1"},
        {"query": "database design patterns", "client_id": "client-2"},
        {"query": "API security guidelines", "filter_tags": ["security"]},
        {"query": "cloud architecture best practices"},
        {"query": "microservices deployment strategies", "file_id": "file-123"}
    ]
    
    try:
        print(f"🚀 Running {len(concurrent_queries)} queries concurrently...")
        
        start_time = asyncio.get_event_loop().time()
        
        # Execute all queries concurrently
        results = await async_retriever.batch_retrieve(concurrent_queries)
        
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time
        
        print(f"✅ Completed in {total_time:.2f} seconds")
        print(f"   Average: {total_time/len(concurrent_queries):.2f}s per query")
        
        # Show results summary
        for i, (query_params, docs) in enumerate(zip(concurrent_queries, results)):
            print(f"\n   Query {i+1}: {query_params['query'][:50]}...")
            print(f"   Results: {len(docs)} documents")
            if docs:
                best_score = max(doc.metadata.get('combined_score', 0) for doc in docs)
                print(f"   Best Score: {best_score:.3f}")
                
    except Exception as e:
        print(f"❌ Error during async retrieval: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        async_retriever.sync_retriever.close()

def test_connection():
    """Test basic Cassandra connection."""
    
    print("🔗 Testing Cassandra Connection")
    print("=" * 30)
    
    try:
        retriever = get_cassandra_retriever(
            keyspace=os.getenv('CASSANDRA_KEYSPACE', 'vectordb'),
            hosts=[os.getenv('CASSANDRA_HOST', '127.0.0.1')],
            port=int(os.getenv('CASSANDRA_PORT', '9042')),
            k=1
        )
        
        print("✅ Cassandra connection successful")
        print(f"   Keyspace: {retriever.keyspace}")
        print(f"   Host: {os.getenv('CASSANDRA_HOST', '127.0.0.1')}")
        print(f"   Port: {os.getenv('CASSANDRA_PORT', '9042')}")
        
        retriever.close()
        return True
        
    except Exception as e:
        print(f"❌ Cassandra connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Cassandra Vector Retriever Test Suite")
    print("=" * 60)
    
    # Check environment variables
    required_vars = ["OPENAI_API_KEY"]
    optional_vars = ["CASSANDRA_KEYSPACE", "CASSANDRA_HOST", "CASSANDRA_PORT", 
                     "CASSANDRA_USERNAME", "CASSANDRA_PASSWORD"]
    
    print("\n📋 Environment Check:")
    for var in required_vars:
        if os.getenv(var):
            print(f"   ✅ {var}: Set")
        else:
            print(f"   ❌ {var}: Missing (Required)")
            
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"   ✅ {var}: {value}")
        else:
            print(f"   ⚠️  {var}: Using default")
    
    # Test connection first
    if not test_connection():
        print("\n❌ Cannot proceed without Cassandra connection")
        exit(1)
    
    # Run tests
    test_basic_retrieval()
    test_filtered_retrieval()
    
    # Run async tests
    print("\n" + "="*60)
    asyncio.run(test_async_retrieval())
    
    print("\n🏁 All tests completed!")
