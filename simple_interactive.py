"""Simple interactive terminal for quick testing with basic agent I/O display."""

import os
from typing import Dict, Any
from agents import create_orchestrated_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def display_agent_flow(result: Dict[str, Any]):
    """Display simplified agent flow information."""
    
    print("\n" + "="*60)
    print("🔍 AGENT FLOW SUMMARY")
    print("="*60)
    
    context_obj = result.get('context_object', {})
    
    # Query Understanding Results
    print("\n🧠 QUERY UNDERSTANDING AGENT:")
    qu_result = context_obj.get('query_understanding', {})
    if qu_result:
        print(f"  Intent: {qu_result.get('intent', 'N/A')}")
        print(f"  Confidence: {qu_result.get('confidence', 'N/A')}")
        print(f"  Ambiguous: {qu_result.get('is_ambiguous', 'N/A')}")
        
        enhanced = qu_result.get('enhanced_query', {})
        if enhanced:
            print(f"  Rewritten Query: {enhanced.get('rewritten_query', 'N/A')}")
    else:
        print("  ❌ No output received")
    
    # Planning Results  
    print("\n📋 PLANNING AGENT:")
    planning_result = context_obj.get('planning', {})
    if planning_result:
        print(f"  Action: {planning_result.get('action', 'N/A')}")
        if planning_result.get('message_to_user'):
            print(f"  Message: {planning_result['message_to_user']}")
        plan = planning_result.get('plan', [])
        if plan:
            print(f"  Plan Steps: {len(plan)}")
            for i, step in enumerate(plan, 1):
                print(f"    {i}. {step.get('tool', 'N/A')}: {step.get('description', 'N/A')}")
    else:
        print("  ❌ No output received")
    
    # Execution Results
    print("\n⚡ EXECUTION AGENT:")
    execution_result = context_obj.get('execution', {})
    if execution_result:
        fused_context = execution_result.get('fused_context', '')
        sources = execution_result.get('sources', [])
        print(f"  Response Length: {len(fused_context)} characters")
        print(f"  Sources: {len(sources)}")
        print(f"  Preview: {fused_context[:150]}{'...' if len(fused_context) > 150 else ''}")
    else:
        print("  ⚪ Not executed (clarification needed or error)")

def main():
    """Simple interactive loop."""
    
    print("🤖 Multi-Agent RAG System - Simple Interactive Mode")
    print("=" * 60)
    print("Type 'quit' to exit")
    
    # Environment check
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
    
    # Create orchestrator
    orchestrator = create_orchestrated_chain()
    conversation_history = []
    
    while True:
        try:
            # Get user input
            print("\n" + "-" * 60)
            query = input("Enter your query: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            print(f"\n🎯 Processing: '{query}'")
            
            # Prepare context
            user_context = {
                "conversation_history": conversation_history[-3:],  # Last 3 interactions
                "short_term_memory": {},
                "user_feedback": []
            }
            
            # Run pipeline
            result = orchestrator.invoke({
                "user_query": query,
                "user_context": user_context
            })
            
            # Display results
            status = result.get('status', 'unknown')
            print(f"\n📊 Status: {status.upper()}")
            
            if status == 'completed':
                print(f"\n✅ Response:")
                print(f"{result.get('fused_context', 'No response')}")
                
            elif status == 'clarification_needed':
                print(f"\n❓ Clarification needed:")
                print(f"{result.get('message', 'No message')}")
                
            elif status == 'error':
                print(f"\n❌ Error:")
                print(f"{result.get('error', 'Unknown error')}")
            
            # Show agent flow
            display_agent_flow(result)
            
            # Update conversation history
            conversation_history.append({
                "user": query,
                "assistant": result.get('fused_context', result.get('message', 'Error occurred'))
            })
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
