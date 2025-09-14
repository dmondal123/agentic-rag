"""Interactive terminal interface for the Multi-Agent RAG System with detailed agent I/O."""

import json
import os
from typing import Dict, Any
from agents import create_orchestrated_chain, ContextObject
from agents.query_understanding_agent import QueryUnderstandingAgent
from agents.planning_agent import PlanningAgent 
from agents.execution_agent import ExecutionAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class InteractiveRAGSystem:
    """Interactive terminal interface with detailed agent monitoring."""
    
    def __init__(self):
        self.orchestrator = create_orchestrated_chain()
        # Create individual agents for detailed monitoring
        self.query_understanding_agent = QueryUnderstandingAgent()
        self.planning_agent = PlanningAgent()
        self.execution_agent = ExecutionAgent()
        
    def print_separator(self, title: str, char: str = "=", width: int = 60):
        """Print a formatted separator with title."""
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
        
    def print_agent_input(self, agent_name: str, input_data: Any):
        """Print formatted agent input."""
        self.print_separator(f"📥 {agent_name} INPUT", "─", 50)
        print(f"🔍 Input: {json.dumps(input_data, indent=2, default=str)}")
    
    def print_agent_output(self, agent_name: str, output_data: Any):
        """Print formatted agent output."""
        self.print_separator(f"📤 {agent_name} OUTPUT", "─", 50)
        
        if isinstance(output_data, ContextObject):
            print(f"✅ Current Stage: {output_data.current_stage}")
            print(f"✅ Error Occurred: {output_data.error_occurred}")
            
            if output_data.error_occurred:
                print(f"❌ Error Message: {output_data.error_message}")
                return
            
            # Show latest agent output
            if agent_name == "Query Understanding Agent":
                qu_result = output_data.query_understanding
                if qu_result:
                    print(f"🎯 Intent: {qu_result.get('intent', 'N/A')}")
                    print(f"🎯 Confidence: {qu_result.get('confidence', 'N/A')}")
                    print(f"🎯 Is Ambiguous: {qu_result.get('is_ambiguous', 'N/A')}")
                    
                    if qu_result.get('ambiguity_reason'):
                        print(f"🎯 Ambiguity Reason: {qu_result['ambiguity_reason']}")
                    if qu_result.get('clarification_question'):
                        print(f"🎯 Clarification Question: {qu_result['clarification_question']}")
                    
                    enhanced_query = qu_result.get('enhanced_query')
                    if enhanced_query:
                        print(f"🎯 Enhanced Query:")
                        print(f"   - Original: {enhanced_query.get('original_query', 'N/A')}")
                        print(f"   - Rewritten: {enhanced_query.get('rewritten_query', 'N/A')}")
                        print(f"   - Variations: {enhanced_query.get('query_variations', [])}")
                        print(f"   - Expansion Terms: {enhanced_query.get('expansion_terms', [])}")
            
            elif agent_name == "Planning Agent":
                planning_result = output_data.planning
                if planning_result:
                    print(f"📋 Action: {planning_result.get('action', 'N/A')}")
                    
                    if planning_result.get('message_to_user'):
                        print(f"📋 Message to User: {planning_result['message_to_user']}")
                    
                    plan = planning_result.get('plan')
                    if plan:
                        print(f"📋 Execution Plan ({len(plan)} steps):")
                        for i, step in enumerate(plan, 1):
                            print(f"   Step {i}: {step.get('tool', 'N/A')}")
                            print(f"     - Description: {step.get('description', 'N/A')}")
                            print(f"     - Sub-query: {step.get('sub_query', 'N/A')}")
                            print(f"     - Expected Output: {step.get('expected_output', 'N/A')}")
            
            elif agent_name == "Execution Agent":
                execution_result = output_data.execution
                if execution_result:
                    fused_context = execution_result.get('fused_context', '')
                    sources = execution_result.get('sources', [])
                    
                    print(f"⚡ Fused Context ({len(fused_context)} chars):")
                    print(f"   {fused_context[:200]}{'...' if len(fused_context) > 200 else ''}")
                    
                    print(f"⚡ Sources ({len(sources)} total):")
                    for i, source in enumerate(sources, 1):
                        print(f"   Source {i}: {source.get('source_id', 'N/A')}")
                        print(f"     - Tool: {source.get('tool', 'N/A')}")
                        print(f"     - Sub-query: {source.get('sub_query', 'N/A')}")
                        print(f"     - Content: {str(source.get('content_snippet', ''))[:100]}...")
        else:
            print(f"✅ Output: {json.dumps(output_data, indent=2, default=str)}")
    
    def run_detailed_pipeline(self, user_query: str, user_context: Dict[str, Any]):
        """Run the pipeline with detailed step-by-step monitoring."""
        
        self.print_separator("🚀 STARTING MULTI-AGENT PIPELINE", "=", 80)
        print(f"Query: '{user_query}'")
        
        try:
            # Step 1: Initialize Context
            context = ContextObject(
                user_query=user_query,
                user_context=user_context,
                current_stage="initialization"
            )
            
            self.print_separator("🏗️ CONTEXT INITIALIZATION", "=", 60)
            print(f"✅ Initialized context object")
            print(f"✅ User Query: {user_query}")
            print(f"✅ User Context: {len(user_context)} items")
            
            # Step 2: Query Understanding Agent
            print(f"\n🧠 STEP 1: QUERY UNDERSTANDING AGENT")
            self.print_agent_input("Query Understanding Agent", context)
            
            context = self.query_understanding_agent.invoke(context)
            
            self.print_agent_output("Query Understanding Agent", context)
            
            if context.error_occurred:
                print(f"\n❌ Pipeline halted due to error in Query Understanding Agent")
                return {"status": "error", "error": context.error_message}
            
            # Step 3: Planning Agent
            print(f"\n📋 STEP 2: PLANNING AGENT")
            self.print_agent_input("Planning Agent", context)
            
            context = self.planning_agent.invoke(context)
            
            self.print_agent_output("Planning Agent", context)
            
            if context.error_occurred:
                print(f"\n❌ Pipeline halted due to error in Planning Agent")
                return {"status": "error", "error": context.error_message}
            
            # Check if clarification is needed
            planning_result = context.planning
            if planning_result and planning_result.get("action") == "CLARIFY":
                self.print_separator("❓ CLARIFICATION NEEDED", "!", 60)
                print(f"🔄 Pipeline halted for user clarification")
                return {
                    "status": "clarification_needed",
                    "message": planning_result.get("message_to_user"),
                    "context_object": context.model_dump()
                }
            
            # Step 4: Execution Agent (only if PROCEED_TO_EXECUTE)
            if planning_result and planning_result.get("action") == "PROCEED_TO_EXECUTE":
                print(f"\n⚡ STEP 3: EXECUTION AGENT")
                self.print_agent_input("Execution Agent", context)
                
                context = self.execution_agent.invoke(context)
                
                self.print_agent_output("Execution Agent", context)
                
                if context.error_occurred:
                    print(f"\n❌ Pipeline halted due to error in Execution Agent")
                    return {"status": "error", "error": context.error_message}
                
                # Success!
                self.print_separator("✅ PIPELINE COMPLETED SUCCESSFULLY", "=", 80)
                return {
                    "status": "completed",
                    "fused_context": context.execution.get("fused_context"),
                    "sources": context.execution.get("sources"),
                    "context_object": context.model_dump()
                }
            
            # Unexpected case
            print(f"\n⚠️ Unexpected planning action or missing action")
            return {
                "status": "error",
                "error": "Unexpected planning action or missing action",
                "context_object": context.model_dump()
            }
            
        except Exception as e:
            self.print_separator("💥 PIPELINE ERROR", "!", 60)
            print(f"❌ Exception: {str(e)}")
            return {
                "status": "error",
                "error": f"Pipeline error: {str(e)}",
                "context_object": None
            }
    
    def display_final_result(self, result: Dict[str, Any]):
        """Display the final pipeline result."""
        
        self.print_separator("🎯 FINAL RESULT", "=", 60)
        
        status = result.get("status", "unknown")
        print(f"Status: {status.upper()}")
        
        if status == "completed":
            print(f"\n📝 Response:")
            print(f"{result.get('fused_context', 'No response')}")
            
            sources = result.get('sources', [])
            if sources:
                print(f"\n📚 Sources ({len(sources)} total):")
                for i, source in enumerate(sources, 1):
                    print(f"  {i}. {source.get('source_id', 'Unknown')} ({source.get('tool', 'Unknown tool')})")
        
        elif status == "clarification_needed":
            print(f"\n❓ Clarification Required:")
            print(f"{result.get('message', 'No clarification message')}")
        
        elif status == "error":
            print(f"\n❌ Error:")
            print(f"{result.get('error', 'Unknown error')}")
        
        print()
    
    def run_interactive_session(self):
        """Run the interactive terminal session."""
        
        print("🤖 Multi-Agent RAG System - Interactive Terminal")
        print("=" * 80)
        print("Enter your queries and see detailed agent-by-agent processing!")
        print("Type 'quit', 'exit', or press Ctrl+C to stop.")
        print("=" * 80)
        
        # Check environment setup
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  Warning: OPENAI_API_KEY not set. LLM calls will fail.")
            print("Set your API key: export OPENAI_API_KEY='your-key-here'")
            print()
        
        session_history = []
        
        while True:
            try:
                # Get user input
                print("\n" + "─" * 60)
                user_query = input("🎤 Enter your query: ").strip()
                
                if not user_query:
                    print("Please enter a valid query.")
                    continue
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                # Prepare user context with session history
                user_context = {
                    "conversation_history": session_history[-5:],  # Keep last 5 interactions
                    "short_term_memory": {
                        "session_length": len(session_history),
                        "last_query": session_history[-1].get("user") if session_history else None,
                        "summary": "",
                        "recent_turns": [],
                        "current_topic": "none"
                    },
                    "user_feedback": []
                }
                
                # Run the detailed pipeline
                result = self.run_detailed_pipeline(user_query, user_context)
                
                # Display final result
                self.display_final_result(result)
                
                # Add to session history
                session_entry = {
                    "user": user_query,
                    "status": result.get("status"),
                    "response": result.get("fused_context", result.get("message", "No response"))
                }
                session_history.append(session_entry)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}")
                print("Continuing session...")

def main():
    """Main entry point for interactive terminal."""
    
    try:
        interactive_system = InteractiveRAGSystem()
        interactive_system.run_interactive_session()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start interactive session: {str(e)}")

if __name__ == "__main__":
    main()
