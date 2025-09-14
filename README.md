# Multi-Agent RAG System

A sophisticated multi-agent orchestration system for Retrieval-Augmented Generation (RAG) using LangChain and ChatLiteLLM.

## Architecture

The system consists of four main agents:

### 1. 🎯 Orchestration Agent
- **Input**: `user_query`, `user_context` (conversation_history, short_term_memory, user_feedback)
- **Role**: Manages the stateful `context_object` and controls flow between agents
- **Flow Control**: Halts processing if planning agent returns "CLARIFY" action

### 2. 🧠 Query Understanding Agent  
- **Input**: `context_object`
- **Tasks**:
  - Intent classification: `abcd_1`, `abcd_2`, `abcd_3`, `abcd_4`, `abcd_5`, `UNKNOWN`
  - Ambiguity detection (vagueness, unresolved entities, missing info)
  - Query enhancement (pronoun resolution, keyword extraction, variations)
- **Output**: Structured JSON with intent, confidence, ambiguity flags, enhanced query

### 3. 📋 Planning Agent
- **Input**: `context_object` (with query understanding results)
- **Tasks**:
  - Handle ambiguous queries with clarification requests
  - Tool selection from: `tool1`, `tool2`, `tool3`, `tool4`
  - Generate step-by-step execution plan
- **Output**: Action (`CLARIFY`/`PROCEED_TO_EXECUTE`), plan, message

### 4. ⚡ Execution Agent
- **Input**: `context_object` (with planning results)
- **Tasks**:
  - Execute planned tools with generated sub-queries
  - Collect and synthesize data from multiple sources
  - Embed citations for all information
- **Output**: Fused context with comprehensive source mapping

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   MODEL_NAME=gpt-4o
   ```

## Usage

### Basic Usage

```python
from main import run_multi_agent_rag

# Simple query
result = run_multi_agent_rag(
    user_query="What are the benefits of renewable energy?",
    user_context={
        "conversation_history": [],
        "short_term_memory": {},
        "user_feedback": []
    }
)

if result['status'] == 'completed':
    print(result['fused_context'])
    print("Sources:", result['sources'])
elif result['status'] == 'clarification_needed':
    print("Clarification:", result['message'])
```

### Response Types

#### 1. Successful Completion
```json
{
    "status": "completed",
    "fused_context": "Comprehensive response with citations [Source: source_id]",
    "sources": [
        {
            "source_id": "unique_id",
            "tool": "tool_name", 
            "sub_query": "query_used",
            "content_snippet": "relevant_content"
        }
    ],
    "context_object": {...}
}
```

#### 2. Clarification Needed
```json
{
    "status": "clarification_needed",
    "message": "What specific aspect would you like to know about?",
    "context_object": {...}
}
```

#### 3. Error Handling
```json
{
    "status": "error", 
    "error": "Error description",
    "context_object": {...}
}
```

## Running the System

### 🎮 Interactive Terminal Interfaces

```bash
# Detailed agent-by-agent monitoring (recommended for development)
python interactive_terminal.py

# Simple interactive mode (quick testing)
python simple_interactive.py

# Demo with pre-defined examples
python main.py

# Comprehensive tests
python test_system.py
```

### 🔍 Interactive Features

**`interactive_terminal.py`** - Full detailed monitoring:
- Step-by-step agent execution with input/output display  
- Real-time pipeline visualization
- Complete context object inspection
- Error handling and debugging information
- Session history tracking

**`simple_interactive.py`** - Quick testing interface:
- Clean, minimal output focused on results
- Agent flow summary after each query
- Conversation history maintenance
- Fast iteration for testing queries

### 📝 Demo Mode
The `main.py` script runs pre-defined example queries demonstrating both clear and ambiguous query handling.

## System Features

✅ **Stateful Context Management**: Persistent context object across all agents  
✅ **Intent Classification**: 5-category classification with confidence scoring  
✅ **Ambiguity Detection**: Intelligent detection of vague or unclear queries  
✅ **Dynamic Tool Selection**: Context-aware tool selection from available options  
✅ **Citation Tracking**: Complete source mapping for all synthesized information  
✅ **Error Handling**: Comprehensive error handling at each stage  
✅ **LangChain Integration**: Built entirely on LangChain RunnableChain architecture  
✅ **ChatLiteLLM Support**: Flexible LLM backend with GPT-4o default  

## Configuration

The system uses environment variables for configuration:
- `MODEL_NAME`: LLM model to use (default: "gpt-4o")
- `OPENAI_API_KEY`: Your OpenAI API key

## Extension Points

The system is designed for easy extension:

1. **Tool Integration**: Add new tools by extending the ExecutionAgent
2. **Intent Categories**: Modify intent classification in QueryUnderstandingAgent  
3. **System Prompts**: Update agent prompts for domain-specific behavior
4. **Output Formats**: Customize response structures as needed

## Architecture Diagram

### RunnableBranch Flow Control
```
User Query + Context
        ↓
┌─────────────────────────────────────────────────┐
│              Orchestration Agent                │
│         (RunnableBranch Architecture)           │
└─────────────────────────────────────────────────┘
        ↓
┌─────────────────────┐
│    Initialize       │ → ContextObject Creation
│     Context         │
└─────────────────────┘
        ↓
┌─────────────────────┐
│Query Understanding  │ → Intent, Confidence, Ambiguity  
│     Agent          │   Enhanced Query, Clarification
└─────────────────────┘
        ↓
    RunnableBranch
   ┌─────┴─────┐
   │           │
   ▼           ▼
[Error?]   [Continue]
   │           │
   ▼           ▼
[Error]   ┌─────────────────────┐
Response  │   Planning Agent    │ → CLARIFY/PROCEED_TO_EXECUTE
          │                     │   Tool Selection, Execution Plan
          └─────────────────────┘
                    ↓
              RunnableBranch
         ┌───────┬────────┬───────┐
         │       │        │       │
         ▼       ▼        ▼       ▼
    [Error?] [CLARIFY?] [PROCEED] 
         │       │        │       
         ▼       ▼        ▼       
    [Error]  [Clarify]  ┌─────────────────────┐
    Response Response   │  Execution Agent    │ → Fused Context
                        │                     │   Source Citations  
                        └─────────────────────┘
                                  ↓
                            RunnableBranch
                           ┌─────┴─────┐
                           │           │
                           ▼           ▼
                       [Error?]   [Success]
                           │           │
                           ▼           ▼
                       [Error]    [Completed]
                       Response    Response
```

## RunnableBranch Implementation

The orchestration agent uses **LangChain's RunnableBranch** for elegant conditional flow control:

### Key Features:
- **Declarative Chaining**: Flow logic expressed as chain composition rather than imperative code
- **Nested Branching**: Multiple levels of conditional routing based on agent outputs  
- **Error Handling**: Dedicated branches for error states at each stage
- **Condition Functions**: Pure functions for decision logic (`_should_clarify`, `_check_for_errors`)
- **Response Formatting**: Dedicated formatters for each response type

### Chain Structure:
```python
chain = (
    RunnableLambda(_initialize_context)
    | query_understanding_agent
    | RunnableBranch(
        (error_condition, error_handler),
        planning_agent | RunnableBranch(
            (error_condition, error_handler),
            (clarify_condition, clarification_handler), 
            execution_agent | RunnableBranch(
                (error_condition, error_handler),
                success_handler
            )
        )
    )
)
```

This approach provides:
- **Composability**: Easy to modify flow logic
- **Testability**: Pure functions for conditions  
- **Readability**: Clear flow visualization
- **LangChain Native**: Fully leverages LangChain's declarative paradigm

## Technical Requirements Met

- ✅ LangChain-only implementation
- ✅ **RunnableBranch** for conditional agent chaining  
- ✅ **ChatLiteLLM** integration with GPT-4o
- ✅ Environment-based model configuration
- ✅ Exact output structure compliance
- ✅ Stateful context object management
- ✅ Declarative conditional flow control (CLARIFY halt)
