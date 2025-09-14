# Usage Examples

## Interactive Terminal Interfaces

### 🔍 Detailed Agent Monitoring (`interactive_terminal.py`)

Perfect for development and debugging - shows complete input/output for each agent:

```bash
python interactive_terminal.py
```

**Sample Session:**
```
🤖 Multi-Agent RAG System - Interactive Terminal
================================================================================
Enter your queries and see detailed agent-by-agent processing!
Type 'quit', 'exit', or press Ctrl+C to stop.
================================================================================

────────────────────────────────────────────────────────────────
🎤 Enter your query: What are the benefits of solar energy?

============================== 🚀 STARTING MULTI-AGENT PIPELINE ==============================
Query: 'What are the benefits of solar energy?'

============================= 🏗️ CONTEXT INITIALIZATION =============================
✅ Initialized context object
✅ User Query: What are the benefits of solar energy?
✅ User Context: 3 items

🧠 STEP 1: QUERY UNDERSTANDING AGENT
────────────── 📥 Query Understanding Agent INPUT ───────────────
🔍 User Query: What are the benefits of solar energy?
🔍 Current Stage: initialization
🔍 User Context:
   - conversation_history: 0 items
   - short_term_memory: 2 items
   - user_feedback: 0 items

────────────── 📤 Query Understanding Agent OUTPUT ──────────────
✅ Current Stage: query_understanding_complete
✅ Error Occurred: False
🎯 Intent: abcd_1
🎯 Confidence: 0.95
🎯 Is Ambiguous: False
🎯 Enhanced Query:
   - Original: What are the benefits of solar energy?
   - Rewritten: solar energy benefits advantages renewable power
   - Variations: ['benefits of solar power', 'advantages of photovoltaic systems']
   - Expansion Terms: ['renewable', 'sustainable', 'clean energy', 'photovoltaic']

📋 STEP 2: PLANNING AGENT
──────────────── 📥 Planning Agent INPUT ────────────────
🔍 User Query: What are the benefits of solar energy?
🔍 Current Stage: query_understanding_complete
🔍 Previous Query Understanding: Available

──────────────── 📤 Planning Agent OUTPUT ───────────────
✅ Current Stage: planning_complete
✅ Error Occurred: False
📋 Action: PROCEED_TO_EXECUTE
📋 Execution Plan (3 steps):
   Step 1: tool1
     - Description: Search internal knowledge base for solar energy information
     - Sub-query: solar energy benefits advantages renewable power
     - Expected Output: Technical and economic benefits of solar systems

⚡ STEP 3: EXECUTION AGENT
─────────────── 📥 Execution Agent INPUT ────────────────
🔍 User Query: What are the benefits of solar energy?
🔍 Current Stage: planning_complete
🔍 Previous Query Understanding: Available
🔍 Previous Planning: Available

─────────────── 📤 Execution Agent OUTPUT ───────────────
✅ Current Stage: execution_complete
✅ Error Occurred: False
⚡ Fused Context (450 chars):
   Solar energy offers numerous benefits including cost savings, environmental sustainability [Source: source_tool1_1], reduced carbon footprint [Source: source_tool2_2], and energy independence...
⚡ Sources (3 total):
   Source 1: source_tool1_1
     - Tool: tool1
     - Sub-query: solar energy benefits advantages renewable power
     - Content: Mock result from tool1: This is simulated content for query 'solar energy benefits advantages...

================================= ✅ PIPELINE COMPLETED SUCCESSFULLY =================================

================================ 🎯 FINAL RESULT ===============================
Status: COMPLETED

📝 Response:
Solar energy offers numerous benefits including cost savings, environmental sustainability [Source: source_tool1_1], reduced carbon footprint [Source: source_tool2_2], and energy independence from traditional grid systems [Source: source_tool3_3]. The technology has become increasingly affordable and efficient over recent years.

📚 Sources (3 total):
  1. source_tool1_1 (tool1)
  2. source_tool2_2 (tool2) 
  3. source_tool3_3 (tool3)
```

### ⚡ Simple Interactive Mode (`simple_interactive.py`)

Quick testing with clean output focused on results:

```bash
python simple_interactive.py
```

**Sample Session:**
```
🤖 Multi-Agent RAG System - Simple Interactive Mode
============================================================
Type 'quit' to exit

────────────────────────────────────────────────────────
Enter your query: How does solar energy work?

🎯 Processing: 'How does solar energy work?'

📊 Status: COMPLETED

✅ Response:
Solar energy works through photovoltaic cells that convert sunlight directly into electricity [Source: source_tool1_1]. When photons hit the silicon cells, they knock electrons loose, creating an electrical current [Source: source_tool2_2]. This DC electricity is then converted to AC power for home use through inverters [Source: source_tool3_3].

============================================================
🔍 AGENT FLOW SUMMARY
============================================================

🧠 QUERY UNDERSTANDING AGENT:
  Intent: abcd_3
  Confidence: 0.92
  Ambiguous: False
  Rewritten Query: solar energy photovoltaic process electricity generation

📋 PLANNING AGENT:
  Action: PROCEED_TO_EXECUTE
  Plan Steps: 3
    1. tool1: Search for solar energy technical information
    2. tool2: Find photovoltaic process explanations
    3. tool3: Retrieve energy conversion details

⚡ EXECUTION AGENT:
  Response Length: 287 characters
  Sources: 3
  Preview: Solar energy works through photovoltaic cells that convert sunlight directly into electricity [Source: source_tool1_1]. When photons hit the silicon cells, they knock electrons loose...
```

## Example Query Types

### 1. Clear Factual Queries
- "What are the benefits of renewable energy?"
- "How do solar panels generate electricity?"
- "What is the efficiency of wind turbines?"

### 2. Analytical Queries (Trigger abcd_2 intent)
- "Compare solar vs wind energy efficiency"
- "What are the pros and cons of different renewable sources?"
- "Analyze the cost-effectiveness of hydroelectric power"

### 3. Procedural Queries (Trigger abcd_3 intent)
- "How do you install a solar panel system?"
- "What steps are needed to build a wind farm?"
- "How to maintain a hydroelectric generator?"

### 4. Ambiguous Queries (Trigger clarification)
- "How does it work?" (without context)
- "What about the costs?" (vague reference)
- "Tell me more" (unclear topic)

### 5. Complex Multi-step Queries (Trigger abcd_5 intent)
- "Explain renewable energy, compare different types, and recommend the best for residential use"
- "What are sustainable energy sources, how do they work, and what are the installation costs?"

## Environment Setup

Before running the interactive interfaces:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your API key:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export MODEL_NAME="gpt-4o"  # Optional, defaults to gpt-4o
   ```

3. **Run the interface:**
   ```bash
   # For detailed monitoring
   python interactive_terminal.py
   
   # For quick testing  
   python simple_interactive.py
   ```

## Troubleshooting

### Common Issues:

1. **Import Error**: Make sure `langchain-litellm` is installed
2. **API Key Missing**: Set `OPENAI_API_KEY` environment variable
3. **Model Not Found**: Check that your API key supports GPT-4o model

### Testing Without API Key:

The interfaces will run without an API key but will show errors when making LLM calls. This is useful for testing the flow structure and agent connectivity.
