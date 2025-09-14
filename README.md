# 🧠 Learnd - Simplified Adaptive Learning MCP Server

A streamlined **single-file** Model Context Protocol server that learns from every interaction and provides intelligent context to enhance LLM responses.

## 🚀 Key Features

- **🔄 Auto-Learning**: Automatically extracts and stores concepts from every text input
- **🎯 Smart Context**: Finds relevant learned knowledge to enhance responses  
- **📝 Simple Storage**: In-memory concept storage with frequency tracking
- **⚡ Single File**: Everything in one `mcp_server.py` file for easy deployment
- **🛠️ Self-Documenting**: Clear explanations in every function

## 📁 Project Structure

```
learnd/
├── mcp_server.py          # Complete MCP server (ONLY FILE NEEDED)
├── .env                   # Environment variables
├── pyproject.toml         # Dependencies
└── README.md              # This file
```

## 🛠️ Setup

1. **Clone and setup:**
   ```bash
   git clone <your-repo>
   cd learnd
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Configure environment:**
   ```bash
   cp env.template .env
   # Edit .env with your API keys (optional for basic functionality)
   ```

4. **Start the server:**
   ```bash
   uv run fastmcp run mcp_server:mcp
   ```

## 🧠 How It Works

### Core Learning Cycle

1. **Learn**: `learn_from_text()` extracts concepts from any text
2. **Store**: Concepts stored with frequency tracking  
3. **Retrieve**: `get_relevant_context()` finds relevant learned knowledge
4. **Enhance**: Provides context to improve LLM responses

### Automatic Learning

The system automatically learns from:
- ✅ User questions and inputs
- ✅ Technical terms and concepts  
- ✅ Domain-specific vocabulary
- ✅ Frequent topics and patterns

## 🔧 MCP Tools

### 🧠 `learn_from_text`
**Learn concepts from any text**
```json
Input: {"text": "I'm building a React app with authentication"}
Output: {
  "success": true,
  "concepts_learned": ["react", "authentication", "app building"],
  "total_concepts": 3
}
```

### 🔍 `get_relevant_context` 
**Find relevant learned knowledge**
```json
Input: {"query": "How do I deploy my app?"}
Output: {
  "relevant_concepts": [
    {"text": "deployment", "frequency": 5, "similarity": 0.9}
  ],
  "context_summary": "Based on previous learning about: deployment (mentioned 5 times)"
}
```

### 🤖 `smart_response_with_learning`
**Complete learning + context pipeline**
```json
Input: {"user_input": "Help me with React routing"}
Output: {
  "learning_results": {...},
  "relevant_context": {...},
  "enhanced_prompt": "User Input: Help me with React routing\n\nRelevant Context: react (mentioned 10 times), routing (mentioned 3 times)..."
}
```

### 📊 `get_learning_stats`
**View what the system has learned**
```json
Output: {
  "total_concepts": 25,
  "top_concepts": [
    {"text": "react", "frequency": 10},
    {"text": "deployment", "frequency": 8}
  ]
}
```

### 🔄 `reset_learning` 
**Clear all learned data**

### ❤️ `health_check`
**Verify system health**

## 🎯 Usage Patterns

### Pattern 1: Auto-Learning Chat Bot
```python
# For every user interaction:
1. Call smart_response_with_learning(user_input)
2. Use the enhanced_prompt for your LLM
3. System automatically learns and provides context
```

### Pattern 2: Knowledge Building
```python  
# Learn from documents/content:
1. Call learn_from_text(document_content)
2. Later use get_relevant_context(user_question)
3. Provide context-aware responses
```

### Pattern 3: Topic Tracking
```python
# Monitor learning progress:
1. Call get_learning_stats() regularly
2. See what topics users discuss most
3. Identify knowledge gaps
```

## ⚡ Quick Start Example

```bash
# Start server
uv run fastmcp run mcp_server:mcp

# Test learning (using MCP client):
learn_from_text {"text": "I need help with Python web development using FastAPI"}
# → Learns: "python", "web development", "fastapi"

get_relevant_context {"query": "How do I create APIs?"}  
# → Returns: Relevant context about "python", "web development", "fastapi"

smart_response_with_learning {"user_input": "Show me FastAPI examples"}
# → Learns from input + provides enhanced prompt with previous context
```

## 🔧 Configuration

### Environment Variables (Optional)
```bash
# For advanced features (not required for basic functionality):
MISTRAL_API_KEY=your_key_here
QDRANT_URL=your_qdrant_url  
QDRANT_API_KEY=your_qdrant_key
```

### Deployment
Simply deploy `mcp_server.py` - it contains everything needed:
- ✅ No external dependencies on complex packages
- ✅ Simple in-memory storage
- ✅ Built-in error handling
- ✅ Self-contained functionality

## 🎉 Key Improvements

### ✅ Solved Issues:
- **Actually learns**: Concepts are extracted and stored automatically
- **Single file**: No complex package structure  
- **Clear functions**: Each tool has obvious purpose and examples
- **Auto-called**: `smart_response_with_learning` handles everything
- **Simple storage**: In-memory storage that actually works
- **Enhanced explanations**: Every function explains WHAT, WHEN, HOW

### 🔥 Perfect for:
- Chat applications that learn from conversations
- Knowledge base building from documents  
- Context-aware LLM responses
- Topic tracking and analysis
- Rapid prototyping and deployment

---

**🚀 Deploy `mcp_server.py` and start learning from every interaction!**