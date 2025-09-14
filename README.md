# 🧠 Learnd - Simplified AI Learning MCP Server

A **clean and simple** Model Context Protocol server following the official Mistral + Qdrant tutorial architecture. Provides intelligent memory and learning capabilities for AI agents.

## 🚀 Key Features

- **🤖 Mistral AI Integration**: Uses official Mistral client for concept extraction and embeddings
- **☁️ Qdrant Cloud Storage**: Official Qdrant client for vector storage and similarity search
- **🔄 Auto-Learning**: Extract and store concepts from interactions automatically
- **🎯 Memory Retrieval**: Find relevant stored information using semantic search
- **⚡ Tutorial-Based**: Follows official Mistral + Qdrant integration patterns
- **🛠️ Production Ready**: Simplified, reliable architecture

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
   # Edit .env with your API keys (REQUIRED for full AI functionality)
   ```
   
   **Required environment variables:**
   ```bash
   MISTRAL_API_KEY=your_mistral_api_key
   QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
   QDRANT_API_KEY=your_qdrant_api_key
   ```

4. **Start the server:**
   ```bash
   uv run fastmcp run mcp_server:mcp
   ```

## 🧠 How It Works

### Simplified Learning Pipeline

1. **Concept Extraction**: Mistral AI extracts key concepts from user interactions
2. **Vector Embedding**: Mistral's embedding model creates semantic vectors
3. **Vector Storage**: Qdrant Cloud stores embeddings with metadata
4. **Semantic Search**: Vector similarity search finds relevant memories
5. **Context Enhancement**: Retrieved context enhances LLM responses

### Architecture (Following Tutorial)

- **🤖 Mistral AI**: Official client for concept extraction and embeddings
- **☁️ Qdrant Cloud**: Official client for vector database operations
- **🔗 MCP Tools**: Simple `qdrant_store` and `qdrant_find` operations
- **📊 Smart Retrieval**: Semantic similarity with score thresholds
- **🧠 Auto-Learning**: Automated concept extraction from interactions

## 🔧 MCP Tools

### 🗃️ `qdrant_store`
**Store information in vector memory**
```json
Input: {"text": "React is a JavaScript library for building user interfaces", "context": "web development"}
Output: {
  "success": true,
  "message": "Stored successfully with ID: abc-123",
  "stored_text": "React is a JavaScript library...",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 🔍 `qdrant_find` 
**Find similar information from memory**
```json
Input: {"query": "JavaScript frameworks", "max_results": 3}
Output: {
  "results": [
    {"text": "React is a JavaScript library...", "score": 0.95, "stored_at": "2024-01-15T10:30:00Z"}
  ],
  "total_found": 1,
  "query": "JavaScript frameworks"
}
```

### 🧠 `learn_from_interaction`
**Extract and store concepts from interactions**
```json
Input: {"user_input": "How do I deploy a React app?", "llm_response": "You can deploy using Vercel...", "importance": "high"}
Output: {
  "success": true,
  "concepts_stored": ["react deployment", "vercel", "web hosting"],
  "total_stored": 3
}
```

### 🎯 `get_relevant_context`
**Get learned context for queries**
```json
Input: {"query": "deployment help", "max_concepts": 5}
Output: {
  "context_found": true,
  "formatted_context": "Based on previous learning: 'react deployment' (relevance: 0.95), 'vercel' (relevance: 0.89)...",
  "concepts": [...],
  "total_found": 2
}
```

### ❤️ `get_system_status`
**Check system health and configuration**

### 🗑️ `clear_memory`
**Clear all stored memories (destructive!)**

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