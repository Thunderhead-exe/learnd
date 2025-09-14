# Learnd MCP Server - Usage Guide

## 🚀 Quick Start

### 1. Setup and Configuration

First, ensure your environment is configured:

```bash
# Copy environment template
./copy-env.sh

# Edit .env with your credentials
nano .env
```

Required environment variables:
- `MISTRAL_API_KEY` - Your Mistral AI API key
- `QDRANT_URL` - Your Qdrant Cloud cluster URL  
- `QDRANT_API_KEY` - Your Qdrant Cloud API key

### 2. Start the MCP Server

```bash
# Install dependencies
uv sync

# Start the server
uv run fastmcp run learnd.mcp_server:mcp
```

The server will start on `http://localhost:8000` and expose 8 MCP tools for adaptive learning.

### 3. Verify Setup

Test that everything is working:

```bash
# Run validation
uv run python scripts/validate.py

# Run usage demonstration
uv run python examples/mcp_usage_demo.py
```

## 🧠 Available MCP Tools

### Core Learning Tools

#### 1. `learn_from_interaction`
**Purpose**: Learn from user-LLM interactions  
**Use**: Call after each user interaction  
**Example**:
```json
{
  "user_input": "How does machine learning work?",
  "llm_response": "Machine learning is...",
  "feedback_score": 0.9
}
```

#### 2. `get_relevant_context`
**Purpose**: Get relevant context for LLM augmentation  
**Use**: Call before generating LLM responses  
**Example**:
```json
{
  "query": "Explain neural networks",
  "max_concepts": 5,
  "similarity_threshold": 0.7
}
```

### Knowledge Management Tools

#### 3. `extract_concepts`
**Purpose**: Extract concepts from text  
**Use**: Process documents or analyze content  

#### 4. `search_concepts`
**Purpose**: Search knowledge base by similarity  
**Use**: Explore existing knowledge  

### System Management Tools

#### 5. `get_system_stats`
**Purpose**: Monitor system health and performance  
**Use**: Regular health checks  

#### 6. `rebalance_knowledge`
**Purpose**: Optimize layer performance  
**Use**: When primary layer utilization > 90%  

#### 7. `cleanup_old_concepts`
**Purpose**: Remove unused concepts  
**Use**: Regular maintenance  

## 📋 Integration Patterns

### Pattern 1: Real-time Learning Chat Bot

```python
import asyncio
from mcp_client import MCPClient

async def handle_chat_message(user_message: str) -> str:
    # 1. Get relevant context
    context_response = await mcp_client.call_tool("get_relevant_context", {
        "query": user_message,
        "max_concepts": 5
    })
    
    # 2. Enhance prompt with learned context
    if context_response["total_found"] > 0:
        enhanced_prompt = f"""
        User: {user_message}
        
        Relevant context from previous conversations:
        {context_response["formatted_context"]}
        
        Please provide a helpful response using this context.
        """
    else:
        enhanced_prompt = user_message
    
    # 3. Generate LLM response (replace with your LLM)
    llm_response = await your_llm.generate(enhanced_prompt)
    
    # 4. Learn from this interaction
    await mcp_client.call_tool("learn_from_interaction", {
        "user_input": user_message,
        "llm_response": llm_response,
        "feedback_score": 0.8  # Could be user feedback
    })
    
    return llm_response
```

### Pattern 2: Document Processing Pipeline

```python
async def process_documents(documents: List[str]):
    # Extract concepts from each document
    for doc in documents:
        result = await mcp_client.call_tool("extract_concepts", {
            "text": doc,
            "auto_store": True
        })
        print(f"Extracted {result['total_extracted']} concepts")
    
    # Optimize system after bulk processing
    await mcp_client.call_tool("rebalance_knowledge")
```

### Pattern 3: Knowledge-Enhanced Assistant

```python
async def answer_question(question: str) -> str:
    # Search for related knowledge
    search_result = await mcp_client.call_tool("search_concepts", {
        "query": question,
        "limit": 5
    })
    
    # Get contextual information
    context_result = await mcp_client.call_tool("get_relevant_context", {
        "query": question,
        "max_concepts": 3
    })
    
    # Build informed response
    if context_result["total_found"] > 0:
        prompt = f"""
        Question: {question}
        
        Available knowledge: {context_result["formatted_context"]}
        
        Provide a comprehensive answer using this knowledge.
        """
    else:
        prompt = question
    
    response = await your_llm.generate(prompt)
    
    # Learn from this interaction
    await mcp_client.call_tool("learn_from_interaction", {
        "user_input": question,
        "llm_response": response
    })
    
    return response
```

## 🔧 System Monitoring

### Health Check Script

```python
async def system_health_check():
    stats = await mcp_client.call_tool("get_system_stats")
    
    print(f"System Health: {'✅' if stats['system_healthy'] else '❌'}")
    print(f"Total Concepts: {stats['layers']['total_concepts']}")
    print(f"Primary Utilization: {stats['layers']['utilization_percent']:.1f}%")
    
    # Alert if system needs attention
    if stats['layers']['utilization_percent'] > 90:
        print("⚠️ Consider rebalancing - primary layer near capacity")
        
        # Auto-rebalance
        rebalance_result = await mcp_client.call_tool("rebalance_knowledge")
        print(f"🔄 Rebalanced: {rebalance_result['concepts_promoted']} promoted, {rebalance_result['concepts_demoted']} demoted")
    
    return stats
```

### Maintenance Script

```python
async def weekly_maintenance():
    # Clean up old concepts
    cleanup_result = await mcp_client.call_tool("cleanup_old_concepts", {
        "age_threshold_days": 60
    })
    print(f"🧹 Cleaned up {cleanup_result['concepts_removed']} old concepts")
    
    # Rebalance for optimal performance
    await mcp_client.call_tool("rebalance_knowledge")
    print("⚖️ Knowledge rebalanced")
    
    # Check final stats
    stats = await mcp_client.call_tool("get_system_stats")
    print(f"📊 System health: {stats['layers']['total_concepts']} concepts")
```

## 🎯 Complete Usage Demo

Run the comprehensive demo to see all patterns in action:

```bash
uv run python examples/mcp_usage_demo.py
```

This demo shows:
- ✅ Educational AI assistant scenario
- ✅ Real-time learning from 8 student interactions  
- ✅ Context-aware response enhancement
- ✅ System monitoring and optimization
- ✅ Different integration patterns

## 📊 Performance Tips

### Optimal Settings

- **Primary Layer Capacity**: 500-2000 concepts for best performance
- **Similarity Threshold**: 
  - 0.8+ for high-precision context retrieval
  - 0.5-0.7 for exploratory search
- **Rebalancing**: When primary utilization > 90%
- **Cleanup**: Every 30-90 days based on use case

### Best Practices

1. **Always call `learn_from_interaction`** after user interactions
2. **Use `get_relevant_context`** before LLM calls for better responses  
3. **Monitor with `get_system_stats`** regularly (daily/weekly)
4. **Rebalance periodically** for optimal performance
5. **Cleanup old concepts** monthly to manage storage

## 🔍 Troubleshooting

### Common Issues

**Server won't start:**
- Check environment variables in `.env`
- Verify Qdrant Cloud cluster is running
- Confirm Mistral API key is valid

**No concepts learned:**
- Check Mistral API key and quota
- Verify text input is meaningful
- Check server logs for errors

**Slow responses:**
- Check primary layer utilization
- Run `rebalance_knowledge` if > 90%
- Verify Qdrant Cloud connection

**Memory usage high:**
- Run `cleanup_old_concepts`
- Check total concept count
- Consider adjusting thresholds

### Debug Commands

```bash
# Check environment
uv run python -c "import os; print('MISTRAL_API_KEY:', bool(os.getenv('MISTRAL_API_KEY')))"

# Validate system
uv run python scripts/validate.py

# Test individual components
uv run python examples/mistral_qdrant_demo.py
```

## 🆘 Support

- **Documentation**: See `MCP_TOOLS_REFERENCE.md` for detailed tool docs
- **Examples**: Check `examples/` directory for usage patterns
- **Validation**: Run `scripts/validate.py` to check system health

---

🎉 **You're ready to build adaptive learning systems with Learnd!**

Start with the patterns above and customize for your specific use case. The system learns and improves automatically as you use it.
