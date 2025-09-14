# 🧠 Learnd: Adaptive Continuous Learning MCP Server

**Learnd** is a Model Context Protocol (MCP) server that implements adaptive continuous learning with hierarchical vector storage. It automatically extracts concepts from user interactions using **Mistral AI**, stores them in **Qdrant Cloud**, and provides intelligent retrieval for LLM augmentation.

## 🌟 Features

- **Adaptive Learning**: Automatically extracts and learns concepts from user interactions
- **Hierarchical Storage**: Two-layer system (primary/secondary) with frequency-based migration  
- **Intelligent Clustering**: Groups infrequent concepts using advanced clustering algorithms
- **MCP Integration**: Full Model Context Protocol server for seamless LLM integration
- **Real-time Updates**: Dynamic frequency tracking and layer optimization
- **Cloud-Native**: Built on Qdrant Cloud and Mistral AI for scalability and reliability
- **Lightning Fast**: Powered by `uv` for ultra-fast dependency management

## 📁 Project Structure

```
learnd/
├── learnd/                 # Main package
│   ├── __init__.py
│   ├── mcp_server.py      # 🚀 Main MCP server with 8 tools
│   ├── core.py            # Core learning engine
│   ├── models.py          # Data models and configuration
│   ├── database.py        # Qdrant Cloud integration
│   ├── embeddings.py      # Sentence transformers for embeddings
│   ├── concept_extractor.py  # Mistral AI concept extraction
│   └── clustering.py      # Concept clustering algorithms
├── examples/              # Usage demonstration
│   └── mcp_usage_demo.py  # Complete usage examples
├── tests/                 # Test suite
├── scripts/               # Utility scripts
│   └── validate.py        # System validation
├── env.template          # Environment configuration template
├── copy-env.sh           # Script to create .env from template
├── USAGE_GUIDE.md        # 📚 Complete usage guide
├── pyproject.toml        # uv project configuration
└── README.md
```

## 🏗️ Architecture

### Layer System
- **Primary Layer (L1)**: Fast-access layer for frequently used concepts (configurable capacity)
- **Secondary Layer (L2)**: Comprehensive storage with clustering for less frequent concepts

### Core Components
- **Concept Extractor**: Uses secondary LLM to extract key concepts from text
- **Embedding Manager**: Generates and manages vector embeddings
- **Clustering Manager**: Organizes secondary layer concepts into meaningful clusters
- **Frequency Tracker**: Monitors usage patterns and triggers layer migrations

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (ultra-fast Python package manager)
- [Qdrant Cloud](https://cloud.qdrant.io) account (free tier available)
- [Mistral AI](https://mistral.ai) API key

### Installation

1. **Clone and set up with uv:**
```bash
git clone <repository>
cd learnd
./scripts/setup.sh  # Installs uv and sets up the project
```

2. **Create environment file:**
```bash
./copy-env.sh  # Creates .env from template
```

3. **Set up Qdrant Cloud:**
   - Sign up at [cloud.qdrant.io](https://cloud.qdrant.io) (no credit card required)
   - Create a new cluster (free 1GB tier)
   - Copy your cluster URL and API key

4. **Get Mistral API Key:**
   - Sign up at [mistral.ai](https://mistral.ai)
   - Get your API key from the dashboard

5. **Configure environment:**
```bash
# Edit .env file with your actual credentials:
nano .env  # or use your preferred editor

# Update these lines:
MISTRAL_API_KEY=your_actual_mistral_api_key
QDRANT_URL=https://your-cluster.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=your_actual_qdrant_api_key
```

6. **Start the MCP server:**

For local development:
```bash
uv run fastmcp run learnd.mcp_server:mcp
```

For deployment (avoids import issues & handles read-only filesystems):
```bash
uv run fastmcp run deploy_server:mcp
```

> 💡 **Deployment Note**: If you encounter "[Errno 30] Read-only file system" errors, use `deploy_server.py` which automatically handles serverless/container environments with read-only filesystems.

The server will start on `http://localhost:8000` with 8 MCP tools available.

## 📖 Usage

### 🧠 Available MCP Tools

1. **`learn_from_interaction`** - Primary learning from user interactions
2. **`get_relevant_context`** - Context augmentation for LLM responses  
3. **`extract_concepts`** - Manual concept extraction from text
4. **`search_concepts`** - Knowledge exploration and discovery
5. **`get_system_stats`** - System monitoring and health metrics
6. **`rebalance_knowledge`** - Performance optimization
7. **`cleanup_old_concepts`** - Maintenance and storage management

### Quick Integration Example

```python
import asyncio
from mcp_client import MCPClient

async def enhanced_chat_bot(user_message: str) -> str:
    # 1. Get relevant context from learned knowledge
    context = await mcp_client.call_tool("get_relevant_context", {
        "query": user_message,
        "max_concepts": 5
    })
    
    # 2. Enhance LLM prompt with learned context
    if context["total_found"] > 0:
        enhanced_prompt = f"""
        User: {user_message}
        Context: {context["formatted_context"]}
        """
    else:
        enhanced_prompt = user_message
    
    # 3. Generate LLM response
    response = await your_llm.generate(enhanced_prompt)
    
    # 4. Learn from this interaction
    await mcp_client.call_tool("learn_from_interaction", {
        "user_input": user_message,
        "llm_response": response,
        "feedback_score": 0.8
    })
    
    return response
```

### Complete Usage Guide

📚 **Documentation:**
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Complete usage guide with examples
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Production deployment instructions

**USAGE_GUIDE.md includes:**
- Detailed tool documentation with input/output formats
- 4 integration patterns for different use cases  
- Performance optimization tips
- System monitoring and maintenance
- Complete usage demonstration

**DEPLOYMENT_GUIDE.md includes:**
- Deployment-ready server for production
- Fixing relative import issues  
- Read-only filesystem solutions
- Platform-specific instructions (AWS Lambda, Google Cloud, Azure)
- Docker deployment
- Troubleshooting guide

### 🔧 Quick Deployment Check

Before deploying, run the diagnostic tool to check for common issues:

```bash
uv run python deployment_check.py
```

This will verify:
- ✅ Filesystem permissions
- ✅ Environment variables  
- ✅ Required dependencies
- ✅ Python path configuration
- ✅ Cache setup

## ⚙️ Configuration

### Environment Variables
```bash
MISTRAL_API_KEY=your_mistral_api_key
QDRANT_URL=https://your-cluster.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
COLLECTION_NAME=learnd-concepts
PRIMARY_LAYER_CAPACITY=1000
PROMOTION_THRESHOLD=10
DEMOTION_THRESHOLD=2
```

### Configuration File (`config.json`)
```json
{
  "primary_layer_capacity": 1000,
  "promotion_threshold": 10,
  "demotion_threshold": 2,
  "clustering_method": "kmeans",
  "embedding_model": "all-MiniLM-L6-v2",
  "concept_extraction_model": "mistral-large-latest"
}
```

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run specific test file  
uv run pytest tests/test_core.py

# Run with coverage
uv run pytest --cov=learnd

# Validate installation
uv run python scripts/validate.py
```

## 📊 Monitoring & Maintenance

### System Health
```python
# Check layer statistics
stats = await mcp_client.call_tool("get_layer_statistics")

# Cleanup old concepts
result = await mcp_client.call_tool("cleanup_unused_concepts", {
    "age_threshold_days": 30
})

# Apply frequency decay
await mcp_client.call_tool("apply_frequency_decay")
```

### Performance Optimization
```python
# Optimize vector store
await mcp_client.call_tool("optimize_vector_store")

# Rebalance layers for optimal performance
await mcp_client.call_tool("rebalance_layers")
```

## 🔧 Advanced Features

### Custom Clustering
The system supports multiple clustering algorithms:
- **K-means**: Fast, works well with spherical clusters
- **DBSCAN**: Handles arbitrary cluster shapes and noise
- **Similarity-based**: Simple threshold-based clustering

### Frequency Management
- **Automatic Decay**: Time-based frequency reduction
- **Dynamic Thresholds**: Configurable promotion/demotion criteria
- **Usage Tracking**: Detailed access pattern analysis

### Extensibility
- **Custom Extractors**: Implement domain-specific concept extraction
- **Alternative Embeddings**: Support for various embedding models
- **Pluggable Storage**: Interface for different vector databases

## 📈 Performance Metrics

### Typical Performance (1000 primary concepts)
- **Concept Storage**: < 100ms
- **Retrieval**: < 50ms  
- **Clustering**: < 5s (1000 concepts)
- **Layer Migration**: < 200ms

### Scalability
- **Primary Layer**: Optimized for sub-50ms retrieval
- **Secondary Layer**: Handles millions of concepts with clustering
- **Memory Usage**: Configurable based on layer capacities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details.

## 🆘 Support

- **Documentation**: [Full API Documentation](docs/)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

## 🎯 Roadmap

- [ ] Web UI for system monitoring
- [ ] Multi-tenant support
- [ ] Distributed clustering
- [ ] Advanced concept relationship mapping
- [ ] Integration with more vector databases
- [ ] Real-time concept drift detection

---

**Learnd** - Making AI systems smarter through adaptive continuous learning 🚀
