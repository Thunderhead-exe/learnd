# Learnd MCP Server - Deployment Guide

## 🚀 Deployment Options

### Option 1: Use Deployment-Ready Server (Recommended)

For deployment scenarios, use the standalone `deploy_server.py` which avoids relative import issues:

```bash
# Deploy using the standalone server
uv run fastmcp run deploy_server:mcp
```

### Option 2: Fix Package Structure for Regular Server

If you prefer using the regular `learnd.mcp_server`, ensure proper Python path setup:

```python
import sys
from pathlib import Path

# Add package root to Python path
package_root = Path(__file__).parent
sys.path.insert(0, str(package_root))

# Now import
from learnd.mcp_server import mcp
```

## 🔧 Common Deployment Issues

### Issue 1: "attempted relative import with no known parent package"

**Cause**: Python can't resolve relative imports when the module isn't part of a proper package structure.

**Solutions**:

1. **Use the deployment server** (easiest):
   ```bash
   uv run fastmcp run deploy_server:mcp
   ```

2. **Fix Python path** (if using regular server):
   ```python
   import sys
   import os
   sys.path.insert(0, os.path.dirname(__file__))
   ```

3. **Install as package**:
   ```bash
   pip install -e .
   # Then run
   uv run fastmcp run learnd.mcp_server:mcp
   ```

### Issue 2: Read-Only File System Error

**Symptoms**: `[Errno 30] Read-only file system` or cache-related errors in serverless environments.

**Cause**: Sentence-transformers and other ML libraries try to cache models to the home directory, which is read-only in many deployment environments.

**Solution**: Use the deployment server which automatically handles this:

```bash
# Use the deployment server (handles read-only filesystems)
uv run fastmcp run deploy_server:mcp
```

The deployment server includes:
- Automatic cache directory detection and fallback
- Environment variable configuration for transformers
- Logging optimized for serverless environments
- Health checks that report filesystem status

### Issue 3: Lambda/Serverless Deployment

For AWS Lambda or similar serverless deployments:

1. **Create deployment package**:
   ```bash
   # Create deployment directory
   mkdir deployment
   
   # Copy source files
   cp -r learnd/ deployment/
   cp deploy_server.py deployment/
   cp serverless_config.py deployment/
   cp env.template deployment/.env
   
   # Install dependencies
   cd deployment
   pip install -r ../requirements.txt -t .
   ```

2. **Use deployment server**:
   ```python
   # lambda_handler.py
   import sys
   import os
   
   # Add current directory to path
   sys.path.insert(0, os.path.dirname(__file__))
   
   from deploy_server import mcp
   
   def lambda_handler(event, context):
       # Your Lambda handler logic here
       return mcp.handle_request(event, context)
   ```

### Issue 3: Docker Deployment

Create a `Dockerfile` for containerized deployment:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY pyproject.toml .
RUN pip install uv && uv pip install .

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run deployment server
CMD ["uv", "run", "fastmcp", "run", "deploy_server:mcp"]
```

## 🔍 Debugging Deployment Issues

### Check Python Path

```python
import sys
print("Python path:")
for path in sys.path:
    print(f"  {path}")
```

### Verify Imports

```python
try:
    from learnd.models import LearndConfig
    print("✅ Absolute imports working")
except ImportError as e:
    print(f"❌ Absolute import failed: {e}")

try:
    from .models import LearndConfig
    print("✅ Relative imports working")
except ImportError as e:
    print(f"❌ Relative import failed: {e}")
```

### Test Deployment Server

```bash
# Test the deployment server locally
uv run python deploy_server.py

# Should output:
# 🧠 Learnd MCP Server - Deployment Version
# ✅ Server configuration loaded
# 📊 Available tools: X
# 🚀 Ready for deployment
```

## 📦 Environment Variables for Deployment

Ensure these environment variables are set in your deployment environment:

```bash
# Required
MISTRAL_API_KEY=your_mistral_api_key
QDRANT_URL=https://your-cluster.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key

# Optional (with defaults)
COLLECTION_NAME=learnd-concepts
PRIMARY_LAYER_CAPACITY=1000
PROMOTION_THRESHOLD=10
DEMOTION_THRESHOLD=2
EMBEDDING_MODEL=all-MiniLM-L6-v2
MISTRAL_MODEL=mistral-large-latest
```

## 🎯 Deployment Checklist

- [ ] Environment variables configured
- [ ] Qdrant Cloud cluster accessible
- [ ] Mistral API key valid and has quota
- [ ] Python path properly set
- [ ] Dependencies installed
- [ ] Using deployment server (`deploy_server.py`) or fixed imports
- [ ] Health check endpoint working (`health_check` tool)
- [ ] Logs configured for monitoring

## 🔧 Platform-Specific Instructions

### AWS Lambda

```python
# lambda_function.py
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from deploy_server import mcp

def lambda_handler(event, context):
    # Process MCP request
    result = mcp.process_request(event)
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

### Google Cloud Functions

```python
# main.py
import functions_framework
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from deploy_server import mcp

@functions_framework.http
def learnd_mcp(request):
    return mcp.handle_http_request(request)
```

### Azure Functions

```python
# __init__.py
import azure.functions as func
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from deploy_server import mcp

def main(req: func.HttpRequest) -> func.HttpResponse:
    return mcp.handle_azure_request(req)
```

## 📊 Monitoring Deployment

### Health Check

The deployment server includes a health check tool:

```bash
# Test health check
curl http://your-deployment-url/health_check
```

Expected response:
```json
{
  "status": "healthy",
  "service": "learnd-mcp",
  "version": "0.1.0"
}
```

### Logging

Monitor these log messages:
- `"Learnd core system initialized"` - System started successfully
- `"Failed to initialize"` - Startup error
- `"Failed to get relevant context"` - Runtime error

## 🆘 Troubleshooting

### Common Error Messages

1. **"attempted relative import with no known parent package"**
   - Use `deploy_server.py` instead of `learnd.mcp_server`

2. **"[Errno 30] Read-only file system" / cache errors**
   - Use `deploy_server.py` which handles read-only filesystems
   - Check health_check endpoint for filesystem status
   - Environment should show `filesystem_readonly: true` in health check

3. **"ModuleNotFoundError: No module named 'learnd'"**
   - Add package to Python path or install with `pip install -e .`

4. **"Failed to initialize Qdrant"**
   - Check `QDRANT_URL` and `QDRANT_API_KEY` environment variables

5. **"Mistral API key not provided"**
   - Set `MISTRAL_API_KEY` environment variable

6. **Sentence-transformers cache errors**
   - The deployment server automatically handles this
   - Models will use temp directory or fallback to minimal caching

### Get Support

- Check logs for specific error messages
- Verify environment variables are set correctly
- Test with the deployment server locally first
- Ensure all dependencies are included in deployment package

---

✅ **Your Learnd MCP server is now ready for production deployment!**
