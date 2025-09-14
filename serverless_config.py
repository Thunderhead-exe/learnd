#!/usr/bin/env python3
"""
Serverless Configuration for Learnd MCP Server

This module configures the environment for serverless/container deployments
where the file system might be read-only.
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path


def configure_for_serverless():
    """Configure environment for serverless deployment."""
    
    # 1. Configure transformers/sentence-transformers for read-only filesystems
    configure_transformers_cache()
    
    # 2. Configure logging for serverless
    configure_logging()
    
    # 3. Suppress warnings
    configure_warnings()
    
    # 4. Set Python path
    configure_python_path()
    
    print("✅ Serverless environment configured")


def configure_transformers_cache():
    """Configure transformers cache for read-only filesystems."""
    
    # Prevent automatic model downloads
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    
    # Try to set writable cache directories
    try:
        # Check if we can write to temp
        tmp_dir = tempfile.gettempdir()
        test_file = os.path.join(tmp_dir, 'test_write')
        
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        
        # If successful, use temp directory
        os.environ['TRANSFORMERS_CACHE'] = tmp_dir
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = tmp_dir
        os.environ['HF_HOME'] = tmp_dir
        print(f"📁 Using cache directory: {tmp_dir}")
        
    except (OSError, PermissionError):
        # If temp is not writable, disable caching
        os.environ['TRANSFORMERS_CACHE'] = '/dev/null'
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/dev/null'
        os.environ['HF_HOME'] = '/dev/null'
        print("⚠️  Cache disabled due to read-only filesystem")


def configure_logging():
    """Configure logging for serverless environments."""
    from loguru import logger
    
    # Remove default file handler
    logger.remove()
    
    # Add stderr handler for cloud logs
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="INFO",
        serialize=False,
        backtrace=True,
        diagnose=True
    )
    
    # Add structured logging for cloud monitoring
    logger.add(
        sys.stdout,
        format="{time} | {level} | {message}",
        level="ERROR",
        filter=lambda record: record["level"].name == "ERROR"
    )


def configure_warnings():
    """Suppress warnings in production."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Specific warnings to suppress
    warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
    warnings.filterwarnings("ignore", message=".*torch.distributed.*")
    warnings.filterwarnings("ignore", message=".*transformers.*")


def configure_python_path():
    """Configure Python path for package imports."""
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))


def get_deployment_config():
    """Get deployment-specific configuration."""
    return {
        'read_only_filesystem': check_filesystem_readonly(),
        'temp_dir_writable': check_temp_writable(),
        'cache_available': check_cache_available(),
        'python_path': sys.path,
        'environment_vars': {
            'TRANSFORMERS_CACHE': os.environ.get('TRANSFORMERS_CACHE'),
            'SENTENCE_TRANSFORMERS_HOME': os.environ.get('SENTENCE_TRANSFORMERS_HOME'),
            'HF_HOME': os.environ.get('HF_HOME')
        }
    }


def check_filesystem_readonly():
    """Check if the filesystem is read-only."""
    try:
        test_file = 'test_write_permission'
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        return False
    except (OSError, PermissionError):
        return True


def check_temp_writable():
    """Check if temp directory is writable."""
    try:
        tmp_dir = tempfile.gettempdir()
        test_file = os.path.join(tmp_dir, 'test_temp_write')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        return True
    except (OSError, PermissionError):
        return False


def check_cache_available():
    """Check if model cache is available."""
    cache_dir = os.environ.get('TRANSFORMERS_CACHE')
    if not cache_dir or cache_dir == '/dev/null':
        return False
    
    try:
        return os.path.exists(cache_dir) and os.access(cache_dir, os.W_OK)
    except:
        return False


if __name__ == "__main__":
    # Configure and show deployment info
    configure_for_serverless()
    
    config = get_deployment_config()
    print("\n📊 Deployment Configuration:")
    print(f"  Read-only filesystem: {config['read_only_filesystem']}")
    print(f"  Temp dir writable: {config['temp_dir_writable']}")
    print(f"  Cache available: {config['cache_available']}")
    print(f"  Cache location: {config['environment_vars']['TRANSFORMERS_CACHE']}")
    
    if config['read_only_filesystem']:
        print("\n⚠️  Read-only filesystem detected!")
        print("   - Model caching disabled")
        print("   - Using fallback configurations")
        print("   - Logging to stderr/stdout only")
    else:
        print("\n✅ Writable filesystem available")
        print("   - Full caching enabled")
        print("   - Standard configurations")
