#!/usr/bin/env python3
"""
Deployment Diagnostics for Learnd MCP Server

Run this script to diagnose common deployment issues before starting the server.
"""

import os
import sys
import tempfile
import json
from pathlib import Path


def run_deployment_diagnostics():
    """Run comprehensive deployment diagnostics."""
    print("🔍 Learnd MCP Server - Deployment Diagnostics")
    print("=" * 50)
    
    results = {
        "filesystem": check_filesystem(),
        "environment": check_environment_variables(),
        "python": check_python_environment(),
        "imports": check_imports(),
        "cache": check_cache_setup()
    }
    
    # Summary
    print("\n📊 SUMMARY")
    print("=" * 20)
    
    issues = []
    warnings = []
    
    for category, result in results.items():
        if result["status"] == "error":
            issues.append(f"{category}: {result['message']}")
        elif result["status"] == "warning":
            warnings.append(f"{category}: {result['message']}")
        else:
            print(f"✅ {category.title()}: {result['message']}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   {warning}")
    
    if issues:
        print("\n❌ ISSUES:")
        for issue in issues:
            print(f"   {issue}")
        print("\nRecommendation: Use deploy_server.py for production deployment")
    else:
        print("\n🎉 All checks passed! Your deployment should work correctly.")
    
    return len(issues) == 0


def check_filesystem():
    """Check filesystem permissions."""
    try:
        # Check if current directory is writable
        test_file = "test_write_permission"
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        
        # Check temp directory
        temp_dir = tempfile.gettempdir()
        temp_test = os.path.join(temp_dir, "test_temp_write")
        with open(temp_test, 'w') as f:
            f.write("test")
        os.remove(temp_test)
        
        return {
            "status": "ok",
            "message": "Filesystem is writable",
            "details": {
                "current_dir_writable": True,
                "temp_dir_writable": True,
                "temp_dir": temp_dir
            }
        }
        
    except (OSError, PermissionError) as e:
        # Try just temp directory
        try:
            temp_dir = tempfile.gettempdir()
            temp_test = os.path.join(temp_dir, "test_temp_write")
            with open(temp_test, 'w') as f:
                f.write("test")
            os.remove(temp_test)
            
            return {
                "status": "warning",
                "message": "Current directory read-only, but temp writable",
                "details": {
                    "current_dir_writable": False,
                    "temp_dir_writable": True,
                    "temp_dir": temp_dir
                }
            }
        except:
            return {
                "status": "error",
                "message": "Read-only filesystem detected",
                "details": {
                    "current_dir_writable": False,
                    "temp_dir_writable": False,
                    "recommendation": "Use deploy_server.py"
                }
            }


def check_environment_variables():
    """Check required environment variables."""
    required_vars = {
        'MISTRAL_API_KEY': 'Mistral AI API key',
        'QDRANT_URL': 'Qdrant Cloud URL',
        'QDRANT_API_KEY': 'Qdrant API key'
    }
    
    missing = []
    present = []
    
    for var, description in required_vars.items():
        if os.getenv(var):
            present.append(f"{var} ({description})")
        else:
            missing.append(f"{var} ({description})")
    
    if missing:
        return {
            "status": "error",
            "message": f"Missing {len(missing)} required environment variables",
            "details": {
                "missing": missing,
                "present": present
            }
        }
    else:
        return {
            "status": "ok",
            "message": "All required environment variables present",
            "details": {"present": present}
        }


def check_python_environment():
    """Check Python environment and path."""
    python_info = {
        "version": sys.version,
        "executable": sys.executable,
        "path": sys.path[:3]  # Show first 3 paths
    }
    
    # Check if package is in path
    current_dir = str(Path(__file__).parent)
    in_path = current_dir in sys.path
    
    return {
        "status": "ok" if in_path else "warning",
        "message": f"Python {sys.version_info.major}.{sys.version_info.minor}, package {'in' if in_path else 'not in'} path",
        "details": python_info
    }


def check_imports():
    """Check if key imports work."""
    import_tests = [
        ("fastmcp", "FastMCP framework"),
        ("qdrant_client", "Qdrant database client"),
        ("sentence_transformers", "Sentence transformers for embeddings"),
        ("mistralai", "Mistral AI client"),
        ("dotenv", "Environment variable loading")
    ]
    
    failed_imports = []
    success_imports = []
    
    for module, description in import_tests:
        try:
            __import__(module)
            success_imports.append(f"{module} ({description})")
        except ImportError:
            failed_imports.append(f"{module} ({description})")
    
    if failed_imports:
        return {
            "status": "error",
            "message": f"Failed to import {len(failed_imports)} required modules",
            "details": {
                "failed": failed_imports,
                "success": success_imports
            }
        }
    else:
        return {
            "status": "ok",
            "message": "All required modules can be imported",
            "details": {"success": success_imports}
        }


def check_cache_setup():
    """Check cache directory setup."""
    cache_vars = [
        'TRANSFORMERS_CACHE',
        'SENTENCE_TRANSFORMERS_HOME',
        'HF_HOME'
    ]
    
    cache_info = {}
    for var in cache_vars:
        value = os.environ.get(var)
        if value:
            writable = os.path.exists(value) and os.access(value, os.W_OK) if value != '/dev/null' else False
            cache_info[var] = {"value": value, "writable": writable}
    
    if not cache_info:
        return {
            "status": "warning",
            "message": "No cache directories configured",
            "details": {"recommendation": "Use deploy_server.py for automatic cache setup"}
        }
    else:
        return {
            "status": "ok",
            "message": f"Cache configured ({len(cache_info)} directories)",
            "details": cache_info
        }


if __name__ == "__main__":
    success = run_deployment_diagnostics()
    sys.exit(0 if success else 1)
