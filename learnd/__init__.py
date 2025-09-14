"""
Learnd: Adaptive Continuous Learning MCP Server

A Model Context Protocol server that implements adaptive continuous learning
with hierarchical vector storage and frequency-based concept management.

Main Components:
- LearndMCPServer: Primary MCP server with all learning tools
- LearndCore: Core learning engine and concept management
- Models: Data structures for concepts, clusters, and configuration

Usage:
    from learnd import LearndMCPServer
    
    server = LearndMCPServer()
    await server.initialize()
    await server.run(port=8000)
"""

__version__ = "0.1.0"
__author__ = "Learnd Team"

from .models import Concept, Cluster, FrequencyTracker, LearndConfig
from .mcp_server import mcp
from .core import LearndCore

__all__ = [
    "Concept",
    "Cluster", 
    "FrequencyTracker",
    "LearndConfig",
    "mcp",
    "LearndCore"
]
