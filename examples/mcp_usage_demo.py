#!/usr/bin/env python3
"""
Learnd MCP Server Usage Demonstration

This example shows how to use the Learnd MCP server for adaptive continuous 
learning in a real-world scenario. It simulates an AI assistant that learns 
from user interactions and improves over time.

Scenario: Educational AI Assistant
- Students ask questions about various topics
- The AI learns concepts from each interaction
- Knowledge accumulates and improves response quality
- System adapts to frequent topics and optimizes performance
"""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime
import random

# Simulated MCP client for demonstration
class MockMCPClient:
    """
    Mock MCP client that simulates calling the Learnd MCP server tools.
    In real usage, you would use an actual MCP client library.
    """
    
    def __init__(self):
        self.learned_concepts = set()
        self.interaction_count = 0
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate MCP tool calls with realistic responses."""
        
        if tool_name == "learn_from_interaction":
            # Simulate learning from interaction
            user_input = params.get("user_input", "")
            concepts = self._extract_mock_concepts(user_input)
            self.learned_concepts.update(concepts)
            self.interaction_count += 1
            
            return {
                "success": True,
                "concepts_learned": concepts,
                "concepts_count": len(concepts),
                "learning_summary": f"Learned {len(concepts)} concepts from interaction"
            }
        
        elif tool_name == "get_relevant_context":
            # Simulate retrieving relevant context
            query = params.get("query", "")
            relevant = self._find_relevant_concepts(query)
            
            formatted_context = ""
            if relevant:
                formatted_context = f"Relevant learned concepts: {', '.join(relevant)}"
            
            return {
                "relevant_concepts": [
                    {
                        "text": concept,
                        "frequency": random.randint(1, 20),
                        "similarity_score": random.uniform(0.7, 0.95)
                    } for concept in relevant
                ],
                "formatted_context": formatted_context,
                "total_found": len(relevant)
            }
        
        elif tool_name == "get_system_stats":
            # Simulate system statistics
            return {
                "layers": {
                    "primary_size": min(50, len(self.learned_concepts)),
                    "secondary_size": max(0, len(self.learned_concepts) - 50),
                    "total_concepts": len(self.learned_concepts),
                    "utilization_percent": min(100, (len(self.learned_concepts) / 100) * 100)
                },
                "most_frequent_concept": {
                    "text": "machine learning" if "machine learning" in self.learned_concepts else None,
                    "frequency": random.randint(5, 25)
                } if self.learned_concepts else None,
                "system_healthy": True
            }
        
        elif tool_name == "search_concepts":
            # Simulate concept search
            query = params.get("query", "")
            matches = self._find_relevant_concepts(query)
            
            return {
                "results": [
                    {
                        "concept": {
                            "text": concept,
                            "frequency": random.randint(1, 15),
                            "layer": 1 if random.random() > 0.7 else 2
                        },
                        "similarity_score": random.uniform(0.6, 0.9),
                        "rank": i + 1
                    } for i, concept in enumerate(matches[:5])
                ],
                "total_results": len(matches)
            }
        
        elif tool_name == "rebalance_knowledge":
            # Simulate rebalancing
            return {
                "rebalance_completed": True,
                "concepts_promoted": random.randint(0, 5),
                "concepts_demoted": random.randint(0, 8),
                "primary_utilization": min(100, (len(self.learned_concepts) / 100) * 100)
            }
        
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    def _extract_mock_concepts(self, text: str) -> List[str]:
        """Extract concepts from text (simplified simulation)."""
        concept_keywords = {
            "machine learning": ["machine learning", "ML", "algorithm", "model", "training"],
            "neural networks": ["neural", "network", "deep learning", "neuron", "layer"],
            "python programming": ["python", "programming", "code", "function", "variable"],
            "data science": ["data", "analysis", "statistics", "visualization", "pandas"],
            "web development": ["web", "HTML", "CSS", "JavaScript", "frontend", "backend"],
            "databases": ["database", "SQL", "query", "table", "data storage"],
            "artificial intelligence": ["AI", "artificial intelligence", "intelligent", "automation"],
            "software engineering": ["software", "engineering", "development", "architecture"],
            "cloud computing": ["cloud", "AWS", "Azure", "deployment", "scalability"],
            "cybersecurity": ["security", "encryption", "authentication", "vulnerability"]
        }
        
        text_lower = text.lower()
        found_concepts = []
        
        for concept, keywords in concept_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_concepts.append(concept)
        
        return found_concepts[:5]  # Limit to 5 concepts per interaction
    
    def _find_relevant_concepts(self, query: str) -> List[str]:
        """Find relevant concepts from learned set."""
        query_lower = query.lower()
        relevant = []
        
        for concept in self.learned_concepts:
            if any(word in concept.lower() for word in query_lower.split()):
                relevant.append(concept)
        
        return relevant[:10]  # Limit results


class EducationalAssistant:
    """
    Educational AI Assistant that demonstrates adaptive learning with MCP.
    """
    
    def __init__(self):
        self.mcp_client = MockMCPClient()
        self.conversation_history = []
    
    async def handle_student_question(self, question: str, student_id: str = "student_001") -> Dict[str, Any]:
        """
        Handle a student question with adaptive learning.
        
        This demonstrates the complete flow:
        1. Get relevant context from previous learning
        2. Generate response (simulated)
        3. Learn from the interaction
        4. Return enhanced response
        """
        
        print(f"\n📚 Student Question: {question}")
        
        # Step 1: Get relevant context from learned concepts
        context_result = await self.mcp_client.call_tool(
            "get_relevant_context",
            {
                "query": question,
                "max_concepts": 5,
                "similarity_threshold": 0.7
            }
        )
        
        relevant_context = context_result.get("formatted_context", "")
        print(f"🧠 Retrieved Context: {relevant_context or 'No relevant context found'}")
        
        # Step 2: Generate response (simulated - in real usage, call your main LLM)
        base_response = self._generate_educational_response(question)
        
        # Enhance response with learned context
        if relevant_context:
            enhanced_response = f"{base_response}\n\n💡 Based on previous discussions: {relevant_context}"
        else:
            enhanced_response = base_response
        
        print(f"🤖 AI Response: {enhanced_response}")
        
        # Step 3: Learn from this interaction
        learning_result = await self.mcp_client.call_tool(
            "learn_from_interaction",
            {
                "user_input": question,
                "llm_response": enhanced_response,
                "feedback_score": random.uniform(0.7, 0.95),  # Simulated positive feedback
                "context": f"Educational question from {student_id}"
            }
        )
        
        concepts_learned = learning_result.get("concepts_learned", [])
        if concepts_learned:
            print(f"📖 Learned: {', '.join(concepts_learned)}")
        
        # Store interaction
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "student_id": student_id,
            "question": question,
            "response": enhanced_response,
            "concepts_learned": concepts_learned,
            "context_used": bool(relevant_context)
        }
        self.conversation_history.append(interaction)
        
        return interaction
    
    def _generate_educational_response(self, question: str) -> str:
        """Generate educational response (simplified simulation)."""
        question_lower = question.lower()
        
        if "machine learning" in question_lower or "ml" in question_lower:
            return "Machine learning is a subset of AI that enables computers to learn from data without explicit programming. It involves algorithms that improve automatically through experience."
        
        elif "neural network" in question_lower or "deep learning" in question_lower:
            return "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information in layers."
        
        elif "python" in question_lower and "programming" in question_lower:
            return "Python is a versatile programming language known for its simplicity and readability. It's widely used in data science, web development, and automation."
        
        elif "data science" in question_lower:
            return "Data science combines statistics, programming, and domain expertise to extract insights from data. It involves data collection, cleaning, analysis, and visualization."
        
        elif "web development" in question_lower:
            return "Web development involves creating websites and web applications. It includes frontend (user interface) and backend (server-side) development."
        
        else:
            return "That's an interesting question! Let me help you understand this topic better. Could you provide more specific details about what you'd like to learn?"
    
    async def demonstrate_system_growth(self) -> None:
        """Demonstrate how the system grows and adapts over time."""
        
        print("\n" + "="*60)
        print("🎓 EDUCATIONAL AI ASSISTANT - ADAPTIVE LEARNING DEMO")
        print("="*60)
        
        # Simulate a series of student interactions
        student_questions = [
            ("What is machine learning?", "student_001"),
            ("How do neural networks work?", "student_002"),
            ("Can you explain Python programming?", "student_001"),
            ("What's the difference between supervised and unsupervised learning?", "student_003"),
            ("How do I start with data science?", "student_002"),
            ("What are the applications of machine learning?", "student_001"),
            ("How do I build a neural network in Python?", "student_004"),
            ("What tools are used in data science?", "student_003"),
        ]
        
        print(f"📋 Processing {len(student_questions)} student interactions...")
        
        for i, (question, student_id) in enumerate(student_questions, 1):
            print(f"\n--- Interaction {i} ---")
            await self.handle_student_question(question, student_id)
            
            # Show system stats every few interactions
            if i % 3 == 0:
                await self.show_system_stats()
            
            # Brief pause for readability
            await asyncio.sleep(0.5)
        
        print(f"\n🎉 Completed {len(student_questions)} interactions!")
        
        # Final system analysis
        await self.analyze_learning_progress()
    
    async def show_system_stats(self) -> None:
        """Display current system statistics."""
        stats = await self.mcp_client.call_tool("get_system_stats", {})
        
        print(f"\n📊 System Statistics:")
        print(f"   Total Concepts: {stats['layers']['total_concepts']}")
        print(f"   Primary Layer: {stats['layers']['primary_size']}")
        print(f"   Secondary Layer: {stats['layers']['secondary_size']}")
        print(f"   Utilization: {stats['layers']['utilization_percent']:.1f}%")
        
        if stats.get('most_frequent_concept'):
            concept = stats['most_frequent_concept']
            print(f"   Most Frequent: '{concept['text']}' (freq: {concept['frequency']})")
    
    async def analyze_learning_progress(self) -> None:
        """Analyze the learning progress and demonstrate system capabilities."""
        
        print("\n" + "="*60)
        print("📈 LEARNING PROGRESS ANALYSIS")
        print("="*60)
        
        # Show final stats
        await self.show_system_stats()
        
        # Demonstrate concept search
        print(f"\n🔍 Concept Search Examples:")
        
        search_queries = ["machine learning", "python", "data"]
        for query in search_queries:
            result = await self.mcp_client.call_tool(
                "search_concepts",
                {"query": query, "limit": 3}
            )
            
            print(f"\n   Query: '{query}'")
            for r in result.get("results", [])[:3]:
                concept = r["concept"]
                print(f"   • {concept['text']} (similarity: {r['similarity_score']:.2f}, freq: {concept['frequency']})")
        
        # Demonstrate system optimization
        print(f"\n⚖️ Performing Knowledge Rebalancing:")
        rebalance_result = await self.mcp_client.call_tool("rebalance_knowledge", {})
        
        if rebalance_result.get("rebalance_completed"):
            print(f"   ✅ Rebalancing completed")
            print(f"   📈 Concepts promoted: {rebalance_result['concepts_promoted']}")
            print(f"   📉 Concepts demoted: {rebalance_result['concepts_demoted']}")
            print(f"   🎯 Primary utilization: {rebalance_result['primary_utilization']:.1f}%")
        
        # Show conversation summary
        print(f"\n💬 Conversation Summary:")
        print(f"   Total interactions: {len(self.conversation_history)}")
        
        students = set(conv['student_id'] for conv in self.conversation_history)
        print(f"   Unique students: {len(students)}")
        
        context_used = sum(1 for conv in self.conversation_history if conv['context_used'])
        print(f"   Interactions with context: {context_used}/{len(self.conversation_history)}")
        
        all_concepts = set()
        for conv in self.conversation_history:
            all_concepts.update(conv['concepts_learned'])
        print(f"   Unique concepts learned: {len(all_concepts)}")
        
        print(f"\n🎯 Key Benefits Demonstrated:")
        print(f"   ✅ Automatic concept extraction from student questions")
        print(f"   ✅ Context-aware response enhancement")
        print(f"   ✅ Continuous learning and knowledge accumulation")
        print(f"   ✅ Adaptive system optimization")
        print(f"   ✅ Improved responses over time")


async def demonstrate_mcp_integration_patterns():
    """
    Demonstrate different MCP integration patterns for various use cases.
    """
    
    print("\n" + "="*60)
    print("🔧 MCP INTEGRATION PATTERNS")
    print("="*60)
    
    mcp_client = MockMCPClient()
    
    print("\n1️⃣ PATTERN: Real-time Learning During Conversation")
    print("-" * 50)
    print("Use Case: Chat applications, customer support")
    print("Implementation:")
    print("```python")
    print("# After each user interaction")
    print("await mcp_client.call_tool('learn_from_interaction', {")
    print("    'user_input': user_message,")
    print("    'llm_response': ai_response,")
    print("    'feedback_score': user_feedback")
    print("})")
    print("```")
    
    # Demonstrate
    result = await mcp_client.call_tool("learn_from_interaction", {
        "user_input": "How do I optimize database queries?",
        "feedback_score": 0.9
    })
    print(f"Result: {result['learning_summary']}")
    
    print("\n2️⃣ PATTERN: Context Augmentation for Enhanced Responses")
    print("-" * 50)
    print("Use Case: Knowledge-based assistants, domain experts")
    print("Implementation:")
    print("```python")
    print("# Before generating LLM response")
    print("context = await mcp_client.call_tool('get_relevant_context', {")
    print("    'query': user_query,")
    print("    'max_concepts': 5")
    print("})")
    print("enhanced_prompt = f\"{user_query}\\n\\nContext: {context['formatted_context']}\"")
    print("```")
    
    # Demonstrate
    context_result = await mcp_client.call_tool("get_relevant_context", {
        "query": "database optimization techniques"
    })
    print(f"Context retrieved: {context_result['total_found']} relevant concepts")
    
    print("\n3️⃣ PATTERN: Batch Learning from Documents")
    print("-" * 50)
    print("Use Case: Document processing, knowledge base building")
    print("Implementation:")
    print("```python")
    print("# Process multiple documents")
    print("for document in documents:")
    print("    await mcp_client.call_tool('extract_concepts', {")
    print("        'text': document.content,")
    print("        'context': document.metadata,")
    print("        'auto_store': True")
    print("    })")
    print("```")
    
    print("\n4️⃣ PATTERN: System Monitoring and Optimization")
    print("-" * 50)
    print("Use Case: Production systems, performance monitoring")
    print("Implementation:")
    print("```python")
    print("# Regular system health checks")
    print("stats = await mcp_client.call_tool('get_system_stats')")
    print("if stats['layers']['utilization_percent'] > 90:")
    print("    await mcp_client.call_tool('rebalance_knowledge')")
    print("```")
    
    # Demonstrate
    stats = await mcp_client.call_tool("get_system_stats", {})
    print(f"System health: {stats['layers']['total_concepts']} concepts, {stats['layers']['utilization_percent']:.1f}% utilization")


async def main():
    """Run the complete MCP usage demonstration."""
    
    print("🧠 Learnd MCP Server - Usage Demonstration")
    print("🚀 Adaptive Continuous Learning in Action")
    print("=" * 60)
    
    # Create educational assistant
    assistant = EducationalAssistant()
    
    # Demonstrate the complete learning cycle
    await assistant.demonstrate_system_growth()
    
    # Show different integration patterns
    await demonstrate_mcp_integration_patterns()
    
    print("\n" + "="*60)
    print("✨ DEMONSTRATION COMPLETE")
    print("="*60)
    print("\n🎯 What you've seen:")
    print("   • Automatic concept extraction from user inputs")
    print("   • Context-aware response enhancement")
    print("   • Continuous learning and knowledge accumulation") 
    print("   • Adaptive system optimization and rebalancing")
    print("   • Multiple MCP integration patterns")
    
    print("\n🚀 Ready to integrate Learnd MCP Server?")
    print("   1. Start the server: uv run python main.py")
    print("   2. Connect your MCP client")
    print("   3. Begin learning with: learn_from_interaction")
    print("   4. Enhance responses with: get_relevant_context")
    print("   5. Monitor system with: get_system_stats")
    
    print("\n📚 Available MCP Tools:")
    tools = [
        "learn_from_interaction - Primary learning from user interactions",
        "get_relevant_context - Context augmentation for LLM responses", 
        "extract_concepts - Manual concept extraction",
        "search_concepts - Knowledge exploration and discovery",
        "get_system_stats - System monitoring and health",
        "rebalance_knowledge - Performance optimization",
        "cleanup_old_concepts - Maintenance and storage management"
    ]
    
    for tool in tools:
        print(f"   • {tool}")


if __name__ == "__main__":
    asyncio.run(main())
