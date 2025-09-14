"""
Concept extraction using Mistral LLM for the Learnd system.
"""

import asyncio
import json
from typing import List, Optional, Dict, Any
from loguru import logger

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    logger.warning("Mistral not available for concept extraction")

try:
    from .models import LearndConfig
except ImportError:
    from learnd.models import LearndConfig


class ConceptExtractor:
    """Extracts concepts from text using Mistral LLM."""
    
    def __init__(self, config: LearndConfig):
        self.config = config
        self.client: Optional[Mistral] = None
        
    async def initialize(self) -> None:
        """Initialize the Mistral client for concept extraction."""
        try:
            if not MISTRAL_AVAILABLE:
                raise ValueError("Mistral library not available")
                
            if not self.config.mistral_api_key:
                raise ValueError("Mistral API key not provided")
                
            self.client = Mistral(api_key=self.config.mistral_api_key)
            logger.info("Concept extractor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize concept extractor: {e}")
            raise
    
    async def extract_concepts(self, text: str, context: Optional[str] = None) -> List[str]:
        """Extract concepts from input text."""
        try:
            if not self.client:
                raise ValueError("Concept extractor not initialized")
            
            # Build the prompt for concept extraction
            prompt = self._build_extraction_prompt(text, context)
            
            response = await self.client.chat.complete_async(
                model=self.config.concept_extraction_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting key concepts from text. Extract the most important concepts, ideas, and entities mentioned. Return only a JSON list of concept strings."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse the response
            concepts = self._parse_extraction_response(response.choices[0].message.content)
            
            # Limit number of concepts
            if len(concepts) > self.config.max_concepts_per_input:
                concepts = concepts[:self.config.max_concepts_per_input]
            
            logger.debug(f"Extracted {len(concepts)} concepts from text")
            return concepts
            
        except Exception as e:
            logger.error(f"Failed to extract concepts: {e}")
            return []
    
    def _build_extraction_prompt(self, text: str, context: Optional[str] = None) -> str:
        """Build the prompt for concept extraction."""
        prompt = f"""Extract the key concepts, ideas, entities, and important terms from the following text.

Text to analyze:
{text}
"""
        
        if context:
            prompt += f"""
Additional context:
{context}
"""
        
        prompt += """
Requirements:
1. Extract only the most important and meaningful concepts
2. Include entities, technical terms, key ideas, and themes
3. Keep concepts concise (1-5 words each)
4. Avoid common words unless they're domain-specific
5. Return as a JSON list of strings
6. Maximum 10 concepts

Example format: ["machine learning", "neural networks", "data preprocessing", "model evaluation"]

Concepts:"""
        
        return prompt
    
    def _parse_extraction_response(self, response_text: str) -> List[str]:
        """Parse the LLM response to extract concepts."""
        try:
            # Try to parse as JSON first
            try:
                concepts = json.loads(response_text.strip())
                if isinstance(concepts, list):
                    return [str(concept).strip() for concept in concepts if concept.strip()]
            except json.JSONDecodeError:
                pass
            
            # Fallback: extract from text patterns
            concepts = []
            lines = response_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Handle numbered lists
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                    concept = line.split('.', 1)[1].strip()
                    concepts.append(concept)
                
                # Handle bullet points
                elif line.startswith(('-', '*', '•')):
                    concept = line[1:].strip()
                    concepts.append(concept)
                
                # Handle quoted strings
                elif '"' in line:
                    import re
                    quoted = re.findall(r'"([^"]*)"', line)
                    concepts.extend(quoted)
            
            # Clean and filter concepts
            cleaned_concepts = []
            for concept in concepts:
                concept = concept.strip(' "\',')
                if concept and len(concept) > 2:  # Filter very short concepts
                    cleaned_concepts.append(concept)
            
            return cleaned_concepts[:self.config.max_concepts_per_input]
            
        except Exception as e:
            logger.error(f"Failed to parse extraction response: {e}")
            return []
    
    async def validate_concept(self, concept: str, original_text: str) -> bool:
        """Validate if a concept is relevant to the original text."""
        try:
            if not self.client:
                return True  # Default to valid if extractor not available
            
            prompt = f"""Is the concept "{concept}" relevant and accurately extracted from this text?

Text: {original_text}

Respond with only "YES" or "NO"."""
            
            response = await self.client.chat.complete_async(
                model=self.config.concept_extraction_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are validating concept extraction. Respond only with YES or NO."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().upper()
            return result == "YES"
            
        except Exception as e:
            logger.error(f"Failed to validate concept: {e}")
            return True  # Default to valid on error
    
    async def extract_and_validate(self, text: str, context: Optional[str] = None) -> List[str]:
        """Extract concepts and validate their relevance."""
        try:
            # Extract concepts
            concepts = await self.extract_concepts(text, context)
            
            # Validate concepts (in parallel for efficiency)
            validation_tasks = [
                self.validate_concept(concept, text) 
                for concept in concepts
            ]
            
            validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Filter valid concepts
            valid_concepts = []
            for concept, is_valid in zip(concepts, validation_results):
                if isinstance(is_valid, bool) and is_valid:
                    valid_concepts.append(concept)
                elif not isinstance(is_valid, bool):
                    # If validation failed, include the concept anyway
                    valid_concepts.append(concept)
            
            logger.info(f"Validated {len(valid_concepts)}/{len(concepts)} concepts")
            return valid_concepts
            
        except Exception as e:
            logger.error(f"Failed to extract and validate concepts: {e}")
            return []
