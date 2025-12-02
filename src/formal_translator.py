"""Generate formal translations from gen-z examples using slang descriptions."""

import re
from typing import Dict, List, Optional


class FormalTranslator:
    """Translates gen-z slang examples to formal language."""
    
    def __init__(self, slang_dict: Optional[Dict[str, Dict]] = None):
        """
        Initialize translator with slang dictionary.
        
        Args:
            slang_dict: Dictionary mapping slang terms to their descriptions and context
        """
        self.slang_dict = slang_dict or {}
    
    def update_slang_dict(self, slang_dict: Dict[str, Dict]):
        """Update the slang dictionary."""
        self.slang_dict = slang_dict
    
    def translate_example(self, example: str, slang_term: str, description: str, context: str) -> str:
        """
        Translate a gen-z example to formal language.
        
        Args:
            example: The gen-z example text
            slang_term: The slang term being used
            description: Description of what the slang means
            context: Context for when/how it's used
            
        Returns:
            Formal translation of the example
        """
        if not example or not example.strip():
            return ""
        
        # Start with the example
        formal_text = example
        
        # Replace the slang term with its description
        # Handle case-insensitive replacement, but preserve word boundaries
        # First, try exact match with word boundaries
        pattern = re.compile(r'\b' + re.escape(slang_term) + r'\b', re.IGNORECASE)
        
        # Try to replace in context
        # If description is short, use it directly; otherwise, paraphrase
        if len(description) < 50:
            replacement = description
        else:
            # Extract key meaning from description
            replacement = self._extract_key_meaning(description)
        
        # Replace slang term
        formal_text = pattern.sub(replacement, formal_text)
        
        # If no replacement happened, try without word boundaries (for cases like "L+ratio")
        if formal_text == example:
            pattern2 = re.compile(re.escape(slang_term), re.IGNORECASE)
            formal_text = pattern2.sub(replacement, formal_text)
        
        # Clean up common gen-z patterns
        formal_text = self._clean_genz_patterns(formal_text)
        
        # Ensure proper capitalization and punctuation
        formal_text = self._formalize_text(formal_text)
        
        return formal_text
    
    def _extract_key_meaning(self, description: str) -> str:
        """Extract key meaning from a longer description."""
        # Remove common prefixes
        description = description.strip()
        
        # If it starts with common patterns, extract the core meaning
        if description.lower().startswith('shorthand for'):
            return description.replace('Shorthand for', '').replace('shorthand for', '').strip().split(',')[0]
        elif description.lower().startswith('another way'):
            return description.split('.')[0].replace('Another way of saying', '').replace('another way of saying', '').strip()
        elif ',' in description:
            # Take the first part before comma
            return description.split(',')[0].strip()
        else:
            # Take first sentence
            return description.split('.')[0].strip()
    
    def _clean_genz_patterns(self, text: str) -> str:
        """Clean up common gen-z text patterns."""
        # Remove excessive punctuation
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        
        # Fix common abbreviations in context
        replacements = {
            r'\bfr\b': 'for real',
            r'\bno cap\b': 'no lie',
            r'\bngl\b': 'not gonna lie',
            r'\btbh\b': 'to be honest',
            r'\bimo\b': 'in my opinion',
            r'\bimho\b': 'in my humble opinion',
            r'\birl\b': 'in real life',
            r'\bomw\b': 'on my way',
            r'\brn\b': 'right now',
            r'\bwyd\b': 'what are you doing',
            r'\bwyd\b': 'what you doing',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _formalize_text(self, text: str) -> str:
        """Make text more formal."""
        # Ensure proper sentence capitalization
        sentences = re.split(r'([.!?]\s+)', text)
        formalized = []
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                # Capitalize first letter
                if sentence[0].isalpha():
                    sentence = sentence[0].upper() + sentence[1:]
                formalized.append(sentence)
            else:
                formalized.append(sentence)
        
        result = ''.join(formalized)
        
        # Fix spacing
        result = re.sub(r'\s+', ' ', result)
        result = result.strip()
        
        return result
    
    def batch_translate(self, examples: List[Dict]) -> List[Dict]:
        """
        Translate multiple examples.
        
        Args:
            examples: List of dicts with 'example', 'slang', 'description', 'context'
            
        Returns:
            List of dicts with added 'formal_translation' field
        """
        results = []
        for ex in examples:
            formal = self.translate_example(
                ex.get('example', ''),
                ex.get('slang', ''),
                ex.get('description', ''),
                ex.get('context', '')
            )
            ex['formal_translation'] = formal
            results.append(ex)
        return results

