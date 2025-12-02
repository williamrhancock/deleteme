"""Gender detection from text using linguistic patterns and name-based detection."""

import re
from typing import Dict, Literal, Optional
import gender_guesser.detector as gender_detector


class GenderDetector:
    """Detects gender from text using linguistic patterns and name detection."""
    
    def __init__(self):
        self.name_detector = gender_detector.Detector()
        # Common first names that might appear in examples
        self.common_names = {
            'male': ['james', 'john', 'robert', 'michael', 'william', 'david', 'richard', 
                    'joseph', 'thomas', 'charles', 'chris', 'daniel', 'matthew', 'anthony',
                    'mark', 'donald', 'steven', 'paul', 'andrew', 'joshua', 'kenneth',
                    'kevin', 'brian', 'george', 'edward', 'ronald', 'timothy', 'jason',
                    'jeffrey', 'ryan', 'jacob', 'gary', 'nicholas', 'eric', 'stephen',
                    'jonathan', 'larry', 'justin', 'scott', 'brandon', 'benjamin', 'samuel'],
            'female': ['mary', 'patricia', 'jennifer', 'linda', 'elizabeth', 'barbara',
                      'susan', 'jessica', 'sarah', 'karen', 'nancy', 'lisa', 'betty',
                      'margaret', 'sandra', 'ashley', 'kimberly', 'emily', 'donna',
                      'michelle', 'dorothy', 'carol', 'amanda', 'melissa', 'deborah',
                      'stephanie', 'rebecca', 'sharon', 'laura', 'cynthia', 'kathleen',
                      'amy', 'angela', 'shirley', 'anna', 'brenda', 'pamela', 'emma',
                      'nicole', 'helen', 'samantha', 'frances', 'christine', 'marie',
                      'janet', 'catherine', 'jane', 'carrie', 'julie', 'heather']
        }
    
    def detect(self, text: str) -> Dict[str, any]:
        """
        Detect gender from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with:
                - gender: "male", "female", or "unknown"
                - confidence: float from 0.0 to 1.0
                - method: "pronouns", "name", "patterns", or "unknown"
        """
        if not text or not text.strip():
            return {
                "gender": "unknown",
                "confidence": 0.0,
                "method": "unknown"
            }
        
        text_lower = text.lower()
        
        # Method 1: Check for pronouns
        male_pronouns = [' he ', ' his ', ' him ', ' himself ']
        female_pronouns = [' she ', ' her ', ' hers ', ' herself ']
        
        male_count = sum(1 for pronoun in male_pronouns if pronoun in text_lower)
        female_count = sum(1 for pronoun in female_pronouns if pronoun in text_lower)
        
        if male_count > female_count and male_count > 0:
            return {
                "gender": "male",
                "confidence": min(0.8, 0.5 + (male_count * 0.1)),
                "method": "pronouns"
            }
        elif female_count > male_count and female_count > 0:
            return {
                "gender": "female",
                "confidence": min(0.8, 0.5 + (female_count * 0.1)),
                "method": "pronouns"
            }
        
        # Method 2: Extract potential names and check
        # Look for capitalized words that might be names
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        for word in words:
            word_lower = word.lower()
            if word_lower in self.common_names['male']:
                return {
                    "gender": "male",
                    "confidence": 0.7,
                    "method": "name"
                }
            elif word_lower in self.common_names['female']:
                return {
                    "gender": "female",
                    "confidence": 0.7,
                    "method": "name"
                }
            else:
                # Try gender-guesser
                try:
                    guess = self.name_detector.get_gender(word)
                    if guess in ['male', 'mostly_male']:
                        return {
                            "gender": "male",
                            "confidence": 0.6 if guess == 'mostly_male' else 0.8,
                            "method": "name"
                        }
                    elif guess in ['female', 'mostly_female']:
                        return {
                            "gender": "female",
                            "confidence": 0.6 if guess == 'mostly_female' else 0.8,
                            "method": "name"
                        }
                except:
                    pass
        
        # Method 3: Check for self-references and patterns
        self_ref_patterns = {
            'male': [r'\bi\'m\s+\w+', r'\bim\s+\w+', r'\bi am\s+\w+'],
            'female': []
        }
        
        # If no clear indicators, return unknown
        return {
            "gender": "unknown",
            "confidence": 0.0,
            "method": "unknown"
        }
    
    def get_gender(self, text: str) -> Literal["male", "female", "unknown"]:
        """Get just the gender label."""
        return self.detect(text)["gender"]

