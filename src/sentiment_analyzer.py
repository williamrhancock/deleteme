"""Sentiment analysis using VADER and TextBlob."""

from typing import Dict, Literal
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


class SentimentAnalyzer:
    """Analyzes sentiment of text using VADER and TextBlob."""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with:
                - sentiment: "positive", "negative", or "neutral"
                - sentiment_score: float from -1.0 to 1.0
                - confidence: float from 0.0 to 1.0
                - vader_compound: VADER compound score
                - textblob_polarity: TextBlob polarity score
        """
        if not text or not text.strip():
            return {
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "vader_compound": 0.0,
                "textblob_polarity": 0.0
            }
        
        # VADER analysis
        vader_scores = self.vader.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        # TextBlob analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        
        # Combine scores (weighted average)
        combined_score = (vader_compound * 0.6) + (textblob_polarity * 0.4)
        
        # Determine sentiment category
        if combined_score >= 0.1:
            sentiment = "positive"
        elif combined_score <= -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Calculate confidence (absolute value of score)
        confidence = abs(combined_score)
        
        return {
            "sentiment": sentiment,
            "sentiment_score": round(combined_score, 4),
            "confidence": round(confidence, 4),
            "vader_compound": round(vader_compound, 4),
            "textblob_polarity": round(textblob_polarity, 4)
        }
    
    def get_sentiment_label(self, text: str) -> Literal["positive", "negative", "neutral"]:
        """Get just the sentiment label."""
        return self.analyze(text)["sentiment"]
    
    def get_sentiment_score(self, text: str) -> float:
        """Get just the sentiment score."""
        return self.analyze(text)["sentiment_score"]

