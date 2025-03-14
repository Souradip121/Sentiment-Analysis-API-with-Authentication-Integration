import os
from typing import Dict, Optional, Tuple
from enum import Enum
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class SentimentAnalyzer:
    """
    Class to analyze sentiment of text using NLTK's VADER.
    Can be extended to use other models or external APIs.
    """
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
    
    def analyze_with_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze text sentiment using NLTK's VADER.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict containing sentiment scores (neg, neu, pos, compound)
        """
        return self.vader.polarity_scores(text)
    
    def get_sentiment_label(self, compound_score: float) -> SentimentLabel:
        """
        Convert compound score to sentiment label.
        
        Args:
            compound_score: The compound sentiment score from VADER
            
        Returns:
            SentimentLabel enum value (POSITIVE, NEGATIVE, or NEUTRAL)
        """
        if compound_score >= 0.05:
            return SentimentLabel.POSITIVE
        elif compound_score <= -0.05:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL
    
    def analyze(self, text: str) -> Dict[str, any]:
        """
        Analyze text and return sentiment scores and label.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict containing sentiment scores and label
        """
        # Get scores from VADER
        vader_scores = self.analyze_with_vader(text)
        compound_score = vader_scores['compound']
        
        # Get sentiment label
        sentiment_label = self.get_sentiment_label(compound_score)
        
        return {
            "scores": {
                "positive": vader_scores['pos'],
                "negative": vader_scores['neg'],
                "neutral": vader_scores['neu'],
                "compound": compound_score
            },
            "label": sentiment_label,
            "text": text
        }


# Singleton instance
sentiment_analyzer = SentimentAnalyzer()

def analyze_text(text: str) -> Dict[str, any]:
    """
    Analyze the sentiment of text using the SentimentAnalyzer.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dict containing sentiment analysis results
    """
    try:
        result = sentiment_analyzer.analyze(text)
        return result
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        raise Exception(f"Failed to analyze text: {str(e)}")