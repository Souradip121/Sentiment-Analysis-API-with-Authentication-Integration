import os
import logging
import json
import requests
from typing import Dict, Optional, Any, Tuple
from enum import Enum
from dotenv import load_dotenv
import time
import backoff

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API keys and endpoints
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

class APIProvider(str, Enum):
    """Enum for supported API providers"""
    HUGGINGFACE = "huggingface"
    
class SentimentResult(dict):
    """Class to standardize sentiment analysis results from different providers"""
    def __init__(self, 
                 provider: APIProvider, 
                 text: str, 
                 sentiment: str, 
                 score: float, 
                 raw_response: Any = None):
        super().__init__(
            provider=provider,
            text=text,
            sentiment=sentiment,
            score=score,
            raw_response=raw_response
        )

class ExternalAPIError(Exception):
    """Custom exception for API errors"""
    def __init__(self, provider: str, message: str, status_code: Optional[int] = None):
        self.provider = provider
        self.message = message
        self.status_code = status_code
        super().__init__(f"{provider} API Error: {message} (Status: {status_code})")

class SentimentAnalysisAPI:
    """
    Class to handle external API calls for sentiment analysis.
    Currently supports HuggingFace APIs.
    """
    def __init__(self):
        pass
    
    @backoff.on_exception(backoff.expo, 
                         (requests.exceptions.RequestException, ExternalAPIError),
                         max_tries=3)
    def analyze_with_huggingface(self, text: str) -> SentimentResult:
        """
        Analyze sentiment using HuggingFace Inference API.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentResult with standardized response
            
        Raises:
            ExternalAPIError: If API call fails
        """
        if not HUGGINGFACE_API_KEY:
            raise ExternalAPIError("HuggingFace", "API key not set", 400)
        
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": text,
            "options": {"wait_for_model": True}
        }
        
        try:
            response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
            
            # Check for errors
            if response.status_code != 200:
                raise ExternalAPIError(
                    "HuggingFace", 
                    f"Request failed with status {response.status_code}: {response.text}", 
                    response.status_code
                )
            
            result = response.json()
            
            # Parse HuggingFace response format
            if isinstance(result, list) and len(result) > 0:
                result = result[0]  # Get first result
                
            # Extract sentiment and score
            if "label" in result:
                sentiment = result["label"].lower()
                score = result.get("score", 0.0)
                
                # Map HuggingFace labels to our standardized format
                if sentiment == "positive":
                    sentiment = "positive"
                    score_sign = 1.0
                elif sentiment == "negative":
                    sentiment = "negative"
                    score_sign = -1.0
                else:
                    sentiment = "neutral"
                    score_sign = 0.0
                    
                # Ensure score has correct sign
                score = abs(score) * score_sign
                
                return SentimentResult(
                    provider=APIProvider.HUGGINGFACE,
                    text=text,
                    sentiment=sentiment,
                    score=score,
                    raw_response=result
                )
            else:
                raise ExternalAPIError(
                    "HuggingFace", 
                    f"Unexpected response format: {result}", 
                    200
                )
                
        except requests.exceptions.RequestException as e:
            logger.error(f"HuggingFace API request error: {str(e)}")
            raise ExternalAPIError("HuggingFace", str(e))

    def analyze(self, text: str, provider: Optional[APIProvider] = None) -> SentimentResult:
        """
        Analyze sentiment using specified provider.
        
        Args:
            text: Text to analyze
            provider: Specific provider to use (currently only supports HuggingFace)
            
        Returns:
            SentimentResult with standardized response
            
        Raises:
            ExternalAPIError: If API call fails
        """
        # If provider is specified, ensure it's valid
        if provider is not None and provider != APIProvider.HUGGINGFACE:
            raise ExternalAPIError("Provider", f"Unsupported provider: {provider}", 400)
        
        # Use HuggingFace
        if HUGGINGFACE_API_KEY:
            try:
                return self.analyze_with_huggingface(text)
            except ExternalAPIError as e:
                raise ExternalAPIError("HuggingFace", str(e), getattr(e, 'status_code', 500))
        
        # If we got here, no provider is available
        raise ExternalAPIError(
            "All providers", 
            "No sentiment analysis providers available", 
            500
        )

# Create singleton instance
external_api = SentimentAnalysisAPI()

def analyze_with_external_api(text: str, provider: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze text sentiment using external APIs.
    
    Args:
        text: Text to analyze
        provider: API provider to use (currently only "huggingface" is supported)
        
    Returns:
        Dict containing sentiment analysis results
        
    Raises:
        Exception: If analysis fails
    """
    try:
        # Convert provider string to enum if specified
        api_provider = None
        if provider:
            try:
                api_provider = APIProvider(provider.lower())
            except ValueError:
                logger.warning(f"Invalid provider '{provider}', defaulting to HuggingFace")
                api_provider = APIProvider.HUGGINGFACE
        
        # Perform analysis
        result = external_api.analyze(text, api_provider)
        
        return {
            "text": text,
            "provider": result["provider"],
            "sentiment": result["sentiment"],
            "score": result["score"],
            "confidence": abs(result["score"]) if result["score"] is not None else None
        }
    
    except Exception as e:
        logger.error(f"External API analysis failed: {str(e)}")
        raise Exception(f"External sentiment analysis failed: {str(e)}")