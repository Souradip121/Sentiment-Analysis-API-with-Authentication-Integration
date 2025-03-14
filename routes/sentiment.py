from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

from database import get_db
import models
from auth import get_current_active_user
from services.sentiment_analyzer import analyze_text
from services.external_api import analyze_with_external_api

router = APIRouter()

class SentimentRequest(BaseModel):
    text: str
    provider: Optional[str] = None  # "local" or specific external provider

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    score: float
    confidence: Optional[float] = None
    provider: Optional[str] = None
    
    class Config:
        orm_mode = True

@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(
    request: SentimentRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """
    Analyze the sentiment of the provided text.
    Uses local NLTK analyzer by default or external APIs if specified.
    """
    try:
        # Use external API if provider is specified
        if request.provider and request.provider != "local":
            result = analyze_with_external_api(request.text, request.provider)
            
            # Save analysis to database
            db_analysis = models.Analysis(
                text=request.text,
                sentiment_score=result["score"],
                sentiment_label=result["sentiment"],
                user_id=current_user.id
            )
            db.add(db_analysis)
            db.commit()
            
            return result
        
        # Use local analyzer
        else:
            result = analyze_text(request.text)
            
            # Save analysis to database
            db_analysis = models.Analysis(
                text=request.text,
                sentiment_score=result["scores"]["compound"],
                sentiment_label=result["label"],
                user_id=current_user.id
            )
            db.add(db_analysis)
            db.commit()
            
            return {
                "text": request.text,
                "sentiment": result["label"],
                "score": result["scores"]["compound"],
                "confidence": abs(result["scores"]["compound"]),
                "provider": "local"
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing sentiment: {str(e)}"
        )

@router.get("/history", response_model=List[SentimentResponse])
async def get_analysis_history(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user)
):
    """Get sentiment analysis history for the current user"""
    analyses = db.query(models.Analysis).filter(models.Analysis.user_id == current_user.id).all()
    
    return [
        {
            "text": analysis.text,
            "sentiment": analysis.sentiment_label,
            "score": analysis.sentiment_score,
            "confidence": abs(analysis.sentiment_score),
            "provider": "unknown"  # We don't store provider in the current model
        }
        for analysis in analyses
    ]
