from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Import database
from database import engine
import models

# Import routers
from routes import sentiment, auth

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis with authentication",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["authentication"])
app.include_router(sentiment.router, prefix="/sentiment", tags=["sentiment"])

@app.get("/")
def root():
    return {
        "message": "Sentiment Analysis API",
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
