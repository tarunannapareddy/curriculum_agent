from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import os
from dotenv import load_dotenv
from curriculum_agent import LlamaCurriculumAgent, CurriculumInput

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Language Learning Curriculum API",
    description="API for generating language learning curriculum content",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the curriculum agent
try:
    agent = LlamaCurriculumAgent()
except Exception as e:
    print(f"Warning: Could not initialize curriculum agent: {e}")
    agent = None

# Pydantic models for API requests and responses
class CurriculumRequest(BaseModel):
    target_language: str
    scenario: str

class CurriculumQuestion(BaseModel):
    question: str
    expected_response: str

class CorrectionExample(BaseModel):
    incorrect_phrase: str
    correct_phrase: str
    explanation: str

class CurriculumResponse(BaseModel):
    scenario_scene: str
    curriculum_questions: List[CurriculumQuestion]
    correction_examples: List[CorrectionExample]

class HealthResponse(BaseModel):
    status: str
    message: str
    agent_available: bool
    cache_stats: dict = {}

class AvailableScenariosResponse(BaseModel):
    scenarios: List[str]
    languages: List[str]

class CacheStatsResponse(BaseModel):
    knowledge_cache_size: int
    real_world_cache_size: int
    cached_combinations: List[str]
    cache_hit_rate: str

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    cache_stats = agent.get_cache_stats() if agent else {}
    return HealthResponse(
        status="healthy",
        message="Language Learning Curriculum API is running",
        agent_available=agent is not None,
        cache_stats=cache_stats
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    cache_stats = agent.get_cache_stats() if agent else {}
    return HealthResponse(
        status="healthy",
        message="Language Learning Curriculum API is running",
        agent_available=agent is not None,
        cache_stats=cache_stats
    )

@app.get("/available-scenarios", response_model=AvailableScenariosResponse)
async def get_available_scenarios():
    """Get available scenarios and languages from the knowledge base"""
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="Curriculum agent is not available. Please check your API key and try again."
        )
    
    try:
        # Get scenarios and languages from the agent's knowledge base
        scenarios, languages = agent.get_available_scenarios_and_languages()
        
        return AvailableScenariosResponse(
            scenarios=scenarios,
            languages=languages
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving available scenarios and languages: {str(e)}"
        )

@app.get("/cache-stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Get cache statistics"""
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="Curriculum agent is not available."
        )
    
    return CacheStatsResponse(**agent.get_cache_stats())

@app.post("/generate-curriculum", response_model=CurriculumResponse)
async def generate_curriculum(request: CurriculumRequest):
    """Generate curriculum content for a specific language and scenario"""
    
    if agent is None:
        raise HTTPException(
            status_code=503, 
            detail="Curriculum agent is not available. Please check your API key and try again."
        )
    
    # Validate input
    if not request.target_language or not request.scenario:
        raise HTTPException(
            status_code=400,
            detail="Both target_language and scenario are required"
        )
    
    try:
        # Create curriculum input
        curriculum_input = CurriculumInput(
            target_language=request.target_language,
            scenario=request.scenario
        )
        
        # Generate curriculum
        curriculum = agent.generate_curriculum(curriculum_input)
        
        # Convert to response format
        return CurriculumResponse(
            scenario_scene=curriculum.scenario_scene,
            curriculum_questions=[
                CurriculumQuestion(
                    question=q["question"],
                    expected_response=q["expected_response"]
                ) for q in curriculum.curriculum_questions
            ],
            correction_examples=[
                CorrectionExample(
                    incorrect_phrase=c["incorrect_phrase"],
                    correct_phrase=c["correct_phrase"],
                    explanation=c["explanation"]
                ) for c in curriculum.correction_examples
            ]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating curriculum: {str(e)}"
        )

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {
        "message": "API is working!",
        "agent_available": agent is not None,
        "api_key_set": bool(os.getenv("GOOGLE_API_KEY"))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 