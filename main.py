from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import logging
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from thefuzz import process, fuzz
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from typing import List, Dict, Any, Optional
import uvicorn
import gc
import psutil
from contextlib import asynccontextmanager
import nltk
from utils import (
    FileUtils,
    KeywordUtils,
    EmbeddingUtils,
    DeepseekUtils
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Application state management
class AppState:
    __instance = None
    
    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.initialized = False
        return cls.__instance
    
    def initialize(self):
        if not self.initialized:
            self.st_model = None
            self.kw_model = None
            self.file_utils = None
            self.keyword_utils = None
            self.embedding_utils = None
            self.deepseek_utils = None
            self.df = None
            self.label_to_indices = {}
            self.has_clusters = False
            self.similarity_threshold = 0.5
            self.fuzzy_cutoff = 70
            self.initialized = True

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async context manager for app lifespan"""
    await startup_event()
    yield
    shutdown_event()

app = FastAPI(title="ESCO Skill Matcher API", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "memory_usage": f"{psutil.virtual_memory().percent}%"
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://esco-skills-matcher.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CourseDescription(BaseModel):
    text: str

class NLPResult(BaseModel):
    skill: str
    score: float

class AnalysisResponse(BaseModel):
    keywords: List[str]
    nlpResults: List[NLPResult]
    llmResults: List[str]
    hybridResults: List[str]

async def startup_event():
    """Initialize application state with memory optimizations"""
    app_state.initialize()
    
    # Initialize utilities
    app_state.file_utils = FileUtils()
    app_state.keyword_utils = KeywordUtils()
    app_state.embedding_utils = EmbeddingUtils()
    
    # Initialize Deepseek
    load_dotenv()
    app_state.deepseek_utils = DeepseekUtils(
        processed_skills_path='./data/processed_skills.pkl',
        api_key=os.environ.get("DEEPSEEK_API_KEY", "")
    )
    
    # Load data with memory optimizations
    try:
        app_state.df = pd.read_pickle('./data/clustered_processed_skills.pkl')[
            ['preferredLabel', 'altLabels', 'hiddenLabels', 'embeddings', 'cluster']
        ]
        app_state.has_clusters = True
    except FileNotFoundError:
        app_state.df = pd.read_pickle('./data/processed_skills.pkl')[
            ['preferredLabel', 'altLabels', 'hiddenLabels', 'embeddings']
        ]
        app_state.has_clusters = False
    
    # Reduce embedding size
    if 'embeddings' in app_state.df.columns:
        app_state.df['embeddings'] = app_state.df['embeddings']
    
    # Build label index
    for idx, row in app_state.df.iterrows():
        labels = []
        if pd.notnull(row['preferredLabel']):
            labels.append(row['preferredLabel'].lower())
        if pd.notnull(row['altLabels']):
            labels.extend(label.strip().lower() for label in str(row['altLabels']).split(','))
        if pd.notnull(row['hiddenLabels']):
            labels.extend(label.strip().lower() for label in str(row['hiddenLabels']).split(','))
        
        for label in labels:
            if label not in app_state.label_to_indices:
                app_state.label_to_indices[label] = []
            app_state.label_to_indices[label].append(idx)
    
    logging.info("API startup complete")

def shutdown_event():
    """Cleanup on shutdown"""
    if hasattr(app_state, 'st_model') and app_state.st_model is not None:
        del app_state.st_model
    if hasattr(app_state, 'kw_model') and app_state.kw_model is not None:
        del app_state.kw_model
    gc.collect()
    logging.info("API shutdown complete")

def get_memory_usage():
    return psutil.virtual_memory().percent / 100

def load_models_if_needed():
    """Lazy load models when needed"""
    if app_state.st_model is None:
        app_state.st_model = SentenceTransformer('all-mpnet-base-v2')
    if app_state.kw_model is None:
        app_state.kw_model = KeyBERT(app_state.st_model)

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_course(course: CourseDescription):
    """Analyze course description with memory management"""
    if not course.text.strip():
        raise HTTPException(status_code=400, detail="Course description is required")
    
    if get_memory_usage() > 0.85:  # 85% memory threshold
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable due to high memory usage"
        )
    
    try:
        # Lazy load models
        load_models_if_needed()
        
        # Process input with memory cleanup
        try:
            keywords = app_state.keyword_utils.extract_keywords(
                course.text, 
                app_state.kw_model, 
                top_n=10
            )
            embedding = app_state.embedding_utils.compute_embeddings(
                course.text, 
                app_state.st_model
            )  # Reduced dimension
            
            # Fuzzy matching
            fuzzy_matches = []
            for keyword in keywords:
                matches = process.extractBests(
                    keyword,
                    list(app_state.label_to_indices.keys()),
                    scorer=fuzz.token_set_ratio,
                    score_cutoff=app_state.fuzzy_cutoff,
                    limit=50
                )
                fuzzy_matches.extend(matches)

            # Process results with memory efficiency
            nlp_results = process_fuzzy_matches(fuzzy_matches, embedding)
            llm_results = await get_llm_results(course.text)
            hybrid_results = await get_hybrid_results(course.text, nlp_results)
            
            return AnalysisResponse(
                keywords=keywords,
                nlpResults=[NLPResult(skill=r["skill"], score=r["score"]) for r in nlp_results],
                llmResults=llm_results if llm_results else [],
                hybridResults=hybrid_results if hybrid_results else []
            )
        finally:
            gc.collect()
    except Exception as e:
        logging.error(f"Error analyzing course: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")

def process_fuzzy_matches(fuzzy_matches, embedding):
    """Process fuzzy matches with memory efficiency"""
    fuzzy_results = []
    for match_text, match_score in fuzzy_matches:
        if match_text.lower() in app_state.label_to_indices:
            for idx in app_state.label_to_indices[match_text.lower()]:
                row = app_state.df.iloc[idx]
                sim = app_state.embedding_utils.calculate_similarity(
                    embedding, 
                    row['embeddings']
                )
                if sim >= app_state.similarity_threshold:
                    fuzzy_results.append({
                        'preferredLabel': row.get('preferredLabel', 'N/A'),
                        'similarity': sim,
                        'fuzzy_score': match_score,
                        'cluster': row.get('cluster', -1)
                    })

    # Process and deduplicate results
    nlp_results = []
    seen_skills = set()
    
    if fuzzy_results:
        results_df = pd.DataFrame(fuzzy_results)
        results_df = results_df.sort_values(
            by=['similarity', 'fuzzy_score'],
            ascending=[False, False]
        ).drop_duplicates('preferredLabel')
        
        for _, row in results_df.head(10).iterrows():
            if row['preferredLabel'] not in seen_skills:
                nlp_results.append({
                    "skill": row['preferredLabel'],
                    "score": float(row['similarity'])
                })
                seen_skills.add(row['preferredLabel'])
    
    return nlp_results

async def get_llm_results(text: str):
    """Get LLM results with error handling"""
    try:
        return await app_state.deepseek_utils.get_pure_recommendations(
            teacher_input=text,
            max_results=10,
            fuzzy_threshold=70
        )
    except Exception as e:
        logging.error(f"Error getting LLM results: {str(e)}")
        return []

async def get_hybrid_results(text: str, nlp_results: List[Dict]):
    """Get hybrid results with memory efficiency"""
    if not app_state.has_clusters or not nlp_results:
        return []
    
    try:
        # Get cluster skills
        candidate_skills = {res['skill'] for res in nlp_results}
        result_clusters = {res['cluster'] for res in nlp_results if res.get('cluster', -1) != -1}
        
        for cluster_id in result_clusters:
            cluster_skills = app_state.df[app_state.df['cluster'] == cluster_id]['preferredLabel'].head(3)
            candidate_skills.update(cluster_skills.tolist())
        
        return await app_state.deepseek_utils.get_hybrid_recommendations(
            teacher_input=text,
            nlp_cluster_skills=list(candidate_skills)[:10],  # Limited input size
            max_results=5,  # Reduced from 10
            fuzzy_threshold=0
        )
    except Exception as e:
        logging.error(f"Error getting hybrid results: {str(e)}")
        return []

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1  # We manage our own concurrency
    )