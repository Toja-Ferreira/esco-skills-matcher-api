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
from typing import List, Dict, Any, Optional, Tuple
import uvicorn
import gc
import psutil
from contextlib import asynccontextmanager
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from utils import (
    FileUtils,
    ESCOProcessor,
    KeywordUtils,
    EmbeddingUtils,
    DeepseekUtils,
    NLPSkillFinder
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
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = False
    
    def initialize(self):
        if not self.initialized:
            self.st_model = None
            self.kw_model = None
            self.file_utils = FileUtils()
            self.keyword_utils = KeywordUtils()
            self.embedding_utils = EmbeddingUtils()
            self.deepseek_utils = None
            self.nlp_skill_finder = None
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
    url: str = ""

class AnalysisResponse(BaseModel):
    keywords: List[str]
    nlpResults: List[NLPResult]
    llmResults: List[Dict[str, str]]
    hybridResults: List[Dict[str, str]]

async def startup_event():
    """Initialize application state with memory optimizations"""
    app_state.initialize()
    
    # Load models
    load_models_if_needed()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize Deepseek
    app_state.deepseek_utils = DeepseekUtils(
        processed_skills_path='./data/processed_skills.pkl',
        api_key=os.environ.get("DEEPSEEK_API_KEY", "")
    )
    
    # Load data with memory optimizations
    try:
        app_state.df = pd.read_pickle('./data/clustered_processed_skills.pkl')[
            ['preferredLabel', 'altLabels', 'hiddenLabels', 'embeddings', 'cluster', 'conceptUri']
        ]
        app_state.has_clusters = True
    except FileNotFoundError:
        app_state.df = pd.read_pickle('./data/processed_skills.pkl')[
            ['preferredLabel', 'altLabels', 'hiddenLabels', 'embeddings', 'conceptUri']
        ]
        app_state.has_clusters = False
    
    # Initialize NLP Skill Finder
    app_state.nlp_skill_finder = NLPSkillFinder(app_state.df)
    
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
    """Analyze course description with strict quality thresholds"""
    if not course.text.strip():
        raise HTTPException(status_code=400, detail="Empty input")
    
    try:
        # Initialize thresholds (adjust these as needed)
        MIN_SIMILARITY = 50   # 50% similarity minimum
        FUZZY_CUTOFF = 75      # 75/100 fuzzy match minimum
        TOP_N_RESULTS = 10     # Max results per method
        
        # 1. Keyword Extraction
        keywords = KeywordUtils.extract_keywords(
            course.text, 
            app_state.kw_model, 
            top_n=10,
            diversity=0.7  # More diverse keywords
        )
        
        # 2. Generate Embedding
        embedding = EmbeddingUtils.compute_embeddings(
            course.text, 
            app_state.st_model,
            preprocess=True
        )
        
        # 3. Strict Fuzzy Matching
        fuzzy_results = []
        for keyword in keywords:
            matches = process.extractBests(
                keyword,
                list(app_state.label_to_indices.keys()),
                scorer=fuzz.token_set_ratio,
                score_cutoff=FUZZY_CUTOFF,
                limit=30  # More candidates for strict filtering
            )
            for match_text, match_score in matches:
                for idx in app_state.label_to_indices[match_text.lower()]:
                    row = app_state.df.iloc[idx]
                    sim = cosine_similarity([embedding], [row['embeddings']])[0][0]
                    fuzzy_results.append({
                        "skill": row['preferredLabel'],
                        "score": round(sim * 100, 2),
                        "match_score": match_score,
                        "source": "fuzzy"
                    })
        
        # 4. High-Quality Semantic Matching
        similarities = cosine_similarity([embedding], np.vstack(app_state.df['embeddings']))[0]
        semantic_results = []
        for idx in np.argsort(similarities)[-100:][::-1]:  # Check top 100
            score = similarities[idx]
            row = app_state.df.iloc[idx]
            semantic_results.append({
                "skill": row['preferredLabel'],
                "score": round(score * 100, 2),
                "source": "semantic"
            })
            if len(semantic_results) >= TOP_N_RESULTS:
                break
        
        # 5. Cluster-Augmented Results (if available)
        cluster_results = []
        if app_state.has_clusters and (fuzzy_results or semantic_results):
            # Get clusters from ALL valid results
            result_clusters = set()
            
            # Check fuzzy results (if any)
            for res in fuzzy_results:
                if 'cluster' in res and res['cluster'] != -1:
                    result_clusters.add(res['cluster'])
            
            # Check semantic results (if any)
            for res in semantic_results:
                # Find the skill in dataframe to get its cluster
                skill_rows = app_state.df[app_state.df['preferredLabel'] == res['skill']]
                if not skill_rows.empty:
                    cluster_id = skill_rows.iloc[0]['cluster']
                    if cluster_id != -1:
                        result_clusters.add(cluster_id)
            
            # Get skills from relevant clusters
            for cluster_id in result_clusters:
                cluster_skills = app_state.df[app_state.df['cluster'] == cluster_id]
                for _, row in cluster_skills.iterrows():
                    sim = cosine_similarity([embedding], [row['embeddings']])[0][0]
                    cluster_results.append({
                        "skill": row['preferredLabel'],
                        "score": round(sim * 100, 2),
                        "source": f"cluster_{cluster_id}"
                    })
        
        # 6. Combine and Strictly Filter Results
        combined = fuzzy_results + semantic_results + cluster_results
        combined = [res for res in combined if res["score"] >= MIN_SIMILARITY]
        seen = set()
        final_nlp = []
        
        for res in sorted(combined, key=lambda x: x["score"], reverse=True):
            skill = res["skill"]
            if skill not in seen:
                seen.add(skill)
                final_nlp.append(res)
                if len(final_nlp) >= TOP_N_RESULTS * 2:  # Allow more for LLM
                    break
        
        # 7. LLM Processing with Quality Control
        llm_results = await app_state.deepseek_utils.get_pure_recommendations(
            input_text=course.text,
            max_results=TOP_N_RESULTS,
            fuzzy_threshold=80
        )
        
        # 8. Hybrid Approach with Curated Input
        candidate_skills = {res["skill"] for res in final_nlp}
        hybrid_results = await app_state.deepseek_utils.get_hybrid_recommendations(
            input_text=course.text,
            nlp_results=list(candidate_skills),
            max_results=TOP_N_RESULTS,
            fuzzy_threshold=80
        )
        
        def get_skill_url(skill_name: str) -> str:
            return ESCOProcessor.get_skill_url(app_state.df, skill_name)
    
        return AnalysisResponse(
            keywords=keywords,
            nlpResults=[
                NLPResult(
                    skill=r["skill"], 
                    score=r["score"],
                    url=get_skill_url(r["skill"])  # Add URL
                ) for r in final_nlp[:TOP_N_RESULTS]
            ],
            llmResults=[
                {"skill": skill, "url": get_skill_url(skill)}
                for skill in llm_results[:TOP_N_RESULTS]
            ],
            hybridResults=[
                {"skill": skill, "url": get_skill_url(skill)}
                for skill in hybrid_results[:TOP_N_RESULTS]
            ]
        )
        
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Skill analysis failed - please try again"
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1  # We manage our own concurrency
    )