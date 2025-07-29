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
import time
import json
from collections import defaultdict
from contextlib import asynccontextmanager
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from utils import (
    FileUtils,
    ESCOProcessor,
    KeywordUtils,
    EmbeddingUtils,
    DeepseekUtils,
    NLPSkillFinder,
    TextPreprocessor
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
            self._models_loaded = False
            self._data_loaded = False
            self.st_model = None
            self.kw_model = None
            self.file_utils = None
            self.keyword_utils = None
            self.embedding_utils = None
            self.deepseek_utils = None
            self.nlp_skill_finder = None
            self.df = None
            self.label_to_indices = {}
            self.has_clusters = False
            self.initialized = True

    def load_models(self):
        if not hasattr(self, 'st_model'):
            self.st_model = SentenceTransformer('all-mpnet-base-v2')
            self.kw_model = KeyBERT(self.st_model)
            self.file_utils = FileUtils()
            self.keyword_utils = KeywordUtils()
            self.embedding_utils = EmbeddingUtils()
            self._models_loaded = True

    def load_data(self):
        if not hasattr(self, 'df'):
            try:
                self.df = pd.read_pickle('./data/clustered_processed_skills.pkl')[
                    ['preferredLabel', 'altLabels', 'hiddenLabels', 'embeddings', 'skill_keywords', 'cluster', 'conceptUri']
                ]
                self.has_clusters = True
            except FileNotFoundError:
                self.df = pd.read_pickle('./data/processed_skills.pkl')[
                    ['preferredLabel', 'altLabels', 'hiddenLabels', 'embeddings', 'skill_keywords', 'conceptUri']
                ]
                self.has_clusters = False
            
            self.label_to_indices = {}
            for idx, row in self.df.iterrows():
                labels = []
                if pd.notnull(row['preferredLabel']):
                    labels.append(row['preferredLabel'].lower())
                if pd.notnull(row['altLabels']):
                    labels.extend(label.strip().lower() for label in str(row['altLabels']).split(','))
                if pd.notnull(row['hiddenLabels']):
                    labels.extend(label.strip().lower() for label in str(row['hiddenLabels']).split(','))
                
                for label in labels:
                    if label not in self.label_to_indices:
                        self.label_to_indices[label] = []
                    self.label_to_indices[label].append(idx)
            
            self._data_loaded = True

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
    nlpResults: List[NLPResult]
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
            ['preferredLabel', 'altLabels', 'hiddenLabels', 'embeddings', 'skill_keywords', 'cluster', 'conceptUri']
        ]
        app_state.has_clusters = True
    except FileNotFoundError:
        app_state.df = pd.read_pickle('./data/processed_skills.pkl')[
            ['preferredLabel', 'altLabels', 'hiddenLabels', 'embeddings', 'skill_keywords', 'conceptUri']
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
async def analyze_course(course: CourseDescription, request: Request):
    """Analyze course description with strict quality thresholds"""
    if not course.text.strip():
        raise HTTPException(status_code=400, detail="Empty input")
        
    # Initialize debug logging
    debug_data = {
        'input_text': course.text,
        'timings': {},
        'stages': {}
    }
    start_time = time.time()
    
    try:
        # Initialize thresholds (adjust these as needed)
        MIN_SIMILARITY = 50    # 50% similarity minimum
        FUZZY_CUTOFF = 90      #
        TOP_N_RESULTS = 10     # Max results per method
            
        debug_data['parameters'] = {
            'MIN_SIMILARITY': MIN_SIMILARITY,
            'FUZZY_CUTOFF': FUZZY_CUTOFF,
            'TOP_N_RESULTS': TOP_N_RESULTS
        }
    
        # 1. Extract and filter keywords from query text
        kw_start = time.time()
        query_keywords = KeywordUtils.extract_keywords(
            course.text,
            app_state.kw_model,
            top_n=30,
            ngram_range=(2, 2)       # bi-grams only, same as skill side
        )
        
        all_skill_keywords = app_state.df['skill_keywords'].tolist()
        debug_data['timings']['keyword_processing'] = time.time() - kw_start
        debug_data['stages']['keywords'] = {
            'query_keywords': list(query_keywords),
            'total_skills': len(all_skill_keywords),
            'sample_skill_keywords': all_skill_keywords[:3] if all_skill_keywords else []
        }
    
        # 2. Generate Embedding
        emb_start = time.time()
        embedding = EmbeddingUtils.compute_embeddings(
            course.text, 
            app_state.st_model,
            preprocess=True
        )
        debug_data['timings']['embedding_generation'] = time.time() - emb_start
        debug_data['stages']['embedding'] = {
            'dimensions': len(embedding),
            'sample_values': embedding[:3].tolist()  # First 3 values as sample
        }
    
        fuzzy_start = time.time()
        candidate_map   = {}                # idx → kw_strength
        keyword_hits    = defaultdict(list) # kw_string → list[idx]


        for query_kw in query_keywords:                       # outer loop: query bigrams
            for idx, skill_kws in enumerate(all_skill_keywords):
                if not skill_kws: 
                    continue

                best_score = max(
                    fuzz.token_sort_ratio(query_kw, sk_kw)    # strict bi-gram match
                    for sk_kw in skill_kws
                )
                if best_score >= FUZZY_CUTOFF:
                    kw_strength = best_score / 100.0
                    # keep strongest
                    candidate_map[idx] = max(candidate_map.get(idx, 0.0), kw_strength)
                    keyword_hits[query_kw].append(idx)


        # ---------------------------------------------------------------------
        # 4.  For *only* those candidates, compute cosine & blended score
        # ---------------------------------------------------------------------
        fuzzy_results = []

        for idx, kw_strength in candidate_map.items():
            row = app_state.df.iloc[idx]
            sim = cosine_similarity([embedding], [row['embeddings']])[0][0]    # 0-1
            blended = 0.8 * sim + 0.2 * kw_strength

            fuzzy_results.append({
                "skill":         row['preferredLabel'],
                "score":         round(blended * 100, 2),
                "embedding_sim": round(sim * 100, 2),
                "kw_strength":   round(kw_strength*100, 2),
                "source":        "keyword+cosine"
            })
        
        debug_data['timings']['fuzzy_matching'] = time.time() - fuzzy_start
        debug_data['stages']['fuzzy_matching'] = {
            'total_matches': len(fuzzy_results),
            'sample_matches': fuzzy_results[:3] if fuzzy_results else []
        }
    
        # 4. High-Quality Semantic Matching
        semantic_start = time.time()
        similarities = cosine_similarity([embedding], np.vstack(app_state.df['embeddings']))[0]
        semantic_results = []
        top_indices = np.argsort(similarities)[-100:][::-1]  # Check top 100
        for idx in top_indices:
            score = similarities[idx]
            row = app_state.df.iloc[idx]
            result = {
                "skill": row['preferredLabel'],
                "score": round(score * 100, 2),
                "source": "semantic"
            }
            semantic_results.append(result)
            if len(semantic_results) >= TOP_N_RESULTS:
                break
        
        debug_data['timings']['semantic_matching'] = time.time() - semantic_start
        debug_data['stages']['semantic_matching'] = {
            'top_scores': [r['score'] for r in semantic_results],
            'skills_found': [r['skill'] for r in semantic_results],
            'similarity_range': {
                'min': float(similarities.min()),
                'max': float(similarities.max()),
                'mean': float(similarities.mean())
            }
        }
    
        # 5. Cluster-Augmented Results (if available)
        cluster_start = time.time()
        cluster_results = []
        if app_state.has_clusters and (fuzzy_results or semantic_results):
            # Get clusters from ALL valid results
            result_clusters = set()
            
            # Check fuzzy results (if any)
            for res in fuzzy_results:
                skill_rows = app_state.df[app_state.df['preferredLabel'] == res['skill']]
                if not skill_rows.empty:
                    cluster_id = skill_rows.iloc[0]['cluster']
                    if cluster_id != -1:
                        result_clusters.add(cluster_id)
            
            # Check semantic results (if any)
            for res in semantic_results:
                skill_rows = app_state.df[app_state.df['preferredLabel'] == res['skill']]
                if not skill_rows.empty:
                    cluster_id = skill_rows.iloc[0]['cluster']
                    if cluster_id != -1:
                        result_clusters.add(cluster_id)
        
            # Get skills from relevant clusters
            cluster_skills = []
            for cluster_id in result_clusters:
                cluster_skills.extend(
                    app_state.df[app_state.df['cluster'] == cluster_id]
                    .to_dict('records')
                )
            
            for row in cluster_skills:
                sim = cosine_similarity([embedding], [row['embeddings']])[0][0]
                cluster_results.append({
                    "skill": row['preferredLabel'],
                    "score": round(sim * 100, 2),
                    "source": f"cluster_{row['cluster']}"
                })
        
        debug_data['timings']['cluster_processing'] = time.time() - cluster_start
        debug_data['stages']['cluster_results'] = {
            'clusters_used': list(result_clusters) if app_state.has_clusters else [],
            'skills_from_clusters': [r['skill'] for r in cluster_results],
            'cluster_match_count': len(cluster_results)
        }
    
        # 6. Combine and Strictly Filter Results
        combine_start = time.time()
        combined = fuzzy_results + semantic_results + cluster_results
        seen = set()
        final_nlp = []
        
        for res in sorted(combined, key=lambda x: x["score"], reverse=True):
            skill = res["skill"]
            if skill not in seen:
                seen.add(skill)
                if res["score"] >= MIN_SIMILARITY:
                    final_nlp.append(res)
        
        debug_data['timings']['result_combination'] = time.time() - combine_start
        debug_data['stages']['combined_results'] = {
            'total_candidates': len(combined),
            'unique_skills': len(seen),
            'final_candidates': [r['skill'] for r in final_nlp],
            'score_range': {
                'min': min(r['score'] for r in final_nlp) if final_nlp else 0,
                'max': max(r['score'] for r in final_nlp) if final_nlp else 0,
                'avg': sum(r['score'] for r in final_nlp)/len(final_nlp) if final_nlp else 0
            }
        }
        
        # 8. Hybrid Approach with Curated Input
        hybrid_start = time.time()
        candidate_skills = {res["skill"] for res in final_nlp}
        hybrid_results = await app_state.deepseek_utils.get_hybrid_recommendations(
            input_text=course.text,
            nlp_results=list(candidate_skills),
            max_results=TOP_N_RESULTS,
            fuzzy_threshold=80
        )
        debug_data['timings']['hybrid_processing'] = time.time() - hybrid_start
        debug_data['stages']['hybrid_results'] = {
            'input_candidates': len(candidate_skills),
            'final_skills': hybrid_results,
            'new_skills_added': len(set(hybrid_results) - candidate_skills)
        }
        
        # Detailed analysis logging
        total_time = time.time() - start_time
        logging.info(f"\n=== Analysis Report ===")
        logging.info(f"Completed in {total_time:.2f}s")
        
        logging.info(f"\nQuery keywords ({len(query_keywords)}):")
        for kw in query_keywords:
            idxs = keyword_hits.get(kw, [])
            logging.info(f"- '{kw}': Matched {len(idxs)} skills")
            if idxs:
                sample = [app_state.df.iloc[i]['preferredLabel'] for i in idxs]
                logging.info(f"  Sample matches: {sample}")

        # Top matches breakdown
        logging.info("\nNLP Matches (Score Breakdown):")
        for i, match in enumerate(final_nlp, 1):
            logging.info(f"{i}. {match['skill']} (Score: {match['score']:.1f})")
            logging.info(f"   Source: {match['source']}")
            if 'match_score' in match:
                logging.info(f"   Keyword match: {match['match_score']}%")

        # Cluster influence analysis
        if app_state.has_clusters:
            logging.info("\n=== Cluster Analysis ===")
            cluster_stats = defaultdict(lambda: {'count': 0, 'skills': []})
            
            # Analyze top 10 results
            for match in final_nlp[:10]:
                skill_rows = app_state.df[app_state.df['preferredLabel'] == match['skill']]
                if not skill_rows.empty:
                    cluster_id = skill_rows.iloc[0]['cluster']
                    cluster_stats[cluster_id]['count'] += 1
                    cluster_stats[cluster_id]['skills'].append(match['skill'])
            
            # Log cluster distribution
            logging.info("Cluster Distribution in Top 10 Results:")
            for cluster_id, stats in sorted(cluster_stats.items(), 
                                         key=lambda x: x[1]['count'], 
                                         reverse=True):
                logging.info(f"- Cluster {cluster_id}: {stats['count']} skills")
                logging.info(f"  Sample skills: {stats['skills'][:3]}")
            
            # Check for cluster dominance
            dominant_cluster = max(cluster_stats.items(), key=lambda x: x[1]['count'])
            if dominant_cluster[1]['count'] > 5:  # More than half from one cluster
                logging.warning(f"Cluster {dominant_cluster[0]} dominates results - consider reviewing cluster composition")

        # LLM reasoning
        logging.info("\nLLM Analysis:")
        logging.info(f"- Hybrid recommendations: {hybrid_results[:10]}")
        if set(hybrid_results) - set([r['skill'] for r in final_nlp]):
            logging.info("LLM added new skills not found by NLP methods")

        logging.info("\nFinal Top Recommendations:")
        for i, skill in enumerate(hybrid_results[:10], 1):
            logging.info(f"{i}. {skill}")
    
        def get_skill_url(skill_name: str) -> str:
            return ESCOProcessor.get_skill_url(app_state.df, skill_name)
    
        return AnalysisResponse(
            nlpResults=[
                NLPResult(
                    skill=r["skill"], 
                    score=r["score"],
                    url=get_skill_url(r["skill"])  # Add URL
                ) for r in final_nlp[:TOP_N_RESULTS]
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
