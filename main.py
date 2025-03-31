from fastapi import FastAPI, HTTPException
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

# Import your utility files
from utils import FileUtils, KeywordUtils, EmbeddingUtils, DeepseekUtils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(title="ESCO Skill Matcher API")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://esco-skills-matcher.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class CourseDescription(BaseModel):
    text: str

# Response models
class NLPResult(BaseModel):
    skill: str
    score: float

class AnalysisResponse(BaseModel):
    keywords: List[str]
    nlpResults: List[NLPResult]
    llmResults: List[str]
    hybridResults: List[str]

# Global variables for models and data
st_model = None
kw_model = None
file_utils = None
keyword_utils = None
embedding_utils = None
deepseek_utils = None
df = None
label_to_indices = {}
has_clusters = False
similarity_threshold = 0.5
fuzzy_cutoff = 70

@app.on_event("startup")
async def startup_event():
    """Initialize models and load data on startup"""
    global st_model, kw_model, file_utils, keyword_utils, embedding_utils, deepseek_utils, df, label_to_indices, has_clusters
    
    # Initialize models
    logging.info("Initializing models...")
    st_model = SentenceTransformer('all-mpnet-base-v2')
    kw_model = KeyBERT(model='all-mpnet-base-v2')
    file_utils = FileUtils()
    keyword_utils = KeywordUtils()
    embedding_utils = EmbeddingUtils()
    logging.info("Models initialized.")
    
    # Initialize Deepseek
    logging.info("Initializing Deepseek...")
    load_dotenv()  # Load environment variables from .env file
    deepseek_utils = DeepseekUtils(
        processed_skills_path='./data/processed_skills.pkl',
        api_key=os.environ("DEEPSEEK_API_KEY")
    )
    logging.info("Deepseek initialized.")
    
    # Load data
    try:
        df = file_utils.load_data('./data/clustered_processed_skills.pkl')
        logging.info("Loaded clustered skills data")
        has_clusters = True
    except FileNotFoundError:
        df = file_utils.load_data('./data/processed_skills.pkl')
        logging.info("Loaded regular skills data (no clusters)")
        has_clusters = False
    
    # Prepare label index for fuzzy matching
    for idx, row in df.iterrows():
        labels = []
        if pd.notnull(row['preferredLabel']):
            labels.append(row['preferredLabel'].lower())
        if pd.notnull(row['altLabels']):
            labels.extend([label.strip().lower() for label in str(row['altLabels']).split(',')])
        if pd.notnull(row['hiddenLabels']):
            labels.extend([label.strip().lower() for label in str(row['hiddenLabels']).split(',')])
        for label in labels:
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
    
    logging.info("API startup complete")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_course(course: CourseDescription):
    """Analyze course description and return skills"""
    global st_model, kw_model, file_utils, keyword_utils, embedding_utils, deepseek_utils, df, label_to_indices
    
    if not course.text.strip():
        raise HTTPException(status_code=400, detail="Course description is required")
    
    try:
        # Process input
        keywords = keyword_utils.extract_keywords(course.text, kw_model, top_n=10)
        logging.info(f"Extracted Keywords: {keywords}")
        embedding = embedding_utils.compute_embeddings(course.text, st_model)
        
        # 1. Get NLP results (using fuzzy and embedding matches)
        fuzzy_matches = []
        for keyword in keywords:
            matches = process.extractBests(
                keyword,
                list(label_to_indices.keys()),
                scorer=fuzz.token_set_ratio,
                score_cutoff=fuzzy_cutoff,
                limit=50
            )
            fuzzy_matches.extend(matches)

        fuzzy_results = []
        for match_text, match_score in fuzzy_matches:
            if match_text.lower() in label_to_indices:
                for idx in label_to_indices[match_text.lower()]:
                    row = df.iloc[idx]
                    sim = embedding_utils.calculate_similarity(embedding, row['embeddings'])
                    if sim >= similarity_threshold:
                        fuzzy_results.append({
                            'preferredLabel': row.get('preferredLabel', 'N/A'),
                            'similarity': sim,
                            'fuzzy_score': match_score,
                            'definition': row.get('definition', ''),
                            'cluster': row.get('cluster', -1)
                        })
        
        # Process embedding results
        similarities = embedding_utils.batch_similarity(
            embedding,
            np.vstack(df['embeddings'])
        )
        
        top_indices = np.argsort(similarities)[-50:][::-1]
        embedding_results = []
        for idx in top_indices:
            sim = similarities[idx]
            if sim > similarity_threshold:
                row = df.iloc[idx]
                embedding_results.append((sim, row))
        
        # Combine and process results
        nlp_results = []
        seen_skills = set()
        
        # Add fuzzy results first
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
        
        # Add embedding results next
        for sim, row in embedding_results:
            if row['preferredLabel'] not in seen_skills and len(nlp_results) < 10:
                nlp_results.append({
                    "skill": row['preferredLabel'],
                    "score": float(sim)
                })
                seen_skills.add(row['preferredLabel'])
        
        # 2. Get LLM results
        llm_results = await deepseek_utils.get_pure_recommendations(
            teacher_input=course.text,
            max_results=10,
            fuzzy_threshold=70
        )
        
        # 3. Get hybrid recommendations
        # Combine skills from all previous methods
        candidate_skills = set()
        
        # Add fuzzy match skills
        for result in fuzzy_results:
            if isinstance(result, dict):
                candidate_skills.add(result['preferredLabel'])
        
        # Add embedding match skills
        for sim, row in embedding_results:
            candidate_skills.add(row['preferredLabel'])
        
        # Add cluster skills if available
        if has_clusters:
            result_clusters = set()
            
            # From fuzzy results
            for result in fuzzy_results:
                if isinstance(result, dict) and result.get('cluster', -1) != -1:
                    result_clusters.add(result['cluster'])
            
            # From embedding results
            for _, row in embedding_results:
                if hasattr(row, 'get') and row.get('cluster', -1) != -1:
                    result_clusters.add(row['cluster'])
            
            # Add skills from relevant clusters
            for cluster_id in result_clusters:
                cluster_skills = df[df['cluster'] == cluster_id]
                candidate_skills.update(cluster_skills['preferredLabel'].tolist())
        
        hybrid_results = await deepseek_utils.get_hybrid_recommendations(
            teacher_input=course.text,
            nlp_cluster_skills=list(candidate_skills),
            max_results=10,
            fuzzy_threshold=0  # Exact match for filtered skills
        )
        
        # Return combined results
        return AnalysisResponse(
            keywords=keywords,
            nlpResults=[NLPResult(skill=r["skill"], score=r["score"]) for r in nlp_results],
            llmResults=llm_results if llm_results else [],
            hybridResults=hybrid_results if hybrid_results else []
        )
    
    except Exception as e:
        logging.error(f"Error analyzing course: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)