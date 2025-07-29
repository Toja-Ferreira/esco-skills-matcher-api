import pandas as pd
import re
import nltk
import os
import json
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from tqdm import tqdm
import httpx
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import lru_cache
from thefuzz import fuzz, process
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize NLTK resources quietly
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class TextPreprocessor:
    """Handles text preprocessing for different purposes"""
    
    @staticmethod
    def clean_text(text: str, lower_case: bool = True, remove_punctuation: bool = True) -> str:
        """Basic text cleaning"""
        if lower_case:
            text = text.lower()
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    @staticmethod
    def preprocess_for_keywords(text: str) -> str:
        """Preprocess text for keyword extraction (lemmatization)"""
        text = TextPreprocessor.clean_text(text)
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    @staticmethod
    def preprocess_for_embeddings(text: str) -> str:
        """Minimal preprocessing for embeddings"""
        return TextPreprocessor.clean_text(text)

class FileUtils:
    """Handles all file operations"""
    
    @staticmethod
    def load_data(path: str, file_type: str = None) -> pd.DataFrame:
        """
        Load data from file with automatic format detection
        
        Args:
            path: File path to load
            file_type: Explicit file type ('csv', 'pkl', 'json')
            
        Returns:
            Loaded DataFrame
        """
        if file_type is None:
            file_type = path.split('.')[-1].lower()
        
        try:
            if file_type == 'csv':
                return pd.read_csv(path)
            elif file_type == 'pkl':
                return pd.read_pickle(path)
            elif file_type == 'json':
                return pd.read_json(path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise IOError(f"Failed to load {path}: {str(e)}")

    @staticmethod
    def save_data(data: Union[pd.DataFrame, dict], path: str):
        """
        Save data to file with format detection
        
        Args:
            data: DataFrame or dictionary to save
            path: Destination path
        """
        file_type = path.split('.')[-1].lower()
        
        try:
            if file_type == 'csv' and isinstance(data, pd.DataFrame):
                data.to_csv(path, index=False)
            elif file_type == 'pkl':
                with open(path, 'wb') as f:
                    pickle.dump(data, f)
            elif file_type == 'json':
                if isinstance(data, pd.DataFrame):
                    data.to_json(path)
                else:
                    with open(path, 'w') as f:
                        json.dump(data, f)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise IOError(f"Failed to save {path}: {str(e)}")

class ESCOProcessor:
    """Combined ESCO data processing functionality"""
    
    def __init__(self, n_clusters: int = 50):
        """
        Initialize with models
        
        Args:
            n_clusters: Number of clusters for skill grouping
        """
        self.st_model = SentenceTransformer('all-mpnet-base-v2', device)
        self.kw_model = KeyBERT(model='all-mpnet-base-v2')
        self.n_clusters = n_clusters
        self.kmeans = None
        
    @staticmethod
    def combine_text_fields(row: pd.Series) -> str:
        """Combine all relevant text fields from an ESCO skill row"""
        components = []
        text_fields = [
            'preferredLabel', 'altLabels', 'hiddenLabels',
            'scopeNote', 'definition', 'description'
        ]
        
        for field in text_fields:
            if pd.notnull(row[field]):
                if field.endswith('Labels'):
                    components.extend(
                        label.strip() 
                        for label in str(row[field]).split(',')
                    )
                else:
                    components.append(str(row[field]))
        
        return ' '.join(components)

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full preprocessing pipeline on a DataFrame"""
        # Process text and extract features
        df['combined_text'] = df.apply(self.combine_text_fields, axis=1)
        df['skill_keywords'] = self._extract_keywords_batch(df['combined_text'])
        df['embeddings'] = self._generate_embeddings(df['combined_text'])
        
        # Cluster skills based on embeddings
        embeddings_array = np.vstack(df['embeddings'].values)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        df['cluster'] = self.kmeans.fit_predict(embeddings_array)
        
        return df
    
    def _extract_keywords_batch(self, texts: pd.Series) -> List[List[str]]:
        """Process keywords in batches"""
        batch_size = 256
        keywords = []
        for i in tqdm(range(0, len(texts), batch_size),
                      desc="Extracting Keywords",
                      unit="batch"):
            batch = texts.iloc[i:i+batch_size].tolist()
            keywords.extend([
                KeywordUtils.extract_keywords(text, self.kw_model)
                for text in batch
            ])
        return keywords
    
    def _generate_embeddings(self, texts: pd.Series) -> List[np.ndarray]:
        """Generate embeddings with batch processing, memory-optimized"""
        embeddings = []
        batch_size = 256  # Adjust based on your GPU memory
        
        for i in tqdm(range(0, len(texts), batch_size),
                    desc="Generating Embeddings",
                    unit="batch"):
            batch = texts.iloc[i:i+batch_size].tolist()
            batch_embeddings = self.st_model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.extend([batch_embeddings[j] for j in range(len(batch_embeddings))])
        
        return embeddings
    
    @staticmethod
    def get_skill_url(df: pd.DataFrame, skill_name: str) -> str:
        """
        Search DataFrame for skill and return its ESCO URL
        Args:
            df: Loaded DataFrame from PKL
            skill_name: Skill name to look up
        Returns:
            ESCO URL string or empty string if not found
        """
        try:
            # First try exact match with preferredLabel
            exact_match = df[df['preferredLabel'].str.lower() == skill_name.lower()]
            if not exact_match.empty:
                uri = exact_match.iloc[0]['conceptUri']
                return uri  # Return the full URI from conceptUri column
            
            # Then try altLabels if exists (comma-separated)
            if 'altLabels' in df.columns:
                for _, row in df.iterrows():
                    if pd.notna(row['altLabels']):
                        alt_labels = [label.strip().lower() for label in str(row['altLabels']).split(',')]
                        if skill_name.lower() in alt_labels:
                            return row['conceptUri']
            
            return ""  # Return empty if no match found
            
        except Exception as e:
            logging.error(f"URL lookup failed for {skill_name}: {str(e)}")
            return ""

class KeywordUtils:
    """Keyword extraction and processing utilities"""
    
    @staticmethod
    def extract_keywords(
        text: str, 
        model: KeyBERT,
        top_n: int = 18,
        ngram_range: tuple = (1, 3)
    ) -> List[str]:
        """
        Extract meaningful multi-word phrases as keywords
        Returns:
            List of keywords sorted by relevance (score desc, then length desc)
        """
        preprocessed = TextPreprocessor.preprocess_for_keywords(text)
        keywords = model.extract_keywords(
            preprocessed,
            keyphrase_ngram_range=ngram_range,
            stop_words='english',
            use_mmr=False,
            top_n=top_n*2  # Get more candidates for filtering
        )
        
        # Filter and prioritize multi-word phrases
        filtered = []
        for kw, score in keywords:
            words = kw.split()
            filtered.append((kw, score * (1 + 0.2*len(words))))
        
        # Sort by adjusted score descending, then phrase length descending
        filtered_sorted = sorted(filtered, key=lambda x: (-x[1], -len(x[0].split())))
        return [kw for kw, _ in filtered_sorted][:top_n]

    @staticmethod
    def deduplicate_keywords(keywords: List[str]) -> List[str]:
        """Remove redundant keyword variations"""
        seen = set()
        unique = []
        for kw in keywords:
            normalized = ' '.join(sorted(kw.split()))
            if normalized not in seen:
                seen.add(normalized)
                unique.append(kw)
        return unique

    @staticmethod
    def filter_frequent_keywords(
        skill_keywords: List[List[str]], 
        threshold: float = 0.5
    ) -> List[List[str]]:
        """Remove overly frequent keywords"""
        keyword_counts = defaultdict(int)
        total_skills = len(skill_keywords)
        
        for kws in skill_keywords:
            for kw in set(kws):
                keyword_counts[kw] += 1
        
        max_allowed = threshold * total_skills
        return [
            [kw for kw in kws if keyword_counts[kw] <= max_allowed]
            for kws in skill_keywords
        ]

    @staticmethod
    def keyword_similarity(query: str, skill_keywords: List[List[str]]) -> List[Tuple[int, float]]:
        """
        Compare query to pre-calculated skill keywords (mostly bi-grams)
        Returns: List of (skill_index, score) tuples sorted by relevance
        
        Args:
            query: Input text to match against skills
            skill_keywords: List of keyword lists (one per skill)
            
        Process:
        1. Extract bi-grams from query using same method as skill keywords
        2. For each skill's keywords:
           - Find best fuzzy match for each query keyword
           - Score matches above 65% similarity threshold
           - Weight scores by match quality and keyword length
        3. Normalize scores by number of query keywords
        4. Return sorted results by descending score
        """
        if not skill_keywords:
            return []

        # Extract bi-grams from query using same method as skill keywords
        query_keywords = KeywordUtils.extract_keywords(
            query,
            KeyBERT(SentenceTransformer('all-mpnet-base-v2')),
            top_n=15,
            ngram_range=(2, 2)  # Focus only on bi-grams to match skill_keywords format
        )
        
        if not query_keywords:
            return []
            
        results = []
        for i, skill_kws in enumerate(skill_keywords):
            if not skill_kws:
                continue
                
            # Calculate overlap between query bi-grams and skill bi-grams
            overlap_score = 0.0
            for q_kw in query_keywords:
                # Find best matching skill keyword
                best_match = process.extractOne(
                    q_kw.lower(),
                    [sk_kw.lower() for sk_kw in skill_kws],
                    scorer=fuzz.token_sort_ratio
                )
                if best_match and best_match[1] >= 65:  # Minimum 65% match
                    # Weight by match quality and keyword length
                    overlap_score += best_match[1] * (1 + len(q_kw.split())/10.0)
            
            if overlap_score > 0:
                # Normalize by number of query keywords
                normalized_score = overlap_score / len(query_keywords)
                results.append((i, float(normalized_score)))
            
        return sorted(results, key=lambda x: x[1], reverse=True)

class EmbeddingUtils:
    """Embedding-related utilities"""
    
    @staticmethod
    def compute_embeddings(
        text: str, 
        model,
        preprocess: bool = True
    ) -> np.ndarray:
        """Generate embeddings for text"""
        if preprocess:
            text = TextPreprocessor.preprocess_for_embeddings(text)
        return model.encode([text], convert_to_tensor=False)[0]

    @staticmethod
    def calculate_similarity(
        emb1: np.ndarray, 
        emb2: np.ndarray
    ) -> float:
        """Compute cosine similarity between embeddings"""
        return cosine_similarity([emb1], [emb2])[0][0]

    @staticmethod
    def batch_similarity(
        query_embedding: np.ndarray,
        embeddings: List[np.ndarray]
    ) -> List[Tuple[int, float]]:
        """
        Calculate similarities for a batch of embeddings
        Returns: List of (index, similarity_score) tuples
        """
        similarities = []
        embeddings_array = np.vstack(embeddings)
        scores = cosine_similarity([query_embedding], embeddings_array)[0]
        
        for i, score in enumerate(scores):
            similarities.append((i, float(score)))
            
        return sorted(similarities, key=lambda x: x[1], reverse=True)

class NLPSkillFinder:
    """Combined NLP approach for skill recommendation"""
    
    def __init__(self, df: pd.DataFrame, model_path: str = 'all-mpnet-base-v2'):
        """
        Initialize with processed DataFrame
        
        Args:
            df: DataFrame with processed skills (must have embeddings, keywords, cluster)
            model_path: SentenceTransformer model path
        """
        self.df = df
        self.model = SentenceTransformer(model_path)
        
    def _analyze_query(self, query: str) -> dict:
        """Analyze query text features for dynamic weighting"""
        features = {
            'length': len(query),
            'word_count': len(query.split()),
            'keyword_density': len(KeywordUtils.extract_keywords(query, KeyBERT(self.model))) / max(1, len(query.split()))
        }
        return features
        
    def _get_dynamic_weights(self, features: dict) -> dict:
        """Determine weights based on query features"""
        if features['length'] < 50:
            return {'embedding': 0.4, 'keyword': 0.5, 'cluster': 0.1}
        elif features['keyword_density'] > 0.3:
            return {'embedding': 0.5, 'keyword': 0.4, 'cluster': 0.1}
        else:
            return {'embedding': 0.7, 'keyword': 0.2, 'cluster': 0.1}
            
    def _phase1_recall(self, query: str, top_n: int = 50) -> List[Tuple[int, float]]:
        """Broad recall phase with relaxed thresholds"""
        # Get embedding matches
        query_embedding = EmbeddingUtils.compute_embeddings(query, self.model)
        embedding_matches = EmbeddingUtils.batch_similarity(
            query_embedding, 
            self.df['embeddings'].tolist()
        )[:top_n*2]
        
        # Get keyword matches
        keyword_matches = KeywordUtils.keyword_similarity(
            query, 
            self.df['skill_keywords'].tolist()
        )[:top_n*2]
        
        return embedding_matches, keyword_matches
        
    def _phase2_refine(self, candidates: List[int], query: str, weights: dict) -> List[Tuple[int, float]]:
        """Precision refinement phase"""
        refined = []
        query_embedding = EmbeddingUtils.compute_embeddings(query, self.model)
        
        for idx in candidates:
            row = self.df.iloc[idx]
            emb_score = cosine_similarity(
                [query_embedding],
                [row['embeddings']]
            )[0][0]
            
            kw_similarity = KeywordUtils.keyword_similarity(query, [row['skill_keywords']])
            kw_score = max(kw_similarity[0][1] if kw_similarity else 0, 0)
            
            cluster_score = 1.0 if 'cluster' in row and row['cluster'] in self._get_relevant_clusters(query) else 0.0
            
            combined = (
                weights['embedding'] * emb_score +
                weights['keyword'] * kw_score +
                weights.get('cluster', 0) * cluster_score
            )
            refined.append((idx, combined))
            
        return sorted(refined, key=lambda x: x[1], reverse=True)
        
    def _get_relevant_clusters(self, query: str, n_clusters: int = 3) -> set:
        """Identify most relevant clusters for query"""
        if not hasattr(self, 'cluster_centers_'):
            # Get cluster centers from KMeans if available in the dataframe
            if 'cluster' in self.df.columns:
                # Calculate cluster centers from embeddings
                cluster_centers = []
                for cluster_id in sorted(self.df['cluster'].unique()):
                    cluster_embeddings = np.vstack(
                        self.df[self.df['cluster'] == cluster_id]['embeddings'].values
                    )
                    cluster_centers.append(cluster_embeddings.mean(axis=0))
                self.cluster_centers_ = cluster_centers
            else:
                # Fallback to empty set if no clusters available
                return set()
                
        query_embedding = EmbeddingUtils.compute_embeddings(query, self.model)
        similarities = [
            (i, cosine_similarity([query_embedding], [center])[0][0])
            for i, center in enumerate(self.cluster_centers_)
        ]
        return {i for i, _ in sorted(similarities, key=lambda x: x[1], reverse=True)[:n_clusters]}
        
    def find_skills(
        self, 
        query: str, 
        top_n: int = 20,
        embedding_weight: float = None,
        keyword_weight: float = None,
        cluster_weight: float = None
    ) -> List[str]:
        """Two-phase skill finding with recall + refinement"""
        # Phase 1: Broad recall
        embedding_matches, keyword_matches = self._phase1_recall(query, top_n*3)
        
        # Combine and deduplicate candidates
        candidate_indices = set()
        for idx, _ in embedding_matches[:top_n*2]:
            candidate_indices.add(idx)
        for idx, _ in keyword_matches[:top_n*2]:
            candidate_indices.add(idx)
            
        # Phase 2: Precision refinement
        features = self._analyze_query(query)
        weights = self._get_dynamic_weights(features)
        refined = self._phase2_refine(list(candidate_indices), query, weights)
        """Find skills with dynamic weighting based on query characteristics"""
        # Set default weights if not provided
        if embedding_weight is None or keyword_weight is None or cluster_weight is None:
            features = self._analyze_query(query)
            weights = self._get_dynamic_weights(features)
            embedding_weight = weights['embedding']
            keyword_weight = weights['keyword'] 
            cluster_weight = weights.get('cluster', 0)
        """
        Find skills using combined NLP approaches
        
        Args:
            query: User query string
            top_n: Number of skills to return
            embedding_weight: Weight for embedding similarity
            keyword_weight: Weight for keyword similarity
            cluster_weight: Weight for cluster membership
            
        Returns:
            List of relevant skill names
        """
        # 1. Get embedding similarity scores
        query_embedding = EmbeddingUtils.compute_embeddings(query, self.model)
        embedding_results = EmbeddingUtils.batch_similarity(
            query_embedding, 
            self.df['embeddings'].tolist()
        )
        
        # 2. Get keyword similarity scores
        keyword_results = KeywordUtils.keyword_similarity(
            query, 
            self.df['skill_keywords'].tolist()
        )
        
        # 3. Determine most relevant clusters
        top_indices = [idx for idx, _ in embedding_results[:50]]
        cluster_counts = self.df.iloc[top_indices]['cluster'].value_counts()
        relevant_clusters = set(cluster_counts.nlargest(3).index.tolist())
        
        # 4. Compute combined scores
        combined_scores = {}
        for i in range(len(self.df)):
            emb_score = next((score for idx, score in embedding_results if idx == i), 0)
            kw_score = next((score for idx, score in keyword_results if idx == i), 0)
            cluster_score = 1.0 if self.df.iloc[i]['cluster'] in relevant_clusters else 0.0
            
            # Weighted combination
            combined_scores[i] = (
                embedding_weight * emb_score +
                keyword_weight * kw_score +
                cluster_weight * cluster_score
            )
        
        # 5. Get top skills
        top_indices = sorted(combined_scores.keys(), 
                           key=lambda idx: combined_scores[idx], 
                           reverse=True)[:top_n]
        
        return self.df.iloc[top_indices]['preferredLabel'].tolist()

class DeepseekUtils:
    """Ultra-optimized ESCO recommender with fuzzy matching and runtime validation"""
    
    def __init__(self, processed_skills_path: str, api_key: str = None):
        """
        Args:
            processed_skills_path: Path to your processed skills file (pkl/json)
            api_key: Deepseek API key (falls back to DEEPSEEK_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        # Load and index skills
        self.processed_skills = self._load_skills(processed_skills_path)
        self._skill_index = self._build_case_insensitive_index()

    def _load_skills(self, path: str) -> set[str]:
        """Load all skill names from processed file"""
        if path.endswith('.pkl'):
            df = pd.read_pickle(path)
        elif path.endswith('.json'):
            df = pd.read_json(path)
        else:
            raise ValueError("File must be .pkl or .json")
        
        # Extract all skill name variations
        skills = set(df['preferredLabel'].dropna())
        if 'altLabels' in df.columns:
            skills.update(df['altLabels'].explode().dropna())
        if 'hiddenLabels' in df.columns:
            skills.update(df['hiddenLabels'].explode().dropna())
        return skills

    def _build_case_insensitive_index(self) -> Dict[str, str]:
        """Create {lowercase: original} mapping for 100+ faster lookups"""
        return {skill.lower(): skill for skill in self.processed_skills}

    @lru_cache(maxsize=10_000)
    def _validate_skill(self, skill_name: str, fuzzy_threshold: int = 0) -> Optional[str]:
        """
        Validate skill with optional fuzzy matching
        Args:
            fuzzy_threshold: 0-100, 0=exact match only
        Returns:
            Original ESCO skill name if valid, None otherwise
        """
        # Fast exact match check
        normalized = skill_name.lower()
        if normalized in self._skill_index:
            return self._skill_index[normalized]
        
        # Fuzzy match if enabled
        if fuzzy_threshold > 0:
            best_match, best_score = None, 0
            for esco_skill in self.processed_skills:
                score = fuzz.ratio(normalized, esco_skill.lower())
                if score > best_score and score >= fuzzy_threshold:
                    best_match, best_score = esco_skill, score
            return best_match
        
        return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _query_deepseek(self, prompt: str) -> Dict:
        """Base API call with system role enforcement for ESCO compliance"""
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": """You are an ESCO taxonomy expert specializing in skills classification across all domains.
                    
                    Your tasks:
                    1. Identify relevant skills from the ESCO framework
                    2. RANK skills from most to least relevant
                    3. REMOVE any skills that are not applicable
                    4. Maintain exact ESCO skill names
                    5. Also consider:
                       - Transferable skills
                       - Soft skills
                       - Other domain-specific skills"""
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 500,
            "response_format": {"type": "json_object"}
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()

    async def get_pure_recommendations(
        self,
        input_text: str,
        max_results: int = 10,
        fuzzy_threshold: int = 85
    ) -> List[str]:
        """
        Mode 1: User input only
        Args:
            input_text: User request for skills
            fuzzy_threshold: 0-100 similarity score (0=exact match only)
        """
        prompt = f"""Analyze this request and identify the most relevant skills from the ESCO framework:
        
        Request: "{input_text}"
        
        Return up to {max_results} official ESCO skills that best match this request.
        
        Requirements:
        - ONLY return exact ESCO skill names
        - RANK from most to least relevant
        - Focus on both domain-specific and transferable skills
        - JSON format: {{"skills": ["most_relevant_skill", "second_most", ...]}}
        """
        
        response = await self._query_deepseek(prompt)
        return self._parse_response(response, fuzzy_threshold=fuzzy_threshold, max_results=max_results)

    async def get_hybrid_recommendations(
        self,
        input_text: str,
        nlp_results: List[str],
        max_results: int = 10,
        fuzzy_threshold: int = 0
    ) -> List[str]:
        """
        Improved Mode 2: Pass user input + all NLP results to LLM for re-ranking
        
        Args:
            input_text: User query
            nlp_results: Results from NLP approach
            max_results: Maximum results to return
            fuzzy_threshold: Fuzzy matching threshold for validation
        """
        prompt = f"""Course Description: "{input_text}"
        
        Analyze this course description and identify the most relevant skills from this list:
        {json.dumps(nlp_results, indent=2)}
        
        Instructions:
        1. RANK skills from most to least relevant for the course description
        2. REMOVE any skills that are not applicable or relevant to the course description
        3. FOCUS on both domain-specific and transferable skills
        
        JSON format: {{"skills": ["most_relevant_skill", "second_most_relevant", ...]}}
        """
        
        response = await self._query_deepseek(prompt)
        return self._parse_response(
            response,
            fuzzy_threshold=fuzzy_threshold,  # Validates against ESCO
            max_results=max_results
        )

    def _parse_response(
        self,
        response: Dict,
        context_skills: Optional[List[str]] = None,
        fuzzy_threshold: int = 0,
        max_results: int = 10
    ) -> List[str]:
        """
        Validate and filter API response
        Args:
            context_skills: If provided, skills must be subset of these
            fuzzy_threshold: 0-100 similarity score
            max_results: Maximum number of results to return
        """
        try:
            content = json.loads(response["choices"][0]["message"]["content"])
            raw_skills = content["skills"]
            
            validated = []
            for skill in raw_skills:
                # Validate against ESCO taxonomy
                esco_skill = self._validate_skill(skill, fuzzy_threshold)
                if not esco_skill:
                    continue
                
                # Validate against context if provided
                if context_skills:
                    if fuzzy_threshold > 0:
                        # Fuzzy match against context
                        context_match = None
                        for candidate in context_skills:
                            if fuzz.ratio(esco_skill.lower(), candidate.lower()) >= fuzzy_threshold:
                                context_match = candidate
                                break
                        if not context_match:
                            continue
                    elif esco_skill not in context_skills:
                        continue
                
                validated.append(esco_skill)
                if len(validated) >= max_results:
                    break
            
            return validated
            
        except (KeyError, json.JSONDecodeError) as e:
            logging.error(f"Response parsing failed: {e}")
            return []
