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
from typing import List, Dict, Optional, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import httpx
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import lru_cache
from thefuzz import fuzz

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
    
    def __init__(self):
        """Initialize with models"""
        self.st_model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
        self.kw_model = KeyBERT(model='all-mpnet-base-v2')
    
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
        df['combined_text'] = df.apply(self.combine_text_fields, axis=1)
        df['skill_keywords'] = self._extract_keywords_batch(df['combined_text'])
        df['embeddings'] = self._generate_embeddings(df['combined_text'])
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

class KeywordUtils:
    """Keyword extraction and processing utilities"""
    
    @staticmethod
    def extract_keywords(
        text: str, 
        model: KeyBERT,
        top_n: int = 10,
        ngram_range: tuple = (1, 2),
        diversity: float = 0.6
    ) -> List[str]:
        """Extract keywords with reduced redundancy"""
        preprocessed = TextPreprocessor.preprocess_for_keywords(text)
        keywords = model.extract_keywords(
            preprocessed,
            keyphrase_ngram_range=ngram_range,
            stop_words='english',
            use_mmr=True,
            diversity=diversity,
            top_n=top_n*2
        )
        raw_keywords = [kw[0] for kw in keywords]
        return KeywordUtils.deduplicate_keywords(raw_keywords)[:top_n]

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
        embeddings: np.ndarray
    ) -> np.ndarray:
        """Calculate similarities for a batch of embeddings"""
        return cosine_similarity([query_embedding], embeddings)[0]

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
        skills.update(df['altLabels'].explode().dropna())
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
                    "content": """You are a strict ESCO taxonomy expert specializing in programming skills. 
                    Rules:
                    1. Only recommend skills actually relevant to the request
                    2. Never recommend obscure/unrelated technologies 
                    3. Maintain exact ESCO skill names"""
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
        teacher_input: str,
        max_results: int = 10,
        fuzzy_threshold: int = 85
    ) -> List[str]:
        """
        Mode 1: Teacher input only with fuzzy validation
        Args:
            fuzzy_threshold: 0-100 similarity score (0=exact match only)
        """
        prompt = f"""Return up to {max_results} official ESCO skills for: "{teacher_input}"
        Requirements:
        - ONLY exact ESCO skill names
        - JSON format: {{"skills": ["skill1", ...]}}
        """
        
        response = await self._query_deepseek(prompt)
        return self._parse_response(response, fuzzy_threshold=fuzzy_threshold, max_results=max_results)

    async def get_hybrid_recommendations(
        self,
        teacher_input: str,
        nlp_cluster_skills: List[str],
        max_results: int = 10,
        fuzzy_threshold: int = 0
    ) -> List[str]:
        """
        Improved Mode 2: Teacher input + NLP context with smarter filtering
        """
        # Step 1: Pre-filter skills by relevance to input
        input_lower = teacher_input.lower()
        filtered_skills = [
            skill for skill in nlp_cluster_skills 
            if ("python" in input_lower and "python" in skill.lower()) or
            fuzz.partial_ratio(input_lower, skill.lower()) > 60
        ]
        
        # Step 2: Limit to top 50 most relevant skills to avoid overwhelming LLM
        filtered_skills = sorted(
            filtered_skills,
            key=lambda x: fuzz.token_set_ratio(input_lower, x.lower()),
            reverse=True
        )[:10]
        
        prompt = f"""From these programming skills related to {teacher_input}:
        {json.dumps(filtered_skills, indent=2)}
        
        Select the top {max_results} most specifically relevant skills.
        Rules:
        1. Prioritize skills mentioning "Python" explicitly
        2. Then consider general programming concepts
        3. Exclude completely unrelated technologies
        4. JSON format: {{"skills": ["skill1", ...]}}
        """
        
        response = await self._query_deepseek(prompt)
        return self._parse_response(response, context_skills=filtered_skills, fuzzy_threshold=fuzzy_threshold)

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