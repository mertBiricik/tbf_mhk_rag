"""
Advanced Retrieval System for Basketball RAG
Implements optimized retrieval strategies for better performance and accuracy.
"""

import time
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json

import torch
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class RetrievalResult:
    content: str
    metadata: Dict[str, Any]
    score: float
    rerank_score: Optional[float] = None

class QueryExpander:
    """Expand queries with basketball-specific terminology."""
    
    def __init__(self):
        self.basketball_expansions = {
            # Turkish expansions
            'faul': ['faul', 'kişisel faul', 'ihlal', 'penalty'],
            'saha': ['saha', 'koridor', 'alan', 'court'],
            'şut': ['şut', 'atış', 'basket', 'shot'],
            'oyuncu': ['oyuncu', 'basketbolcu', 'player'],
            'zaman': ['zaman', 'süre', 'dakika', 'saniye', 'time'],
            
            # English expansions
            'foul': ['foul', 'personal foul', 'violation', 'penalty'],
            'court': ['court', 'field', 'area', 'saha'],
            'shot': ['shot', 'basket', 'scoring', 'şut'],
            'player': ['player', 'athlete', 'oyuncu'],
            'time': ['time', 'duration', 'minute', 'second', 'zaman']
        }
    
    def expand_query(self, query: str) -> str:
        """Expand query with relevant basketball terms."""
        expanded_terms = []
        query_lower = query.lower()
        
        for base_term, expansions in self.basketball_expansions.items():
            if base_term in query_lower:
                # Add most relevant expansion
                expanded_terms.extend(expansions[:2])
        
        if expanded_terms:
            return f"{query} {' '.join(set(expanded_terms))}"
        return query

class HybridReranker:
    """Re-rank results using multiple signals."""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # Keep basketball-specific terms
            ngram_range=(1, 2)
        )
        self.is_fitted = False
    
    def fit(self, documents: List[str]):
        """Fit TF-IDF vectorizer on document corpus."""
        self.tfidf_vectorizer.fit(documents)
        self.is_fitted = True
    
    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Re-rank results using hybrid scoring."""
        if not self.is_fitted or not results:
            return results
        
        # Extract texts for TF-IDF
        texts = [result.content for result in results]
        
        try:
            # TF-IDF similarity
            query_tfidf = self.tfidf_vectorizer.transform([query])
            docs_tfidf = self.tfidf_vectorizer.transform(texts)
            
            tfidf_scores = cosine_similarity(query_tfidf, docs_tfidf)[0]
            
            # Combine with semantic scores
            for i, result in enumerate(results):
                semantic_score = result.score
                tfidf_score = tfidf_scores[i]
                
                # Weighted combination (favor semantic for basketball domain)
                combined_score = 0.7 * semantic_score + 0.3 * tfidf_score
                
                # Boost for basketball-specific metadata
                if self._is_basketball_relevant(result.metadata):
                    combined_score *= 1.1
                
                result.rerank_score = combined_score
            
            # Sort by rerank score
            results.sort(key=lambda x: x.rerank_score or x.score, reverse=True)
            
        except Exception as e:
            logging.warning(f"Reranking failed: {e}")
        
        return results
    
    def _is_basketball_relevant(self, metadata: Dict[str, Any]) -> bool:
        """Check if metadata indicates high basketball relevance."""
        doc_type = metadata.get('doc_type', '').lower()
        year = metadata.get('year', 0)
        
        # Boost recent rule changes
        if doc_type == 'changes' and year >= 2024:
            return True
        
        # Boost interpretations
        if doc_type == 'interpretations':
            return True
        
        return False

class CachedRetriever:
    """Add caching layer for frequent queries."""
    
    def __init__(self, cache_size: int = 100):
        self.cache_size = cache_size
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _hash_query(self, query: str, top_k: int) -> str:
        """Create hash for query caching."""
        content = f"{query.lower().strip()}_{top_k}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached(self, query: str, top_k: int) -> Optional[List[RetrievalResult]]:
        """Get cached results if available."""
        cache_key = self._hash_query(query, top_k)
        
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]['results']
        
        self.cache_misses += 1
        return None
    
    def cache_results(self, query: str, top_k: int, results: List[RetrievalResult]):
        """Cache query results."""
        cache_key = self._hash_query(query, top_k)
        
        # LRU eviction if cache full
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[cache_key] = {
            'results': results,
            'timestamp': time.time()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }

class AdvancedBasketballRetriever:
    """Advanced retrieval system with multiple optimization strategies."""
    
    def __init__(self, 
                 vector_db_path: str = "./vector_db/chroma_db",
                 collection_name: str = "basketball_rules",
                 embedding_model_name: str = "BAAI/bge-m3",
                 enable_caching: bool = True,
                 enable_reranking: bool = True,
                 enable_query_expansion: bool = True):
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.query_expander = QueryExpander() if enable_query_expansion else None
        self.cached_retriever = CachedRetriever() if enable_caching else None
        self.reranker = HybridReranker() if enable_reranking else None
        
        # Load embedding model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model.to(device)
        
        # Connect to vector database
        client = chromadb.PersistentClient(path=vector_db_path)
        self.collection = client.get_collection(name=collection_name)
        
        # Initialize reranker if enabled
        if self.reranker:
            self._initialize_reranker()
        
        self.logger.info(f"AdvancedRetriever initialized with {self.collection.count()} documents")
    
    def _initialize_reranker(self):
        """Initialize reranker with document corpus."""
        try:
            # Get all documents for TF-IDF fitting
            all_docs = self.collection.get()
            
            # Handle different ChromaDB return formats
            documents = None
            if isinstance(all_docs, dict) and 'documents' in all_docs:
                documents = all_docs['documents']
            elif hasattr(all_docs, 'documents'):
                documents = all_docs.documents
            
            if documents and len(documents) > 0:
                # Flatten documents if they're nested
                if isinstance(documents[0], list):
                    documents = [doc for sublist in documents for doc in sublist]
                
                self.reranker.fit(documents)
                self.logger.info(f"Reranker initialized with {len(documents)} documents")
            else:
                self.logger.warning("No documents found for reranker initialization")
                
        except Exception as e:
            self.logger.warning(f"Reranker initialization failed: {e}")
            # Continue without reranker rather than failing
    
    def retrieve(self, 
                query: str, 
                top_k: int = 7,
                score_threshold: float = 0.3,
                use_expansion: bool = True,
                use_cache: bool = True,
                use_reranking: bool = True) -> List[RetrievalResult]:
        """
        Advanced retrieval with multiple optimization strategies.
        
        Args:
            query: User query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            use_expansion: Whether to expand query
            use_cache: Whether to use caching
            use_reranking: Whether to re-rank results
        
        Returns:
            List of retrieval results
        """
        start_time = time.time()
        
        # Check cache first
        if use_cache and self.cached_retriever:
            cached_results = self.cached_retriever.get_cached(query, top_k)
            if cached_results:
                self.logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_results
        
        # Expand query if enabled
        expanded_query = query
        if use_expansion and self.query_expander:
            expanded_query = self.query_expander.expand_query(query)
            if expanded_query != query:
                self.logger.debug(f"Query expanded: {query} -> {expanded_query}")
        
        # Generate embedding and search
        query_embedding = self.embedding_model.encode([expanded_query])
        
        search_results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(top_k * 2, 20),  # Get more for reranking
            include=['documents', 'metadatas', 'distances']
        )
        
        # Convert to RetrievalResult objects
        results = []
        if search_results['documents'] and search_results['documents'][0]:
            for doc, metadata, distance in zip(
                search_results['documents'][0],
                search_results['metadatas'][0],
                search_results['distances'][0]
            ):
                # Convert distance to similarity score
                similarity_score = 1.0 - distance
                
                if similarity_score >= score_threshold:
                    results.append(RetrievalResult(
                        content=doc,
                        metadata=metadata,
                        score=similarity_score
                    ))
        
        # Re-rank if enabled
        if use_reranking and self.reranker and results:
            results = self.reranker.rerank(query, results)
        
        # Limit to top_k
        results = results[:top_k]
        
        # Cache results
        if use_cache and self.cached_retriever:
            self.cached_retriever.cache_results(query, top_k, results)
        
        retrieval_time = time.time() - start_time
        self.logger.debug(f"Retrieved {len(results)} results in {retrieval_time:.3f}s")
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get retrieval performance statistics."""
        stats = {
            'total_documents': self.collection.count(),
            'embedding_model': self.embedding_model._modules['0'].config_name if hasattr(self.embedding_model, '_modules') else 'unknown'
        }
        
        if self.cached_retriever:
            stats['cache'] = self.cached_retriever.get_stats()
        
        return stats
    
    def warm_up_cache(self, common_queries: List[str]):
        """Pre-populate cache with common queries."""
        if not self.cached_retriever:
            return
        
        self.logger.info(f"Warming up cache with {len(common_queries)} queries...")
        for query in common_queries:
            self.retrieve(query, use_cache=False)  # Don't use cache for warmup
        
        self.logger.info("Cache warmup completed") 