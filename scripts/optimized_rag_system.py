#!/usr/bin/env python3
"""
Optimized Basketball RAG System
Integrates all performance optimizations and advanced features.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.advanced_retriever import AdvancedBasketballRetriever, RetrievalResult
from src.generation.optimized_generator import OptimizedGenerator, GenerationRequest, GenerationResponse
from src.document_processing.optimized_processor import OptimizedDocumentProcessor
from src.utils.config import Config

class OptimizedBasketballRAG:
    """
    Optimized Basketball RAG System with all performance enhancements.
    
    Features:
    - Advanced retrieval with query expansion, re-ranking, and caching
    - Optimized document processing with semantic chunking
    - Enhanced generation with specialized prompts and response caching
    - Comprehensive performance monitoring
    """
    
    def __init__(self, config_path: str = "config/optimized_config.yaml"):
        self.logger = self._setup_logging()
        
        # Load optimized configuration
        self.config = Config(config_path)
        
        # Initialize optimized components
        self.retriever = None
        self.generator = None
        self.processor = None
        
        # Performance tracking
        self.query_count = 0
        self.total_response_time = 0.0
        self.cache_hit_count = 0
        
        self.logger.info("OptimizedBasketballRAG initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup optimized logging."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler
            os.makedirs("logs", exist_ok=True)
            file_handler = logging.FileHandler("logs/optimized_rag.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def initialize_system(self):
        """Initialize all optimized components."""
        self.logger.info("🚀 Initializing Optimized Basketball RAG System...")
        
        # Initialize advanced retriever
        self._initialize_retriever()
        
        # Initialize optimized generator
        self._initialize_generator()
        
        # Initialize optimized processor
        self._initialize_processor()
        
        # System warmup
        self._warm_up_system()
        
        self.logger.info("✅ Optimized system initialization complete")
    
    def _initialize_retriever(self):
        """Initialize advanced retrieval system."""
        self.logger.info("📊 Initializing advanced retriever...")
        
        retrieval_config = self.config.get('retrieval', {})
        
        self.retriever = AdvancedBasketballRetriever(
            vector_db_path=self.config.get('vector_db.persist_directory', './vector_db/chroma_db'),
            collection_name=self.config.get('vector_db.collection_name', 'basketball_rules'),
            embedding_model_name=self.config.get('models.embeddings.name', 'BAAI/bge-m3'),
            enable_caching=retrieval_config.get('enable_caching', True),
            enable_reranking=retrieval_config.get('enable_reranking', True),
            enable_query_expansion=retrieval_config.get('enable_query_expansion', True)
        )
        
        self.logger.info("✅ Advanced retriever initialized")
    
    def _initialize_generator(self):
        """Initialize optimized generation system."""
        self.logger.info("🧠 Initializing optimized generator...")
        
        generation_config = self.config.get('generation', {})
        
        self.generator = OptimizedGenerator(
            llm_model=self.config.get('models.llm.name', 'llama3.1:8b-instruct-q4_K_M'),
            base_url=self.config.get('models.llm.base_url', 'http://localhost:11434'),
            enable_caching=generation_config.get('enable_caching', True),
            enable_streaming=generation_config.get('enable_streaming', False)
        )
        
        self.logger.info("✅ Optimized generator initialized")
    
    def _initialize_processor(self):
        """Initialize optimized document processor."""
        self.logger.info("📚 Initializing optimized processor...")
        
        self.processor = OptimizedDocumentProcessor(self.config.config)
        
        self.logger.info("✅ Optimized processor initialized")
    
    def _warm_up_system(self):
        """Warm up the system for optimal performance."""
        self.logger.info("🔥 Warming up system...")
        
        warmup_queries = self.config.get('performance.warmup_queries', [
            "5 faul yapan oyuncuya ne olur?",
            "Basketbol sahasının boyutları nelerdir?",
            "What happens when a player gets 5 fouls?"
        ])
        
        # Warm up retriever
        if self.retriever:
            self.retriever.warm_up_cache(warmup_queries)
        
        # Warm up generator
        if self.generator:
            self.generator.warm_up(warmup_queries)
        
        self.logger.info("✅ System warmup complete")
    
    def process_query(self, 
                     query: str,
                     language: str = 'auto',
                     use_optimizations: bool = True) -> Dict[str, Any]:
        """
        Process query with all optimizations.
        
        Args:
            query: User question
            language: Language preference ('auto', 'turkish', 'english')
            use_optimizations: Whether to use all optimization features
            
        Returns:
            Dictionary with answer, sources, metadata, and performance info
        """
        start_time = time.time()
        
        try:
            # Step 1: Advanced Retrieval
            retrieval_start = time.time()
            retrieval_results = self.retriever.retrieve(
                query=query,
                top_k=self.config.get('retrieval.top_k', 7),
                score_threshold=self.config.get('retrieval.score_threshold', 0.3),
                use_expansion=use_optimizations,
                use_cache=use_optimizations,
                use_reranking=use_optimizations
            )
            retrieval_time = time.time() - retrieval_start
            
            if not retrieval_results:
                return self._create_no_results_response(query, language)
            
            # Convert to format expected by generator
            context_chunks = [
                {
                    'content': result.content,
                    'metadata': result.metadata,
                    'score': result.rerank_score or result.score
                }
                for result in retrieval_results
            ]
            
            # Step 2: Optimized Generation
            generation_start = time.time()
            generation_request = GenerationRequest(
                query=query,
                context_chunks=context_chunks,
                language=language,
                max_tokens=self.config.get('models.llm.max_tokens', 2048),
                temperature=self.config.get('models.llm.temperature', 0.1),
                use_cache=use_optimizations
            )
            
            generation_response = self.generator.generate(generation_request)
            generation_time = time.time() - generation_start
            
            # Step 3: Compile response
            total_time = time.time() - start_time
            
            # Update performance tracking
            self.query_count += 1
            self.total_response_time += total_time
            if generation_response.from_cache:
                self.cache_hit_count += 1
            
            response = {
                'answer': generation_response.answer,
                'sources': generation_response.sources,
                'language': generation_response.language,
                'confidence_score': generation_response.confidence_score,
                'performance': {
                    'total_time': total_time,
                    'retrieval_time': retrieval_time,
                    'generation_time': generation_time,
                    'retrieved_chunks': len(retrieval_results),
                    'from_cache': generation_response.from_cache
                },
                'metadata': {
                    'query_expansion_used': use_optimizations,
                    'reranking_used': use_optimizations,
                    'optimization_enabled': use_optimizations
                }
            }
            
            self.logger.info(f"Query processed in {total_time:.3f}s (retrieval: {retrieval_time:.3f}s, generation: {generation_time:.3f}s)")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            return self._create_error_response(query, str(e))
    
    def _create_no_results_response(self, query: str, language: str) -> Dict[str, Any]:
        """Create response when no retrieval results found."""
        if language == 'english':
            answer = "I couldn't find relevant information for your question. Please try rephrasing or asking about specific basketball rules."
        else:
            answer = "Sorunuz için uygun bilgi bulamadım. Lütfen sorunuzu yeniden ifade edin veya belirli basketbol kuralları hakkında soru sorun."
        
        return {
            'answer': answer,
            'sources': [],
            'language': language,
            'confidence_score': 0.1,
            'performance': {'total_time': 0.0, 'retrieval_time': 0.0, 'generation_time': 0.0},
            'metadata': {'error': 'no_results_found'}
        }
    
    def _create_error_response(self, query: str, error: str) -> Dict[str, Any]:
        """Create response for system errors."""
        return {
            'answer': f"System error occurred: {error}",
            'sources': [],
            'language': 'english',
            'confidence_score': 0.0,
            'performance': {'total_time': 0.0, 'retrieval_time': 0.0, 'generation_time': 0.0},
            'metadata': {'error': error}
        }
    
    def get_system_performance(self) -> Dict[str, Any]:
        """Get comprehensive system performance statistics."""
        avg_response_time = self.total_response_time / self.query_count if self.query_count > 0 else 0
        cache_hit_rate = self.cache_hit_count / self.query_count if self.query_count > 0 else 0
        
        performance = {
            'query_statistics': {
                'total_queries': self.query_count,
                'avg_response_time': avg_response_time,
                'total_response_time': self.total_response_time,
                'cache_hit_rate': cache_hit_rate
            }
        }
        
        # Add component-specific performance
        if self.retriever:
            performance['retrieval'] = self.retriever.get_performance_stats()
        
        if self.generator:
            performance['generation'] = self.generator.get_performance_stats()
        
        # Hardware information
        hardware_info = self.config.get_hardware_info()
        performance['hardware'] = hardware_info
        
        return performance
    
    def benchmark_system(self, test_queries: List[str]) -> Dict[str, Any]:
        """Run benchmark tests on the optimized system."""
        self.logger.info(f"🔍 Running benchmark with {len(test_queries)} queries...")
        
        results = []
        start_time = time.time()
        
        for i, query in enumerate(test_queries):
            self.logger.info(f"Processing query {i+1}/{len(test_queries)}: {query[:50]}...")
            
            query_start = time.time()
            result = self.process_query(query)
            query_time = time.time() - query_start
            
            results.append({
                'query': query,
                'response_time': query_time,
                'confidence': result.get('confidence_score', 0),
                'retrieved_chunks': result.get('performance', {}).get('retrieved_chunks', 0),
                'from_cache': result.get('performance', {}).get('from_cache', False)
            })
        
        total_benchmark_time = time.time() - start_time
        
        # Calculate benchmark statistics
        response_times = [r['response_time'] for r in results]
        confidences = [r['confidence'] for r in results]
        cache_hits = sum(1 for r in results if r['from_cache'])
        
        benchmark_stats = {
            'total_queries': len(test_queries),
            'total_time': total_benchmark_time,
            'avg_response_time': sum(response_times) / len(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'avg_confidence': sum(confidences) / len(confidences),
            'cache_hit_rate': cache_hits / len(test_queries),
            'queries_per_second': len(test_queries) / total_benchmark_time
        }
        
        self.logger.info(f"✅ Benchmark complete: {benchmark_stats['avg_response_time']:.3f}s avg, {benchmark_stats['queries_per_second']:.1f} QPS")
        
        return {
            'statistics': benchmark_stats,
            'detailed_results': results
        }
    
    def optimize_database(self):
        """Optimize the vector database for better performance."""
        self.logger.info("🔧 Optimizing database...")
        
        if not self.processor:
            self.logger.error("Processor not initialized")
            return False
        
        try:
            # Get document configurations
            document_configs = self.config.get('basketball.document_types', {})
            
            # Process documents with optimizations
            optimized_chunks = self.processor.process_all_documents(document_configs)
            
            self.logger.info(f"✅ Database optimization complete: {len(optimized_chunks)} optimized chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Database optimization failed: {e}")
            return False

def main():
    """Main function for testing the optimized system."""
    print("🏀 Optimized Basketball RAG System")
    print("=" * 50)
    
    # Initialize optimized system
    rag = OptimizedBasketballRAG()
    rag.initialize_system()
    
    # Test queries
    test_queries = [
        "5 faul yapan oyuncuya ne olur?",
        "Basketbol sahasının boyutları nelerdir?",
        "What happens when a player gets 5 fouls?",
        "2024 yılında hangi kurallar değişti?",
        "Şut saati kuralı nasıl işler?"
    ]
    
    print("\n🧪 Testing optimized system...")
    
    # Process test queries
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        
        result = rag.process_query(query)
        
        print(f"   Answer: {result['answer'][:100]}...")
        print(f"   Time: {result['performance']['total_time']:.3f}s")
        print(f"   Confidence: {result['confidence_score']:.2f}")
        print(f"   Sources: {len(result['sources'])}")
        print(f"   From Cache: {result['performance']['from_cache']}")
    
    # Performance statistics
    print("\n📊 System Performance:")
    performance = rag.get_system_performance()
    
    query_stats = performance['query_statistics']
    print(f"   Total Queries: {query_stats['total_queries']}")
    print(f"   Avg Response Time: {query_stats['avg_response_time']:.3f}s")
    print(f"   Cache Hit Rate: {query_stats['cache_hit_rate']:.1%}")
    
    # Hardware info
    hardware = performance['hardware']
    print(f"   GPU: {hardware['gpu_name']}")
    print(f"   VRAM: {hardware['vram_gb']:.1f} GB")
    
    print("\n✅ Optimized system test complete!")

if __name__ == "__main__":
    main() 