#!/usr/bin/env python3
"""
Comprehensive Performance Evaluation for Basketball RAG System

This script evaluates:
1. RAG vs Base LLM performance comparison
2. Accuracy and quality metrics  
3. Optimization and performance analysis
4. Multilingual performance assessment

Usage:
    python scripts/evaluate_performance.py
    python scripts/evaluate_performance.py --output results/
    python scripts/evaluate_performance.py --full --benchmark
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import project modules
from src.utils.config import Config

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/evaluation.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)

class PerformanceEvaluator:
    """Comprehensive performance evaluation system."""
    
    def __init__(self, config_path: str = None):
        """Initialize the evaluator."""
        self.config = Config(config_path)
        self.logger = setup_logging()
        self.results = []
        self.start_time = time.time()
        
        # Load test queries
        self.test_queries = self._load_test_queries()
        
        # Initialize models
        self.vectorstore = None
        self.rag_chain = None
        self.base_llm = None
        
    def _load_test_queries(self) -> List[Dict]:
        """Load test queries for evaluation."""
        test_file = self.config.get('evaluation.test_queries_file', './data/test_queries.json')
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            queries = data.get('basketball_rules_queries', [])
            self.logger.info(f"✅ Loaded {len(queries)} test queries")
            return queries
        except Exception as e:
            self.logger.error(f"❌ Failed to load test queries: {e}")
            return []

    def setup_models(self):
        """Initialize RAG and base LLM models."""
        self.logger.info("🤖 Setting up models for evaluation...")
        
        try:
            # Setup Vector Database
            self.logger.info("📊 Loading vector database...")
            embedding_model = HuggingFaceEmbeddings(
                model_name=self.config.get('models.embeddings.name', 'BAAI/bge-m3'),
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            
            persist_directory = self.config.get('vector_db.persist_directory', './vector_db/chroma_db')
            if not Path(persist_directory).exists():
                self.logger.error(f"❌ Vector database not found at {persist_directory}")
                return False
                
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_model,
                collection_name=self.config.get('vector_db.collection_name', 'basketball_rules')
            )
            
            # Setup Base LLM
            self.logger.info("🧠 Setting up base LLM...")
            self.base_llm = Ollama(
                model=self.config.get('models.llm.name', 'llama3.1:8b-instruct-q4_K_M'),
                base_url=self.config.get('models.llm.base_url', 'http://localhost:11434'),
                temperature=self.config.get('models.llm.temperature', 0.1)
            )
            
            # Test base LLM connection
            test_response = self.base_llm.invoke("Test")
            if not test_response:
                self.logger.error("❌ Base LLM connection failed")
                return False
            
            # Setup RAG Chain (simplified)
            from langchain.chains import RetrievalQA
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.get('retrieval.top_k', 7)}
            )
            
            self.rag_chain = RetrievalQA.from_chain_type(
                llm=self.base_llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            self.logger.info("✅ Models setup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model setup failed: {e}")
            return False

    def evaluate_retrieval_performance(self) -> Dict[str, float]:
        """Evaluate retrieval component performance."""
        self.logger.info("🔍 Evaluating retrieval performance...")
        
        if not self.vectorstore:
            self.logger.error("❌ Vector database not available")
            return {}
        
        retrieval_times = []
        relevance_scores = []
        
        for query_data in self.test_queries[:10]:  # Sample for speed
            query = query_data.get('query_turkish', '')
            if not query:
                continue
            
            # Measure retrieval time
            start_time = time.time()
            docs = self.vectorstore.similarity_search_with_score(query, k=7)
            retrieval_time = time.time() - start_time
            retrieval_times.append(retrieval_time)
            
            # Simple relevance scoring based on score
            if docs:
                avg_score = np.mean([score for _, score in docs])
                relevance_scores.append(1.0 / (1.0 + avg_score))  # Convert distance to relevance
        
        metrics = {
            'avg_retrieval_time': np.mean(retrieval_times) if retrieval_times else 0.0,
            'min_retrieval_time': np.min(retrieval_times) if retrieval_times else 0.0,
            'max_retrieval_time': np.max(retrieval_times) if retrieval_times else 0.0,
            'avg_relevance_score': np.mean(relevance_scores) if relevance_scores else 0.0,
            'retrieval_success_rate': len(retrieval_times) / len(self.test_queries[:10])
        }
        
        self.logger.info(f"📊 Retrieval Performance: {metrics['avg_retrieval_time']:.3f}s avg time")
        return metrics

    def evaluate_rag_vs_base_llm(self) -> Dict[str, Any]:
        """Compare RAG system against base LLM performance."""
        self.logger.info("⚖️ Comparing RAG vs Base LLM performance...")
        
        if not self.rag_chain or not self.base_llm:
            self.logger.error("❌ Models not available for comparison")
            return {}
        
        comparison_results = []
        rag_wins = 0
        base_wins = 0
        ties = 0
        
        # Sample queries for comparison
        sample_queries = self.test_queries[:8]  # Limit for evaluation speed
        
        for query_data in sample_queries:
            for lang_key in ['query_turkish', 'query_english']:
                query = query_data.get(lang_key, '')
                if not query:
                    continue
                
                lang = 'turkish' if 'turkish' in lang_key else 'english'
                
                # RAG Response
                rag_start = time.time()
                try:
                    rag_response = self.rag_chain.invoke({"query": query})
                    rag_answer = rag_response.get('result', '')
                    rag_sources = rag_response.get('source_documents', [])
                    rag_time = time.time() - rag_start
                    rag_success = len(rag_answer.strip()) > 10
                except Exception as e:
                    self.logger.warning(f"RAG failed for '{query[:50]}...': {e}")
                    rag_answer = ""
                    rag_sources = []
                    rag_time = 0
                    rag_success = False
                
                # Base LLM Response
                base_start = time.time()
                try:
                    base_answer = self.base_llm.invoke(query)
                    base_time = time.time() - base_start
                    base_success = len(base_answer.strip()) > 10
                except Exception as e:
                    self.logger.warning(f"Base LLM failed for '{query[:50]}...': {e}")
                    base_answer = ""
                    base_time = 0
                    base_success = False
                
                # Quality scoring
                rag_quality = self._score_response_quality(rag_answer, rag_sources, query_data, lang)
                base_quality = self._score_response_quality(base_answer, [], query_data, lang)
                
                # Determine winner
                if rag_quality > base_quality + 1.0:  # RAG needs clear advantage
                    winner = 'rag'
                    rag_wins += 1
                elif base_quality > rag_quality + 1.0:
                    winner = 'base_llm'
                    base_wins += 1
                else:
                    winner = 'tie'
                    ties += 1
                
                result = {
                    'query': query[:100],
                    'language': lang,
                    'category': query_data.get('category', 'unknown'),
                    'rag_time': rag_time,
                    'base_time': base_time,
                    'rag_quality': rag_quality,
                    'base_quality': base_quality,
                    'rag_sources_count': len(rag_sources),
                    'rag_has_citations': '[' in rag_answer and ']' in rag_answer,
                    'winner': winner,
                    'rag_success': rag_success,
                    'base_success': base_success
                }
                
                comparison_results.append(result)
                self.results.append(result)
        
        total_comparisons = len(comparison_results)
        
        metrics = {
            'total_comparisons': total_comparisons,
            'rag_wins': rag_wins,
            'base_llm_wins': base_wins,
            'ties': ties,
            'rag_win_rate': rag_wins / total_comparisons if total_comparisons > 0 else 0,
            'base_win_rate': base_wins / total_comparisons if total_comparisons > 0 else 0,
            'tie_rate': ties / total_comparisons if total_comparisons > 0 else 0,
            'avg_rag_time': np.mean([r['rag_time'] for r in comparison_results if r['rag_time'] > 0]),
            'avg_base_time': np.mean([r['base_time'] for r in comparison_results if r['base_time'] > 0]),
            'avg_rag_quality': np.mean([r['rag_quality'] for r in comparison_results]),
            'avg_base_quality': np.mean([r['base_quality'] for r in comparison_results])
        }
        
        self.logger.info(f"🏆 RAG Win Rate: {metrics['rag_win_rate']:.1%}")
        return metrics

    def _score_response_quality(self, 
                              answer: str, 
                              sources: List, 
                              query_data: Dict, 
                              language: str) -> float:
        """Score response quality (0-10 scale)."""
        if not answer or len(answer.strip()) < 5:
            return 0.0
        
        score = 0.0
        
        # Length appropriateness
        word_count = len(answer.split())
        if 20 <= word_count <= 150:
            score += 3.0
        elif 10 <= word_count <= 200:
            score += 2.0
        elif word_count >= 5:
            score += 1.0
        
        # Has citations (good for RAG)
        if '[' in answer and ']' in answer:
            score += 2.0
        
        # Category relevance
        category = query_data.get('category', '')
        category_keywords = {
            'fouls': ['faul', 'foul', 'ceza', 'penalty'],
            'timing': ['saniye', 'second', 'süre', 'time', 'saat', 'clock'],
            'court_specifications': ['saha', 'court', 'boyut', 'dimension'],
            'rule_changes': ['değişiklik', 'change', '2024', 'yeni', 'new'],
            'substitutions': ['değişim', 'substitution', 'oyuncu', 'player']
        }
        
        keywords = category_keywords.get(category, [])
        keyword_matches = sum(1 for kw in keywords if kw.lower() in answer.lower())
        score += min(keyword_matches * 0.5, 2.0)
        
        # Source utilization (for RAG)
        if sources and len(sources) > 0:
            score += 1.0
        
        # Language consistency
        if language == 'turkish' and any(tr_word in answer.lower() 
                                       for tr_word in ['basketbol', 'oyuncu', 'faul', 'kural']):
            score += 1.0
        elif language == 'english' and any(en_word in answer.lower() 
                                         for en_word in ['basketball', 'player', 'foul', 'rule']):
            score += 1.0
        
        # Completeness (not too brief)
        if word_count >= 30:
            score += 1.0
        
        return min(score, 10.0)

    def analyze_optimization_performance(self) -> Dict[str, Any]:
        """Analyze system optimization and performance characteristics."""
        self.logger.info("⚡ Analyzing optimization performance...")
        
        # GPU Analysis
        gpu_metrics = self._analyze_gpu_performance()
        
        # Response Time Analysis
        if self.results:
            rag_times = [r['rag_time'] for r in self.results if r.get('rag_time', 0) > 0]
            base_times = [r['base_time'] for r in self.results if r.get('base_time', 0) > 0]
            
            time_metrics = {
                'rag_avg_time': np.mean(rag_times) if rag_times else 0,
                'rag_median_time': np.median(rag_times) if rag_times else 0,
                'rag_min_time': np.min(rag_times) if rag_times else 0,
                'rag_max_time': np.max(rag_times) if rag_times else 0,
                'base_avg_time': np.mean(base_times) if base_times else 0,
                'speed_improvement': (np.mean(base_times) - np.mean(rag_times)) / np.mean(base_times) 
                                   if base_times and rag_times else 0
            }
        else:
            time_metrics = {}
        
        # Quality Analysis
        if self.results:
            rag_qualities = [r['rag_quality'] for r in self.results if 'rag_quality' in r]
            base_qualities = [r['base_quality'] for r in self.results if 'base_quality' in r]
            
            quality_metrics = {
                'rag_avg_quality': np.mean(rag_qualities) if rag_qualities else 0,
                'base_avg_quality': np.mean(base_qualities) if base_qualities else 0,
                'quality_improvement': (np.mean(rag_qualities) - np.mean(base_qualities)) / 10 
                                     if rag_qualities and base_qualities else 0,
                'quality_consistency': 1 - (np.std(rag_qualities) / np.mean(rag_qualities)) 
                                     if rag_qualities and np.mean(rag_qualities) > 0 else 0
            }
        else:
            quality_metrics = {}
        
        # Source Utilization
        source_counts = [r['rag_sources_count'] for r in self.results if 'rag_sources_count' in r]
        source_metrics = {
            'avg_sources_used': np.mean(source_counts) if source_counts else 0,
            'source_utilization_rate': len([s for s in source_counts if s > 0]) / len(source_counts) 
                                     if source_counts else 0
        }
        
        # Overall optimization score
        optimization_score = self._calculate_optimization_score(time_metrics, quality_metrics, gpu_metrics)
        
        return {
            'gpu_metrics': gpu_metrics,
            'time_metrics': time_metrics,
            'quality_metrics': quality_metrics,
            'source_metrics': source_metrics,
            'optimization_score': optimization_score
        }

    def _analyze_gpu_performance(self) -> Dict[str, Any]:
        """Analyze GPU performance and utilization."""
        gpu_metrics = {}
        
        if torch.cuda.is_available():
            gpu_metrics['cuda_available'] = True
            gpu_metrics['gpu_count'] = torch.cuda.device_count()
            gpu_metrics['gpu_name'] = torch.cuda.get_device_name()
            
            # Memory stats
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            max_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            gpu_metrics['memory_allocated_gb'] = memory_allocated
            gpu_metrics['memory_reserved_gb'] = memory_reserved
            gpu_metrics['total_memory_gb'] = max_memory
            gpu_metrics['memory_utilization'] = memory_allocated / max_memory if max_memory > 0 else 0
            
        else:
            gpu_metrics['cuda_available'] = False
            gpu_metrics['using_cpu'] = True
        
        return gpu_metrics

    def _calculate_optimization_score(self, 
                                   time_metrics: Dict, 
                                   quality_metrics: Dict, 
                                   gpu_metrics: Dict) -> float:
        """Calculate overall optimization score (0-100)."""
        score = 0.0
        
        # Time performance (30 points)
        avg_time = time_metrics.get('rag_avg_time', float('inf'))
        if avg_time <= 1.0:
            score += 30
        elif avg_time <= 3.0:
            score += 25
        elif avg_time <= 5.0:
            score += 20
        elif avg_time <= 10.0:
            score += 15
        else:
            score += 5
        
        # Quality performance (40 points)
        avg_quality = quality_metrics.get('rag_avg_quality', 0)
        score += min(avg_quality * 4, 40)
        
        # GPU utilization (20 points)
        if gpu_metrics.get('cuda_available', False):
            memory_util = gpu_metrics.get('memory_utilization', 0)
            if 0.2 <= memory_util <= 0.8:  # Optimal range
                score += 20
            elif 0.1 <= memory_util <= 0.9:
                score += 15
            else:
                score += 10
        else:
            score += 5  # CPU usage
        
        # Consistency bonus (10 points)
        quality_consistency = quality_metrics.get('quality_consistency', 0)
        score += max(quality_consistency * 10, 0)
        
        return min(score, 100)

    def generate_comprehensive_report(self, output_dir: str = "./evaluation_results") -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        self.logger.info("📋 Generating comprehensive evaluation report...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run all evaluations
        retrieval_metrics = self.evaluate_retrieval_performance()
        comparison_metrics = self.evaluate_rag_vs_base_llm()
        optimization_metrics = self.analyze_optimization_performance()
        
        # Generate report
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'evaluation_duration': time.time() - self.start_time,
                'total_queries_tested': len(self.results),
                'system_config': {
                    'llm_model': self.config.get('models.llm.name', 'unknown'),
                    'embedding_model': self.config.get('models.embeddings.name', 'unknown'),
                    'top_k': self.config.get('retrieval.top_k', 7),
                    'chunk_size': self.config.get('document_processing.chunk_size', 800)
                }
            },
            'performance_summary': {
                'retrieval_performance': retrieval_metrics,
                'rag_vs_base_comparison': comparison_metrics,
                'optimization_analysis': optimization_metrics
            },
            'detailed_results': self.results,
            'recommendations': self._generate_recommendations(comparison_metrics, optimization_metrics)
        }
        
        # Save main report
        report_file = output_path / "performance_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save CSV for detailed analysis
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = output_path / "detailed_results.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
        # Generate summary markdown
        self._generate_markdown_summary(report, output_path / "summary.md")
        
        self.logger.info(f"✅ Report saved to: {output_path}")
        return report

    def _generate_recommendations(self, 
                                comparison_metrics: Dict, 
                                optimization_metrics: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Performance recommendations
        rag_win_rate = comparison_metrics.get('rag_win_rate', 0)
        if rag_win_rate < 0.6:
            recommendations.append("🎯 RAG system needs improvement - consider tuning retrieval parameters")
        elif rag_win_rate > 0.8:
            recommendations.append("🎉 Excellent RAG performance! System significantly outperforms base LLM")
        
        # Time optimization
        avg_time = optimization_metrics.get('time_metrics', {}).get('rag_avg_time', 0)
        if avg_time > 5:
            recommendations.append("⚡ Consider reducing top_k or chunk_size for faster responses")
        elif avg_time < 1:
            recommendations.append("🔧 Excellent response time - consider increasing quality parameters")
        
        # GPU utilization
        gpu_util = optimization_metrics.get('gpu_metrics', {}).get('memory_utilization', 0)
        if gpu_util < 0.3:
            recommendations.append("💾 GPU underutilized - consider larger models or batch processing")
        elif gpu_util > 0.9:
            recommendations.append("⚠️ High GPU usage - monitor for memory issues")
        
        # Quality improvements
        avg_quality = optimization_metrics.get('quality_metrics', {}).get('rag_avg_quality', 0)
        if avg_quality < 6:
            recommendations.append("📈 Quality could be improved with better prompts or reranking")
        
        if not recommendations:
            recommendations.append("✅ System performance is excellent - no immediate optimizations needed")
        
        return recommendations

    def _generate_markdown_summary(self, report: Dict, output_file: Path):
        """Generate a markdown summary of the evaluation."""
        summary = f"""# Basketball RAG System - Performance Evaluation Summary

## 📊 Evaluation Overview

**Date**: {report['metadata']['timestamp'][:19]}  
**Duration**: {report['metadata']['evaluation_duration']:.1f} seconds  
**Queries Tested**: {report['metadata']['total_queries_tested']}  

## 🏆 Key Results

### RAG vs Base LLM Comparison
- **RAG Win Rate**: {report['performance_summary']['rag_vs_base_comparison'].get('rag_win_rate', 0):.1%}
- **Average RAG Quality**: {report['performance_summary']['optimization_analysis']['quality_metrics'].get('rag_avg_quality', 0):.1f}/10
- **Average Response Time**: {report['performance_summary']['optimization_analysis']['time_metrics'].get('rag_avg_time', 0):.2f}s

### Retrieval Performance
- **Average Retrieval Time**: {report['performance_summary']['retrieval_performance'].get('avg_retrieval_time', 0):.3f}s
- **Retrieval Success Rate**: {report['performance_summary']['retrieval_performance'].get('retrieval_success_rate', 0):.1%}

### System Optimization
- **Overall Optimization Score**: {report['performance_summary']['optimization_analysis'].get('optimization_score', 0):.1f}/100
- **GPU Utilization**: {report['performance_summary']['optimization_analysis']['gpu_metrics'].get('memory_utilization', 0):.1%}

## 💡 Recommendations

"""
        
        for rec in report['recommendations']:
            summary += f"- {rec}\n"
        
        summary += f"""
## 🔧 System Configuration

- **LLM Model**: {report['metadata']['system_config']['llm_model']}
- **Embedding Model**: {report['metadata']['system_config']['embedding_model']}
- **Top K**: {report['metadata']['system_config']['top_k']}
- **Chunk Size**: {report['metadata']['system_config']['chunk_size']}

---
*Generated by Basketball RAG Performance Evaluator*
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Basketball RAG System Performance')
    parser.add_argument('--output', default='./evaluation_results', 
                       help='Output directory for results')
    parser.add_argument('--config', default=None, 
                       help='Config file path')
    parser.add_argument('--full', action='store_true', 
                       help='Run full evaluation (slower but more comprehensive)')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run benchmark tests')
    
    args = parser.parse_args()
    
    print("🏀 Basketball RAG System - Performance Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = PerformanceEvaluator(args.config)
    
    # Setup models
    if not evaluator.setup_models():
        print("❌ Failed to setup models. Check Ollama and vector database.")
        return False
    
    print("✅ Models loaded successfully")
    print(f"📊 Testing with {len(evaluator.test_queries)} queries")
    
    # Run evaluation
    try:
        report = evaluator.generate_comprehensive_report(args.output)
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎉 EVALUATION COMPLETED!")
        print("=" * 60)
        
        comparison = report['performance_summary']['rag_vs_base_comparison']
        optimization = report['performance_summary']['optimization_analysis']
        
        print(f"🏆 RAG Win Rate: {comparison.get('rag_win_rate', 0):.1%}")
        print(f"⚡ Avg Response Time: {optimization['time_metrics'].get('rag_avg_time', 0):.2f}s")
        print(f"📈 Quality Score: {optimization['quality_metrics'].get('rag_avg_quality', 0):.1f}/10")
        print(f"🎯 Optimization Score: {optimization.get('optimization_score', 0):.1f}/100")
        
        print(f"\n📁 Results saved to: {args.output}")
        
        # Print recommendations
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"   {rec}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 