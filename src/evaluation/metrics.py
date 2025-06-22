"""
Basketball RAG System - Comprehensive Performance Metrics and Analysis
"""

import time
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

# Try to download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.config import Config
from ..utils.language_detection import detect_language

class RAGPerformanceAnalyzer:
    """Comprehensive RAG system performance analyzer."""
    
    def __init__(self, config_path: str = None):
        """Initialize the performance analyzer."""
        self.config = Config(config_path)
        self.logger = self._setup_logging()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.results = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for metrics."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger

    def evaluate_retrieval_quality(self, 
                                 vectorstore, 
                                 test_queries: List[Dict], 
                                 top_k: int = 7) -> Dict[str, float]:
        """Evaluate retrieval component quality."""
        self.logger.info("🔍 Evaluating retrieval quality...")
        
        precision_scores = []
        recall_scores = []
        mrr_scores = []  # Mean Reciprocal Rank
        retrieval_times = []
        
        for query_data in test_queries:
            query = query_data.get('query_turkish', '')
            expected_sources = query_data.get('expected_sources', [])
            
            if not query or not expected_sources:
                continue
                
            # Retrieve documents
            start_time = time.time()
            retrieved_docs = vectorstore.similarity_search_with_score(query, k=top_k)
            retrieval_time = time.time() - start_time
            retrieval_times.append(retrieval_time)
            
            # Extract source information from retrieved docs
            retrieved_sources = []
            for doc, score in retrieved_docs:
                source = doc.metadata.get('document_type', 'unknown')
                retrieved_sources.append(source)
            
            # Calculate precision and recall
            retrieved_set = set(retrieved_sources)
            expected_set = set(expected_sources)
            
            if len(retrieved_set) > 0:
                precision = len(retrieved_set & expected_set) / len(retrieved_set)
                precision_scores.append(precision)
            
            if len(expected_set) > 0:
                recall = len(retrieved_set & expected_set) / len(expected_set)
                recall_scores.append(recall)
            
            # Calculate MRR
            for i, source in enumerate(retrieved_sources):
                if source in expected_sources:
                    mrr_scores.append(1.0 / (i + 1))
                    break
            else:
                mrr_scores.append(0.0)
        
        return {
            'precision': np.mean(precision_scores) if precision_scores else 0.0,
            'recall': np.mean(recall_scores) if recall_scores else 0.0,
            'f1_score': 2 * np.mean(precision_scores) * np.mean(recall_scores) / 
                       (np.mean(precision_scores) + np.mean(recall_scores)) 
                       if precision_scores and recall_scores else 0.0,
            'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
            'avg_retrieval_time': np.mean(retrieval_times) if retrieval_times else 0.0
        }

    def evaluate_rag_vs_base_llm(self, 
                                rag_chain, 
                                base_llm, 
                                test_queries: List[Dict]) -> Dict[str, Any]:
        """Compare RAG system performance against base LLM."""
        self.logger.info("⚖️ Comparing RAG vs Base LLM performance...")
        
        rag_results = []
        base_llm_results = []
        comparison_metrics = {
            'rag_better': 0,
            'base_better': 0,
            'tied': 0
        }
        
        for query_data in test_queries:
            query_turkish = query_data.get('query_turkish', '')
            query_english = query_data.get('query_english', '')
            
            # Test both languages
            for query, lang in [(query_turkish, 'turkish'), (query_english, 'english')]:
                if not query:
                    continue
                    
                # RAG Response
                rag_start = time.time()
                try:
                    rag_response = rag_chain.invoke({"question": query})
                    rag_answer = rag_response.get('answer', '')
                    rag_sources = rag_response.get('source_documents', [])
                    rag_time = time.time() - rag_start
                    rag_success = True
                except Exception as e:
                    self.logger.warning(f"RAG failed for query '{query}': {e}")
                    rag_answer = ""
                    rag_sources = []
                    rag_time = 0
                    rag_success = False
                
                # Base LLM Response
                base_start = time.time()
                try:
                    base_response = base_llm.invoke(query)
                    base_answer = base_response if isinstance(base_response, str) else str(base_response)
                    base_time = time.time() - base_start
                    base_success = True
                except Exception as e:
                    self.logger.warning(f"Base LLM failed for query '{query}': {e}")
                    base_answer = ""
                    base_time = 0
                    base_success = False
                
                # Analyze responses
                result = {
                    'query': query,
                    'language': lang,
                    'category': query_data.get('category', 'unknown'),
                    'difficulty': query_data.get('difficulty', 'unknown'),
                    'rag_answer': rag_answer,
                    'base_answer': base_answer,
                    'rag_time': rag_time,
                    'base_time': base_time,
                    'rag_sources_count': len(rag_sources),
                    'rag_success': rag_success,
                    'base_success': base_success,
                    'rag_has_citations': '[' in rag_answer and ']' in rag_answer,
                    'base_has_citations': '[' in base_answer and ']' in base_answer,
                    'rag_length': len(rag_answer.split()),
                    'base_length': len(base_answer.split())
                }
                
                # Quality scoring (simple heuristic)
                rag_quality = self._score_answer_quality(rag_answer, rag_sources, query_data)
                base_quality = self._score_answer_quality(base_answer, [], query_data)
                
                result['rag_quality_score'] = rag_quality
                result['base_quality_score'] = base_quality
                
                if rag_quality > base_quality:
                    comparison_metrics['rag_better'] += 1
                    result['winner'] = 'rag'
                elif base_quality > rag_quality:
                    comparison_metrics['base_better'] += 1
                    result['winner'] = 'base_llm'
                else:
                    comparison_metrics['tied'] += 1
                    result['winner'] = 'tie'
                
                self.results.append(result)
        
        total_comparisons = sum(comparison_metrics.values())
        if total_comparisons > 0:
            comparison_metrics['rag_win_rate'] = comparison_metrics['rag_better'] / total_comparisons
            comparison_metrics['base_win_rate'] = comparison_metrics['base_better'] / total_comparisons
            comparison_metrics['tie_rate'] = comparison_metrics['tied'] / total_comparisons
        
        return comparison_metrics

    def _score_answer_quality(self, 
                            answer: str, 
                            sources: List, 
                            query_data: Dict) -> float:
        """Score answer quality based on multiple factors."""
        if not answer or len(answer.strip()) < 10:
            return 0.0
        
        score = 0.0
        
        # Length appropriateness (not too short, not too verbose)
        word_count = len(answer.split())
        if 20 <= word_count <= 200:
            score += 2.0
        elif 10 <= word_count <= 300:
            score += 1.0
        
        # Has citations (for RAG)
        if '[' in answer and ']' in answer:
            score += 2.0
        
        # Contains relevant keywords based on category
        category = query_data.get('category', '')
        category_keywords = {
            'fouls': ['faul', 'foul', 'ceza', 'penalty'],
            'timing': ['saniye', 'second', 'süre', 'time', 'saat', 'clock'],
            'court_specifications': ['saha', 'court', 'boyut', 'dimension', 'metre', 'meter'],
            'rule_changes': ['değişiklik', 'change', '2024', 'yeni', 'new'],
            'substitutions': ['değişim', 'substitution', 'oyuncu', 'player']
        }
        
        keywords = category_keywords.get(category, [])
        for keyword in keywords:
            if keyword.lower() in answer.lower():
                score += 0.5
        
        # Turkish/English language consistency
        detected_lang = detect_language(answer)
        query_lang = detect_language(query_data.get('query_turkish', ''))
        if detected_lang == query_lang:
            score += 1.0
        
        # Source usage (for RAG)
        if sources and len(sources) > 0:
            score += 1.0
        
        return min(score, 10.0)  # Cap at 10

    def measure_accuracy_metrics(self, 
                               rag_chain,
                               ground_truth_file: str = None) -> Dict[str, float]:
        """Measure various accuracy metrics."""
        self.logger.info("📊 Measuring accuracy metrics...")
        
        if ground_truth_file and Path(ground_truth_file).exists():
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                ground_truth_data = json.load(f)
        else:
            self.logger.warning("No ground truth file provided, using heuristic evaluation")
            return self._heuristic_accuracy_evaluation(rag_chain)
        
        rouge_scores = []
        bleu_scores = []
        semantic_similarities = []
        
        # Load embedding model for semantic similarity
        embedding_model = SentenceTransformer('BAAI/bge-m3')
        
        for item in ground_truth_data.get('test_cases', []):
            query = item.get('query', '')
            expected_answer = item.get('expected_answer', '')
            
            if not query or not expected_answer:
                continue
            
            # Get RAG response
            try:
                response = rag_chain.invoke({"question": query})
                generated_answer = response.get('answer', '')
            except Exception as e:
                self.logger.warning(f"Failed to get response for '{query}': {e}")
                continue
            
            if not generated_answer:
                continue
            
            # ROUGE Score
            rouge_score = self.rouge_scorer.score(expected_answer, generated_answer)
            rouge_scores.append({
                'rouge1': rouge_score['rouge1'].fmeasure,
                'rouge2': rouge_score['rouge2'].fmeasure,
                'rougeL': rouge_score['rougeL'].fmeasure
            })
            
            # BLEU Score
            try:
                reference = [word_tokenize(expected_answer.lower())]
                candidate = word_tokenize(generated_answer.lower())
                bleu_score = sentence_bleu(reference, candidate)
                bleu_scores.append(bleu_score)
            except Exception:
                pass
            
            # Semantic Similarity
            try:
                embeddings = embedding_model.encode([expected_answer, generated_answer])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                semantic_similarities.append(similarity)
            except Exception:
                pass
        
        # Calculate averages
        metrics = {}
        if rouge_scores:
            metrics['rouge1'] = np.mean([r['rouge1'] for r in rouge_scores])
            metrics['rouge2'] = np.mean([r['rouge2'] for r in rouge_scores])
            metrics['rougeL'] = np.mean([r['rougeL'] for r in rouge_scores])
        
        if bleu_scores:
            metrics['bleu'] = np.mean(bleu_scores)
        
        if semantic_similarities:
            metrics['semantic_similarity'] = np.mean(semantic_similarities)
        
        return metrics

    def _heuristic_accuracy_evaluation(self, rag_chain) -> Dict[str, float]:
        """Heuristic evaluation when no ground truth is available."""
        
        # Use test queries from config
        test_queries_file = self.config.get('evaluation.test_queries_file', './data/test_queries.json')
        
        if not Path(test_queries_file).exists():
            self.logger.error(f"Test queries file not found: {test_queries_file}")
            return {}
        
        with open(test_queries_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        queries = test_data.get('basketball_rules_queries', [])
        
        response_success_rate = 0
        citation_accuracy = 0
        language_consistency = 0
        content_relevance = 0
        
        total_tests = 0
        
        for query_data in queries:
            for query_key in ['query_turkish', 'query_english']:
                query = query_data.get(query_key, '')
                if not query:
                    continue
                
                total_tests += 1
                
                try:
                    response = rag_chain.invoke({"question": query})
                    answer = response.get('answer', '')
                    sources = response.get('source_documents', [])
                    
                    # Response success
                    if answer and len(answer.strip()) > 10:
                        response_success_rate += 1
                    
                    # Citation accuracy
                    if sources and ('[' in answer and ']' in answer):
                        citation_accuracy += 1
                    
                    # Language consistency
                    query_lang = detect_language(query)
                    answer_lang = detect_language(answer)
                    if query_lang == answer_lang:
                        language_consistency += 1
                    
                    # Content relevance (simple keyword matching)
                    category = query_data.get('category', '')
                    if self._is_content_relevant(answer, category):
                        content_relevance += 1
                        
                except Exception as e:
                    self.logger.warning(f"Failed evaluation for '{query}': {e}")
        
        if total_tests == 0:
            return {}
        
        return {
            'response_success_rate': response_success_rate / total_tests,
            'citation_accuracy': citation_accuracy / total_tests,
            'language_consistency': language_consistency / total_tests,
            'content_relevance': content_relevance / total_tests,
            'overall_accuracy': (response_success_rate + citation_accuracy + 
                               language_consistency + content_relevance) / (4 * total_tests)
        }

    def _is_content_relevant(self, answer: str, category: str) -> bool:
        """Check if answer content is relevant to the category."""
        answer_lower = answer.lower()
        
        relevance_keywords = {
            'fouls': ['faul', 'foul', 'ceza', 'penalty', 'ihlal', 'violation'],
            'timing': ['saniye', 'second', 'süre', 'time', 'saat', 'clock', 'zaman'],
            'court_specifications': ['saha', 'court', 'boyut', 'dimension', 'metre', 'meter', 'çizgi', 'line'],
            'rule_changes': ['değişiklik', 'change', '2024', 'yeni', 'new', 'güncelleme', 'update'],
            'substitutions': ['değişim', 'substitution', 'oyuncu', 'player', 'değiştir', 'replace'],
            'game_flow': ['oyun', 'game', 'süre', 'time', 'periyot', 'period', 'yarı', 'half'],
            'officiating': ['hakem', 'referee', 'karar', 'decision', 'çağrı', 'call'],
            'technology': ['video', 'teknoloji', 'technology', 'sistem', 'system']
        }
        
        keywords = relevance_keywords.get(category, [])
        return any(keyword in answer_lower for keyword in keywords)

    def analyze_optimization_performance(self) -> Dict[str, Any]:
        """Analyze system optimization and performance characteristics."""
        self.logger.info("⚡ Analyzing optimization performance...")
        
        # GPU Memory Analysis
        gpu_metrics = self._analyze_gpu_performance()
        
        # Response Time Analysis
        response_times = [r['rag_time'] for r in self.results if r.get('rag_time', 0) > 0]
        time_metrics = {
            'avg_response_time': np.mean(response_times) if response_times else 0,
            'median_response_time': np.median(response_times) if response_times else 0,
            'min_response_time': np.min(response_times) if response_times else 0,
            'max_response_time': np.max(response_times) if response_times else 0,
            'response_time_std': np.std(response_times) if response_times else 0
        }
        
        # Quality vs Speed Trade-off
        quality_scores = [r['rag_quality_score'] for r in self.results if 'rag_quality_score' in r]
        quality_metrics = {
            'avg_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'quality_consistency': 1 - (np.std(quality_scores) / np.mean(quality_scores)) 
                                 if quality_scores and np.mean(quality_scores) > 0 else 0
        }
        
        # Source Utilization
        source_counts = [r['rag_sources_count'] for r in self.results if 'rag_sources_count' in r]
        source_metrics = {
            'avg_sources_used': np.mean(source_counts) if source_counts else 0,
            'source_utilization_rate': len([s for s in source_counts if s > 0]) / len(source_counts) 
                                     if source_counts else 0
        }
        
        return {
            'gpu_metrics': gpu_metrics,
            'time_metrics': time_metrics,
            'quality_metrics': quality_metrics,
            'source_metrics': source_metrics,
            'optimization_score': self._calculate_optimization_score(
                time_metrics, quality_metrics, gpu_metrics
            )
        }

    def _analyze_gpu_performance(self) -> Dict[str, Any]:
        """Analyze GPU performance metrics."""
        gpu_metrics = {}
        
        if torch.cuda.is_available():
            gpu_metrics['cuda_available'] = True
            gpu_metrics['gpu_count'] = torch.cuda.device_count()
            gpu_metrics['current_device'] = torch.cuda.current_device()
            gpu_metrics['gpu_name'] = torch.cuda.get_device_name()
            
            # Memory stats
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            gpu_metrics['memory_allocated_gb'] = memory_allocated
            gpu_metrics['memory_reserved_gb'] = memory_reserved
            gpu_metrics['max_memory_allocated_gb'] = max_memory_allocated
            
            # Calculate memory efficiency
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_metrics['total_memory_gb'] = total_memory
            gpu_metrics['memory_utilization'] = memory_allocated / total_memory
            
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
        avg_time = time_metrics.get('avg_response_time', float('inf'))
        if avg_time <= 1.0:
            score += 30
        elif avg_time <= 3.0:
            score += 25
        elif avg_time <= 5.0:
            score += 20
        elif avg_time <= 10.0:
            score += 15
        else:
            score += 10
        
        # Quality performance (40 points)
        avg_quality = quality_metrics.get('avg_quality_score', 0)
        score += min(avg_quality * 4, 40)
        
        # GPU utilization efficiency (20 points)
        if gpu_metrics.get('cuda_available', False):
            memory_util = gpu_metrics.get('memory_utilization', 0)
            if 0.3 <= memory_util <= 0.8:  # Optimal range
                score += 20
            elif 0.1 <= memory_util <= 0.9:
                score += 15
            else:
                score += 10
        else:
            score += 5  # CPU usage
        
        # Consistency bonus (10 points)
        quality_consistency = quality_metrics.get('quality_consistency', 0)
        score += quality_consistency * 10
        
        return min(score, 100)

    def generate_comprehensive_report(self, output_file: str = None) -> Dict[str, Any]:
        """Generate comprehensive performance analysis report."""
        self.logger.info("📋 Generating comprehensive performance report...")
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_evaluations': len(self.results),
                'config': dict(self.config.config) if hasattr(self.config, 'config') else {}
            },
            'summary': {},
            'detailed_results': self.results,
            'recommendations': []
        }
        
        if not self.results:
            self.logger.warning("No evaluation results available for report generation")
            return report
        
        # Summary statistics
        successful_responses = len([r for r in self.results if r.get('rag_success', False)])
        avg_response_time = np.mean([r['rag_time'] for r in self.results if r.get('rag_time', 0) > 0])
        avg_quality = np.mean([r['rag_quality_score'] for r in self.results if 'rag_quality_score' in r])
        
        report['summary'] = {
            'success_rate': successful_responses / len(self.results) if self.results else 0,
            'avg_response_time': avg_response_time,
            'avg_quality_score': avg_quality,
            'rag_win_rate': len([r for r in self.results if r.get('winner') == 'rag']) / len(self.results),
            'languages_tested': list(set([r.get('language', '') for r in self.results])),
            'categories_tested': list(set([r.get('category', '') for r in self.results]))
        }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report['summary'])
        
        # Save report if output file specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Report saved to: {output_path}")
        
        return report

    def _generate_recommendations(self, summary: Dict) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Response time recommendations
        avg_time = summary.get('avg_response_time', 0)
        if avg_time > 5.0:
            recommendations.append(
                "⚡ Consider reducing chunk_size or top_k to improve response times"
            )
        elif avg_time > 3.0:
            recommendations.append(
                "🔧 Response time could be optimized with model quantization or GPU acceleration"
            )
        
        # Quality recommendations
        avg_quality = summary.get('avg_quality_score', 0)
        if avg_quality < 6.0:
            recommendations.append(
                "📈 Consider increasing top_k or improving prompt engineering for better quality"
            )
        
        # Success rate recommendations
        success_rate = summary.get('success_rate', 0)
        if success_rate < 0.9:
            recommendations.append(
                "🛠️ Investigate failed queries and improve error handling"
            )
        
        if not recommendations:
            recommendations.append("✅ System performance is optimal - no immediate changes needed")
        
        return recommendations

    def export_results_to_csv(self, output_file: str):
        """Export detailed results to CSV for further analysis."""
        if not self.results:
            self.logger.warning("No results to export")
            return
        
        df = pd.DataFrame(self.results)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        self.logger.info(f"Results exported to CSV: {output_path}")

# Example usage function
def run_full_evaluation(config_path: str = None, 
                       output_dir: str = "./evaluation_results") -> Dict[str, Any]:
    """Run complete evaluation suite."""
    
    # This would be called from a separate script that imports your RAG components
    # analyzer = RAGPerformanceAnalyzer(config_path)
    
    # Load your RAG chain and base LLM here
    # rag_chain = load_rag_chain()
    # base_llm = load_base_llm()
    # vectorstore = load_vectorstore()
    
    # Run evaluations
    # retrieval_metrics = analyzer.evaluate_retrieval_quality(vectorstore, test_queries)
    # comparison_metrics = analyzer.evaluate_rag_vs_base_llm(rag_chain, base_llm, test_queries)
    # accuracy_metrics = analyzer.measure_accuracy_metrics(rag_chain)
    # optimization_metrics = analyzer.analyze_optimization_performance()
    
    # Generate reports
    # report = analyzer.generate_comprehensive_report(f"{output_dir}/performance_report.json")
    # analyzer.export_results_to_csv(f"{output_dir}/detailed_results.csv")
    
    # return report
    
    pass  # Placeholder for the actual implementation 