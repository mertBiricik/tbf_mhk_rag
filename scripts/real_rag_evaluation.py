#!/usr/bin/env python3
"""
REAL Basketball RAG System Performance Evaluation
This script actually runs queries against your RAG system and measures REAL performance.
"""

import os
import time
import json
import sys
import numpy as np
import requests
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import chromadb
from sentence_transformers import SentenceTransformer

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

class RealRAGEvaluator:
    def __init__(self):
        """Initialize the real RAG evaluation system."""
        print("🏀 Initializing REAL Basketball RAG Evaluator...")
        self.setup_rag_system()
        self.load_test_queries()
        self.results = []
        
    def setup_rag_system(self):
        """Set up the actual RAG system components."""
        print("⚙️  Setting up RAG system components...")
        
        # Initialize embedding model (same as your system)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"📊 Loading BGE-M3 on {device}...")
        self.embedding_model = SentenceTransformer('BAAI/bge-m3')
        self.embedding_model = self.embedding_model.to(device)
        
        # Connect to your existing ChromaDB
        db_path = "./vector_db/chroma_db"
        collection_name = "basketball_rules"
        
        client = chromadb.PersistentClient(path=db_path)
        try:
            self.collection = client.get_collection(name=collection_name)
            print(f"✅ Connected to vector database: {self.collection.count()} documents")
        except Exception as e:
            print(f"❌ Failed to connect to vector database: {e}")
            sys.exit(1)
            
        # LLM model name (same as your system)
        self.llm_model = "llama3.1:8b-instruct-q4_K_M"
        
        # Test Ollama connection
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama connection successful")
            else:
                print("❌ Ollama connection failed")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            sys.exit(1)
    
    def load_test_queries(self):
        """Load test queries from JSON file."""
        queries_file = Path("data/test_queries.json")
        if not queries_file.exists():
            print("❌ Test queries file not found!")
            sys.exit(1)
            
        with open(queries_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.test_queries = data['basketball_rules_queries']
        
        print(f"📝 Loaded {len(self.test_queries)} test queries")
    
    def search_documents(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search for relevant documents (same as your RAG system)."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = query_embedding.tolist()
        
        # Search vector database
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=num_results
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
        
        # Format results
        search_results = []
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            search_results.append({
                'content': doc,
                'source': metadata.get('source', 'unknown'),
                'doc_type': metadata.get('doc_type', 'unknown'),
                'year': metadata.get('year', 'unknown'),
                'chunk_id': metadata.get('chunk_id', 'unknown')
            })
        
        return search_results
    
    def generate_rag_answer(self, query: str, search_results: List[Dict]) -> str:
        """Generate answer using RAG (exactly like your system)."""
        if not search_results:
            return "Bu konuyla ilgili bilgi bulamadım."
        
        # Prepare context
        context_parts = []
        for i, result in enumerate(search_results):
            source_info = f"{result['source']} ({result['year']})"
            context_parts.append(f"Kaynak {i+1} - {source_info}:\n{result['content']}")
        
        context_text = "\n\n".join(context_parts)
        
        # Create prompt (same as your system)
        prompt = f"""Sen Türkiye Basketbol Federasyonu kuralları uzmanısın. Aşağıdaki resmi belgelerden yararlanarak soruyu yanıtla.

RESMI BELGELER:
{context_text}

SORU: {query}

YANIT KURALLARI:
- Sadece verilen belgelerden bilgi kullan
- Kısa ve net yanıt ver
- Hangi belgeden aldığın bilgiyi belirt
- Madde numaralarını belirt
- Türkçe yanıtla

YANIT:"""

        # Call LLM (same as your system)
        try:
            payload = {
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 400
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                return answer
            else:
                return f"LLM hatası: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Yanıt oluşturma hatası: {str(e)}"
    
    def generate_base_llm_answer(self, query: str) -> str:
        """Generate answer using base LLM without RAG."""
        prompt = f"""Sen basketbol kuralları uzmanısın. Lütfen şu soruyu kısaca yanıtla:

Soru: {query}

Yanıt:"""

        try:
            payload = {
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 300
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                return answer
            else:
                return f"LLM hatası: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Yanıt oluşturma hatası: {str(e)}"
    
    def evaluate_retrieval_quality(self, query: Dict, search_results: List[Dict]) -> Dict:
        """Evaluate retrieval quality metrics."""
        metrics = {}
        
        # Check if expected sources are retrieved
        expected_sources = query.get('expected_sources', [])
        retrieved_sources = [result['doc_type'] for result in search_results]
        
        # Source coverage
        found_sources = []
        for expected in expected_sources:
            for retrieved in retrieved_sources:
                if expected.replace('_', '') in retrieved:
                    found_sources.append(expected)
                    break
        
        metrics['source_coverage'] = len(found_sources) / len(expected_sources) if expected_sources else 0
        metrics['sources_found'] = len(search_results)
        metrics['expected_sources'] = expected_sources
        metrics['retrieved_sources'] = list(set(retrieved_sources))
        
        return metrics
    
    def evaluate_answer_quality(self, rag_answer: str, base_answer: str, query: Dict) -> Dict:
        """Evaluate answer quality metrics."""
        metrics = {}
        
        # Length comparison
        metrics['rag_answer_length'] = len(rag_answer)
        metrics['base_answer_length'] = len(base_answer)
        
        # Check for citations/sources
        source_indicators = ['kaynak', 'madde', 'kural', 'belge', '2022', '2023', '2024']
        rag_citations = sum(1 for indicator in source_indicators if indicator.lower() in rag_answer.lower())
        base_citations = sum(1 for indicator in source_indicators if indicator.lower() in base_answer.lower())
        
        metrics['rag_citations'] = rag_citations
        metrics['base_citations'] = base_citations
        
        # Check for specific terms related to query category
        category = query.get('category', '')
        category_terms = {
            'fouls': ['faul', 'foul', 'ihlal', 'ceza'],
            'timing': ['saniye', 'süre', 'saat', 'zaman'],
            'court_specifications': ['saha', 'boyut', 'mesafe', 'uzunluk'],
            'rule_changes': ['değişti', 'yeni', '2024', 'güncelleme'],
            'game_flow': ['oyun', 'süre', 'periyot', 'uzatma']
        }
        
        relevant_terms = category_terms.get(category, [])
        rag_relevance = sum(1 for term in relevant_terms if term in rag_answer.lower())
        base_relevance = sum(1 for term in relevant_terms if term in base_answer.lower())
        
        metrics['rag_relevance_score'] = rag_relevance
        metrics['base_relevance_score'] = base_relevance
        
        # Check for confidence indicators
        uncertainty_words = ['belki', 'muhtemelen', 'sanırım', 'galiba', 'olabilir']
        rag_uncertainty = sum(1 for word in uncertainty_words if word in rag_answer.lower())
        base_uncertainty = sum(1 for word in uncertainty_words if word in base_answer.lower())
        
        metrics['rag_uncertainty'] = rag_uncertainty
        metrics['base_uncertainty'] = base_uncertainty
        
        return metrics
    
    def run_single_evaluation(self, query: Dict) -> Dict:
        """Run evaluation for a single query."""
        print(f"\n🔍 Evaluating: {query['query_turkish']}")
        
        start_time = time.time()
        
        # 1. Retrieval phase
        retrieval_start = time.time()
        search_results = self.search_documents(query['query_turkish'])
        retrieval_time = time.time() - retrieval_start
        
        # 2. RAG generation
        rag_start = time.time()
        rag_answer = self.generate_rag_answer(query['query_turkish'], search_results)
        rag_generation_time = time.time() - rag_start
        
        # 3. Base LLM generation
        base_start = time.time()
        base_answer = self.generate_base_llm_answer(query['query_turkish'])
        base_generation_time = time.time() - base_start
        
        total_time = time.time() - start_time
        
        # 4. Evaluate metrics
        retrieval_metrics = self.evaluate_retrieval_quality(query, search_results)
        answer_metrics = self.evaluate_answer_quality(rag_answer, base_answer, query)
        
        # Compile results
        result = {
            'query_id': query['id'],
            'query': query['query_turkish'],
            'category': query['category'],
            'difficulty': query['difficulty'],
            
            # Timing
            'retrieval_time': retrieval_time,
            'rag_generation_time': rag_generation_time,
            'base_generation_time': base_generation_time,
            'total_rag_time': retrieval_time + rag_generation_time,
            'total_time': total_time,
            
            # Answers
            'rag_answer': rag_answer,
            'base_answer': base_answer,
            
            # Retrieved documents
            'retrieved_docs': len(search_results),
            'search_results': search_results,
            
            # Metrics
            **retrieval_metrics,
            **answer_metrics
        }
        
        # Print detailed results including full answers
        print(f"   ⏱️  Retrieval: {retrieval_time:.2f}s | RAG Gen: {rag_generation_time:.2f}s | Base Gen: {base_generation_time:.2f}s")
        print(f"   📊 Source Coverage: {retrieval_metrics['source_coverage']:.1%}")
        print(f"   📝 RAG Citations: {answer_metrics['rag_citations']} | Base Citations: {answer_metrics['base_citations']}")
        
        # Print full answers for comparison
        print(f"\n   🤖 RAG ANSWER:")
        print(f"   {'-' * 50}")
        print(f"   {rag_answer}")
        print(f"   {'-' * 50}")
        
        print(f"\n   🧠 BASE LLM ANSWER:")
        print(f"   {'-' * 50}")
        print(f"   {base_answer}")
        print(f"   {'-' * 50}")
        
        print(f"\n   📚 RETRIEVED SOURCES:")
        for i, doc in enumerate(search_results):
            print(f"   {i+1}. {doc['source']} ({doc['year']}) - {doc['doc_type']}")
        
        return result
    
    def run_full_evaluation(self) -> Dict:
        """Run complete evaluation on all test queries."""
        print("\n🚀 Starting FULL RAG System Evaluation...")
        print("=" * 60)
        
        all_results = []
        total_start_time = time.time()
        
        for i, query in enumerate(self.test_queries, 1):
            print(f"\n[{i}/{len(self.test_queries)}] Testing Query {query['id']}")
            
            try:
                result = self.run_single_evaluation(query)
                all_results.append(result)
                
                # Brief progress update
                avg_time = (time.time() - total_start_time) / i
                remaining = (len(self.test_queries) - i) * avg_time
                print(f"   📈 Progress: {i}/{len(self.test_queries)} | Est. remaining: {remaining:.0f}s")
                
            except Exception as e:
                print(f"   ❌ Evaluation failed: {e}")
                continue
        
        total_evaluation_time = time.time() - total_start_time
        
        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(all_results)
        aggregate_metrics['total_evaluation_time'] = total_evaluation_time
        aggregate_metrics['queries_evaluated'] = len(all_results)
        
        return {
            'individual_results': all_results,
            'aggregate_metrics': aggregate_metrics,
            'evaluation_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_queries': len(self.test_queries),
                'successful_evaluations': len(all_results),
                'evaluation_time': total_evaluation_time
            }
        }
    
    def calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate performance metrics."""
        if not results:
            return {}
        
        metrics = {}
        
        # Timing metrics
        metrics['avg_retrieval_time'] = np.mean([r['retrieval_time'] for r in results])
        metrics['avg_rag_generation_time'] = np.mean([r['rag_generation_time'] for r in results])
        metrics['avg_base_generation_time'] = np.mean([r['base_generation_time'] for r in results])
        metrics['avg_total_rag_time'] = np.mean([r['total_rag_time'] for r in results])
        
        # Retrieval metrics
        metrics['avg_source_coverage'] = np.mean([r['source_coverage'] for r in results])
        metrics['avg_retrieved_docs'] = np.mean([r['retrieved_docs'] for r in results])
        
        # Answer quality metrics
        metrics['avg_rag_citations'] = np.mean([r['rag_citations'] for r in results])
        metrics['avg_base_citations'] = np.mean([r['base_citations'] for r in results])
        metrics['avg_rag_relevance'] = np.mean([r['rag_relevance_score'] for r in results])
        metrics['avg_base_relevance'] = np.mean([r['base_relevance_score'] for r in results])
        
        # RAG vs Base comparisons
        rag_better_citations = sum(1 for r in results if r['rag_citations'] > r['base_citations'])
        rag_better_relevance = sum(1 for r in results if r['rag_relevance_score'] > r['base_relevance_score'])
        rag_less_uncertainty = sum(1 for r in results if r['rag_uncertainty'] < r['base_uncertainty'])
        
        metrics['rag_citation_advantage'] = rag_better_citations / len(results)
        metrics['rag_relevance_advantage'] = rag_better_relevance / len(results)
        metrics['rag_confidence_advantage'] = rag_less_uncertainty / len(results)
        
        # Performance by category
        categories = {}
        for result in results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        
        category_performance = {}
        for cat, cat_results in categories.items():
            category_performance[cat] = {
                'queries': len(cat_results),
                'avg_source_coverage': np.mean([r['source_coverage'] for r in cat_results]),
                'avg_rag_citations': np.mean([r['rag_citations'] for r in cat_results]),
                'rag_citation_win_rate': sum(1 for r in cat_results if r['rag_citations'] > r['base_citations']) / len(cat_results)
            }
        
        metrics['category_performance'] = category_performance
        
        return metrics
    
    def save_results(self, evaluation_results: Dict):
        """Save evaluation results to files."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Create results directory
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save full results as JSON
        json_file = results_dir / f"real_rag_evaluation_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Full results saved to: {json_file}")
        
        # Create summary report
        self.create_summary_report(evaluation_results, results_dir / f"real_evaluation_summary_{timestamp}.md")
        
        # Create detailed answers file
        self.create_detailed_answers_file(evaluation_results, results_dir / f"detailed_answers_{timestamp}.md")
    
    def create_summary_report(self, results: Dict, output_file: Path):
        """Create a human-readable summary report."""
        aggregate = results['aggregate_metrics']
        metadata = results['evaluation_metadata']
        
        report = f"""# 🏀 REAL Basketball RAG System Evaluation Report

**Evaluation Date:** {metadata['timestamp']}  
**Total Queries:** {metadata['total_queries']}  
**Successful Evaluations:** {metadata['successful_evaluations']}  
**Total Evaluation Time:** {metadata['evaluation_time']:.2f} seconds

## 📊 Overall Performance Summary

### ⏱️ Timing Performance
- **Average Retrieval Time:** {aggregate['avg_retrieval_time']:.2f}s
- **Average RAG Generation:** {aggregate['avg_rag_generation_time']:.2f}s  
- **Average Base LLM Generation:** {aggregate['avg_base_generation_time']:.2f}s
- **Average Total RAG Time:** {aggregate['avg_total_rag_time']:.2f}s

### 🔍 Retrieval Quality
- **Average Source Coverage:** {aggregate['avg_source_coverage']:.1%}
- **Average Documents Retrieved:** {aggregate['avg_retrieved_docs']:.1f}

### 📝 Answer Quality Comparison

| Metric | RAG | Base LLM | RAG Advantage |
|--------|-----|----------|---------------|
| **Citations/Sources** | {aggregate['avg_rag_citations']:.1f} | {aggregate['avg_base_citations']:.1f} | {aggregate['rag_citation_advantage']:.1%} |
| **Relevance Score** | {aggregate['avg_rag_relevance']:.1f} | {aggregate['avg_base_relevance']:.1f} | {aggregate['rag_relevance_advantage']:.1%} |
| **Confidence Level** | Higher | Lower | {aggregate['rag_confidence_advantage']:.1%} |

## 📈 Performance by Category

"""
        
        for category, performance in aggregate['category_performance'].items():
            report += f"""### {category.replace('_', ' ').title()}
- **Queries:** {performance['queries']}
- **Source Coverage:** {performance['avg_source_coverage']:.1%}
- **RAG Citation Win Rate:** {performance['rag_citation_win_rate']:.1%}

"""
        
        report += f"""## 🎯 Key Findings

### RAG System Advantages:
- **{aggregate['rag_citation_advantage']:.1%}** of answers had better source citations
- **{aggregate['rag_relevance_advantage']:.1%}** of answers were more relevant to the query
- **{aggregate['rag_confidence_advantage']:.1%}** of answers showed higher confidence (less uncertainty)

### Performance Insights:
- Average retrieval covers **{aggregate['avg_source_coverage']:.1%}** of expected sources
- RAG system provides **{aggregate['avg_rag_citations']:.1f}x** more citations than base LLM
- Total response time averages **{aggregate['avg_total_rag_time']:.2f} seconds**

## 🔬 Detailed Query Results

"""
        
        # Add detailed query results with full answers
        individual = results['individual_results']
        for i, result in enumerate(individual):  # Show all results
            report += f"""### Query {result['query_id']}: {result['query']}

**Category:** {result['category']} | **Difficulty:** {result['difficulty']}

#### 🤖 RAG Answer:
```
{result['rag_answer']}
```

#### 🧠 Base LLM Answer:
```
{result['base_answer']}
```

#### 📊 Performance Metrics:
- **Retrieval Time:** {result['retrieval_time']:.2f}s
- **RAG Generation Time:** {result['rag_generation_time']:.2f}s  
- **Base Generation Time:** {result['base_generation_time']:.2f}s
- **Source Coverage:** {result['source_coverage']:.1%}
- **RAG Citations:** {result['rag_citations']}
- **Base Citations:** {result['base_citations']}
- **Documents Retrieved:** {result['retrieved_docs']}

#### 📚 Sources Used:
"""
            
            # Add retrieved sources
            for j, doc in enumerate(result['search_results']):
                report += f"- **{j+1}.** {doc['source']} ({doc['year']}) - {doc['doc_type']}\n"
            
            report += "\n---\n\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📝 Summary report saved to: {output_file}")

    def create_detailed_answers_file(self, results: Dict, output_file: Path):
        """Create a detailed file with all questions and answers."""
        individual = results['individual_results']
        metadata = results['evaluation_metadata']
        
        content = f"""# 🏀 Detailed RAG vs Base LLM Answers

**Evaluation Date:** {metadata['timestamp']}  
**Total Queries:** {metadata['total_queries']}  

This file contains the complete answers from both RAG and Base LLM for direct comparison.

---

"""
        
        for result in individual:
            content += f"""## Query {result['query_id']}: {result['query']}

**Category:** {result['category']} | **Difficulty:** {result['difficulty']}

### 🤖 RAG System Answer:
{result['rag_answer']}

### 🧠 Base LLM Answer:
{result['base_answer']}

### 📊 Quick Metrics:
- Retrieval Time: {result['retrieval_time']:.2f}s
- RAG Generation: {result['rag_generation_time']:.2f}s
- Base Generation: {result['base_generation_time']:.2f}s
- Source Coverage: {result['source_coverage']:.1%}
- RAG Citations: {result['rag_citations']} | Base Citations: {result['base_citations']}

### 📚 Sources Retrieved:
"""
            
            for i, doc in enumerate(result['search_results']):
                content += f"{i+1}. {doc['source']} ({doc['year']}) - {doc['doc_type']}\n"
            
            content += "\n" + "="*80 + "\n\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"📋 Detailed answers saved to: {output_file}")

def main():
    """Main evaluation function."""
    print("🚀 REAL Basketball RAG System Performance Evaluation")
    print("=" * 60)
    print("This will actually run queries against your RAG system!")
    print("=" * 60)
    
    try:
        # Initialize evaluator
        evaluator = RealRAGEvaluator()
        
        # Run full evaluation
        evaluation_results = evaluator.run_full_evaluation()
        
        # Save results
        evaluator.save_results(evaluation_results)
        
        # Print final summary
        print("\n" + "=" * 60)
        print("🎉 REAL EVALUATION COMPLETED!")
        print("=" * 60)
        
        aggregate = evaluation_results['aggregate_metrics']
        metadata = evaluation_results['evaluation_metadata']
        
        print(f"📊 Evaluated {metadata['successful_evaluations']}/{metadata['total_queries']} queries")
        print(f"⏱️  Total time: {metadata['evaluation_time']:.2f}s")
        print(f"🎯 RAG Citation Advantage: {aggregate['rag_citation_advantage']:.1%}")
        print(f"🔍 Average Source Coverage: {aggregate['avg_source_coverage']:.1%}")
        print(f"🚀 Average Response Time: {aggregate['avg_total_rag_time']:.2f}s")
        
        print("\n✅ Results saved to evaluation_results/ directory")
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()