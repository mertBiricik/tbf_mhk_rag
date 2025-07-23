"""
Optimized Generation Pipeline for Basketball RAG
Implements advanced prompt engineering, caching, and streaming.
"""

import time
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Iterator, Tuple
from dataclasses import dataclass
from pathlib import Path

import requests

# Language detection function (simplified implementation)
def detect_language(text: str) -> str:
    """Simple language detection for Turkish vs English."""
    # Turkish specific characters and common words
    turkish_chars = set('çğıöşüÇĞIİÖŞÜ')
    turkish_words = {'ve', 'bir', 'bu', 'için', 'olan', 'ile', 'den', 'dır', 'lar', 'ler', 'nın', 'nin', 
                     'basketbol', 'oyuncu', 'faul', 'kural', 'sahası', 'atış', 'saha', 'oyun', 'yapan', 
                     'zaman', 'süre', 'dakika', 'saniye', 'şut', 'top', 'takım', 'oyuncuya', 'nedir', 
                     'nasıl', 'nelerdir', 'hangi', 'değişti', 'kuralları', 'boyutları'}
    
    # English common words
    english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                     'basketball', 'player', 'foul', 'rule', 'court', 'shot', 'game', 'team', 'what',
                     'how', 'when', 'where', 'why', 'rules', 'dimensions', 'changed', 'which'}
    
    text_lower = text.lower()
    
    # Count Turkish characters
    turkish_char_count = sum(1 for char in text if char in turkish_chars)
    
    # Count Turkish words
    turkish_word_count = sum(1 for word in turkish_words if word in text_lower)
    
    # Count English words
    english_word_count = sum(1 for word in english_words if word in text_lower)
    
    # Decision logic
    if turkish_char_count > 0 or turkish_word_count > english_word_count:
        return 'turkish'
    else:
        return 'english'

@dataclass
class GenerationRequest:
    query: str
    context_chunks: List[Dict[str, Any]]
    language: str = 'auto'
    max_tokens: int = 2048
    temperature: float = 0.1
    use_cache: bool = True

@dataclass 
class GenerationResponse:
    answer: str
    sources: List[str]
    language: str
    generation_time: float
    from_cache: bool = False
    confidence_score: float = 0.0

class PromptTemplateManager:
    """Manage optimized prompts for different query types."""
    
    def __init__(self):
        self.templates = {
            'foul_rules': {
                'turkish': """Sen Türkiye Basketbol Federasyonu kurallarının uzmanısın. Aşağıdaki resmi belgelerden yalnızca faul kurallarıyla ilgili soruya yanıt ver.

RESMI BELGELER:
{context}

SORU: {query}

YANIT KURALLARI:
- Sadece verilen belgelerden bilgi kullan
- Faul türünü açıkça belirt
- Madde numaralarını dahil et
- Kesin ve net yanıt ver
- Türkçe yanıtla

YANIT:""",
                'english': """You are a Türkiye Basketball Federation rules expert. Answer the foul-related question using only the official documents below.

OFFICIAL DOCUMENTS:
{context}

QUESTION: {query}

ANSWER RULES:
- Use only information from provided documents
- Clearly specify foul type
- Include article numbers
- Give precise and clear answer
- Answer in English

ANSWER:"""
            },
            'court_specs': {
                'turkish': """Sen basketbol saha ölçüleri uzmanısın. Aşağıdaki resmi belgelerden saha boyutları/özellikleri sorusuna yanıt ver.

RESMI BELGELER:
{context}

SORU: {query}

YANIT KURALLARI:
- Sadece verilen belgelerden ölçü bilgilerini kullan
- Tam ölçüleri metre cinsinden ver
- Kaynak belgeyi belirt
- Net sayısal değerler kullan
- Türkçe yanıtla

YANIT:""",
                'english': """You are a basketball court specifications expert. Answer the court dimensions/features question using the official documents below.

OFFICIAL DOCUMENTS:
{context}

QUESTION: {query}

ANSWER RULES:
- Use only measurement information from provided documents
- Give exact measurements in meters
- Specify source document
- Use precise numerical values
- Answer in English

ANSWER:"""
            },
            'rule_changes': {
                'turkish': """Sen 2024 basketbol kural değişiklikleri uzmanısın. Aşağıdaki resmi belgelerden kural değişiklikleriyle ilgili soruya yanıt ver.

RESMI BELGELER:
{context}

SORU: {query}

YANIT KURALLARI:
- Sadece 2024 değişikliklerini vurgula
- Eski kural ile yeni kuralı karşılaştır
- Değişikliğin ne zaman yürürlüğe girdiğini belirt
- Kaynak belgeyi dahil et
- Türkçe yanıtla

YANIT:""",
                'english': """You are a 2024 basketball rule changes expert. Answer the rule changes question using the official documents below.

OFFICIAL DOCUMENTS:
{context}

QUESTION: {query}

ANSWER RULES:
- Focus only on 2024 changes
- Compare old rule with new rule
- Specify when change takes effect
- Include source document
- Answer in English

ANSWER:"""
            },
            'general': {
                'turkish': """Sen Türkiye Basketbol Federasyonu kuralları uzmanısın. Aşağıdaki resmi belgelerden soruya yanıt ver.

RESMI BELGELER:
{context}

SORU: {query}

YANIT KURALLARI:
- Sadece verilen belgelerden bilgi kullan
- Kaynak belgeyi [Kaynak: ...] formatında belirt
- Net ve anlaşılır yanıt ver
- Madde numaralarını dahil et
- Türkçe yanıtla

YANIT:""",
                'english': """You are a Türkiye Basketball Federation rules expert. Answer the question using the official documents below.

OFFICIAL DOCUMENTS:
{context}

QUESTION: {query}

ANSWER RULES:
- Use only information from provided documents
- Cite source documents as [Source: ...]
- Give clear and understandable answer
- Include article numbers
- Answer in English

ANSWER:"""
            }
        }
    
    def get_template(self, query: str, language: str) -> str:
        """Get the most appropriate template for query and language."""
        query_lower = query.lower()
        
        # Determine query type
        if any(term in query_lower for term in ['faul', 'foul']):
            template_type = 'foul_rules'
        elif any(term in query_lower for term in ['saha', 'court', 'boyut', 'dimension', 'ölçü', 'measurement']):
            template_type = 'court_specs'
        elif any(term in query_lower for term in ['2024', 'değişiklik', 'change', 'yeni', 'new']):
            template_type = 'rule_changes'
        else:
            template_type = 'general'
        
        return self.templates[template_type][language]

class ResponseCache:
    """Cache generation responses for improved performance."""
    
    def __init__(self, cache_size: int = 200, cache_ttl: int = 3600):
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl  # Time to live in seconds
        self.cache = {}
        self.access_times = {}
        
    def _hash_request(self, request: GenerationRequest) -> str:
        """Create hash for caching."""
        # Create a deterministic hash from query and context
        context_hash = hashlib.md5(
            str(sorted([chunk['content'][:100] for chunk in request.context_chunks])).encode()
        ).hexdigest()[:8]
        
        query_hash = hashlib.md5(request.query.lower().encode()).hexdigest()[:8]
        
        return f"{query_hash}_{context_hash}_{request.language}"
    
    def get(self, request: GenerationRequest) -> Optional[GenerationResponse]:
        """Get cached response if available and not expired."""
        cache_key = self._hash_request(request)
        
        if cache_key in self.cache:
            cached_time, response = self.cache[cache_key]
            
            # Check if expired
            if time.time() - cached_time < self.cache_ttl:
                self.access_times[cache_key] = time.time()
                response.from_cache = True
                return response
            else:
                # Remove expired entry
                del self.cache[cache_key]
                if cache_key in self.access_times:
                    del self.access_times[cache_key]
        
        return None
    
    def put(self, request: GenerationRequest, response: GenerationResponse):
        """Cache the response."""
        cache_key = self._hash_request(request)
        
        # LRU eviction if cache full
        if len(self.cache) >= self.cache_size:
            # Remove least recently used entry
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[cache_key] = (time.time(), response)
        self.access_times[cache_key] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'max_size': self.cache_size,
            'ttl_seconds': self.cache_ttl
        }

class ContextOptimizer:
    """Optimize context for better generation quality."""
    
    def __init__(self, max_context_length: int = 4000):
        self.max_context_length = max_context_length
        
    def optimize_context(self, 
                        query: str, 
                        context_chunks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """Optimize context for generation."""
        if not context_chunks:
            return "", []
        
        # Sort chunks by relevance
        sorted_chunks = self._sort_by_relevance(query, context_chunks)
        
        # Build context within length limit
        context_parts = []
        sources = []
        current_length = 0
        
        for i, chunk in enumerate(sorted_chunks):
            content = chunk['content']
            metadata = chunk['metadata']
            
            # Format source info
            source_info = self._format_source(metadata, i + 1)
            formatted_chunk = f"{source_info}:\n{content}"
            
            # Check if adding this chunk would exceed limit
            if current_length + len(formatted_chunk) > self.max_context_length:
                break
            
            context_parts.append(formatted_chunk)
            sources.append(f"{metadata.get('file_name', 'unknown')} ({metadata.get('year', 'unknown')})")
            current_length += len(formatted_chunk)
        
        context_text = "\n\n".join(context_parts)
        return context_text, sources
    
    def _sort_by_relevance(self, 
                          query: str, 
                          chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort chunks by relevance to query."""
        def relevance_score(chunk):
            content = chunk['content'].lower()
            metadata = chunk['metadata']
            score = chunk.get('score', 0.5)
            
            # Boost for query keywords
            query_words = query.lower().split()
            keyword_matches = sum(1 for word in query_words if word in content)
            score += keyword_matches * 0.1
            
            # Boost for basketball relevance
            if metadata.get('contains_foul_rules') and 'faul' in query.lower():
                score += 0.2
            if metadata.get('contains_court_specs') and any(term in query.lower() for term in ['saha', 'court']):
                score += 0.2
            if metadata.get('document_type') == 'changes' and '2024' in query:
                score += 0.3
            
            # Boost for high priority documents
            priority = metadata.get('priority', 1)
            score += priority * 0.1
            
            return score
        
        return sorted(chunks, key=relevance_score, reverse=True)
    
    def _format_source(self, metadata: Dict[str, Any], index: int) -> str:
        """Format source information for context."""
        doc_type = metadata.get('document_type', 'unknown')
        year = metadata.get('year', 'unknown')
        
        type_names = {
            'rules': 'Basketbol Kuralları',
            'changes': 'Kural Değişiklikleri', 
            'interpretations': 'Resmi Yorumlar'
        }
        
        type_name = type_names.get(doc_type, doc_type.title())
        return f"Kaynak {index} - {type_name} {year}"

class OptimizedGenerator:
    """Optimized generation pipeline with caching and streaming."""
    
    def __init__(self, 
                 llm_model: str = "llama3.1:8b-instruct-q4_K_M",
                 base_url: str = "http://localhost:11434",
                 enable_caching: bool = True,
                 enable_streaming: bool = False):
        
        self.logger = logging.getLogger(__name__)
        self.llm_model = llm_model
        self.base_url = base_url
        self.enable_streaming = enable_streaming
        
        # Initialize components
        self.prompt_manager = PromptTemplateManager()
        self.context_optimizer = ContextOptimizer()
        self.cache = ResponseCache() if enable_caching else None
        
        # Performance tracking
        self.generation_count = 0
        self.total_generation_time = 0.0
        self.cache_hits = 0
        
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate response with all optimizations."""
        start_time = time.time()
        
        # Check cache first
        if request.use_cache and self.cache:
            cached_response = self.cache.get(request)
            if cached_response:
                self.cache_hits += 1
                self.logger.debug("Cache hit for query")
                return cached_response
        
        # Detect language if auto
        if request.language == 'auto':
            request.language = detect_language(request.query)
        
        # Optimize context
        context_text, sources = self.context_optimizer.optimize_context(
            request.query, request.context_chunks
        )
        
        if not context_text:
            return self._create_no_context_response(request)
        
        # Generate response
        if self.enable_streaming:
            response = self._generate_streaming(request, context_text, sources)
        else:
            response = self._generate_standard(request, context_text, sources)
        
        # Calculate generation time
        response.generation_time = time.time() - start_time
        
        # Cache response
        if request.use_cache and self.cache:
            self.cache.put(request, response)
        
        # Update stats
        self.generation_count += 1
        self.total_generation_time += response.generation_time
        
        return response
    
    def _generate_standard(self, 
                          request: GenerationRequest,
                          context_text: str,
                          sources: List[str]) -> GenerationResponse:
        """Standard generation without streaming."""
        
        # Get appropriate prompt template
        template = self.prompt_manager.get_template(request.query, request.language)
        
        # Create prompt
        prompt = template.format(
            context=context_text,
            query=request.query
        )
        
        # Generate response
        try:
            response = self._call_llm(prompt, request)
            
            # Extract answer and calculate confidence
            answer = response.strip()
            confidence = self._calculate_confidence(answer, sources)
            
            return GenerationResponse(
                answer=answer,
                sources=sources,
                language=request.language,
                generation_time=0.0,  # Will be set by caller
                confidence_score=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return self._create_error_response(request, str(e))
    
    def _generate_streaming(self, 
                           request: GenerationRequest,
                           context_text: str,
                           sources: List[str]) -> GenerationResponse:
        """Streaming generation (if supported by LLM)."""
        # For now, fall back to standard generation
        # Streaming can be implemented based on specific LLM API
        return self._generate_standard(request, context_text, sources)
    
    def _call_llm(self, prompt: str, request: GenerationRequest) -> str:
        """Call the LLM with optimized parameters."""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "stop": ["SORU:", "QUESTION:", "YANIT:", "ANSWER:"]
            },
            "stream": False
        }
        
        try:
            response = requests.post(
                url, 
                json=payload, 
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"LLM API error: {e}")
        except Exception as e:
            raise Exception(f"LLM call failed: {e}")
    
    def _calculate_confidence(self, answer: str, sources: List[str]) -> float:
        """Calculate confidence score for the answer."""
        if not answer or len(answer.strip()) < 10:
            return 0.1
        
        score = 0.5  # Base score
        
        # Length appropriateness
        word_count = len(answer.split())
        if 20 <= word_count <= 200:
            score += 0.2
        elif 10 <= word_count <= 300:
            score += 0.1
        
        # Has sources
        if sources:
            score += 0.2
        
        # Contains citations
        if '[' in answer and ']' in answer:
            score += 0.1
        
        # Basketball terminology presence
        basketball_terms = ['basketbol', 'basketball', 'faul', 'foul', 'saha', 'court', 'oyuncu', 'player']
        term_count = sum(1 for term in basketball_terms if term.lower() in answer.lower())
        score += min(term_count * 0.05, 0.2)
        
        return min(score, 1.0)
    
    def _create_no_context_response(self, request: GenerationRequest) -> GenerationResponse:
        """Create response when no context is available."""
        if request.language == 'english':
            answer = "I don't have enough information to answer this question. Please try rephrasing or asking about specific basketball rules."
        else:
            answer = "Bu soruyu yanıtlamak için yeterli bilgim yok. Lütfen sorunuzu yeniden ifade edin veya belirli basketbol kuralları hakkında soru sorun."
        
        return GenerationResponse(
            answer=answer,
            sources=[],
            language=request.language,
            generation_time=0.0,
            confidence_score=0.1
        )
    
    def _create_error_response(self, request: GenerationRequest, error: str) -> GenerationResponse:
        """Create response for errors."""
        if request.language == 'english':
            answer = f"Sorry, I encountered an error while processing your question: {error}"
        else:
            answer = f"Üzgünüm, sorunuzu işlerken bir hata oluştu: {error}"
        
        return GenerationResponse(
            answer=answer,
            sources=[],
            language=request.language,
            generation_time=0.0,
            confidence_score=0.0
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get generation performance statistics."""
        avg_time = self.total_generation_time / self.generation_count if self.generation_count > 0 else 0
        cache_hit_rate = self.cache_hits / self.generation_count if self.generation_count > 0 else 0
        
        stats = {
            'total_generations': self.generation_count,
            'avg_generation_time': avg_time,
            'total_generation_time': self.total_generation_time,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate
        }
        
        if self.cache:
            stats['cache'] = self.cache.get_stats()
        
        return stats
    
    def warm_up(self, sample_queries: List[str]):
        """Warm up the generation pipeline."""
        self.logger.info("Warming up generation pipeline...")
        
        for query in sample_queries[:3]:  # Limit warmup queries
            try:
                # Simple test request
                test_request = GenerationRequest(
                    query=query,
                    context_chunks=[{
                        'content': 'Test content for warmup',
                        'metadata': {'document_type': 'test'},
                        'score': 0.5
                    }],
                    use_cache=False  # Don't cache warmup
                )
                
                self._call_llm("Test prompt", test_request)
                self.logger.debug(f"Warmup query successful: {query[:50]}...")
                
            except Exception as e:
                self.logger.warning(f"Warmup failed for query '{query}': {e}")
        
        self.logger.info("Generation pipeline warmup completed") 