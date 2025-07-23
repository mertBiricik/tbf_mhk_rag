"""
Optimized Document Processing for Basketball RAG
Implements advanced chunking strategies and parallel processing.
"""

import re
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time

import numpy as np
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticChunker:
    """Chunk documents based on semantic similarity."""
    
    def __init__(self, embedding_model_name: str = "BAAI/bge-m3"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.similarity_threshold = 0.7
        
    def chunk_by_semantics(self, text: str, max_chunk_size: int = 800) -> List[str]:
        """Split text into semantically coherent chunks."""
        # First split by sentences
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [text]
        
        # Generate embeddings for sentences
        embeddings = self.embedding_model.encode(sentences)
        
        # Find semantic boundaries
        chunks = []
        current_chunk = [sentences[0]]
        current_length = len(sentences[0])
        
        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            similarity = cosine_similarity(
                [embeddings[i-1]], [embeddings[i]]
            )[0][0]
            
            sentence_length = len(sentences[i])
            
            # Decide whether to start new chunk
            should_split = (
                similarity < self.similarity_threshold or  # Low semantic similarity
                current_length + sentence_length > max_chunk_size  # Size limit
            )
            
            if should_split and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
                current_length = sentence_length
            else:
                current_chunk.append(sentences[i])
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with basketball-specific rules."""
        # Basketball-specific sentence boundaries
        patterns = [
            r'(?<=\.)\s+(?=[A-ZÜĞŞÇÖ])',  # Turkish uppercase after period
            r'(?<=\.)\s+(?=Madde\s+\d+)',  # Rule articles
            r'(?<=\.)\s+(?=Rule\s+\d+)',   # English rules
            r'(?<=\.)\s+(?=Kural\s+\d+)',  # Turkish rules
            r'(?<=:)\s+(?=[A-ZÜĞŞÇÖ])',    # After colons
        ]
        
        sentences = [text]
        for pattern in patterns:
            new_sentences = []
            for sentence in sentences:
                parts = re.split(pattern, sentence)
                new_sentences.extend([p.strip() for p in parts if p.strip()])
            sentences = new_sentences
        
        return [s for s in sentences if len(s) > 10]  # Filter very short sentences

class ParallelDocumentProcessor:
    """Process documents in parallel for better performance."""
    
    def __init__(self, 
                 use_semantic_chunking: bool = True,
                 num_workers: Optional[int] = None):
        self.logger = logging.getLogger(__name__)
        self.use_semantic_chunking = use_semantic_chunking
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        
        if use_semantic_chunking:
            self.semantic_chunker = SemanticChunker()
        
        # Enhanced text splitter
        self.text_splitter = EnhancedBasketballTextSplitter()
        
    def process_documents_parallel(self, 
                                 documents: List[Document]) -> List[Document]:
        """Process multiple documents in parallel."""
        start_time = time.time()
        
        self.logger.info(f"Processing {len(documents)} documents with {self.num_workers} workers...")
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._process_single_document, doc)
                for doc in documents
            ]
            
            all_chunks = []
            for future in futures:
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    self.logger.error(f"Document processing failed: {e}")
        
        processing_time = time.time() - start_time
        self.logger.info(f"Processed {len(all_chunks)} chunks in {processing_time:.2f}s")
        
        return all_chunks
    
    def _process_single_document(self, document: Document) -> List[Document]:
        """Process a single document with optimizations."""
        # Clean and preprocess
        cleaned_content = self._deep_clean_content(document.page_content)
        
        # Choose chunking strategy
        if self.use_semantic_chunking:
            chunks = self.semantic_chunker.chunk_by_semantics(cleaned_content)
        else:
            chunks = self.text_splitter.split_text(cleaned_content)
        
        # Create document objects with enhanced metadata
        chunk_documents = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip very short chunks
                continue
            
            # Enhanced metadata
            metadata = document.metadata.copy()
            metadata.update({
                'chunk_id': i,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk),
                'chunk_words': len(chunk.split()),
                'rule_density': self._calculate_rule_density(chunk),
                'semantic_coherence': self._calculate_coherence(chunk)
            })
            
            # Extract basketball-specific features
            basketball_features = self._extract_basketball_features(chunk)
            metadata.update(basketball_features)
            
            chunk_documents.append(Document(
                page_content=chunk,
                metadata=metadata
            ))
        
        return chunk_documents
    
    def _deep_clean_content(self, content: str) -> str:
        """Enhanced content cleaning for basketball documents."""
        # Original cleaning
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r'\s+', ' ', content)
        
        # Basketball-specific cleaning
        replacements = {
            # Fix encoding issues
            'Ä±': 'ı', 'Å': 'ş', 'Ä': 'ğ', 'Ã': 'ü', 
            'Ã§': 'ç', 'Ã¶': 'ö', 'Ä°': 'İ', 'Åž': 'Ş',
            
            # Standardize terminology
            'basketbol sahasÄ±': 'basketbol sahası',
            'oyuncular': 'oyuncular',
            'faul sayÄ±sÄ±': 'faul sayısı',
        }
        
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        # Normalize rule references
        content = re.sub(r'MADDE\s*(\d+)', r'Madde \1', content, flags=re.IGNORECASE)
        content = re.sub(r'RULE\s*(\d+)', r'Rule \1', content, flags=re.IGNORECASE)
        
        # Fix spacing around punctuation
        content = re.sub(r'\s*([,.;:])\s*', r'\1 ', content)
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()
    
    def _calculate_rule_density(self, text: str) -> float:
        """Calculate density of basketball rules in text."""
        rule_patterns = [
            r'Madde\s+\d+',
            r'Rule\s+\d+', 
            r'Kural\s+\d+',
            r'\d+\.\d+',  # Sub-rules
            r'faul|foul',
            r'basketbol|basketball',
            r'saha|court',
            r'oyuncu|player'
        ]
        
        rule_matches = 0
        for pattern in rule_patterns:
            rule_matches += len(re.findall(pattern, text, re.IGNORECASE))
        
        words = len(text.split())
        return rule_matches / words if words > 0 else 0
    
    def _calculate_coherence(self, text: str) -> float:
        """Calculate semantic coherence score."""
        sentences = text.split('.')
        if len(sentences) < 2:
            return 1.0
        
        # Simple coherence based on repeated terms
        words = set(word.lower() for sentence in sentences 
                   for word in sentence.split() if len(word) > 3)
        
        if not words:
            return 0.5
        
        # Count word repetitions across sentences
        total_repetitions = 0
        for word in words:
            count = sum(1 for sentence in sentences if word in sentence.lower())
            if count > 1:
                total_repetitions += count
        
        max_possible = len(sentences) * len(words)
        return total_repetitions / max_possible if max_possible > 0 else 0.5
    
    def _extract_basketball_features(self, text: str) -> Dict[str, Any]:
        """Extract basketball-specific features from text."""
        features = {}
        
        # Rule types
        if re.search(r'faul|foul', text, re.IGNORECASE):
            features['contains_foul_rules'] = True
        if re.search(r'saha|court|boyut|dimension', text, re.IGNORECASE):
            features['contains_court_specs'] = True
        if re.search(r'zaman|time|süre|dakika|saniye', text, re.IGNORECASE):
            features['contains_time_rules'] = True
        if re.search(r'şut|shot|atış', text, re.IGNORECASE):
            features['contains_shooting_rules'] = True
        
        # Extract numeric values (useful for measurements, times, etc.)
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if numbers:
            features['contains_measurements'] = True
            features['numeric_values'] = [float(n) for n in numbers[:5]]  # Limit to first 5
        
        # Language detection
        turkish_chars = sum(1 for c in text if c in 'çğıöşüÇĞIİÖŞÜ')
        features['turkish_char_density'] = turkish_chars / len(text) if text else 0
        
        return features

class EnhancedBasketballTextSplitter(RecursiveCharacterTextSplitter):
    """Enhanced text splitter with basketball-specific optimizations."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        # Basketball-optimized separators
        separators = [
            "\n\nMadde ",      # Turkish rule articles (highest priority)
            "\n\nRule ",       # English rule articles
            "\n\nKural ",      # Turkish rule sections
            "\n\n## ",         # Markdown headers
            "\n\n# ",          # Markdown headers
            "\n\n",            # Paragraph breaks
            "\n• ",            # Bullet points
            "\n",              # Line breaks
            ". ",              # Sentence endings
            "; ",              # Semicolons
            ", ",              # Commas
            " ",               # Word breaks
        ]
        
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=True,
            length_function=len
        )
    
    def split_text(self, text: str) -> List[str]:
        """Enhanced text splitting with basketball-specific rules."""
        # Pre-process to mark important boundaries
        text = self._mark_rule_boundaries(text)
        
        # Use parent implementation
        chunks = super().split_text(text)
        
        # Post-process chunks
        enhanced_chunks = []
        for chunk in chunks:
            # Ensure chunks don't break in middle of rules
            adjusted_chunk = self._adjust_chunk_boundaries(chunk)
            if len(adjusted_chunk.strip()) > 50:  # Minimum chunk size
                enhanced_chunks.append(adjusted_chunk)
        
        return enhanced_chunks
    
    def _mark_rule_boundaries(self, text: str) -> str:
        """Mark rule boundaries to prevent splitting."""
        # Add markers before important rules
        patterns = [
            (r'(Madde\s+\d+[^.]*\.)', r'\n\n\1'),
            (r'(Rule\s+\d+[^.]*\.)', r'\n\n\1'),
            (r'(Kural\s+\d+[^.]*\.)', r'\n\n\1'),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        
        return text
    
    def _adjust_chunk_boundaries(self, chunk: str) -> str:
        """Adjust chunk boundaries to preserve semantic integrity."""
        # If chunk ends mid-sentence, try to include complete sentence
        if not chunk.endswith(('.', '!', '?', ':')):
            sentences = chunk.split('.')
            if len(sentences) > 1:
                # Keep complete sentences only
                complete_sentences = sentences[:-1]
                chunk = '.'.join(complete_sentences) + '.'
        
        return chunk.strip()

class OptimizedDocumentProcessor:
    """Main optimized processor combining all strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize parallel processor
        self.parallel_processor = ParallelDocumentProcessor(
            use_semantic_chunking=config.get('document_processing.use_semantic_chunking', True),
            num_workers=config.get('document_processing.num_workers', None)
        )
    
    def process_all_documents(self, document_configs: Dict[str, Any]) -> List[Document]:
        """Process all documents with optimizations."""
        start_time = time.time()
        
        # Load documents
        documents = self._load_all_documents(document_configs)
        
        # Process in parallel
        all_chunks = self.parallel_processor.process_documents_parallel(documents)
        
        # Post-processing optimizations
        optimized_chunks = self._post_process_chunks(all_chunks)
        
        total_time = time.time() - start_time
        self.logger.info(f"Optimized processing completed: {len(optimized_chunks)} chunks in {total_time:.2f}s")
        
        return optimized_chunks
    
    def _load_all_documents(self, document_configs: Dict[str, Any]) -> List[Document]:
        """Load all configured documents efficiently."""
        documents = []
        source_path = Path('source/txt')
        
        for doc_type, config in document_configs.items():
            file_path = source_path / config['file']
            
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    metadata = {
                        'source': str(file_path),
                        'document_type': config['type'],
                        'year': config['year'],
                        'priority': config['priority'],
                        'language': config.get('language', 'turkish'),
                        'file_name': config['file']
                    }
                    
                    documents.append(Document(page_content=content, metadata=metadata))
                    self.logger.info(f"Loaded {doc_type}: {len(content)} characters")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load {doc_type}: {e}")
        
        return documents
    
    def _post_process_chunks(self, chunks: List[Document]) -> List[Document]:
        """Post-process chunks for quality improvements."""
        # Filter out low-quality chunks
        filtered_chunks = []
        
        for chunk in chunks:
            # Quality filters
            if self._is_high_quality_chunk(chunk):
                # Enhance metadata
                chunk.metadata['processing_timestamp'] = time.time()
                chunk.metadata['quality_score'] = self._calculate_quality_score(chunk)
                filtered_chunks.append(chunk)
        
        # Sort by quality and priority
        filtered_chunks.sort(
            key=lambda x: (x.metadata.get('priority', 0), x.metadata.get('quality_score', 0)),
            reverse=True
        )
        
        self.logger.info(f"Filtered {len(chunks)} -> {len(filtered_chunks)} high-quality chunks")
        return filtered_chunks
    
    def _is_high_quality_chunk(self, chunk: Document) -> bool:
        """Determine if chunk meets quality standards."""
        content = chunk.page_content
        
        # Basic quality checks
        if len(content) < 50:
            return False
        
        # Should contain meaningful basketball content
        basketball_terms = ['basketbol', 'basketball', 'oyuncu', 'player', 'saha', 'court', 
                           'faul', 'foul', 'kural', 'rule', 'madde']
        
        has_basketball_content = any(term.lower() in content.lower() for term in basketball_terms)
        
        # Should not be just numbers or punctuation
        alpha_ratio = sum(c.isalpha() for c in content) / len(content)
        
        return has_basketball_content and alpha_ratio > 0.5
    
    def _calculate_quality_score(self, chunk: Document) -> float:
        """Calculate quality score for chunk ranking."""
        content = chunk.page_content
        metadata = chunk.metadata
        
        score = 0.0
        
        # Length score (optimal around 400-800 chars)
        length = len(content)
        if 200 <= length <= 1000:
            score += 2.0
        elif 100 <= length <= 1500:
            score += 1.0
        
        # Rule density score
        rule_density = metadata.get('rule_density', 0)
        score += min(rule_density * 10, 2.0)
        
        # Priority boost for recent changes
        if metadata.get('document_type') == 'changes' and metadata.get('year', 0) >= 2024:
            score += 1.0
        
        # Basketball feature bonuses
        if metadata.get('contains_foul_rules'):
            score += 0.5
        if metadata.get('contains_court_specs'):
            score += 0.5
        if metadata.get('contains_measurements'):
            score += 0.3
        
        return min(score, 10.0)  # Cap at 10 