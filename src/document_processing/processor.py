"""
Basketball Rules Document Processor
Handles loading, processing, and chunking of Turkish basketball documents.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chardet

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..utils.config import config


class BasketballDocumentProcessor:
    """Process basketball rules documents with Turkish language support."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Load document configurations
        self.document_configs = self.config.get('basketball.document_types', {})
        self.source_path = Path('source/txt')
        
    def detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding."""
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception as e:
            self.logger.warning(f"Could not detect encoding for {file_path}: {e}")
            return 'utf-8'
    
    def load_document(self, doc_type: str) -> Optional[Document]:
        """Load a single document by type."""
        if doc_type not in self.document_configs:
            self.logger.error(f"Unknown document type: {doc_type}")
            return None
            
        doc_config = self.document_configs[doc_type]
        file_path = self.source_path / doc_config['file']
        
        if not file_path.exists():
            self.logger.error(f"Document file not found: {file_path}")
            return None
        
        try:
            # Detect and use appropriate encoding
            encoding = self.detect_encoding(file_path)
            self.logger.info(f"Loading {doc_type} with encoding: {encoding}")
            
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            
            # Create metadata
            metadata = {
                'source': str(file_path),
                'document_type': doc_config['type'],
                'year': doc_config['year'],
                'priority': doc_config['priority'],
                'language': doc_config.get('language', 'turkish'),
                'file_name': doc_config['file']
            }
            
            # Clean content
            content = self._clean_content(content, doc_type)
            
            self.logger.info(f"✅ Loaded {doc_type}: {len(content)} characters")
            return Document(page_content=content, metadata=metadata)
            
        except Exception as e:
            self.logger.error(f"Error loading {doc_type}: {e}")
            return None
    
    def _clean_content(self, content: str, doc_type: str) -> str:
        """Clean and normalize document content."""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r'\s+', ' ', content)
        
        # Normalize Turkish characters
        content = self._normalize_turkish_text(content)
        
        # Remove page numbers and headers/footers
        content = re.sub(r'Sayfa \d+', '', content)
        content = re.sub(r'Page \d+', '', content)
        
        # Clean up basketball-specific formatting
        if 'rules' in doc_type or 'kural' in doc_type:
            content = self._clean_rule_formatting(content)
        
        return content.strip()
    
    def _normalize_turkish_text(self, text: str) -> str:
        """Normalize Turkish text."""
        # Fix common encoding issues
        replacements = {
            'Ä±': 'ı', 'Å': 'ş', 'Ä': 'ğ', 'Ã': 'ü', 
            'Ã§': 'ç', 'Ã¶': 'ö', 'Ä°': 'İ', 'Åž': 'Ş',
            'ÄŸ': 'ğ', 'Ã': 'Ü', 'Ã–': 'Ö', 'Ã‡': 'Ç'
        }
        
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        
        return text
    
    def _clean_rule_formatting(self, content: str) -> str:
        """Clean basketball rule-specific formatting."""
        # Normalize rule numbers
        content = re.sub(r'MADDE\s*(\d+)', r'Madde \1', content, flags=re.IGNORECASE)
        content = re.sub(r'RULE\s*(\d+)', r'Rule \1', content, flags=re.IGNORECASE)
        content = re.sub(r'KURAL\s*(\d+)', r'Kural \1', content, flags=re.IGNORECASE)
        
        # Fix bullet points
        content = re.sub(r'•\s*', '• ', content)
        content = re.sub(r'\*\s*', '• ', content)
        
        return content
    
    def load_all_documents(self) -> List[Document]:
        """Load all configured documents."""
        documents = []
        
        for doc_type in self.document_configs.keys():
            doc = self.load_document(doc_type)
            if doc:
                documents.append(doc)
        
        self.logger.info(f"✅ Loaded {len(documents)} documents total")
        return documents
    
    def chunk_document(self, document: Document) -> List[Document]:
        """Chunk a single document into smaller pieces."""
        chunk_size = self.config.get('document_processing.chunk_size', 800)
        chunk_overlap = self.config.get('document_processing.chunk_overlap', 100)
        separators = self.config.get('document_processing.separators', ["\n\n", "\n", ". ", " "])
        
        # Use basketball-specific chunking
        splitter = BasketballTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
        
        chunks = splitter.split_documents([document])
        
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk.page_content)
            })
            
            # Extract rule numbers if present
            rule_numbers = self._extract_rule_numbers(chunk.page_content)
            if rule_numbers:
                chunk.metadata['rule_numbers'] = rule_numbers
        
        self.logger.info(f"✅ Created {len(chunks)} chunks from {document.metadata.get('document_type', 'document')}")
        return chunks
    
    def _extract_rule_numbers(self, text: str) -> List[str]:
        """Extract rule numbers from text."""
        patterns = [
            r'Madde\s*(\d+(?:\.\d+)?)',
            r'Rule\s*(\d+(?:\.\d+)?)',
            r'Kural\s*(\d+(?:\.\d+)?)',
            r'(\d+)\.(\d+)\.',  # For sub-rules like 5.1
        ]
        
        rule_numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    rule_numbers.append('.'.join(match))
                else:
                    rule_numbers.append(match)
        
        return list(set(rule_numbers))  # Remove duplicates
    
    def process_all_documents(self) -> List[Document]:
        """Load and chunk all documents."""
        all_chunks = []
        
        documents = self.load_all_documents()
        
        for document in documents:
            chunks = self.chunk_document(document)
            all_chunks.extend(chunks)
        
        self.logger.info(f"✅ Total processed chunks: {len(all_chunks)}")
        return all_chunks


class BasketballTextSplitter(RecursiveCharacterTextSplitter):
    """Custom text splitter optimized for basketball rules."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100, separators: List[str] = None):
        if separators is None:
            # Basketball-specific separators
            separators = [
                "\n\nMadde",  # Turkish rule articles
                "\n\nRule",   # English rule articles  
                "\n\nKural",  # Turkish rule sections
                "\n\n",       # Paragraph breaks
                "\n",         # Line breaks
                ". ",         # Sentence endings
                " ",          # Word breaks
            ]
        
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=True
        )
    
    def split_text(self, text: str) -> List[str]:
        """Split text with basketball-specific logic."""
        # First, try to split by rule boundaries
        rule_chunks = self._split_by_rules(text)
        
        # Then apply standard recursive splitting to each rule chunk
        final_chunks = []
        for rule_chunk in rule_chunks:
            if len(rule_chunk) <= self._chunk_size:
                final_chunks.append(rule_chunk)
            else:
                # Use parent class method for oversized chunks
                sub_chunks = super().split_text(rule_chunk)
                final_chunks.extend(sub_chunks)
        
        return final_chunks
    
    def _split_by_rules(self, text: str) -> List[str]:
        """Split text by basketball rule boundaries."""
        # Pattern to match rule beginnings
        rule_pattern = r'((?:Madde|Rule|Kural)\s*\d+(?:\.\d+)?)'
        
        # Find all rule matches
        matches = list(re.finditer(rule_pattern, text, re.IGNORECASE))
        
        if not matches:
            return [text]
        
        chunks = []
        
        # Add content before first rule
        if matches[0].start() > 0:
            chunks.append(text[:matches[0].start()].strip())
        
        # Process each rule section
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            rule_text = text[start:end].strip()
            if rule_text:
                chunks.append(rule_text)
        
        return [chunk for chunk in chunks if chunk.strip()] 