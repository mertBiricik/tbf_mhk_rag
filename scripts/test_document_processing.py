#!/usr/bin/env python3
"""
Test Document Processing Pipeline
Demo script to test basketball document processing without Ollama.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_config_loading():
    """Test configuration loading."""
    print("🔧 Testing Configuration Loading...")
    try:
        from src.utils.config import config
        
        # Test basic config access
        chunk_size = config.get('document_processing.chunk_size', 800)
        collection_name = config.get('vector_db.collection_name', 'basketball_rules')
        
        print(f"✅ Config loaded successfully")
        print(f"   Chunk size: {chunk_size}")
        print(f"   Collection name: {collection_name}")
        
        # Test document types
        doc_types = config.get('basketball.document_types', {})
        print(f"   Document types configured: {len(doc_types)}")
        for doc_type, info in doc_types.items():
            print(f"     - {doc_type}: {info.get('file', 'Unknown')}")
        
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_document_loading():
    """Test document loading from source files."""
    print("\n📚 Testing Document Loading...")
    try:
        from src.document_processing.processor import BasketballDocumentProcessor
        
        processor = BasketballDocumentProcessor()
        
        # Test loading individual documents
        print("📋 Loading basketball rules documents...")
        documents = processor.load_all_documents()
        
        if not documents:
            print("❌ No documents loaded!")
            return False
        
        print(f"✅ Loaded {len(documents)} documents:")
        for i, doc in enumerate(documents):
            doc_type = doc.metadata.get('document_type', 'unknown')
            year = doc.metadata.get('year', 'unknown')
            content_length = len(doc.page_content)
            
            print(f"   {i+1}. {doc_type} ({year}): {content_length:,} characters")
            
            # Show preview
            preview = doc.page_content[:200].replace('\n', ' ')
            print(f"      Preview: {preview}...")
        
        return True, documents
    except Exception as e:
        print(f"❌ Document loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def test_document_chunking(documents):
    """Test document chunking."""
    print("\n✂️  Testing Document Chunking...")
    try:
        from src.document_processing.processor import BasketballDocumentProcessor
        
        processor = BasketballDocumentProcessor()
        
        all_chunks = []
        for doc in documents:
            print(f"📝 Chunking {doc.metadata.get('document_type', 'unknown')}...")
            chunks = processor.chunk_document(doc)
            all_chunks.extend(chunks)
            
            print(f"   Created {len(chunks)} chunks")
            
            # Show example chunk
            if chunks:
                example_chunk = chunks[0]
                chunk_preview = example_chunk.page_content[:150].replace('\n', ' ')
                rule_numbers = example_chunk.metadata.get('rule_numbers', [])
                
                print(f"   Example chunk: {chunk_preview}...")
                if rule_numbers:
                    print(f"   Detected rules: {rule_numbers}")
        
        print(f"\n✅ Total chunks created: {len(all_chunks)}")
        
        # Analyze chunk distribution
        chunk_sizes = [len(chunk.page_content) for chunk in all_chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        min_size = min(chunk_sizes) if chunk_sizes else 0
        max_size = max(chunk_sizes) if chunk_sizes else 0
        
        print(f"📊 Chunk statistics:")
        print(f"   Average size: {avg_size:.0f} characters")
        print(f"   Size range: {min_size} - {max_size} characters")
        
        return True, all_chunks
    except Exception as e:
        print(f"❌ Document chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def test_embedding_model():
    """Test embedding model loading and encoding."""
    print("\n🧠 Testing Embedding Model...")
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        # Load model
        model_name = "BAAI/bge-m3"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"📥 Loading {model_name} on {device}...")
        start_time = time.time()
        model = SentenceTransformer(model_name)
        model = model.to(device)
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Test with basketball terms
        test_texts = [
            "basketbol kuralları faul cezası",
            "basketball rules foul penalty", 
            "şut saati 24 saniye kuralı",
            "shot clock 24 second rule",
            "5 faul diskalifiye edilme",
            "5 fouls disqualification"
        ]
        
        print("🧪 Testing embeddings with Turkish/English basketball terms...")
        start_time = time.time()
        embeddings = model.encode(test_texts)
        encode_time = time.time() - start_time
        
        print(f"✅ Generated embeddings in {encode_time:.2f}s")
        print(f"   Input texts: {len(test_texts)}")
        print(f"   Embedding shape: {embeddings.shape}")
        
        # Test Turkish-English similarity
        import numpy as np
        
        similarities = []
        for i in range(0, len(test_texts), 2):
            if i+1 < len(test_texts):
                sim = np.dot(embeddings[i], embeddings[i+1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
                )
                similarities.append(sim)
                print(f"   '{test_texts[i][:30]}...' <-> '{test_texts[i+1][:30]}...': {sim:.3f}")
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        print(f"   Average Turkish-English similarity: {avg_similarity:.3f}")
        
        return True, model
    except Exception as e:
        print(f"❌ Embedding model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def create_simple_search_demo(chunks, model):
    """Create a simple search demo without vector database."""
    print("\n🔍 Creating Simple Search Demo...")
    try:
        import numpy as np
        
        print("📊 Generating embeddings for all chunks...")
        start_time = time.time()
        
        # Get text content from chunks
        chunk_texts = [chunk.page_content for chunk in chunks]
        
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(chunk_texts), batch_size):
            batch_texts = chunk_texts[i:i + batch_size]
            batch_embeddings = model.encode(batch_texts)
            all_embeddings.extend(batch_embeddings)
            print(f"   Processed {min(i + batch_size, len(chunk_texts))}/{len(chunk_texts)} chunks")
        
        embeddings_array = np.array(all_embeddings)
        process_time = time.time() - start_time
        
        print(f"✅ Generated {len(all_embeddings)} embeddings in {process_time:.2f}s")
        
        # Test queries
        test_queries = [
            "5 faul yapan oyuncuya ne olur?",
            "şut saati kaç saniyedir?",
            "basketbol sahasının boyutları",
            "teknik faul nedir?"
        ]
        
        print("\n🧪 Testing search with basketball queries...")
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            # Generate query embedding
            query_embedding = model.encode([query])[0]
            
            # Calculate similarities
            similarities = np.dot(embeddings_array, query_embedding) / (
                np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top results
            top_indices = np.argsort(similarities)[-3:][::-1]
            
            for idx, chunk_idx in enumerate(top_indices):
                similarity = similarities[chunk_idx]
                chunk = chunks[chunk_idx]
                
                preview = chunk.page_content[:100].replace('\n', ' ')
                doc_type = chunk.metadata.get('document_type', 'unknown')
                rule_numbers = chunk.metadata.get('rule_numbers', [])
                
                print(f"   {idx+1}. Score: {similarity:.3f} | {doc_type}")
                print(f"      Content: {preview}...")
                if rule_numbers:
                    print(f"      Rules: {rule_numbers}")
        
        return True
    except Exception as e:
        print(f"❌ Search demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demo function."""
    print("🏀 Basketball RAG Document Processing Demo")
    print("=" * 60)
    
    start_time = time.time()
    
    # Test steps
    steps = [
        ("Configuration Loading", test_config_loading),
        ("Document Loading", test_document_loading),
    ]
    
    documents = []
    chunks = []
    model = None
    
    # Run basic tests
    for step_name, step_func in steps:
        print(f"\n{'='*60}")
        try:
            if step_name == "Document Loading":
                success, documents = step_func()
            else:
                success = step_func()
            
            if not success:
                print(f"❌ {step_name} failed - stopping demo")
                return False
        except Exception as e:
            print(f"❌ {step_name} failed with error: {e}")
            return False
    
    # Test chunking if we have documents
    if documents:
        print(f"\n{'='*60}")
        try:
            success, chunks = test_document_chunking(documents)
            if not success:
                print("❌ Document chunking failed - stopping demo")
                return False
        except Exception as e:
            print(f"❌ Document chunking failed: {e}")
            return False
    
    # Test embedding model
    print(f"\n{'='*60}")
    try:
        success, model = test_embedding_model()
        if not success:
            print("⚠️  Embedding model failed - skipping search demo")
        elif chunks and model:
            # Create search demo
            print(f"\n{'='*60}")
            create_simple_search_demo(chunks, model)
    except Exception as e:
        print(f"❌ Embedding model test failed: {e}")
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 DEMO SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Total time: {total_time:.2f}s")
    print(f"📚 Documents processed: {len(documents)}")
    print(f"✂️  Chunks created: {len(chunks)}")
    print(f"🧠 Embedding model: {'✅ Loaded' if model else '❌ Failed'}")
    print(f"\n🎯 Next steps:")
    print("   1. Install Ollama for LLM functionality")
    print("   2. Set up vector database (ChromaDB)")
    print("   3. Create complete RAG pipeline")
    print("   4. Build web interface")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 