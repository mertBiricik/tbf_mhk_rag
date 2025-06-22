#!/usr/bin/env python3
"""
Simple Basketball RAG Demo
Tests core functionality without complex imports.
"""

import os
import time
import sys
from pathlib import Path

def test_basic_imports():
    """Test if we can import the key packages."""
    print("🔧 Testing Basic Imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers available")
    except ImportError:
        print("❌ SentenceTransformers not available")
        return False
    
    try:
        import chromadb
        print("✅ ChromaDB available")
    except ImportError:
        print("❌ ChromaDB not available")
        return False
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("✅ LangChain available")
    except ImportError:
        print("❌ LangChain not available")
        return False
    
    return True

def test_source_documents():
    """Test if source documents exist and are readable."""
    print("\n📚 Testing Source Documents...")
    
    source_dir = Path("source/txt")
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return False, []
    
    required_files = [
        "basketbol_oyun_kurallari_2022.txt",
        "basketbol_oyun_kurallari_degisiklikleri_2024.txt", 
        "basketbol_oyun_kurallari_resmi_yorumlar_2023.txt"
    ]
    
    documents = []
    
    for filename in required_files:
        filepath = source_dir / filename
        if not filepath.exists():
            print(f"❌ Missing: {filename}")
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_size = len(content)
            print(f"✅ {filename}: {file_size:,} characters")
            
            # Show preview
            preview = content[:100].replace('\n', ' ')
            print(f"   Preview: {preview}...")
            
            documents.append({
                'filename': filename,
                'content': content,
                'size': file_size
            })
            
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
    
    return len(documents) > 0, documents

def test_text_chunking(documents):
    """Test basic text chunking."""
    print("\n✂️  Testing Text Chunking...")
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Create text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\nMadde", "\n\nRule", "\n\nKural", "\n\n", "\n", ". ", " "]
        )
        
        all_chunks = []
        
        for doc in documents:
            print(f"📝 Chunking {doc['filename']}...")
            
            # Split the text
            chunks = splitter.split_text(doc['content'])
            
            print(f"   Created {len(chunks)} chunks")
            
            # Show example chunk
            if chunks:
                example = chunks[0][:150].replace('\n', ' ')
                print(f"   Example: {example}...")
            
            # Add to all chunks with metadata
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'text': chunk,
                    'source': doc['filename'],
                    'chunk_id': i,
                    'size': len(chunk)
                })
        
        print(f"\n✅ Total chunks: {len(all_chunks)}")
        
        # Statistics
        chunk_sizes = [chunk['size'] for chunk in all_chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        
        print(f"📊 Average chunk size: {avg_size:.0f} characters")
        print(f"   Size range: {min(chunk_sizes)} - {max(chunk_sizes)}")
        
        return True, all_chunks
        
    except Exception as e:
        print(f"❌ Chunking failed: {e}")
        return False, []

def test_embeddings(chunks):
    """Test embedding generation."""
    print("\n🧠 Testing Embeddings...")
    
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        # Load model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"📥 Loading BGE-M3 model on {device}...")
        
        start_time = time.time()
        model = SentenceTransformer('BAAI/bge-m3')
        model = model.to(device)
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Test with Turkish basketball terms
        test_texts = [
            "basketbol kuralları",
            "5 faul kuralı", 
            "şut saati 24 saniye",
            "teknik faul cezası"
        ]
        
        print("🧪 Testing with basketball terms...")
        start_time = time.time()
        embeddings = model.encode(test_texts)
        encode_time = time.time() - start_time
        
        print(f"✅ Generated embeddings in {encode_time:.2f}s")
        print(f"   Shape: {embeddings.shape}")
        
        # Test a few chunks
        print("\n📊 Testing chunk embeddings...")
        sample_chunks = chunks[:5]  # Test first 5 chunks
        chunk_texts = [chunk['text'] for chunk in sample_chunks]
        
        start_time = time.time()
        chunk_embeddings = model.encode(chunk_texts)
        chunk_time = time.time() - start_time
        
        print(f"✅ Generated {len(chunk_embeddings)} chunk embeddings in {chunk_time:.2f}s")
        
        return True, model
        
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_simple_search(chunks, model):
    """Test simple search without vector database."""
    print("\n🔍 Testing Simple Search...")
    
    try:
        import numpy as np
        
        # Take a subset of chunks for demo
        sample_chunks = chunks[:20]  # Use first 20 chunks for demo
        chunk_texts = [chunk['text'] for chunk in sample_chunks]
        
        print(f"📊 Generating embeddings for {len(sample_chunks)} chunks...")
        start_time = time.time()
        chunk_embeddings = model.encode(chunk_texts)
        embed_time = time.time() - start_time
        
        print(f"✅ Embeddings generated in {embed_time:.2f}s")
        
        # Test queries
        test_queries = [
            "5 faul yapan oyuncuya ne olur?",
            "basketbol sahasının boyutları nedir?",
            "şut saati kuralı"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            # Generate query embedding
            query_embedding = model.encode([query])[0]
            
            # Calculate similarities
            similarities = np.dot(chunk_embeddings, query_embedding) / (
                np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top result
            best_idx = np.argmax(similarities)
            best_chunk = sample_chunks[best_idx]
            best_score = similarities[best_idx]
            
            print(f"   Best match (score: {best_score:.3f}):")
            print(f"   Source: {best_chunk['source']}")
            content_preview = best_chunk['text'][:200].replace('\n', ' ')
            print(f"   Content: {content_preview}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def main():
    """Main demo function."""
    print("🏀 Simple Basketball RAG Demo")
    print("=" * 50)
    
    start_time = time.time()
    
    # Test basic imports
    if not test_basic_imports():
        print("❌ Basic imports failed - cannot continue")
        return False
    
    # Test source documents
    success, documents = test_source_documents()
    if not success:
        print("❌ No source documents found")
        return False
    
    # Test chunking
    success, chunks = test_text_chunking(documents)
    if not success:
        print("❌ Chunking failed")
        return False
    
    # Test embeddings
    success, model = test_embeddings(chunks)
    if not success:
        print("⚠️  Embedding test failed - skipping search")
        model = None
    
    # Test search if model loaded
    if model and chunks:
        test_simple_search(chunks, model)
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 DEMO SUMMARY")
    print("=" * 50)
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"📚 Documents: {len(documents)}")
    print(f"✂️  Chunks: {len(chunks)}")
    print(f"🧠 Embeddings: {'✅ Working' if model else '❌ Failed'}")
    
    print("\n🎯 System Status:")
    print("   ✅ Document loading working")
    print("   ✅ Text chunking working") 
    print(f"   {'✅' if model else '❌'} Embedding model working")
    print("   ⏳ Need Ollama for LLM (you're installing)")
    print("   ⏳ Need ChromaDB setup for persistence")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1) 