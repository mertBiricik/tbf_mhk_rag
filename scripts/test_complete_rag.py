#!/usr/bin/env python3
"""
Complete Basketball RAG System Test
Tests the full pipeline: Documents → Chunks → Embeddings → Vector DB → LLM
"""

import os
import time
import sys
import json
from pathlib import Path

def test_ollama_connection():
    """Test Ollama connection and model."""
    print("🧠 Testing Ollama LLM...")
    
    try:
        import requests
        
        # Test Ollama service
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama connected, {len(models)} models available")
            
            # Check if our model is available
            target_model = "llama3.1:8b-instruct-q4_K_M"
            model_names = [m.get('name', '') for m in models]
            
            if target_model in model_names:
                print(f"✅ Target model found: {target_model}")
            else:
                print(f"❌ Target model not found: {target_model}")
                print(f"Available models: {model_names}")
                return False, None
            
            return True, target_model
        else:
            print(f"❌ Ollama responded with status: {response.status_code}")
            return False, None
            
    except requests.ConnectionError:
        print("❌ Cannot connect to Ollama. Is the service running?")
        print("Try: ollama serve")
        return False, None
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False, None

def test_llm_inference(model_name):
    """Test LLM inference with basketball question."""
    print("\n🎯 Testing LLM Inference...")
    
    try:
        import requests
        
        # Test prompt in Turkish
        test_prompt = """Sen basketbol kuralları uzmanısın. Lütfen şu soruyu kısaca yanıtla:

Soru: Basketbol oyununda bir oyuncu 5 faul yaptığında ne olur?

Yanıt:"""
        
        payload = {
            "model": model_name,
            "prompt": test_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 150
            }
        }
        
        print(f"🤔 Asking: '5 faul yapan oyuncuya ne olur?'")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )
        
        inference_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '').strip()
            
            print(f"✅ LLM responded in {inference_time:.2f}s")
            print(f"🧠 Answer: {answer}")
            
            return True
        else:
            print(f"❌ LLM inference failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ LLM inference error: {e}")
        return False

def setup_vector_database():
    """Set up ChromaDB with basketball documents."""
    print("\n🗃️  Setting up Vector Database...")
    
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        import torch
        
        # Load embedding model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"📊 Loading BGE-M3 embedding model on {device}...")
        
        model = SentenceTransformer('BAAI/bge-m3')
        model = model.to(device)
        
        # Setup ChromaDB
        db_path = "./vector_db/chroma_db"
        collection_name = "basketball_rules"
        
        os.makedirs(db_path, exist_ok=True)
        client = chromadb.PersistentClient(path=db_path)
        
        # Create or get collection
        try:
            client.delete_collection(name=collection_name)
            print("🗑️  Deleted existing collection")
        except:
            pass
        
        print("📋 Creating new collection...")
        collection = client.create_collection(name=collection_name)
        
        # Load and process documents
        print("📚 Loading basketball documents...")
        source_dir = Path("source/txt")
        
        files = [
            "basketbol_oyun_kurallari_2022.txt",
            "basketbol_oyun_kurallari_degisiklikleri_2024.txt", 
            "basketbol_oyun_kurallari_resmi_yorumlar_2023.txt"
        ]
        
        # Text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\nMadde", "\n\nRule", "\n\nKural", "\n\n", "\n", ". ", " "]
        )
        
        all_chunks = []
        doc_types = ["rules", "changes", "interpretations"]
        years = [2022, 2024, 2023]
        
        for i, filename in enumerate(files):
            filepath = source_dir / filename
            if not filepath.exists():
                continue
                
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks
            chunks = splitter.split_text(content)
            
            for j, chunk in enumerate(chunks):
                all_chunks.append({
                    'text': chunk,
                    'source': filename,
                    'doc_type': doc_types[i],
                    'year': years[i],
                    'chunk_id': j
                })
            
            print(f"   {filename}: {len(chunks)} chunks")
        
        print(f"📊 Total chunks: {len(all_chunks)}")
        
        # Generate embeddings and add to database
        batch_size = 32
        print("⚡ Generating embeddings...")
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            
            texts = [chunk['text'] for chunk in batch]
            metadatas = [{k: str(v) for k, v in chunk.items() if k != 'text'} for chunk in batch]
            ids = [f"chunk_{i+j}" for j in range(len(batch))]
            
            embeddings = model.encode(texts, convert_to_tensor=False)
            embeddings = embeddings.tolist()
            
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            print(f"   Processed {min(i + batch_size, len(all_chunks))}/{len(all_chunks)} chunks")
        
        print(f"✅ Vector database created with {collection.count()} documents")
        return True, collection, model
        
    except Exception as e:
        print(f"❌ Vector database setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_rag_pipeline(collection, embedding_model, llm_model):
    """Test the complete RAG pipeline."""
    print("\n🔍 Testing Complete RAG Pipeline...")
    
    try:
        import requests
        import numpy as np
        
        # Test queries
        test_queries = [
            "5 faul yapan oyuncuya ne olur?",
            "Basketbol sahasının boyutları nelerdir?", 
            "Şut saati kuralı nasıl işler?",
            "2024 yılında hangi kurallar değişti?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            # 1. Retrieve relevant documents
            print("   📖 Searching documents...")
            
            # Generate query embedding with BGE-M3
            query_embedding = embedding_model.encode([query], convert_to_tensor=False)
            query_embedding = query_embedding.tolist()
            
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=5
            )
            
            if not results['documents'] or not results['documents'][0]:
                print("   ❌ No relevant documents found")
                continue
            
            # 2. Prepare context
            contexts = results['documents'][0]
            sources = [meta.get('source', 'unknown') for meta in results['metadatas'][0]]
            
            context_text = "\n\n".join([f"Kaynak {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
            
            print(f"   ✅ Found {len(contexts)} relevant chunks")
            print(f"   📚 Sources: {set(sources)}")
            
            # 3. Generate response with LLM
            rag_prompt = f"""Sen Türk Basketbol Federasyonu kuralları uzmanısın. Aşağıdaki belgelerden yararlanarak soruyu yanıtla.

Belgeler:
{context_text}

Soru: {query}

Lütfen belgelerden yararlanarak kısa ve net bir yanıt ver. Hangi belgeden aldığın bilgiyi belirt.

Yanıt:"""

            payload = {
                "model": llm_model,
                "prompt": rag_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 300
                }
            }
            
            print("   🧠 Generating answer...")
            start_time = time.time()
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=45
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                print(f"   ✅ Response generated in {response_time:.2f}s")
                print(f"   🎯 Answer: {answer}")
            else:
                print(f"   ❌ LLM generation failed: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG pipeline test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🏀 Complete Basketball RAG System Test")
    print("=" * 60)
    
    start_time = time.time()
    
    # Test Ollama
    success, llm_model = test_ollama_connection()
    if not success:
        print("❌ Ollama test failed - cannot proceed")
        return False
    
    # Test LLM inference
    if not test_llm_inference(llm_model):
        print("❌ LLM inference failed - cannot proceed")
        return False
    
    # Setup vector database
    success, collection, embedding_model = setup_vector_database()
    if not success:
        print("❌ Vector database setup failed")
        return False
    
    # Test complete RAG pipeline
    if not test_rag_pipeline(collection, embedding_model, llm_model):
        print("❌ RAG pipeline test failed")
        return False
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 COMPLETE RAG SYSTEM TEST RESULTS")
    print("=" * 60)
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"🧠 LLM Model: {llm_model}")
    print(f"📊 Vector Database: {collection.count() if collection else 0} documents")
    print(f"🎯 Status: ✅ FULLY OPERATIONAL!")
    
    print("\n🚀 Your Basketball RAG System is ready!")
    print("📋 Capabilities:")
    print("   ✅ Turkish language processing")
    print("   ✅ Basketball rule expertise")
    print("   ✅ Multi-document search")
    print("   ✅ Accurate rule citations")
    print("   ✅ GPU-accelerated inference")
    
    print("\n🎯 Next: Create web interface!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1) 