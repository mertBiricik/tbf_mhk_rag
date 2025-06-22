#!/usr/bin/env python3
"""
Basketball RAG Model Testing Script
Tests LLM and embedding models functionality.
"""

import os
import sys
import logging
import requests
import json
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    import torch
    from sentence_transformers import SentenceTransformer
    from src.utils.config import config
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_ollama_connection():
    """Test Ollama connection and availability."""
    logger = logging.getLogger(__name__)
    
    ollama_url = config.get('models.llm.base_url', 'http://localhost:11434')
    
    try:
        # Test if Ollama is running
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            logger.info(f"✅ Ollama connected. Available models: {len(models)}")
            
            # List available models
            for model in models:
                logger.info(f"   📦 {model.get('name', 'Unknown')}")
            
            return True, models
        else:
            logger.error(f"❌ Ollama responded with status: {response.status_code}")
            return False, []
            
    except requests.ConnectionError:
        logger.error("❌ Cannot connect to Ollama. Make sure it's running:")
        logger.error("   ollama serve")
        return False, []
    except Exception as e:
        logger.error(f"❌ Ollama connection error: {e}")
        return False, []

def download_llm_model():
    """Download the required LLM model."""
    logger = logging.getLogger(__name__)
    
    model_name = config.get('models.llm.name', 'llama3.1:8b-instruct-q4_K_M')
    logger.info(f"📥 Downloading LLM model: {model_name}")
    
    try:
        import subprocess
        result = subprocess.run(
            ['ollama', 'pull', model_name],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"✅ Model {model_name} downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to download model: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("❌ Ollama command not found. Please install Ollama first.")
        return False

def test_llm_inference():
    """Test LLM inference with a basketball question."""
    logger = logging.getLogger(__name__)
    
    ollama_url = config.get('models.llm.base_url', 'http://localhost:11434')
    model_name = config.get('models.llm.name', 'llama3.1:8b-instruct-q4_K_M')
    
    test_prompt = "Basketbol oyununda 5 faul yapan oyuncuya ne olur? Kısaca açıkla."
    
    payload = {
        "model": model_name,
        "prompt": test_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 100
        }
    }
    
    try:
        logger.info("🧠 Testing LLM inference...")
        start_time = time.time()
        
        response = requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=30
        )
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '').strip()
            
            logger.info(f"✅ LLM inference successful ({inference_time:.2f}s)")
            logger.info(f"   Question: {test_prompt}")
            logger.info(f"   Answer: {answer[:100]}...")
            
            return True, inference_time
        else:
            logger.error(f"❌ LLM inference failed: {response.status_code}")
            return False, 0
            
    except Exception as e:
        logger.error(f"❌ LLM inference error: {e}")
        return False, 0

def test_embedding_model():
    """Test embedding model."""
    logger = logging.getLogger(__name__)
    
    model_name = config.get('models.embeddings.name', 'BAAI/bge-m3')
    device = config.get('models.embeddings.device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        logger.info(f"📊 Loading embedding model: {model_name}")
        
        # Load model
        start_time = time.time()
        model = SentenceTransformer(model_name)
        model = model.to(device)
        load_time = time.time() - start_time
        
        logger.info(f"✅ Embedding model loaded ({load_time:.2f}s) on {device}")
        
        # Test encoding
        test_texts = [
            "basketbol kuralları",
            "basketball rules", 
            "5 faul kuralı",
            "foul rules",
            "şut saati",
            "shot clock"
        ]
        
        start_time = time.time()
        embeddings = model.encode(test_texts)
        encode_time = time.time() - start_time
        
        logger.info(f"✅ Embedding generation successful ({encode_time:.2f}s)")
        logger.info(f"   Input texts: {len(test_texts)}")
        logger.info(f"   Embedding shape: {embeddings.shape}")
        logger.info(f"   Device: {embeddings.device if hasattr(embeddings, 'device') else 'numpy'}")
        
        # Test similarity
        turkish_basketball = embeddings[0]
        english_basketball = embeddings[1]
        similarity = torch.cosine_similarity(
            torch.tensor(turkish_basketball).unsqueeze(0),
            torch.tensor(english_basketball).unsqueeze(0)
        ).item()
        
        logger.info(f"   Turkish-English similarity: {similarity:.3f}")
        
        return True, encode_time, embeddings.shape
        
    except Exception as e:
        logger.error(f"❌ Embedding model error: {e}")
        return False, 0, None

def test_gpu_memory():
    """Test GPU memory usage."""
    logger = logging.getLogger(__name__)
    
    if not torch.cuda.is_available():
        logger.info("ℹ️  CUDA not available, skipping GPU memory test")
        return True
    
    try:
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)
        cached_memory = torch.cuda.memory_reserved(device) / (1024**3)
        
        logger.info(f"💾 GPU Memory Status:")
        logger.info(f"   Total: {total_memory:.2f} GB")
        logger.info(f"   Allocated: {allocated_memory:.2f} GB")
        logger.info(f"   Cached: {cached_memory:.2f} GB")
        logger.info(f"   Free: {total_memory - cached_memory:.2f} GB")
        
        if cached_memory > total_memory * 0.9:
            logger.warning("⚠️  GPU memory usage is high")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GPU memory check error: {e}")
        return False

def main():
    """Main testing function."""
    print("🏀 Basketball RAG Model Testing")
    print("=" * 50)
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Run tests
    tests = [
        ("Ollama Connection", lambda: test_ollama_connection()[0]),
        ("LLM Model Download", download_llm_model),
        ("LLM Inference", lambda: test_llm_inference()[0]),
        ("Embedding Model", lambda: test_embedding_model()[0]),
        ("GPU Memory", test_gpu_memory),
    ]
    
    failed_tests = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        try:
            if not test_func():
                failed_tests.append(test_name)
        except Exception as e:
            logger.error(f"❌ {test_name} failed with error: {e}")
            failed_tests.append(test_name)
    
    # Summary
    print("\n" + "=" * 50)
    if failed_tests:
        logger.error(f"❌ Some tests failed: {', '.join(failed_tests)}")
        logger.error("Please resolve the issues above.")
        return False
    else:
        logger.info("✅ All model tests passed!")
        logger.info("\n🎯 Next step: python scripts/setup_database.py")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 