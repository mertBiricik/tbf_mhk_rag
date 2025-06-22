#!/usr/bin/env python3
"""
Basketball RAG Environment Setup Script
Checks system requirements and sets up the environment.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import torch
import platform

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.utils.config import get_gpu_memory, select_optimal_models
except ImportError:
    # Fallback if config not available yet
    def get_gpu_memory():
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            total_memory_gb = total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(device)
            return total_memory_gb, gpu_name
        return 0.0, "CPU"
    
    def select_optimal_models(vram_gb):
        if vram_gb >= 8.0:
            return {"llm": "llama3.1:8b-instruct-q4_K_M", "embedding": "BAAI/bge-m3"}
        elif vram_gb >= 4.0:
            return {"llm": "llama3.1:3b-instruct-q4_K_M", "embedding": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}
        else:
            return {"llm": "llama3.1:3b-instruct-q4_K_M", "embedding": "sentence-transformers/all-MiniLM-L6-v2"}

def setup_logging():
    """Setup logging for setup process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/setup.log')
        ]
    )

def check_system_requirements():
    """Check if system meets requirements."""
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or python_version.minor < 10:
        logger.error(f"❌ Python 3.10+ required, found {python_version.major}.{python_version.minor}")
        return False
    logger.info(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Hardware detection and model selection
    vram_gb, gpu_name = get_gpu_memory()
    optimal_models = select_optimal_models(vram_gb)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        logger.info(f"✅ CUDA available: {device_count} device(s)")
        logger.info(f"   Device: {gpu_name}")
        logger.info(f"   Memory: {vram_gb:.1f} GB")
        
        # Hardware-specific recommendations
        if vram_gb >= 8.0:
            logger.info("🚀 High VRAM detected - Optimal performance expected")
            logger.info(f"   Recommended LLM: {optimal_models['llm']}")
            logger.info(f"   Recommended Embedding: {optimal_models['embedding']}")
        elif vram_gb >= 4.0:
            logger.warning(f"⚠️  Medium VRAM ({vram_gb:.1f} GB) - Using 3B model for better compatibility")
            logger.info(f"   Recommended LLM: {optimal_models['llm']}")
            logger.info(f"   Recommended Embedding: {optimal_models['embedding']}")
        else:
            logger.warning(f"⚠️  Low VRAM ({vram_gb:.1f} GB) - Performance may be limited")
            logger.info(f"   Recommended LLM: {optimal_models['llm']}")
            logger.info(f"   Recommended Embedding: {optimal_models['embedding']}")
    else:
        logger.warning("⚠️  CUDA not available - will use CPU (slower performance)")
        logger.info(f"   CPU-optimized LLM: {optimal_models['llm']}")
        logger.info(f"   CPU-optimized Embedding: {optimal_models['embedding']}")
    
    # Check system info
    logger.info(f"📊 OS: {platform.system()} {platform.release()}")
    logger.info(f"📊 Architecture: {platform.machine()}")
    
    return True

def check_ollama_installation():
    """Check if Ollama is installed and running."""
    logger = logging.getLogger(__name__)
    
    try:
        # Check if ollama command exists
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, check=True)
        logger.info(f"✅ Ollama installed: {result.stdout.strip()}")
        
        # Check if ollama service is running
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, check=True)
            logger.info("✅ Ollama service is running")
            return True
        except subprocess.CalledProcessError:
            logger.info("🔄 Starting Ollama service...")
            subprocess.run(['ollama', 'serve'], check=False)
            return True
            
    except subprocess.CalledProcessError:
        logger.error("❌ Ollama not found. Please install Ollama first:")
        logger.error("   curl -fsSL https://ollama.ai/install.sh | sh")
        return False
    except FileNotFoundError:
        logger.error("❌ Ollama not found in PATH. Please install Ollama first:")
        logger.error("   curl -fsSL https://ollama.ai/install.sh | sh")
        return False

def create_directories():
    """Create necessary directories."""
    logger = logging.getLogger(__name__)
    
    directories = [
        'logs', 'models', 'models/huggingface', 
        'vector_db', 'data/processed', 'notebooks'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")

def setup_environment_variables():
    """Setup environment variables."""
    logger = logging.getLogger(__name__)
    
    env_vars = {
        'HF_HOME': './models/huggingface',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        # Note: TRANSFORMERS_CACHE is deprecated in favor of HF_HOME
        # Setting HF_HOME automatically handles transformers cache
    }
    
    # Read existing .env or create new one
    env_file = Path('.env')
    env_content = ""
    
    if env_file.exists():
        env_content = env_file.read_text()
    
    # Remove deprecated TRANSFORMERS_CACHE if it exists
    if "TRANSFORMERS_CACHE=" in env_content:
        lines = env_content.split('\n')
        lines = [line for line in lines if not line.startswith('TRANSFORMERS_CACHE=')]
        env_content = '\n'.join(lines)
        logger.info("🔧 Removed deprecated TRANSFORMERS_CACHE variable")
    
    # Add missing environment variables
    for key, value in env_vars.items():
        if f"{key}=" not in env_content:
            env_content += f"{key}={value}\n"
            logger.info(f"🔧 Added environment variable: {key}")
    
    # Write back to .env file
    env_file.write_text(env_content)
    
    # Remove deprecated environment variable from current session
    if 'TRANSFORMERS_CACHE' in os.environ:
        del os.environ['TRANSFORMERS_CACHE']
        logger.info("🔧 Removed TRANSFORMERS_CACHE from current session")
    
    # Set for current session
    for key, value in env_vars.items():
        os.environ[key] = value

def download_optimal_models():
    """Download the optimal models based on hardware detection."""
    logger = logging.getLogger(__name__)
    
    logger.info("🤖 Detecting optimal models for your hardware...")
    
    vram_gb, gpu_name = get_gpu_memory()
    optimal_models = select_optimal_models(vram_gb)
    
    logger.info(f"🎯 Selected models for {gpu_name} ({vram_gb:.1f}GB VRAM):")
    logger.info(f"   LLM: {optimal_models['llm']}")
    logger.info(f"   Embedding: {optimal_models['embedding']}")
    
    # Download LLM model via Ollama
    try:
        logger.info(f"📥 Downloading LLM model: {optimal_models['llm']}")
        result = subprocess.run(['ollama', 'pull', optimal_models['llm']], 
                              capture_output=True, text=True, check=True)
        logger.info("✅ LLM model downloaded successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to download LLM model: {e}")
        logger.error("Please ensure Ollama is running and try manually:")
        logger.error(f"   ollama pull {optimal_models['llm']}")
        return False
    except FileNotFoundError:
        logger.error("❌ Ollama not found. Please install Ollama first.")
        return False
    
    # The embedding model will be downloaded automatically when first used
    logger.info(f"ℹ️  Embedding model ({optimal_models['embedding']}) will be downloaded automatically on first use")
    
    return True

def test_imports():
    """Test critical imports."""
    logger = logging.getLogger(__name__)
    
    critical_imports = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('sentence_transformers', 'SentenceTransformers'),
        ('chromadb', 'ChromaDB'),
        ('langchain', 'LangChain'),
        ('yaml', 'PyYAML'),
    ]
    
    failed_imports = []
    
    for module, name in critical_imports:
        try:
            __import__(module)
            logger.info(f"✅ {name} import successful")
        except ImportError as e:
            logger.error(f"❌ {name} import failed: {e}")
            failed_imports.append(name)
    
    if failed_imports:
        logger.error("❌ Some required packages are missing. Install them with:")
        logger.error("   pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main setup function."""
    print("🏀 Basketball RAG Environment Setup")
    print("=" * 50)
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting environment setup...")
    
    # Run checks
    checks = [
        ("System Requirements", check_system_requirements),
        ("Directory Creation", create_directories),
        ("Environment Variables", setup_environment_variables),
        ("Critical Imports", test_imports),
        ("Ollama Installation", check_ollama_installation),
        ("Optimal Model Download", download_optimal_models),
    ]
    
    failed_checks = []
    
    for check_name, check_func in checks:
        logger.info(f"\n🔍 Running: {check_name}")
        try:
            if not check_func():
                failed_checks.append(check_name)
        except Exception as e:
            logger.error(f"❌ {check_name} failed with error: {e}")
            failed_checks.append(check_name)
    
    # Summary
    print("\n" + "=" * 50)
    if failed_checks:
        logger.error(f"❌ Setup incomplete. Failed checks: {', '.join(failed_checks)}")
        logger.error("Please resolve the issues above and run setup again.")
        return False
    else:
        logger.info("✅ Environment setup completed successfully!")
        
        # Show hardware-specific summary
        vram_gb, gpu_name = get_gpu_memory()
        optimal_models = select_optimal_models(vram_gb)
        logger.info(f"\n🎯 Hardware Configuration Summary:")
        logger.info(f"   GPU: {gpu_name}")
        logger.info(f"   VRAM: {vram_gb:.1f} GB")
        logger.info(f"   LLM Model: {optimal_models['llm']}")
        logger.info(f"   Embedding: {optimal_models['embedding']}")
        
        logger.info("\n🎯 Next steps:")
        logger.info("   1. Run: python scripts/test_hardware_detection.py  # Test hardware detection")
        logger.info("   2. Run: python scripts/test_models.py             # Test model functionality")
        logger.info("   3. Run: python scripts/setup_database.py          # Setup vector database")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 