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
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if device_count > 0 else 0
        logger.info(f"✅ CUDA available: {device_count} device(s)")
        logger.info(f"   Device: {device_name}")
        logger.info(f"   Memory: {total_memory:.1f} GB")
        
        if total_memory < 12:
            logger.warning(f"⚠️  GPU memory ({total_memory:.1f} GB) might be insufficient for optimal performance")
    else:
        logger.warning("⚠️  CUDA not available - will use CPU (slower performance)")
    
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
        'logs', 'models', 'models/huggingface', 'models/transformers_cache',
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
        'TRANSFORMERS_CACHE': './models/transformers_cache',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512'
    }
    
    # Read existing .env or create new one
    env_file = Path('.env')
    env_content = ""
    
    if env_file.exists():
        env_content = env_file.read_text()
    
    # Add missing environment variables
    for key, value in env_vars.items():
        if f"{key}=" not in env_content:
            env_content += f"{key}={value}\n"
            logger.info(f"🔧 Added environment variable: {key}")
    
    # Write back to .env file
    env_file.write_text(env_content)
    
    # Set for current session
    for key, value in env_vars.items():
        os.environ[key] = value

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
        logger.info("\n🎯 Next steps:")
        logger.info("   1. Run: python scripts/test_models.py")
        logger.info("   2. Run: python scripts/setup_database.py")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 