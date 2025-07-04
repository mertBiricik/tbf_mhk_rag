# Basketball RAG Project Setup Instructions for WSL

## System Specifications
- OS: Ubuntu-22,04, it is in a WSL but you are already running in it. you do not have any connection to the windows 11 host.
- GPU: NVIDIA A5000 16GB VRAM
- RAM: 64GB
- Documents: Turkish basketball rules (2022 rules, 2024 changes, 2023 interpretations)

## Project Overview
Build a local RAG (Retrieval-Augmented Generation) system for basketball rules that:
- Runs entirely locally (no API costs)
- Uses GPU acceleration for fast inference
- Supports Turkish and English languages
- Handles multiple document types with version conflicts
- Provides accurate rule references and citations

## Optimal Tech Stack for This Hardware

### Core Components
- LLM: Llama-3.1-8B-Instruct (fits in 16GB VRAM)
- Embeddings: BGE-M3 (multilingual, excellent for Turkish/English)
- Vector Database: Chroma (local, persistent)
- Framework: LangChain + Ollama
- Environment: WSL Ubuntu with conda

### Why This Stack?
- GPU Optimized: Perfect fit for A5000 16GB
- Cost-Free: No external API costs
- Private: Documents stay local
- Multilingual: Excellent Turkish support
- Production Ready: Can handle enterprise workloads

## Automatic Hardware Detection & Model Selection

### Smart VRAM-Based Configuration
The system automatically detects your GPU's VRAM and selects optimal models:

#### VRAM Categories & Model Selection:
- 8GB+ VRAM (RTX 4060, 3070, A5000): Llama 8B + BGE-M3 (optimal performance)
- 6GB+ VRAM (RTX 2060, 1660 Ti): Llama 8B + MiniLM-L12 (very good performance)
- 4GB+ VRAM (GTX 1050 Ti, 1650): Llama 3B + MiniLM-L12 (good performance)
- 2GB+ VRAM (GT 1030): Qwen 1.5B + MiniLM-L6 (fair performance)
- CPU/Low VRAM: Optimized for CPU processing (basic performance)

#### Automatic Benefits:
- No manual configuration - detects and downloads correct models
- Prevents VRAM overflow - never exceeds available memory
- Optimal performance - best models for your hardware
- GTX 1050 Ti support - works perfectly with 4GB VRAM

#### Expected Performance by Hardware:
| GPU | VRAM | LLM Model | Response Time | Quality |
|-----|------|-----------|---------------|---------|
| RTX A5000 | 16GB | Llama 8B | ~0.5s | Excellent |
| GTX 1070 | 8GB | Llama 8B | ~1s | Excellent |
| GTX 1060 | 6GB | Llama 8B | ~1.5s | Very Good |
| GTX 1050 Ti | 4GB | Llama 3B | ~3s | Good |
| GT 1030 | 2GB | Qwen 1.5B | ~8s | Fair |

## Step-by-Step Setup Instructions

### Phase 1: WSL Environment Setup

#### 1.1 Verify WSL and GPU Access
```bash
# Check WSL version
wsl --version

# Verify NVIDIA GPU access in WSL
nvidia-smi

# If nvidia-smi doesn't work, install CUDA drivers for WSL
```

#### 1.2 Create Conda Environment
```bash
# Create new conda environment for the project
conda create -n basketball_rag python=3.11 -y

# Activate environment
conda activate basketball_rag

# Install conda dependencies
conda install -c conda-forge jupyter notebook ipykernel -y
```

#### 1.3 Install Python Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core RAG dependencies
pip install langchain==0.1.0
pip install langchain-community
pip install chromadb
pip install sentence-transformers
pip install transformers
pip install accelerate
pip install bitsandbytes

# Install Ollama for local LLM management
curl -fsSL https://ollama.ai/install.sh | sh

# Install document processing
pip install pypdf2
pip install python-docx
pip install unstructured
pip install tiktoken

# Install evaluation and utilities
pip install rouge-score
pip install datasets
pip install gradio  # For web interface
pip install python-dotenv
```

### Phase 2: Model Setup

#### 2.1 Automatic Model Detection & Download
```bash
# Automatic hardware detection and model download
python scripts/setup_environment.py

# This will:
# 1. Detect your GPU and VRAM
# 2. Select optimal models (Llama 8B for 8GB+, 3B for 4GB+)
# 3. Download the correct models automatically
# 4. Configure everything optimally

# Test hardware detection
python scripts/test_hardware_detection.py

# If you want to manually override (advanced users):
# ollama pull llama3.1:8b-instruct-q4_K_M  # For 8GB+ VRAM
# ollama pull llama3.1:3b-instruct-q4_K_M  # For 4GB+ VRAM
# ollama pull qwen2:1.5b                   # For 2GB+ VRAM
```

#### 2.1.1 Hardware-Specific Recommendations
```bash
# GTX 1050 Ti (4GB) users will get:
# - Llama 3B model (~2GB)
# - MiniLM-L12 embedding (~470MB)
# - Total: ~2.5GB VRAM usage (safe for 4GB)

# RTX A5000 (16GB) users will get:
# - Llama 8B model (~4.9GB)  
# - BGE-M3 embedding (~2GB)
# - Total: ~7GB VRAM usage (optimal for 16GB)
```

#### 2.2 Setup Embedding Model
```python
# Create test script: test_embeddings.py
from sentence_transformers import SentenceTransformer
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Load multilingual embedding model (good for Turkish)
model = SentenceTransformer('BAAI/bge-m3')
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Test with Turkish and English basketball terms
test_texts = [
    "basketbol kuralları",
    "basketball rules",
    "faul", 
    "foul",
    "şut saati",
    "shot clock"
]

embeddings = model.encode(test_texts)
print(f"Embedding shape: {embeddings.shape}")
print("Embedding model working correctly")
```

### Phase 3: Project Structure

#### 3.1 Create Project Directory Structure
```bash
# Navigate to project directory (should already be there)
cd /d/courses/tbf_mhk_rag

# Create additional directories
mkdir -p {data/processed,models,notebooks,src,tests,logs,vector_db}
mkdir -p config

# Create src subdirectories
mkdir -p src/{document_processing,retrieval,generation,evaluation,utils}
```

#### 3.2 Project Structure Should Look Like:
```
tbf_mhk_rag/
├── source/
│   ├── pdf/                 # Original PDF documents
│   └── txt/                 # Converted text documents
├── src/                     # Main source code
│   ├── document_processing/ # Document loading and chunking
│   ├── retrieval/          # Vector search and retrieval
│   ├── generation/         # LLM integration and response generation
│   ├── evaluation/         # Quality metrics and testing
│   └── utils/              # Shared utilities and configuration
├── config/                 # Configuration files
├── data/                   # Processed data and test queries
├── vector_db/             # ChromaDB storage
├── models/                # Downloaded models
├── notebooks/             # Jupyter notebooks for development
├── tests/                 # Test suite
└── logs/                  # Application logs
```

### Phase 4: Document Processing

#### 4.1 Process Basketball Documents
```bash
# Run document processing
python scripts/setup_database.py

# This will:
# 1. Load Turkish basketball documents
# 2. Clean and chunk the text
# 3. Generate embeddings
# 4. Store in ChromaDB vector database
```

#### 4.2 Test Document Processing
```python
# Test retrieval system
from src.retrieval.retriever import BasketballRetriever

retriever = BasketballRetriever()
results = retriever.search("5 faul kuralı")
print(f"Found {len(results)} relevant documents")
```

### Phase 5: Web Interface Setup

#### 5.1 Launch Gradio Interface
```bash
# Start simple web interface
python scripts/launch_web_apps.py gradio

# Access at: http://localhost:7860
```

#### 5.2 Test Complete System
```bash
# Run full system test
python scripts/test_complete_rag.py

# This will test:
# 1. Document retrieval
# 2. LLM generation
# 3. Turkish language support
# 4. Citation accuracy
```

### Phase 6: Advanced Configuration

#### 6.1 Customize Configuration
Edit `config/config.yaml`:
```yaml
# Model settings
models:
  llm:
    name: "llama3.1:8b-instruct-q4_K_M"  # Auto-selected based on hardware
    temperature: 0.1
    max_tokens: 2048

# Basketball-specific settings
basketball:
  rule_priority:
    changes_2024: 3      # Highest priority
    interpretations_2023: 2
    rules_2022: 1        # Base rules
```

#### 6.2 Performance Tuning
```yaml
# For high-performance systems
performance:
  gpu_memory_fraction: 0.8
  batch_size: 32
  cache_embeddings: true

# For lower-end systems (like GTX 1050 Ti)
performance:
  gpu_memory_fraction: 0.6
  batch_size: 16
  cache_embeddings: false
```

## Testing and Validation

### Test Categories

#### Basic Functionality
```bash
python scripts/test_models.py
python scripts/test_document_processing.py
python scripts/test_hardware_detection.py
```

#### Basketball-Specific Tests
```bash
python scripts/test_language_detection.py
python scripts/evaluate_performance.py
```

#### Performance Benchmarks
```bash
python scripts/run_performance_analysis.py
```

## Common Issues and Solutions

### CUDA/GPU Issues
- Verify nvidia-smi works in WSL
- Check CUDA version compatibility
- Ensure sufficient VRAM for selected models

### Model Download Issues
- Check internet connectivity
- Verify Ollama service is running
- Use manual model download as fallback

### Turkish Character Issues
- Ensure UTF-8 encoding
- Verify proper text preprocessing
- Check terminal locale settings

### Performance Issues
- Monitor GPU memory usage
- Adjust batch sizes in configuration
- Consider smaller models for limited hardware

## Expected Results

After successful setup:
- RTX A5000: Sub-second response times with excellent quality
- GTX 1050 Ti: 2-5 second response times with good quality
- Accurate Turkish basketball rule retrieval
- Proper source citations
- Multilingual support (Turkish/English)

## Next Steps

1. Regular model updates via Ollama
2. Document collection expansion
3. Performance monitoring and optimization
4. User feedback integration

The system is designed for production use with the Turkish Basketball Federation, providing reliable, fast, and accurate basketball rule queries. 