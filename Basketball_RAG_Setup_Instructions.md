# Basketball RAG Project Setup Instructions for WSL

## System Specifications
- **OS**: Ubuntu-22,04, it is in a WSL but you are already running in it. you do not have any connection to the windows 11 host.
- **GPU**: NVIDIA A5000 16GB VRAM
- **RAM**: 64GB
- **Documents**: Turkish basketball rules (2022 rules, 2024 changes, 2023 interpretations)

## Project Overview
Build a local RAG (Retrieval-Augmented Generation) system for basketball rules that:
- Runs entirely locally (no API costs)
- Uses GPU acceleration for fast inference
- Supports Turkish and English languages
- Handles multiple document types with version conflicts
- Provides accurate rule references and citations

## Optimal Tech Stack for This Hardware

### Core Components
- **LLM**: Llama-3.1-8B-Instruct (fits in 16GB VRAM)
- **Embeddings**: BGE-M3 (multilingual, excellent for Turkish/English)
- **Vector Database**: Chroma (local, persistent)
- **Framework**: LangChain + Ollama
- **Environment**: WSL Ubuntu with conda

### Why This Stack?
✅ **GPU Optimized**: Perfect fit for A5000 16GB  
✅ **Cost-Free**: No external API costs  
✅ **Private**: Documents stay local  
✅ **Multilingual**: Excellent Turkish support  
✅ **Production Ready**: Can handle enterprise workloads  

## 🤖 Automatic Hardware Detection & Model Selection

### New Feature: Smart VRAM-Based Configuration
The system now automatically detects your GPU's VRAM and selects optimal models:

#### VRAM Categories & Model Selection:
- **8GB+ VRAM** (RTX 4060, 3070, A5000): Llama 8B + BGE-M3 (optimal performance)
- **6GB+ VRAM** (RTX 2060, 1660 Ti): Llama 8B + MiniLM-L12 (very good performance)  
- **4GB+ VRAM** (GTX 1050 Ti, 1650): Llama 3B + MiniLM-L12 (good performance)
- **2GB+ VRAM** (GT 1030): Qwen 1.5B + MiniLM-L6 (fair performance)
- **CPU/Low VRAM**: Optimized for CPU processing (basic performance)

#### Automatic Benefits:
✅ **No manual configuration** - detects and downloads correct models  
✅ **Prevents VRAM overflow** - never exceeds available memory  
✅ **Optimal performance** - best models for your hardware  
✅ **GTX 1050 Ti support** - works perfectly with 4GB VRAM  

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
pip install streamlit  # Alternative interface
pip install python-dotenv
```

### Phase 2: Model Setup

#### 2.1 Automatic Model Detection & Download
```bash
# NEW: Automatic hardware detection and model download
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
print("✅ Embedding model working correctly")
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
│   ├── pdf/
│   └── txt/
├── data/
│   └── processed/
├── models/
├── vector_db/
├── src/
│   ├── document_processing/
│   ├── retrieval/
│   ├── generation/
│   ├── evaluation/
│   └── utils/
├── notebooks/
├── tests/
├── logs/
├── config/
├── requirements.txt
├── README.md
└── Basketball_RAG_Setup_Instructions.md
```

### Phase 4: Core Implementation Files

#### 4.1 Configuration File
Create `config/config.yaml`:
```yaml
# Model Configuration
models:
  llm:
    name: "llama3.1:8b-instruct-q4_K_M"
    base_url: "http://localhost:11434"
    temperature: 0.1
    max_tokens: 2048
    
  embeddings:
    name: "BAAI/bge-m3"
    device: "cuda"
    max_seq_length: 1024

# Vector Database
vector_db:
  path: "./vector_db"
  collection_name: "basketball_rules"
  
# Document Processing
document_processing:
  chunk_size: 800
  chunk_overlap: 100
  separators: ["\n\nMadde", "\n\nRule", "\n\nKural", "\n\n", "\n", ". ", " "]

# Retrieval
retrieval:
  top_k: 7
  score_threshold: 0.3
  
# Basketball-specific settings
basketball:
  languages: ["turkish", "english"]
  document_types:
    rules_2022:
      file: "basketbol_oyun_kurallari_2022.txt"
      type: "rules"
      year: 2022
      priority: 1
    changes_2024:
      file: "basketbol_oyun_kurallari_degisiklikleri_2024.txt"
      type: "changes"
      year: 2024
      priority: 3
    interpretations_2023:
      file: "basketbol_oyun_kurallari_resmi_yorumlar_2023.txt"
      type: "interpretations"
      year: 2023
      priority: 2
```

#### 4.2 Core Implementation Requirements

**Document Processor** (`src/document_processing/processor.py`):
- Load Turkish basketball documents
- Handle UTF-8 encoding properly
- Extract rule numbers and sections
- Add basketball-specific metadata
- Implement semantic chunking for rules

**Vector Store Manager** (`src/retrieval/vector_store.py`):
- Initialize Chroma with BGE-M3 embeddings
- Handle Turkish text encoding
- Implement metadata filtering
- Support incremental updates

**RAG Chain** (`src/generation/rag_chain.py`):
- Integration with local Ollama model
- Basketball-specific prompting
- Turkish/English bilingual support
- Source citation and rule references

**Evaluation Module** (`src/evaluation/evaluator.py`):
- Retrieval quality metrics
- Answer accuracy assessment
- Turkish language evaluation
- Basketball domain-specific tests

#### 4.3 Basketball-Specific Features to Implement

**Rule Conflict Resolution**:
- Priority system: 2024 changes > 2023 interpretations > 2022 base rules
- Version tracking and conflict detection
- Clear indication of rule updates

**Multilingual Query Handling**:
- Turkish query expansion
- Bilingual keyword mapping
- Language detection and response formatting

**Basketball Domain Intelligence**:
- Term extraction (foul types, violations, etc.)
- Rule number recognition and linking
- Context-aware chunking for basketball rules

**Advanced Retrieval**:
- Hybrid search (semantic + keyword)
- Metadata filtering by document type/year
- Re-ranking based on basketball relevance

### Phase 5: Testing and Validation

#### 5.1 Test Queries for Validation
```python
test_queries = [
    # Turkish queries
    "5 faul yapan oyuncuya ne olur?",
    "Şut saati kuralları nelerdir?",
    "Teknik faul ile kişisel faul arasındaki fark nedir?",
    "2024 yılında hangi kurallar değişti?",
    
    # English queries  
    "What happens when a player commits 5 fouls?",
    "What are the shot clock rules?",
    "What is the difference between technical and personal fouls?",
    "What rules changed in 2024?",
    
    # Mixed complexity
    "Oyuncu değişimi sırasında mola alınabilir mi?",
    "Can a timeout be called during substitution?"
]
```

#### 5.2 Performance Benchmarks
Target performance metrics:
- **Retrieval Latency**: < 200ms
- **Generation Latency**: < 3 seconds  
- **Memory Usage**: < 12GB VRAM
- **Accuracy**: > 85% for rule-based questions
- **Turkish Support**: Native-level understanding

### Phase 6: Deployment Options

#### 6.1 Gradio Web Interface
Simple web UI for testing and demonstration

#### 6.2 Streamlit Dashboard
More advanced interface with:
- Document upload capabilities
- Retrieval visualization
- Performance monitoring
- A/B testing interface

#### 6.3 API Endpoint
FastAPI service for integration with other applications

### Phase 7: Optimization Strategies

#### 7.1 Performance Optimization
- Model quantization (4-bit) for faster inference
- Embedding caching for repeated queries
- Batch processing for document ingestion
- GPU memory optimization

#### 7.2 Quality Optimization
- Query expansion with basketball terminology
- Custom re-ranking for basketball domain
- Few-shot prompting for better responses
- Evaluation-driven iterative improvement

## Expected Outcomes

After completing this setup, you will have:
1. **Local RAG System**: Running entirely on your hardware
2. **Basketball Expert**: Specialized in Turkish basketball rules
3. **Version Control**: Handles multiple rule versions intelligently  
4. **High Performance**: GPU-accelerated inference
5. **Production Ready**: Scalable and maintainable architecture

## Hardware Utilization
- **GPU**: 8-12GB VRAM usage (optimal for A5000)
- **RAM**: 16-24GB system RAM usage
- **Storage**: ~5-10GB for models and vector database

## Troubleshooting Section

### Common Issues and Solutions

#### Environment Issues
```bash
# If CUDA not detected
nvidia-smi
# If fails, install CUDA drivers for WSL:
sudo apt update
sudo apt install nvidia-cuda-toolkit

# If conda command not found
export PATH="/opt/miniconda3/bin:$PATH"
source ~/.bashrc
```

#### Model Issues
```bash
# If Ollama connection fails
sudo systemctl start ollama
ollama serve

# If embedding model download fails
export HF_HOME=./models/huggingface
huggingface-cli login  # Optional for private models
```

#### Memory Issues
```bash
# If GPU memory errors, reduce batch size in config.yaml:
models:
  embeddings:
    batch_size: 16  # Reduce from 32

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Performance Monitoring
```bash
# Check system resources
htop
nvidia-smi

# Monitor application logs
tail -f logs/basketball_rag.log
```

## Docker Deployment Option

For reproducible deployment, create `Dockerfile`:
```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip curl

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy project files
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose ports
EXPOSE 7860 8501 11434

# Start script
CMD ["bash", "scripts/docker_start.sh"]
```

## Enhanced Evaluation Framework

### Automated Testing
```bash
# Create comprehensive test suite
python scripts/run_tests.py --full-evaluation

# Performance benchmarking
python scripts/benchmark.py --iterations 100

# Quality assessment
python scripts/evaluate_responses.py --test-set data/test_queries.json
```

## Next Steps for Implementation
1. ✅ Set up project structure and configuration
2. 🔄 Set up WSL environment and conda
3. 🔄 Install all dependencies and test models
4. ⏳ Implement document processing pipeline
5. ⏳ Build vector database with basketball rules
6. ⏳ Create RAG chain with Ollama integration
7. ⏳ Develop web interface (Gradio/Streamlit)
8. ⏳ Test with comprehensive basketball queries
9. ⏳ Optimize performance and fine-tune parameters
10. ⏳ Deploy, monitor, and maintain system

This setup maximizes your hardware capabilities while ensuring the system is production-ready and basketball-domain optimized. The enhanced troubleshooting and monitoring capabilities ensure smooth operation and easy maintenance. 