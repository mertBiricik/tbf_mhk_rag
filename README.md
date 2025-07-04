# Basketball RAG System

Local Retrieval-Augmented Generation system for Turkish basketball rules with multilingual support.

## Features

- Local deployment with no API costs
- GPU optimized for NVIDIA A5000 16GB VRAM
- Native Turkish and English support
- Version control for multiple rule versions and conflicts
- Sub-3 second response times
- Precise rule references and sources

## System Requirements

- OS: Ubuntu 22.04 (WSL2 supported)
- GPU: NVIDIA GPU with 16GB+ VRAM
- RAM: 16GB+ system memory
- Python: 3.11+
- CUDA: 12.1+

## Setup

### Environment Setup

```bash
cd tbf_mhk_rag
conda create -n basketball_rag python=3.11 -y
conda activate basketball_rag
pip install -r requirements.txt
```

### Model Setup

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b-instruct-q4_K_M
python scripts/test_models.py
```

### Database Initialization

```bash
python scripts/setup_database.py
```

### Interface Launch

```bash
python scripts/run_gradio.py
```

## Project Structure

```
tbf_mhk_rag/
├── source/                     # Original documents
│   ├── pdf/                   # PDF sources
│   └── txt/                   # Text sources
├── src/                       # Core application code
│   ├── document_processing/   # Document ingestion
│   ├── retrieval/            # Vector search
│   ├── generation/           # LLM integration
│   ├── evaluation/           # Quality metrics
│   └── utils/                # Shared utilities
├── config/                   # Configuration files
├── data/                     # Processed data
├── vector_db/               # Vector database
├── notebooks/               # Development notebooks
└── tests/                   # Test suite
```

## Documents

- 2022 Basketball Rules (Base rules)
- 2024 Rule Changes (Latest updates)
- 2023 Official Interpretations (Clarifications)

## Technology Stack

- LLM: Llama-3.1-8B-Instruct (Ollama)
- Embeddings: BGE-M3 (Multilingual)
- Vector DB: ChromaDB
- Framework: LangChain
- UI: Gradio

## Performance Benchmarks

- Retrieval: <200ms average
- Generation: <3s average
- Memory Usage: ~12GB VRAM
- Accuracy: >85% on rule queries

## Query Examples

Turkish:
- "5 faul yapan oyuncuya ne olur?"
- "2024 yılında hangi kurallar değişti?"
- "Şut saati kuralları nelerdir?"

English:
- "What happens when a player commits 5 fouls?"
- "What rules changed in 2024?"
- "What are the shot clock rules?"

## Development

### Testing

```bash
pytest tests/
pytest tests/test_retrieval.py
```

### Evaluation

```bash
python scripts/evaluate_system.py
python scripts/generate_report.py
```

## Configuration

Edit `config/config.yaml` to customize:
- Model parameters
- Retrieval settings
- Basketball-specific configurations
- Performance tuning

## Troubleshooting

### Common Issues

1. CUDA Issues: Ensure NVIDIA drivers and CUDA toolkit are installed
2. Memory Errors: Reduce batch size in configuration
3. Turkish Encoding: Ensure UTF-8 encoding for all text files

### Performance Tuning

- Adjust `chunk_size` for better retrieval
- Modify `top_k` for result diversity
- Tune `temperature` for response creativity

## License

MIT License - see LICENSE file for details. 