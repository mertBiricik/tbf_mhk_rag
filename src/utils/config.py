"""Configuration management utilities."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

def get_gpu_memory() -> Tuple[float, str]:
    """
    Get available GPU memory in GB and GPU name.
    
    Returns:
        Tuple[float, str]: (VRAM in GB, GPU name)
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device)
            # Get total memory in bytes, convert to GB
            total_memory = torch.cuda.get_device_properties(device).total_memory
            total_memory_gb = total_memory / (1024**3)
            
            logging.info(f"Detected GPU: {gpu_name}")
            logging.info(f"Total VRAM: {total_memory_gb:.1f} GB")
            
            return total_memory_gb, gpu_name
        else:
            logging.warning("CUDA not available, using CPU")
            return 0.0, "CPU"
    except ImportError:
        logging.warning("PyTorch not available, cannot detect GPU")
        return 0.0, "Unknown"
    except Exception as e:
        logging.error(f"Error detecting GPU: {e}")
        return 0.0, "Unknown"

def select_optimal_models(vram_gb: float) -> Dict[str, str]:
    """
    Select optimal models based on available VRAM.
    
    Args:
        vram_gb: Available VRAM in GB
        
    Returns:
        Dict with recommended model names
    """
    models = {}
    
    # LLM Model Selection
    if vram_gb >= 8.0:
        # High VRAM: Use 8B model
        models["llm"] = "llama3.1:8b-instruct-q4_K_M"
        models["embedding"] = "BAAI/bge-m3"
        logging.info(f"High VRAM ({vram_gb:.1f}GB): Using Llama 8B + BGE-M3")
    elif vram_gb >= 6.0:
        # Medium VRAM: Use 8B model with smaller embedding
        models["llm"] = "llama3.1:8b-instruct-q4_K_M"
        models["embedding"] = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        logging.info(f"Medium VRAM ({vram_gb:.1f}GB): Using Llama 8B + MiniLM-L12")
    elif vram_gb >= 4.0:
        # Low VRAM: Use 3B model
        models["llm"] = "llama3.1:3b-instruct-q4_K_M"
        models["embedding"] = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        logging.info(f"Low VRAM ({vram_gb:.1f}GB): Using Llama 3B + MiniLM-L12")
    elif vram_gb >= 2.0:
        # Very low VRAM: Use smallest models
        models["llm"] = "qwen2:1.5b"
        models["embedding"] = "sentence-transformers/all-MiniLM-L6-v2"
        logging.info(f"Very low VRAM ({vram_gb:.1f}GB): Using Qwen 1.5B + MiniLM-L6")
    else:
        # CPU only or very limited GPU
        models["llm"] = "llama3.1:3b-instruct-q4_K_M"  # Will run on CPU
        models["embedding"] = "sentence-transformers/all-MiniLM-L6-v2"
        logging.info(f"Limited/No VRAM ({vram_gb:.1f}GB): Using CPU models")
    
    return models

class Config:
    """Configuration manager for Basketball RAG system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.vram_gb, self.gpu_name = get_gpu_memory()
        self.optimal_models = select_optimal_models(self.vram_gb)
        self.config = self._load_config()
        self._setup_logging()
        self._apply_hardware_optimizations()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            logging.warning(f"Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
        except yaml.YAMLError as e:
            logging.error(f"Error parsing config file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration with hardware-optimized models."""
        return {
            "models": {
                "llm": {
                    "name": self.optimal_models["llm"],
                    "base_url": "http://localhost:11434",
                    "temperature": 0.1,
                    "max_tokens": 2048
                },
                "embeddings": {
                    "name": self.optimal_models["embedding"],
                    "device": "cuda" if self.vram_gb > 0 else "cpu",
                    "max_seq_length": 1024 if "bge-m3" in self.optimal_models["embedding"] else 512
                }
            },
            "vector_db": {
                "path": "./vector_db",
                "collection_name": "basketball_rules"
            },
            "document_processing": {
                "chunk_size": 800,
                "chunk_overlap": 100,
                "batch_size": max(1, min(8, int(self.vram_gb)))  # Adaptive batch size
            },
            "hardware": {
                "vram_gb": self.vram_gb,
                "gpu_name": self.gpu_name,
                "optimal_models": self.optimal_models
            }
        }
    
    def _apply_hardware_optimizations(self):
        """Apply hardware-specific optimizations to loaded config."""
        # Update models if not explicitly set
        if "models" in self.config:
            if "llm" in self.config["models"]:
                # Update LLM model if using default
                current_llm = self.config["models"]["llm"].get("name", "")
                if current_llm in ["llama3.1:8b-instruct-q4_K_M", "llama3.1:3b-instruct-q4_K_M"]:
                    self.config["models"]["llm"]["name"] = self.optimal_models["llm"]
                    logging.info(f"Updated LLM model to: {self.optimal_models['llm']}")
            
            if "embeddings" in self.config["models"]:
                # Update embedding model if using default
                current_emb = self.config["models"]["embeddings"].get("name", "")
                if current_emb in ["BAAI/bge-m3", "sentence-transformers/all-MiniLM-L6-v2"]:
                    self.config["models"]["embeddings"]["name"] = self.optimal_models["embedding"]
                    self.config["models"]["embeddings"]["device"] = "cuda" if self.vram_gb > 0 else "cpu"
                    logging.info(f"Updated embedding model to: {self.optimal_models['embedding']}")
        
        # Add hardware info
        if "hardware" not in self.config:
            self.config["hardware"] = {
                "vram_gb": self.vram_gb,
                "gpu_name": self.gpu_name,
                "optimal_models": self.optimal_models
            }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        level = log_config.get("level", "INFO")
        format_str = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Create logs directory if it doesn't exist
        log_file = log_config.get("file_path", "./logs/basketball_rag.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=format_str,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def get(self, key: str, default=None):
        """Get configuration value by key path (e.g., 'models.llm.temperature')."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, key: str, value: Any):
        """Update configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self):
        """Save current configuration to file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as file:
            yaml.dump(self.config, file, default_flow_style=False, allow_unicode=True)
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information and optimal model recommendations."""
        return {
            "vram_gb": self.vram_gb,
            "gpu_name": self.gpu_name,
            "optimal_models": self.optimal_models,
            "current_models": {
                "llm": self.get("models.llm.name"),
                "embedding": self.get("models.embeddings.name")
            }
        }

# Global config instance
config = Config() 