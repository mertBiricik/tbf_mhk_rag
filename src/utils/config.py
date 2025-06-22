"""Configuration management utilities."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class Config:
    """Configuration manager for Basketball RAG system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_logging()
    
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
        """Return default configuration."""
        return {
            "models": {
                "llm": {
                    "name": "llama3.1:8b-instruct-q4_K_M",
                    "base_url": "http://localhost:11434",
                    "temperature": 0.1,
                    "max_tokens": 2048
                },
                "embeddings": {
                    "name": "BAAI/bge-m3",
                    "device": "cuda",
                    "max_seq_length": 1024
                }
            },
            "vector_db": {
                "path": "./vector_db",
                "collection_name": "basketball_rules"
            },
            "document_processing": {
                "chunk_size": 800,
                "chunk_overlap": 100
            }
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

# Global config instance
config = Config() 