#!/usr/bin/env python3
"""
Test hardware detection and automatic model selection.
This script demonstrates the VRAM-based model selection feature.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config, get_gpu_memory, select_optimal_models
import logging

def test_hardware_detection():
    """Test the hardware detection functionality."""
    print("🔍 Basketball RAG - Hardware Detection Test")
    print("=" * 50)
    
    # Test GPU memory detection
    vram_gb, gpu_name = get_gpu_memory()
    
    print(f"🖥️  Detected GPU: {gpu_name}")
    print(f"💾 Available VRAM: {vram_gb:.1f} GB")
    print()
    
    # Test model selection
    optimal_models = select_optimal_models(vram_gb)
    print("🤖 Optimal Model Selection:")
    print(f"   LLM Model: {optimal_models['llm']}")
    print(f"   Embedding Model: {optimal_models['embedding']}")
    print()
    
    # Test configuration loading
    print("⚙️  Loading Configuration...")
    config = Config()
    hardware_info = config.get_hardware_info()
    
    print("📊 Hardware Configuration Summary:")
    print(f"   GPU: {hardware_info['gpu_name']}")
    print(f"   VRAM: {hardware_info['vram_gb']:.1f} GB")
    print()
    
    print("🎯 Selected Models:")
    print(f"   LLM: {hardware_info['current_models']['llm']}")
    print(f"   Embedding: {hardware_info['current_models']['embedding']}")
    print()
    
    # Memory usage recommendations
    print("💡 Memory Usage Recommendations:")
    if vram_gb >= 8.0:
        print("   ✅ High VRAM - Can run all models on GPU")
        print("   📈 Expected performance: Excellent")
        print("   ⚡ Response time: ~0.5-1s")
    elif vram_gb >= 6.0:
        print("   ✅ Medium VRAM - LLM on GPU, smaller embedding model")
        print("   📈 Expected performance: Very Good")
        print("   ⚡ Response time: ~1-2s")
    elif vram_gb >= 4.0:
        print("   ⚠️  Low VRAM - Using 3B model")
        print("   📈 Expected performance: Good")
        print("   ⚡ Response time: ~2-5s")
        print("   💾 Recommended: Close other GPU applications")
    elif vram_gb >= 2.0:
        print("   ⚠️  Very Low VRAM - Using smallest models")
        print("   📈 Expected performance: Fair")
        print("   ⚡ Response time: ~5-10s")
        print("   💾 Recommended: Use CPU for LLM if issues occur")
    else:
        print("   ❌ Limited/No GPU - CPU processing")
        print("   📈 Expected performance: Basic")
        print("   ⚡ Response time: ~10-30s")
        print("   💾 Recommended: Consider cloud GPU services")
    
    print()
    
    # Model download recommendations
    print("📥 Model Download Requirements:")
    if "3b" in optimal_models['llm']:
        print("   📦 LLM Model Size: ~2GB (3B parameters)")
    elif "8b" in optimal_models['llm']:
        print("   📦 LLM Model Size: ~4.9GB (8B parameters)")
    elif "1.5b" in optimal_models['llm']:
        print("   📦 LLM Model Size: ~1GB (1.5B parameters)")
    
    if "bge-m3" in optimal_models['embedding']:
        print("   📦 Embedding Model Size: ~2.3GB (BGE-M3)")
    elif "MiniLM-L12" in optimal_models['embedding']:
        print("   📦 Embedding Model Size: ~470MB (MiniLM-L12)")
    elif "MiniLM-L6" in optimal_models['embedding']:
        print("   📦 Embedding Model Size: ~90MB (MiniLM-L6)")
    
    return config

def simulate_different_vram():
    """Simulate different VRAM scenarios for testing."""
    print("\n🧪 Simulating Different VRAM Scenarios:")
    print("=" * 50)
    
    test_scenarios = [
        (16.0, "RTX 4090 / RTX A5000"),
        (12.0, "RTX 4070 Ti / RTX 3060"),
        (8.0, "RTX 4060 / RTX 3070"),
        (6.0, "RTX 2060 / GTX 1660 Ti"),
        (4.0, "GTX 1050 Ti / GTX 1650"),
        (2.0, "GT 1030 / Integrated GPU"),
        (0.0, "CPU Only")
    ]
    
    for vram, gpu_type in test_scenarios:
        models = select_optimal_models(vram)
        print(f"\n💾 {vram:.1f}GB VRAM ({gpu_type}):")
        print(f"   🤖 LLM: {models['llm']}")
        print(f"   🔤 Embedding: {models['embedding']}")

if __name__ == "__main__":
    try:
        # Test hardware detection
        config = test_hardware_detection()
        
        # Simulate different scenarios
        simulate_different_vram()
        
        print("\n✅ Hardware detection test completed successfully!")
        print("\n💡 To manually download the selected models:")
        hardware_info = config.get_hardware_info()
        print(f"   ollama pull {hardware_info['optimal_models']['llm']}")
        print(f"   # Embedding model will auto-download: {hardware_info['optimal_models']['embedding']}")
        
    except Exception as e:
        print(f"❌ Error during hardware detection test: {e}")
        logging.error(f"Hardware detection test failed: {e}")
        sys.exit(1) 