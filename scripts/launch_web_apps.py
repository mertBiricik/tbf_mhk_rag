#!/usr/bin/env python3
"""
Basketball RAG System - Web App Launcher
Choose between Gradio or Streamlit interfaces
"""

import argparse
import subprocess
import sys
import os
import time
import requests

# Add src to path for config imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import hardware detection config
try:
    from src.utils.config import Config, get_gpu_memory, select_optimal_models
    HARDWARE_DETECTION_AVAILABLE = True
except ImportError:
    HARDWARE_DETECTION_AVAILABLE = False

def check_ollama():
    """Check if Ollama is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_vector_db():
    """Check if vector database exists."""
    db_path = "./vector_db/chroma_db"
    return os.path.exists(db_path) and os.listdir(db_path)

def show_hardware_info():
    """Display hardware detection information."""
    if not HARDWARE_DETECTION_AVAILABLE:
        print("⚠️  Hardware detection not available")
        return
    
    try:
        print("🔍 Hardware Detection & Model Selection")
        print("-" * 40)
        
        vram_gb, gpu_name = get_gpu_memory()
        optimal_models = select_optimal_models(vram_gb)
        
        print(f"🖥️  GPU: {gpu_name}")
        print(f"💾 VRAM: {vram_gb:.1f} GB")
        print(f"🤖 Selected LLM: {optimal_models['llm']}")
        print(f"🔤 Selected Embedding: {optimal_models['embedding']}")
        
        # Performance expectations
        if vram_gb >= 8.0:
            print("📈 Expected Performance: ⭐⭐⭐⭐⭐ Excellent (~0.5-1s)")
        elif vram_gb >= 6.0:
            print("📈 Expected Performance: ⭐⭐⭐⭐ Very Good (~1-2s)")
        elif vram_gb >= 4.0:
            print("📈 Expected Performance: ⭐⭐⭐ Good (~2-5s)")
        elif vram_gb >= 2.0:
            print("📈 Expected Performance: ⭐⭐ Fair (~5-10s)")
        else:
            print("📈 Expected Performance: ⭐ Basic (~10-30s)")
            
        print()
        
    except Exception as e:
        print(f"❌ Hardware detection failed: {e}")

def start_gradio():
    """Start Gradio interface."""
    print("🚀 Starting Gradio Web Interface...")
    
    # Show hardware detection info
    show_hardware_info()
    
    print("📱 Interface will be available at: http://localhost:7860")
    print("💡 The web app will automatically use optimal models for your hardware!")
    subprocess.run([sys.executable, "scripts/gradio_app.py"])



def run_system_check():
    """Check if all system components are ready."""
    print("🔍 Basketball RAG System - Startup Check")
    print("=" * 60)
    
    # Check Ollama
    print("🧠 Checking Ollama LLM...")
    if check_ollama():
        print("   ✅ Ollama is running")
    else:
        print("   ❌ Ollama is not running")
        print("   💡 Start with: ollama serve")
        return False
    
    # Check vector database
    print("🗃️  Checking Vector Database...")
    if check_vector_db():
        print("   ✅ Vector database found")
    else:
        print("   ❌ Vector database not found")
        print("   💡 Create with: python scripts/test_complete_rag.py")
        return False
    
    print("✅ All systems ready!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Basketball RAG System Web App Launcher")
    parser.add_argument(
        "interface",
        choices=["gradio", "check", "hardware"],
        help="Choose web interface to start, run system check, or show hardware info"
    )
    
    args = parser.parse_args()
    
    if args.interface == "check":
        run_system_check()
        return
    
    if args.interface == "hardware":
        print("🏀 Basketball RAG System - Hardware Information")
        print("=" * 60)
        show_hardware_info()
        return
    
    print("🏀 Basketball RAG System - Web App Launcher")
    print("=" * 60)
    
    # Run system check
    if not run_system_check():
        print("\n❌ System not ready. Please fix the issues above.")
        return
    
    print("\n🌐 Starting web interface...")
    
    if args.interface == "gradio":
        start_gradio()

if __name__ == "__main__":
    main() 