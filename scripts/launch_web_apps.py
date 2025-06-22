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

def start_gradio():
    """Start Gradio interface."""
    print("🚀 Starting Gradio Web Interface...")
    print("📱 Interface will be available at: http://localhost:7860")
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
        choices=["gradio", "check"],
        help="Choose web interface to start or run system check"
    )
    
    args = parser.parse_args()
    
    if args.interface == "check":
        run_system_check()
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