#!/usr/bin/env python3
"""
Simple Basketball RAG App Launcher
Direct launcher for the basketball rules web interface.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def main():
    print("🏀 Basketball RAG System Launcher")
    print("=" * 40)
    
    try:
        # Try to import and run the basic gradio app
        print("🚀 Starting Basketball Rules Interface...")
        from scripts.gradio_app import main as gradio_main
        gradio_main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Please install required packages:")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        print("\n🔧 Try checking:")
        print("   1. Vector database exists: ./vector_db/")
        print("   2. Source documents exist: ./source/txt/")
        print("   3. Ollama service is running: ollama serve")

if __name__ == "__main__":
    main() 