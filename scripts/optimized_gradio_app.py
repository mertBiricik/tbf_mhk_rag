#!/usr/bin/env python3
"""
Optimized Basketball RAG System - Gradio Web Interface
High-performance web interface with advanced RAG optimizations.
"""

import os
import sys
import time
import gradio as gr
import torch
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def launch_basic_gradio():
    """Launch the basic gradio interface as fallback."""
    print("🔄 Falling back to basic Gradio interface...")
    try:
        # Import and run the basic gradio app
        from scripts.gradio_app import main as basic_main
        basic_main()
    except Exception as e:
        print(f"❌ Basic gradio launch failed: {e}")
        print("Please check if gradio_app.py exists and is working")

def main():
    """Main function to launch the optimized interface."""
    print("🏀 Starting Optimized Basketball RAG Interface...")
    
    # Try to import the optimized system
    try:
        from scripts.optimized_rag_system import OptimizedBasketballRAG
        
        # Try to initialize the optimized system
        rag_system = OptimizedBasketballRAG()
        rag_system.initialize_system()
        print("✅ Optimized system initialized successfully!")
        
        # If successful, use the optimized interface (placeholder for now)
        print("📝 Note: Optimized interface not fully implemented yet.")
        print("🔄 Falling back to basic interface...")
        launch_basic_gradio()
        
    except ImportError as e:
        print(f"⚠️ Optimized system not available: {e}")
        print("🔄 Using basic interface...")
        launch_basic_gradio()
    except Exception as e:
        print(f"❌ Optimized system initialization failed: {e}")
        print("🔄 Falling back to basic interface...")
        launch_basic_gradio()

if __name__ == "__main__":
    main() 