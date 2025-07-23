#!/usr/bin/env python3
"""
Optimized Basketball RAG System Launcher
Unified launcher for all system components and testing.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def check_system():
    """Check system requirements and status."""
    print("🔍 System Check")
    print("=" * 50)
    
    # Check Python version
    print(f"Python Version: {sys.version.split()[0]}")
    
    # Check CUDA availability
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError:
        print("PyTorch: Not installed")
    
    # Check vector database
    db_path = Path("./vector_db/chroma_db")
    print(f"Vector DB: {'✅ Found' if db_path.exists() else '❌ Not found'}")
    
    # Check source documents
    source_path = Path("./source/txt")
    if source_path.exists():
        files = list(source_path.glob("*.txt"))
        print(f"Source Documents: {len(files)} files found")
    else:
        print("Source Documents: ❌ Not found")
    
    # Check performance data
    stats_path = Path("./stats")
    if stats_path.exists():
        csv_files = list((stats_path / "csv").glob("*.csv")) if (stats_path / "csv").exists() else []
        json_files = list((stats_path / "json").glob("*.json")) if (stats_path / "json").exists() else []
        print(f"Performance Data: {len(csv_files)} CSV files, {len(json_files)} JSON files")
    else:
        print("Performance Data: ❌ Not found")
    
    # Check Ollama service
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("Ollama Service: ✅ Running")
        else:
            print("Ollama Service: ❌ Not responding")
    except:
        print("Ollama Service: ❌ Not accessible")

def launch_gradio():
    """Launch the optimized Gradio interface."""
    print("🚀 Launching Optimized Gradio Interface...")
    try:
        from scripts.optimized_gradio_app import main
        main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Launch error: {e}")

def test_system():
    """Run comprehensive system tests."""
    print("🧪 Testing Optimized System...")
    try:
        from scripts.optimized_rag_system import main
        main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Test error: {e}")

def setup_database():
    """Setup the vector database."""
    print("🗃️ Setting up Vector Database...")
    try:
        from scripts.setup_database import main
        main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Setup error: {e}")

def install_requirements():
    """Install required packages."""
    print("📦 Installing Requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")

def benchmark_system():
    """Run performance benchmarks."""
    print("⚡ Running Performance Benchmarks...")
    try:
        from scripts.optimized_rag_system import OptimizedBasketballRAG
        
        # Initialize system
        rag = OptimizedBasketballRAG()
        rag.initialize_system()
        
        # Benchmark queries
        test_queries = [
            "5 faul yapan oyuncuya ne olur?",
            "Basketbol sahasının boyutları nelerdir?",
            "What happens when a player gets 5 fouls?",
            "2024 yılında hangi kurallar değişti?",
            "Şut saati kuralı nasıl işler?",
            "Teknik faul ne zaman verilir?",
            "Oyuncu değişimi nasıl yapılır?",
            "What are the timeout rules?",
            "Diskalifiye edici faul nedir?",
            "How are player substitutions made?"
        ]
        
        # Run benchmark
        results = rag.benchmark_system(test_queries)
        
        # Display results
        stats = results['statistics']
        print(f"\n📊 Benchmark Results:")
        print(f"   Queries Processed: {stats['total_queries']}")
        print(f"   Average Response Time: {stats['avg_response_time']:.3f}s")
        print(f"   Min Response Time: {stats['min_response_time']:.3f}s")
        print(f"   Max Response Time: {stats['max_response_time']:.3f}s")
        print(f"   Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
        print(f"   Queries per Second: {stats['queries_per_second']:.1f}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Benchmark error: {e}")

def analyze_performance():
    """Run comprehensive performance analysis."""
    print("📈 Running Comprehensive Performance Analysis...")
    try:
        # Run the existing performance analysis script
        subprocess.run([sys.executable, "scripts/analyze_rag_performance.py"], check=True)
        print("✅ Performance analysis completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Performance analysis failed: {e}")
    except Exception as e:
        print(f"❌ Analysis error: {e}")

def evaluate_rag():
    """Run detailed RAG evaluation."""
    print("🔬 Running Detailed RAG Evaluation...")
    try:
        # Run the real RAG evaluation script
        subprocess.run([sys.executable, "scripts/real_rag_evaluation.py"], check=True)
        print("✅ RAG evaluation completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ RAG evaluation failed: {e}")
    except Exception as e:
        print(f"❌ Evaluation error: {e}")

def run_full_evaluation():
    """Run comprehensive evaluation suite."""
    print("🧮 Running Full Evaluation Suite...")
    try:
        # Run multiple evaluation scripts
        evaluation_scripts = [
            "scripts/evaluate_performance.py",
            "scripts/run_performance_analysis.py",
            "scripts/real_rag_evaluation.py"
        ]
        
        for script in evaluation_scripts:
            if Path(script).exists():
                print(f"Running {script}...")
                subprocess.run([sys.executable, script], check=True)
            else:
                print(f"⚠️ Script not found: {script}")
        
        print("✅ Full evaluation suite completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation suite failed: {e}")
    except Exception as e:
        print(f"❌ Suite error: {e}")

def view_stats():
    """Display available performance statistics."""
    print("📊 Available Performance Statistics")
    print("=" * 50)
    
    stats_path = Path("./stats")
    if not stats_path.exists():
        print("❌ No stats directory found")
        return
    
    # Show CSV files
    csv_path = stats_path / "csv"
    if csv_path.exists():
        csv_files = list(csv_path.glob("*.csv"))
        print(f"📈 CSV Data Files: {len(csv_files)}")
        for file in sorted(csv_files)[:10]:  # Show first 10
            size = file.stat().st_size / 1024
            print(f"   - {file.name} ({size:.1f} KB)")
        if len(csv_files) > 10:
            print(f"   ... and {len(csv_files) - 10} more files")
    
    # Show JSON files
    json_path = stats_path / "json"
    if json_path.exists():
        json_files = list(json_path.glob("*.json"))
        print(f"📝 JSON Metadata Files: {len(json_files)}")
    
    # Show other files
    other_files = [f for f in stats_path.iterdir() if f.is_file()]
    if other_files:
        print(f"📋 Other Files:")
        for file in other_files:
            size = file.stat().st_size / 1024
            print(f"   - {file.name} ({size:.1f} KB)")

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description="Optimized Basketball RAG System Launcher")
    parser.add_argument("command", choices=[
        "check", "gradio", "test", "setup", "install", "benchmark", 
        "analyze", "evaluate", "eval-full", "stats"
    ], help="Command to execute")
    
    args = parser.parse_args()
    
    print("🏀 Optimized Basketball RAG System")
    print("=" * 50)
    
    if args.command == "check":
        check_system()
    elif args.command == "gradio":
        launch_gradio()
    elif args.command == "test":
        test_system()
    elif args.command == "setup":
        setup_database()
    elif args.command == "install":
        install_requirements()
    elif args.command == "benchmark":
        benchmark_system()
    elif args.command == "analyze":
        analyze_performance()
    elif args.command == "evaluate":
        evaluate_rag()
    elif args.command == "eval-full":
        run_full_evaluation()
    elif args.command == "stats":
        view_stats()

if __name__ == "__main__":
    main() 