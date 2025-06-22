#!/usr/bin/env python3
"""
Basketball RAG System - Gradio Web Interface
Beautiful web interface for Türkiye Basketball Federation rules
"""

import os
import sys
import time
import gradio as gr
import torch
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import re

# Add src to path for config imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set proper environment variables for transformers
os.environ['HF_HOME'] = './models/huggingface'
# Remove deprecated TRANSFORMERS_CACHE if set
if 'TRANSFORMERS_CACHE' in os.environ:
    del os.environ['TRANSFORMERS_CACHE']

# Import hardware detection config
try:
    from src.utils.config import Config
    HARDWARE_DETECTION_AVAILABLE = True
except ImportError:
    print("⚠️  Hardware detection not available - using default models")
    HARDWARE_DETECTION_AVAILABLE = False

def detect_language(text):
    """Simple language detection for Turkish vs English."""
    # Turkish specific characters and common words
    turkish_chars = set('çğıöşüÇĞIİÖŞÜ')
    turkish_words = {'ve', 'bir', 'bu', 'için', 'olan', 'ile', 'den', 'dır', 'lar', 'ler', 'nın', 'nin', 
                     'basketbol', 'oyuncu', 'faul', 'kural', 'sahası', 'atış', 'saha', 'oyun', 'yapan', 
                     'zaman', 'süre', 'dakika', 'saniye', 'şut', 'top', 'takım', 'oyuncuya', 'nedir', 
                     'nasıl', 'nelerdir', 'hangi', 'değişti', 'kuralları', 'boyutları'}
    
    # English common words
    english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                     'basketball', 'player', 'foul', 'rule', 'court', 'shot', 'game', 'team', 'what',
                     'how', 'when', 'where', 'why', 'rules', 'dimensions', 'changed', 'which'}
    
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    
    # Check for Turkish characters
    if any(char in text for char in turkish_chars):
        return 'turkish'
    
    # Count Turkish vs English words
    turkish_score = len(words & turkish_words)
    english_score = len(words & english_words)
    
    # If we have Turkish words or no clear English dominance, assume Turkish
    if turkish_score > 0 or english_score == 0:
        return 'turkish'
    elif english_score > turkish_score:
        return 'english'
    else:
        return 'turkish'  # Default to Turkish

class BasketballRAG:
    def __init__(self):
        self.embedding_model = None
        self.collection = None
        self.config = None
        self.setup_hardware_detection()
        self.setup_system()
    
    def setup_hardware_detection(self):
        """Detect hardware and select optimal models."""
        if HARDWARE_DETECTION_AVAILABLE:
            try:
                # Load config with hardware detection
                self.config = Config()
                hardware_info = self.config.get_hardware_info()
                
                # Get optimal models from hardware detection
                self.llm_model = hardware_info['current_models']['llm']
                self.embedding_model_name = hardware_info['current_models']['embedding']
                
                print(f"🔍 Hardware Detection Results:")
                print(f"   GPU: {hardware_info['gpu_name']}")
                print(f"   VRAM: {hardware_info['vram_gb']:.1f} GB")
                print(f"   Selected LLM: {self.llm_model}")
                print(f"   Selected Embedding: {self.embedding_model_name}")
                
            except Exception as e:
                print(f"⚠️  Hardware detection failed: {e}")
                print("   Using default models...")
                self.setup_default_models()
        else:
            self.setup_default_models()
    
    def setup_default_models(self):
        """Fallback to default models if hardware detection fails."""
        self.llm_model = "llama3.1:8b-instruct-q4_K_M"
        self.embedding_model_name = "BAAI/bge-m3"
        print(f"🎯 Using default models:")
        print(f"   LLM: {self.llm_model}")
        print(f"   Embedding: {self.embedding_model_name}")
    
    def setup_system(self):
        """Initialize the RAG system components."""
        print("🏀 Initializing Basketball RAG System...")
        
        # Load embedding model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"📊 Loading {self.embedding_model_name} on {device}...")
        
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_model = self.embedding_model.to(device)
        
        # Setup ChromaDB
        db_path = "./vector_db/chroma_db"
        collection_name = "basketball_rules"
        
        client = chromadb.PersistentClient(path=db_path)
        
        try:
            self.collection = client.get_collection(name=collection_name)
            print(f"✅ Connected to vector database: {self.collection.count()} documents")
        except:
            print("❌ Vector database not found. Please run setup first.")
            raise Exception("Vector database not initialized")
    
    def search_rules(self, query, num_results=5):
        """Search for relevant basketball rules."""
        if not query.strip():
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = query_embedding.tolist()
        
        # Search vector database
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=num_results
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
        
        # Format results
        search_results = []
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            search_results.append({
                'content': doc,
                'source': metadata.get('source', 'unknown'),
                'doc_type': metadata.get('doc_type', 'unknown'),
                'year': metadata.get('year', 'unknown')
            })
        
        return search_results
    
    def generate_answer(self, query, search_results):
        """Generate answer using LLM and retrieved documents."""
        # Detect language
        query_language = detect_language(query)
        
        if not search_results:
            if query_language == 'english':
                return "Sorry, I couldn't find information about this topic. Please try rephrasing your question."
            else:
                return "Üzgünüm, bu konuyla ilgili bilgi bulamadım. Lütfen sorunuzu farklı şekilde sormayı deneyin."
        
        # Prepare context
        context_parts = []
        for i, result in enumerate(search_results):
            source_info = f"{result['source']} ({result['year']})"
            if query_language == 'english':
                context_parts.append(f"Source {i+1} - {source_info}:\n{result['content']}")
            else:
                context_parts.append(f"Kaynak {i+1} - {source_info}:\n{result['content']}")
        
        context_text = "\n\n".join(context_parts)
        
        # Create language-appropriate prompt
        if query_language == 'english':
            prompt = f"""You are a Türkiye Basketball Federation rules expert. Answer the question using the official documents below.

OFFICIAL DOCUMENTS:
{context_text}

QUESTION: {query}

ANSWER RULES:
- Use only information from the provided documents
- Give a clear and concise answer
- Mention which document you got the information from
- Include article numbers when relevant
- Answer in English

ANSWER:"""
        else:
            prompt = f"""Sen Türkiye Basketbol Federasyonu kuralları uzmanısın. Aşağıdaki resmi belgelerden yararlanarak soruyu yanıtla.

RESMI BELGELER:
{context_text}

SORU: {query}

YANIT KURALLARI:
- Sadece verilen belgelerden bilgi kullan
- Kısa ve net yanıt ver
- Hangi belgeden aldığın bilgiyi belirt
- Madde numaralarını belirt
- Türkçe yanıtla

YANIT:"""

        # Call LLM
        try:
            payload = {
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 400
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                return answer
            else:
                return f"LLM hatası: HTTP {response.status_code}"
                
        except Exception as e:
            return f"Yanıt oluşturma hatası: {str(e)}"
    
    def query_system(self, question):
        """Complete RAG pipeline for answering questions."""
        if not question.strip():
            return "Lütfen bir soru girin.", "Sonuç yok"
        
        start_time = time.time()
        
        # Search for relevant documents
        search_results = self.search_rules(question)
        
        if not search_results:
            return "Bu konuyla ilgili bilgi bulamadım. Lütfen sorunuzu farklı şekilde sormayı deneyin.", "Kaynak bulunamadı"
        
        # Generate answer
        answer = self.generate_answer(question, search_results)
        
        # Prepare sources info
        sources = []
        for result in search_results:
            sources.append(f"📄 {result['source']} ({result['year']}) - {result['doc_type']}")
        
        sources_text = "\n".join(sources)
        
        response_time = time.time() - start_time
        
        # Add timing info
        answer += f"\n\n⏱️ Yanıt süresi: {response_time:.2f} saniye"
        
        return answer, sources_text

# Initialize RAG system
print("🚀 Starting Basketball RAG System...")
try:
    rag_system = BasketballRAG()
    system_ready = True
except Exception as e:
    print(f"❌ System initialization failed: {e}")
    system_ready = False

def answer_question(question):
    """Gradio interface function."""
    if not system_ready:
        return "❌ Sistem hazır değil. Lütfen vector database'i kurun.", "Sistem hatası"
    
    return rag_system.query_system(question)

# Sample questions for users
sample_questions = [
    "5 faul yapan oyuncuya ne olur?",
    "What happens when a player gets 5 fouls?",
    "Basketbol sahasının boyutları nelerdir?",
    "What are basketball court dimensions?",
    "Şut saati kuralı nasıl işler?",
    "How does the shot clock rule work?",
    "2024 yılında hangi kurallar değişti?",
    "Which rules changed in 2024?",
    "Teknik faul ne zaman verilir?",
    "When is a technical foul given?",
    "Üçlük atış çizgisi nereden başlar?",
    "Where does the three-point line start?",
    "Oyuncu değişimi nasıl yapılır?",
    "How is player substitution done?",
    "Free throw kuralları nelerdir?",
    "What are the free throw rules?"
]

# Create Gradio interface
with gr.Blocks(
    title="🏀 Türk Basketbol Federasyonu RAG Sistemi",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .question-box {
        border-radius: 10px;
        border: 2px solid #3498db;
    }
    .answer-box {
        border-radius: 10px;
        background-color: #f8f9fa;
    }
    """
) as app:
    
    gr.HTML("""
    <div class="main-header">
        <h1>🏀 Türkiye Basketbol Federasyonu</h1>
        <h2>Akıllı Kural Danışmanı</h2>
        <p>Basketbol kuralları hakkında Türkçe veya İngilizce sorular sorun!</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="🤔 Sorunuz",
                placeholder="Örnek: 5 faul yapan oyuncuya ne olur?",
                lines=3,
                elem_classes=["question-box"]
            )
            
            with gr.Row():
                submit_btn = gr.Button("🔍 Yanıtla", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Temizle", variant="secondary")
            
            gr.HTML("<h3>💡 Örnek Sorular:</h3>")
            with gr.Row():
                example_btns = []
                for i in range(0, len(sample_questions), 2):
                    with gr.Column():
                        for j in range(2):
                            if i + j < len(sample_questions):
                                btn = gr.Button(
                                    sample_questions[i + j], 
                                    variant="outline",
                                    size="sm"
                                )
                                btn.click(
                                    fn=lambda x=sample_questions[i + j]: x,
                                    outputs=question_input
                                )
        
        with gr.Column(scale=3):
            answer_output = gr.Textbox(
                label="🎯 Yanıt",
                lines=12,
                elem_classes=["answer-box"],
                interactive=False
            )
            
            sources_output = gr.Textbox(
                label="📚 Kaynaklar",
                lines=4,
                interactive=False
            )
    
    # Event handlers
    submit_btn.click(
        fn=answer_question,
        inputs=[question_input],
        outputs=[answer_output, sources_output]
    )
    
    question_input.submit(
        fn=answer_question,
        inputs=[question_input],
        outputs=[answer_output, sources_output]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", ""),
        outputs=[question_input, answer_output, sources_output]
    )
    
    gr.HTML("""
    <div style="text-align: center; margin-top: 20px; color: #7f8c8d;">
        <p>⚡ GPU Hızlandırmalı | 🧠 Llama 3.1 8B | 📊 BGE-M3 Embeddings</p>
        <p>📋 965 Kural Belgesi | 🎯 Türkçe Dil Desteği</p>
    </div>
    """)

if __name__ == "__main__":
    if system_ready:
        print("🎉 Basketball RAG System Ready!")
        print("🌐 Starting Gradio web interface...")
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
    else:
        print("❌ Cannot start web interface - system not ready") 