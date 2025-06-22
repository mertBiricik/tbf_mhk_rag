#!/usr/bin/env python3
"""
Test script for language detection and bilingual responses
"""

import sys
import os
sys.path.append('scripts')

# Import the language detection function
from gradio_app import detect_language, BasketballRAG

def test_language_detection():
    """Test the language detection function."""
    print("🔍 Testing Language Detection")
    print("=" * 40)
    
    test_cases = [
        ("5 faul yapan oyuncuya ne olur?", "turkish"),
        ("What happens when a player gets 5 fouls?", "english"),
        ("Basketbol sahasının boyutları nelerdir?", "turkish"),
        ("What are basketball court dimensions?", "english"),
        ("Şut saati kuralı nasıl işler?", "turkish"),
        ("How does the shot clock rule work?", "english"),
        ("2024 yılında hangi kurallar değişti?", "turkish"),
        ("Which rules changed in 2024?", "english"),
    ]
    
    for question, expected in test_cases:
        detected = detect_language(question)
        status = "✅" if detected == expected else "❌"
        print(f"{status} '{question}' → {detected} (expected: {expected})")
    
    print("\n🎯 Language detection test completed!")

def test_bilingual_responses():
    """Test the RAG system with both languages."""
    print("\n🌐 Testing Bilingual RAG Responses")
    print("=" * 40)
    
    try:
        # Initialize RAG system
        rag = BasketballRAG()
        
        # Test questions in both languages
        test_questions = [
            "5 faul yapan oyuncuya ne olur?",
            "What happens when a player gets 5 fouls?",
            "Basketbol sahasının boyutları nelerdir?",
            "What are basketball court dimensions?"
        ]
        
        for question in test_questions:
            print(f"\n🤔 Question: '{question}'")
            language = detect_language(question)
            print(f"📝 Detected Language: {language}")
            
            # Get response
            answer, sources = rag.query_system(question)
            print(f"🎯 Answer: {answer[:100]}...")
            print(f"📚 Sources found: {len(sources.split(chr(10))) if sources != 'Sonuç yok' else 0}")
            
    except Exception as e:
        print(f"❌ Error testing RAG system: {e}")
        print("💡 Make sure the vector database is set up first")

if __name__ == "__main__":
    test_language_detection()
    test_bilingual_responses() 