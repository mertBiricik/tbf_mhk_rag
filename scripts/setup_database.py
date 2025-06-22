#!/usr/bin/env python3
"""
Basketball RAG Database Setup Script
Processes documents and creates vector database.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from src.utils.config import config
    from src.document_processing.processor import BasketballDocumentProcessor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed and run setup_environment.py first")
    sys.exit(1)

def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_source_documents():
    """Check if source documents exist."""
    logger = logging.getLogger(__name__)
    
    source_path = Path('source/txt')
    if not source_path.exists():
        logger.error(f"❌ Source directory not found: {source_path}")
        return False
    
    required_files = [
        'basketbol_oyun_kurallari_2022.txt',
        'basketbol_oyun_kurallari_degisiklikleri_2024.txt', 
        'basketbol_oyun_kurallari_resmi_yorumlar_2023.txt'
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = source_path / file_name
        if not file_path.exists():
            missing_files.append(file_name)
        else:
            file_size = file_path.stat().st_size
            logger.info(f"✅ Found: {file_name} ({file_size:,} bytes)")
    
    if missing_files:
        logger.error(f"❌ Missing source documents: {missing_files}")
        logger.error("Please ensure all Turkish basketball rules documents are in source/txt/")
        return False
    
    return True

def process_documents():
    """Process and chunk documents."""
    logger = logging.getLogger(__name__)
    
    logger.info("📚 Processing basketball documents...")
    
    try:
        processor = BasketballDocumentProcessor()
        
        # Process all documents
        start_time = time.time()
        chunks = processor.process_all_documents()
        process_time = time.time() - start_time
        
        if not chunks:
            logger.error("❌ No chunks were created from documents")
            return False, []
        
        logger.info(f"✅ Document processing completed in {process_time:.2f}s")
        logger.info(f"   Total chunks: {len(chunks)}")
        
        # Analyze chunks
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        
        logger.info(f"   Total characters: {total_chars:,}")
        logger.info(f"   Average chunk size: {avg_chunk_size:.1f} characters")
        
        # Show document type distribution
        doc_types = {}
        for chunk in chunks:
            doc_type = chunk.metadata.get('document_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        logger.info("   Document type distribution:")
        for doc_type, count in doc_types.items():
            logger.info(f"     {doc_type}: {count} chunks")
        
        return True, chunks
        
    except Exception as e:
        logger.error(f"❌ Error processing documents: {e}")
        return False, []

def create_vector_database(chunks):
    """Create vector database from processed chunks."""
    logger = logging.getLogger(__name__)
    
    logger.info("🗃️  Creating vector database...")
    
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        import torch
        
        # Setup embedding model
        model_name = config.get('models.embeddings.name', 'BAAI/bge-m3')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"📊 Loading embedding model: {model_name} on {device}")
        embedding_model = SentenceTransformer(model_name)
        embedding_model = embedding_model.to(device)
        
        # Setup ChromaDB
        db_path = config.get('vector_db.persist_directory', './vector_db/chroma_db')
        collection_name = config.get('vector_db.collection_name', 'basketball_rules')
        
        # Create directory if it doesn't exist
        Path(db_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=str(db_path))
        
        # Create or get collection
        try:
            collection = client.get_collection(name=collection_name)
            logger.info(f"📋 Found existing collection: {collection_name}")
            # Clear existing collection
            collection.delete()
            collection = client.create_collection(name=collection_name)
            logger.info("🗑️  Cleared existing collection")
        except ValueError:
            collection = client.create_collection(name=collection_name)
            logger.info(f"📋 Created new collection: {collection_name}")
        
        # Process chunks in batches
        batch_size = config.get('models.embeddings.batch_size', 32)
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        logger.info(f"⚡ Processing {len(chunks)} chunks in {total_batches} batches...")
        
        start_time = time.time()
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"   Processing batch {batch_num}/{total_batches}")
            
            # Extract texts and metadata
            texts = [chunk.page_content for chunk in batch_chunks]
            metadatas = [chunk.metadata for chunk in batch_chunks]
            ids = [f"chunk_{i+j}" for j in range(len(batch_chunks))]
            
            # Generate embeddings
            embeddings = embedding_model.encode(texts, convert_to_tensor=False)
            embeddings = embeddings.tolist()  # Convert to list for ChromaDB
            
            # Add to collection
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
        
        embedding_time = time.time() - start_time
        
        logger.info(f"✅ Vector database created successfully!")
        logger.info(f"   Embedding time: {embedding_time:.2f}s")
        logger.info(f"   Database path: {db_path}")
        logger.info(f"   Collection: {collection_name}")
        logger.info(f"   Total documents: {collection.count()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating vector database: {e}")
        return False

def test_vector_database():
    """Test vector database with sample queries."""
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing vector database...")
    
    try:
        import chromadb
        
        # Connect to database
        db_path = config.get('vector_db.persist_directory', './vector_db/chroma_db')
        collection_name = config.get('vector_db.collection_name', 'basketball_rules')
        
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection(name=collection_name)
        
        # Test queries
        test_queries = [
            "basketbol kuralları",
            "5 faul kuralı", 
            "şut saati",
            "teknik faul"
        ]
        
        for query in test_queries:
            logger.info(f"   Testing query: '{query}'")
            
            results = collection.query(
                query_texts=[query],
                n_results=3
            )
            
            if results['documents'] and results['documents'][0]:
                logger.info(f"     ✅ Found {len(results['documents'][0])} results")
                # Show first result preview
                first_result = results['documents'][0][0]
                preview = first_result[:100] + "..." if len(first_result) > 100 else first_result
                logger.info(f"     Preview: {preview}")
            else:
                logger.warning(f"     ⚠️  No results found")
        
        logger.info("✅ Vector database test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing vector database: {e}")
        return False

def main():
    """Main database setup function."""
    print("🏀 Basketball RAG Database Setup")
    print("=" * 50)
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Run setup steps
    steps = [
        ("Source Documents Check", check_source_documents),
        ("Document Processing", lambda: process_documents()[0]),
        ("Vector Database Creation", lambda: create_vector_database(process_documents()[1]) if process_documents()[0] else False),
        ("Database Testing", test_vector_database),
    ]
    
    failed_steps = []
    processed_chunks = None
    
    for step_name, step_func in steps:
        logger.info(f"\n🔍 Running: {step_name}")
        try:
            if step_name == "Document Processing":
                success, processed_chunks = process_documents()
                if not success:
                    failed_steps.append(step_name)
            elif step_name == "Vector Database Creation":
                if processed_chunks:
                    if not create_vector_database(processed_chunks):
                        failed_steps.append(step_name)
                else:
                    logger.error("❌ No processed chunks available")
                    failed_steps.append(step_name)
            else:
                if not step_func():
                    failed_steps.append(step_name)
        except Exception as e:
            logger.error(f"❌ {step_name} failed with error: {e}")
            failed_steps.append(step_name)
    
    # Summary
    print("\n" + "=" * 50)
    if failed_steps:
        logger.error(f"❌ Database setup incomplete. Failed steps: {', '.join(failed_steps)}")
        logger.error("Please resolve the issues above and run setup again.")
        return False
    else:
        logger.info("✅ Database setup completed successfully!")
        logger.info("\n🎯 Next steps:")
        logger.info("   1. Run: python scripts/run_gradio.py")
        logger.info("   2. Test queries in the web interface")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 