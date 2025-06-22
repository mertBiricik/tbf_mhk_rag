# Complete Guide to RAG (Retrieval-Augmented Generation) Projects

## Table of Contents
1. [What is RAG?](#what-is-rag)
2. [Why Use RAG?](#why-use-rag)
3. [RAG Architecture Overview](#rag-architecture-overview)
4. [Core Components](#core-components)
5. [Step-by-Step Implementation](#step-by-step-implementation)
6. [Technical Implementation Details](#technical-implementation-details)
7. [Popular Tools & Frameworks](#popular-tools--frameworks)
8. [Best Practices](#best-practices)
9. [Common Challenges & Solutions](#common-challenges--solutions)
10. [Evaluation & Metrics](#evaluation--metrics)
11. [Real-World Example: Basketball Rules System](#real-world-example-basketball-rules-system)

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI architecture pattern that combines the power of large language models (LLMs) with external knowledge retrieval systems. Instead of relying solely on the pre-trained knowledge within an LLM, RAG systems first retrieve relevant information from external sources and then use that information to generate more accurate, up-to-date, and contextually relevant responses.

### The Problem RAG Solves

Traditional LLMs have several limitations:
- **Knowledge Cutoff**: They only know information up to their training date
- **Hallucination**: They sometimes generate plausible-sounding but incorrect information
- **Domain-Specific Knowledge**: They may lack deep expertise in specialized fields
- **Dynamic Information**: They can't access real-time or frequently updated information
- **Source Attribution**: They can't cite specific sources for their claims

RAG addresses these issues by grounding the LLM's responses in retrieved factual information from your specific knowledge base.

### How RAG Works (Simple Explanation)

Think of RAG like having a research assistant:
1. You ask a question
2. The assistant searches through relevant documents to find information
3. The assistant reads the found information and uses it to write a comprehensive answer
4. You get an answer that's based on actual source material

## Why Use RAG?

### Key Benefits

1. **Accuracy & Reliability**: Responses are grounded in actual source documents, reducing hallucinations
2. **Up-to-date Information**: Can work with recently added documents without retraining models
3. **Transparency**: Can show which sources were used for answers, enabling verification
4. **Domain Expertise**: Can be tailored to specific knowledge domains with specialized documents
5. **Cost-Effective**: No need to retrain expensive large models with new information
6. **Controllable Knowledge**: You control exactly what knowledge the system has access to
7. **Privacy**: Can work with proprietary documents without sending them to external APIs

### Common Use Cases

- **Customer Support**: Answer questions using company documentation, FAQs, and knowledge bases
- **Research Assistance**: Help researchers find and synthesize information from papers and reports
- **Educational Tools**: Create tutoring systems with textbooks, lecture notes, and course materials
- **Legal/Compliance**: Query legal documents, regulations, and compliance frameworks
- **Technical Documentation**: Help developers with API documentation, coding guides, and best practices
- **Content Creation**: Generate content based on brand guidelines, style guides, and source materials
- **Enterprise Knowledge Management**: Make organizational knowledge searchable and accessible

## RAG Architecture Overview

```
User Query → [Query Processing] → [Retrieval System] → [Retrieved Documents] → [LLM + Prompt Engineering] → Response
                                          ↓
                                  [Vector Database]
                                          ↑
                                  [Document Processing Pipeline]
                                          ↑
                                  [Source Documents (PDFs, texts, etc.)]
```

### High-Level Flow

1. **Document Ingestion**: Process and store documents in a searchable format
   - Load documents from various sources
   - Clean and preprocess text
   - Split into manageable chunks
   - Convert to vector embeddings
   - Store in vector database

2. **Query Processing**: User submits a question
   - Clean and preprocess the query
   - Convert query to vector embedding
   - Optionally expand or rephrase query

3. **Retrieval**: Find relevant document chunks
   - Search vector database for similar content
   - Apply filters (date, document type, etc.)
   - Rank results by relevance

4. **Augmentation**: Combine retrieved content with the user query
   - Select best matching chunks
   - Format context for the LLM
   - Create comprehensive prompt

5. **Generation**: LLM generates response using retrieved context
   - Feed context and query to language model
   - Generate human-like response
   - Ensure response stays grounded in source material

6. **Response**: Return answer to user with optional source citations
   - Format final response
   - Include source references
   - Provide confidence indicators

## Core Components

### 1. Document Processing Pipeline

**Purpose**: Convert raw documents into searchable, semantic chunks that preserve meaning while being manageable for retrieval and processing.

**Key Steps**:

**Document Loading**:
- Extract text from various formats (PDF, Word, HTML, Markdown, etc.)
- Handle different encodings and languages
- Preserve important structural information
- Extract metadata (author, date, document type)

**Text Cleaning**:
- Remove formatting artifacts, headers, footers
- Handle special characters and encoding issues
- Normalize whitespace and line breaks
- Remove or handle non-text elements (tables, images)

**Chunking Strategy**:
- Split documents into manageable pieces (typically 500-1500 characters)
- Preserve semantic boundaries (don't break sentences/paragraphs)
- Add overlap between chunks to prevent information loss
- Maintain chunk metadata for tracking

**Example Code**:
```python
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Target size per chunk
    chunk_overlap=200,    # Overlap between chunks
    separators=["\n\n", "\n", ". ", " "]  # Split boundaries
)
chunks = text_splitter.split_documents(documents)
```

### 2. Embedding Models

**Purpose**: Convert text chunks and queries into numerical vectors that capture semantic meaning, enabling similarity-based search.

**How Embeddings Work**:
- Transform text into high-dimensional vectors (typically 768-1536 dimensions)
- Similar concepts have similar vector representations
- Enable semantic search beyond keyword matching
- Allow for mathematical operations on text meaning

**Popular Embedding Options**:

**Commercial APIs**:
- OpenAI `text-embedding-ada-002`: High quality, easy to use
- Cohere Embed: Good multilingual support
- Voyage AI: Specialized for retrieval tasks

**Open Source Models**:
- Sentence-BERT: Various model sizes and specializations
- E5 (Microsoft): Strong performance across tasks
- BGE (BAAI): Chinese company, good multilingual
- Instructor: Supports task-specific instructions

**Selection Criteria**:
- Quality for your domain
- Language support
- Model size vs. performance
- Cost considerations
- Licensing requirements

### 3. Vector Database

**Purpose**: Store document embeddings and efficiently search for similar vectors to enable fast semantic retrieval.

**Key Features**:
- **Similarity Search**: Find semantically similar content using vector operations
- **Scalability**: Handle millions of document chunks efficiently
- **Metadata Filtering**: Filter by document properties (date, type, author)
- **Real-time Updates**: Add/remove documents without rebuilding index
- **Persistence**: Store embeddings durably for repeated use

**Popular Vector Database Options**:

**Managed Services**:
- **Pinecone**: Easy to use, highly scalable, good performance
- **Weaviate**: Open source with cloud options, multi-modal support
- **Qdrant**: Fast, feature-rich, good for production

**Self-Hosted/Local Options**:
- **Chroma**: Great for development, easy Python integration
- **FAISS**: Facebook's library, very fast for local use
- **Milvus**: Open source, highly scalable

**Example Setup**:
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Create embeddings
embeddings = OpenAIEmbeddings()

# Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./vector_db"
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Return top 5 matches
)
```

### 4. Retrieval System

**Purpose**: Find the most relevant document chunks for a given query using various search strategies.

**Retrieval Strategies**:

**Semantic Search**:
- Use vector similarity to find related content
- Good for conceptual matches
- Works across different wording

**Keyword Search**:
- Traditional text search using keywords
- Good for exact matches and specific terms
- Fast and reliable for known terminology

**Hybrid Search**:
- Combine semantic and keyword approaches
- Get benefits of both methods
- Often provides best overall results

**Advanced Techniques**:
- **Re-ranking**: Improve initial retrieval with second-stage ranking
- **Query expansion**: Add related terms to improve recall
- **Metadata filtering**: Use document properties to constrain search
- **Multi-stage retrieval**: First broad search, then focused refinement

### 5. Language Model Integration

**Purpose**: Generate human-like responses using the retrieved context while staying grounded in source material.

**LLM Options**:

**Commercial APIs**:
- **OpenAI GPT-4**: Excellent performance, good instruction following
- **Anthropic Claude**: Long context windows, safety-focused
- **Google Gemini**: Multimodal capabilities, competitive performance

**Open Source Models**:
- **Llama 2/3**: Meta's models, good performance, various sizes
- **Mistral**: French company, efficient models
- **Code Llama**: Specialized for programming tasks

**Integration Considerations**:
- Context window size (how much text can be processed at once)
- Cost per token for API models
- Latency requirements
- Privacy and data handling requirements
- Customization needs

## Step-by-Step Implementation

### Phase 1: Environment Setup

First, set up your development environment with the necessary dependencies:

```bash
# Create virtual environment
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate

# Install core libraries
pip install langchain
pip install openai
pip install chromadb
pip install pypdf2
pip install tiktoken
pip install python-dotenv
```

Create environment configuration:
```python
# .env file
OPENAI_API_KEY=your_openai_api_key_here
```

### Phase 2: Document Processing Implementation

```python
import os
from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_documents_from_directory(directory_path: str) -> List[Document]:
    """Load all supported documents from a directory."""
    documents = []
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        try:
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                # Add filename to metadata
                for doc in docs:
                    doc.metadata['filename'] = filename
                documents.extend(docs)
                
            elif filename.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                for doc in docs:
                    doc.metadata['filename'] = filename
                documents.extend(docs)
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return documents

def create_chunks(documents: List[Document]) -> List[Document]:
    """Split documents into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk information to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        chunk.metadata['chunk_size'] = len(chunk.page_content)
    
    return chunks

# Example usage
documents = load_documents_from_directory("./source/txt/")
chunks = create_chunks(documents)
print(f"Created {len(chunks)} chunks from {len(documents)} documents")
```

### Phase 3: Vector Store Setup

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import dotenv

# Load environment variables
dotenv.load_dotenv()

def create_vector_store(chunks: List[Document], persist_directory: str = "./chroma_db"):
    """Create and populate vector store."""
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Persist the database
    vectorstore.persist()
    
    return vectorstore

def load_vector_store(persist_directory: str = "./chroma_db"):
    """Load existing vector store."""
    embeddings = OpenAIEmbeddings()
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    return vectorstore

# Create vector store
vectorstore = create_vector_store(chunks)
print("Vector store created and populated")
```

### Phase 4: RAG Chain Implementation

```python
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def create_rag_chain(vectorstore, return_sources: bool = True):
    """Create the complete RAG chain."""
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Define prompt template
    prompt_template = """
    You are a helpful assistant that answers questions based on the provided context.
    Use the following pieces of context to answer the question at the end.
    
    If you don't know the answer based on the context, just say that you don't know.
    Don't try to make up an answer.
    
    When possible, reference the specific source document in your answer.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create LLM
    llm = OpenAI(temperature=0)  # Low temperature for consistent answers
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=return_sources,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

# Create RAG system
qa_chain = create_rag_chain(vectorstore)

# Test the system
def ask_question(question: str):
    """Ask a question and get response with sources."""
    result = qa_chain({"query": question})
    
    print(f"Question: {question}")
    print(f"Answer: {result['result']}")
    
    if result.get('source_documents'):
        print("\nSources:")
        for i, doc in enumerate(result['source_documents']):
            print(f"{i+1}. {doc.metadata.get('filename', 'Unknown')} - {doc.page_content[:100]}...")
    
    print("-" * 50)
    return result

# Example usage
ask_question("What are the main basketball rules?")
```

This implementation provides a solid foundation for your basketball rules RAG system. In the next sections, I'll cover advanced techniques, best practices, and how to evaluate and improve your system. 

## Technical Implementation Details

### Advanced Chunking Strategies

**Semantic Chunking**:
```python
def semantic_chunking(text: str, max_chunk_size: int = 1000):
    """Split text at semantic boundaries."""
    import spacy
    
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    chunks = []
    current_chunk = ""
    
    for sent in doc.sents:
        if len(current_chunk) + len(sent.text) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sent.text
        else:
            current_chunk += " " + sent.text
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
```

**Overlapping Windows**:
```python
def create_overlapping_chunks(text: str, chunk_size: int = 1000, overlap: int = 200):
    """Create overlapping chunks to preserve context."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Find last complete sentence
        last_period = chunk.rfind('.')
        if last_period > chunk_size * 0.8:  # If sentence break is reasonably close to end
            chunk = chunk[:last_period + 1]
            end = start + last_period + 1
        
        chunks.append(chunk)
        start = end - overlap
        
        if end >= len(text):
            break
    
    return chunks
```

### Hybrid Search Implementation

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

def create_hybrid_retriever(documents, vectorstore):
    """Combine vector search with keyword search."""
    
    # Vector retriever
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )
    
    # Keyword retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10
    
    # Ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]  # Favor vector search slightly
    )
    
    return ensemble_retriever
```

### Query Expansion

```python
def expand_query(query: str, expansion_terms: int = 3):
    """Expand query with related terms."""
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    # Load model for finding similar terms
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get embedding for original query
    query_embedding = model.encode([query])
    
    # Your domain-specific terms database
    domain_terms = [
        "basketball", "foul", "violation", "shot clock", "free throw",
        "technical foul", "personal foul", "timeout", "substitution"
        # Add more domain-specific terms
    ]
    
    # Find most similar terms
    term_embeddings = model.encode(domain_terms)
    similarities = np.dot(query_embedding, term_embeddings.T)[0]
    
    # Get top similar terms
    top_indices = np.argsort(similarities)[-expansion_terms:]
    expanded_terms = [domain_terms[i] for i in top_indices]
    
    # Create expanded query
    expanded_query = f"{query} {' '.join(expanded_terms)}"
    
    return expanded_query
```

## Popular Tools & Frameworks

### Framework Comparison

| Framework | Strengths | Best For | Learning Curve |
|-----------|-----------|----------|----------------|
| **LangChain** | Extensive ecosystem, many integrations | Rapid prototyping, experimentation | Medium |
| **LlamaIndex** | Document-focused, advanced indexing | Data-heavy applications | Medium |
| **Haystack** | Production-ready, enterprise features | Large-scale deployments | High |
| **Chroma** | Simple setup, local development | Getting started, small projects | Low |

### Vector Database Comparison

| Database | Type | Strengths | Best For |
|----------|------|-----------|----------|
| **Pinecone** | Managed | Easy scaling, high performance | Production apps |
| **Weaviate** | Open/Managed | Multi-modal, GraphQL API | Complex data types |
| **Chroma** | Local/Cloud | Simple setup, Python-friendly | Development, prototyping |
| **Qdrant** | Open/Cloud | Fast, feature-rich | High-performance needs |
| **FAISS** | Local | Very fast, free | Local deployments |

### Embedding Model Comparison

| Model | Dimensions | Strengths | Best For |
|-------|------------|-----------|----------|
| **OpenAI ada-002** | 1536 | High quality, easy to use | General purpose |
| **Sentence-BERT** | 384-768 | Open source, many variants | Custom deployments |
| **E5** | 1024 | Strong performance | Multilingual needs |
| **BGE** | 768-1024 | Good multilingual support | Non-English content |

## Best Practices

### Document Preparation

1. **Clean Your Data**:
   ```python
   def clean_text(text: str) -> str:
       """Clean text for better processing."""
       import re
       
       # Remove extra whitespace
       text = re.sub(r'\s+', ' ', text)
       
       # Remove special characters that don't add meaning
       text = re.sub(r'[^\w\s\.\,\!\?\;]', '', text)
       
       # Fix common OCR errors
       text = text.replace('rn', 'm')  # Common OCR mistake
       
       return text.strip()
   ```

2. **Add Rich Metadata**:
   ```python
   def enrich_metadata(document, filename):
       """Add useful metadata to documents."""
       metadata = document.metadata.copy()
       
       # Extract information from filename
       if "2024" in filename:
           metadata["year"] = 2024
           metadata["version"] = "latest"
       
       # Determine document type
       if "rules" in filename.lower():
           metadata["doc_type"] = "rules"
       elif "interpretation" in filename.lower():
           metadata["doc_type"] = "interpretation"
       
       # Add content-based metadata
       content = document.page_content.lower()
       if "foul" in content:
           metadata["contains_fouls"] = True
       if "shot clock" in content:
           metadata["contains_shot_clock"] = True
       
       document.metadata = metadata
       return document
   ```

### Retrieval Optimization

1. **Implement Re-ranking**:
   ```python
   from sentence_transformers import CrossEncoder
   
   def rerank_documents(query: str, documents, top_k: int = 3):
       """Re-rank retrieved documents for better relevance."""
       
       reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
       
       # Create query-document pairs
       pairs = [[query, doc.page_content] for doc in documents]
       
       # Get relevance scores
       scores = reranker.predict(pairs)
       
       # Sort by score and return top k
       scored_docs = list(zip(documents, scores))
       scored_docs.sort(key=lambda x: x[1], reverse=True)
       
       return [doc for doc, score in scored_docs[:top_k]]
   ```

2. **Metadata Filtering**:
   ```python
   def filtered_retrieval(query: str, vectorstore, filters: dict = None):
       """Retrieve with metadata filters."""
       
       search_kwargs = {"k": 10}
       if filters:
           search_kwargs["filter"] = filters
       
       retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
       return retriever.get_relevant_documents(query)
   
   # Example usage
   recent_docs = filtered_retrieval(
       "basketball fouls", 
       vectorstore, 
       filters={"year": {"$gte": 2023}}
   )
   ```

### Prompt Engineering

1. **Domain-Specific Prompts**:
   ```python
   BASKETBALL_PROMPT = """
   You are an expert basketball referee and rules interpreter. 
   
   Use the following basketball rule excerpts to answer the question accurately.
   Always reference specific rule numbers or sections when available.
   
   If the question involves recent rule changes, prioritize the most recent information.
   If you cannot find the answer in the provided context, say so clearly.
   
   Context: {context}
   
   Question: {question}
   
   Answer (include rule references where applicable):
   """
   ```

2. **Chain of Thought Prompting**:
   ```python
   COT_PROMPT = """
   Let's think through this basketball rules question step by step.
   
   Context: {context}
   Question: {question}
   
   Step 1: Identify the key rule categories involved
   Step 2: Find relevant rule sections in the context
   Step 3: Apply the rules to the specific situation
   Step 4: Provide a clear answer with rule references
   
   Answer:
   """
   ```

### Performance Optimization

1. **Caching Frequent Queries**:
   ```python
   from functools import lru_cache
   import hashlib
   
   class CachedRAG:
       def __init__(self, qa_chain):
           self.qa_chain = qa_chain
           self._cache = {}
       
       def _hash_query(self, query: str) -> str:
           return hashlib.md5(query.encode()).hexdigest()
       
       def ask(self, query: str):
           query_hash = self._hash_query(query)
           
           if query_hash in self._cache:
               return self._cache[query_hash]
           
           result = self.qa_chain({"query": query})
           self._cache[query_hash] = result
           
           return result
   ```

2. **Async Processing**:
   ```python
   import asyncio
   from concurrent.futures import ThreadPoolExecutor
   
   async def async_retrieval(queries: list, qa_chain):
       """Process multiple queries concurrently."""
       
       def process_query(query):
           return qa_chain({"query": query})
       
       with ThreadPoolExecutor(max_workers=5) as executor:
           loop = asyncio.get_event_loop()
           tasks = [
               loop.run_in_executor(executor, process_query, query) 
               for query in queries
           ]
           results = await asyncio.gather(*tasks)
       
       return results
   ```

## Common Challenges & Solutions

### Challenge 1: Poor Retrieval Quality

**Symptoms**:
- Irrelevant documents retrieved
- Missing obvious relevant information
- Inconsistent retrieval results

**Solutions**:
```python
# 1. Improve chunking strategy
def improved_chunking(documents):
    # Use smaller chunks for better precision
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller chunks
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "]
    )
    return text_splitter.split_documents(documents)

# 2. Use hybrid search
hybrid_retriever = create_hybrid_retriever(documents, vectorstore)

# 3. Implement query expansion
expanded_query = expand_query(original_query)
```

### Challenge 2: Context Window Limitations

**Symptoms**:
- "Token limit exceeded" errors
- Truncated context
- Lost relevant information

**Solutions**:
```python
def summarize_context(retrieved_docs, max_tokens: int = 3000):
    """Summarize retrieved context to fit token limits."""
    from langchain.chains.summarize import load_summarize_chain
    from langchain.llms import OpenAI
    
    llm = OpenAI(temperature=0)
    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")
    
    # If context is too long, summarize it
    total_content = " ".join([doc.page_content for doc in retrieved_docs])
    
    if len(total_content) > max_tokens * 4:  # Rough token estimation
        summary = summarize_chain.run(retrieved_docs)
        return summary
    else:
        return total_content
```

### Challenge 3: Inconsistent Response Quality

**Symptoms**:
- Responses vary significantly for similar questions
- Sometimes detailed, sometimes vague
- Inconsistent formatting

**Solutions**:
```python
# 1. Use deterministic settings
llm = OpenAI(temperature=0, seed=42)

# 2. Implement response validation
def validate_response(response: str, min_length: int = 50):
    """Validate response quality."""
    if len(response) < min_length:
        return False, "Response too short"
    
    if "I don't know" in response and len(response) < 100:
        return False, "Non-informative response"
    
    return True, "Valid response"

# 3. Use structured outputs
STRUCTURED_PROMPT = """
Please provide your answer in the following format:

**Answer**: [Your main answer here]

**Rule Reference**: [Specific rule numbers or sections]

**Source**: [Which document this comes from]

Context: {context}
Question: {question}
"""
```

## Evaluation & Metrics

### Retrieval Evaluation

```python
def evaluate_retrieval(test_cases):
    """Evaluate retrieval quality."""
    results = []
    
    for query, expected_docs in test_cases:
        retrieved_docs = retriever.get_relevant_documents(query)
        
        # Calculate metrics
        retrieved_ids = {doc.metadata.get('chunk_id') for doc in retrieved_docs}
        expected_ids = {doc.metadata.get('chunk_id') for doc in expected_docs}
        
        # Precision: How many retrieved docs are relevant
        precision = len(retrieved_ids & expected_ids) / len(retrieved_ids) if retrieved_ids else 0
        
        # Recall: How many relevant docs were retrieved
        recall = len(retrieved_ids & expected_ids) / len(expected_ids) if expected_ids else 0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'query': query,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return results
```

### End-to-End Evaluation

```python
def evaluate_rag_system(test_cases):
    """Evaluate complete RAG system."""
    from rouge_score import rouge_scorer
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = []
    
    for query, expected_answer in test_cases:
        # Get RAG response
        response = qa_chain({"query": query})
        generated_answer = response["result"]
        
        # Calculate ROUGE scores
        rouge_scores = scorer.score(expected_answer, generated_answer)
        
        # Check for hallucination (answer not supported by sources)
        source_content = " ".join([doc.page_content for doc in response["source_documents"]])
        
        results.append({
            'query': query,
            'generated_answer': generated_answer,
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure,
            'sources_used': len(response["source_documents"])
        })
    
    return results
```

## Real-World Example: Basketball Rules System

Based on your project structure, here's a complete implementation for your basketball rules RAG system:

### 1. Basketball-Specific Document Processing

```python
def process_basketball_documents():
    """Process basketball rules documents with domain-specific handling."""
    
    # Load documents
    documents = []
    document_types = {
        "basketbol_oyun_kurallari_2022.txt": {"type": "rules", "year": 2022, "version": "official"},
        "basketbol_oyun_kurallari_degisiklikleri_2024.txt": {"type": "changes", "year": 2024, "version": "updates"},
        "basketbol_oyun_kurallari_resmi_yorumlar_2023.txt": {"type": "interpretations", "year": 2023, "version": "official"}
    }
    
    for filename, metadata in document_types.items():
        file_path = f"source/txt/{filename}"
        
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            
            # Enrich with basketball-specific metadata
            for doc in docs:
                doc.metadata.update(metadata)
                doc.metadata['filename'] = filename
                
                # Add content-based tags
                content_lower = doc.page_content.lower()
                tags = []
                
                basketball_terms = {
                    'foul': ['foul', 'faul'],
                    'shot_clock': ['shot clock', 'şut saati', '24 saniye'],
                    'timeout': ['timeout', 'mola'],
                    'substitution': ['substitution', 'oyuncu değişimi'],
                    'violation': ['violation', 'ihlal'],
                    'technical': ['technical', 'teknik']
                }
                
                for tag, terms in basketball_terms.items():
                    if any(term in content_lower for term in terms):
                        tags.append(tag)
                
                doc.metadata['tags'] = tags
            
            documents.extend(docs)
            print(f"Loaded {len(docs)} documents from {filename}")
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return documents

# Basketball-specific chunking
def basketball_chunking(documents):
    """Chunk documents preserving basketball rule structure."""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller for specific rules
        chunk_overlap=100,
        separators=[
            "\n\nMadde",      # Turkish for "Article"
            "\n\nRule",       # English rules
            "\n\nKural",      # Turkish for "Rule"
            "\n\n",           # Paragraph breaks
            "\n",             # Line breaks
            ". ",             # Sentence breaks
            " "               # Word breaks
        ]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add rule number extraction
    import re
    for chunk in chunks:
        # Try to extract rule numbers
        rule_match = re.search(r'(Madde|Rule|Kural)\s+(\d+)', chunk.page_content)
        if rule_match:
            chunk.metadata['rule_number'] = rule_match.group(2)
    
    return chunks
```

### 2. Basketball-Specific RAG Chain

```python
def create_basketball_rag():
    """Create specialized basketball rules RAG system."""
    
    # Process documents
    documents = process_basketball_documents()
    chunks = basketball_chunking(documents)
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./basketball_db"
    )
    
    # Basketball-specific prompt
    basketball_prompt = PromptTemplate(
        template="""
        You are an expert basketball referee and rules interpreter with deep knowledge of FIBA basketball rules.
        
        Answer the question using the provided basketball rule excerpts. Be precise and reference specific rule numbers when available.
        
        Guidelines:
        - If the question involves rule changes, prioritize the most recent information (2024 updates)
        - For interpretations, refer to official interpretations when available
        - If you cannot find the answer in the provided context, say so clearly
        - Use both Turkish and English rule references if applicable
        - Be specific about situations and exceptions
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """,
        input_variables=["context", "question"]
    )
    
    # Create retriever with basketball-specific settings
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 7}  # More context for complex rules
    )
    
    # Create QA chain
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": basketball_prompt}
    )
    
    return qa_chain, vectorstore

# Enhanced query function for basketball
def ask_basketball_question(question: str, qa_chain, filters=None):
    """Ask basketball-specific questions with enhanced features."""
    
    # Query expansion for basketball terms
    basketball_synonyms = {
        "foul": ["faul", "ihlal", "violation"],
        "shot clock": ["şut saati", "24 saniye", "24 seconds"],
        "timeout": ["mola", "time out"],
        "free throw": ["serbest atış", "free shot"]
    }
    
    expanded_question = question
    for term, synonyms in basketball_synonyms.items():
        if term.lower() in question.lower():
            expanded_question += f" {' '.join(synonyms)}"
    
    # Get response
    result = qa_chain({"query": expanded_question})
    
    print(f"Question: {question}")
    print(f"Answer: {result['result']}")
    
    # Enhanced source display
    if result.get('source_documents'):
        print("\nSources:")
        for i, doc in enumerate(result['source_documents']):
            doc_type = doc.metadata.get('type', 'unknown')
            year = doc.metadata.get('year', 'unknown')
            rule_num = doc.metadata.get('rule_number', '')
            
            source_info = f"{doc_type.title()} ({year})"
            if rule_num:
                source_info += f" - Rule {rule_num}"
            
            print(f"{i+1}. {source_info}")
            print(f"   Content: {doc.page_content[:150]}...")
            print()
    
    return result

# Create the basketball RAG system
basketball_qa, basketball_vectorstore = create_basketball_rag()

# Example usage
print("Basketball Rules RAG System Ready!")
print("Try asking questions like:")
print("- What happens when a player commits 5 fouls?")
print("- What are the shot clock rules?")
print("- What changed in the 2024 rule updates?")
print("- How is a technical foul different from a personal foul?")

# Test questions
test_questions = [
    "What are the rules for player fouls?",
    "How does the shot clock work?",
    "What are the timeout rules?",
    "What constitutes a technical foul?",
    "What changed in the 2024 basketball rules?"
]

for question in test_questions:
    ask_basketball_question(question, basketball_qa)
    print("=" * 80)
```

### 3. Advanced Features for Basketball System

```python
# Specialized retrieval for different query types
def basketball_smart_retrieval(question: str, vectorstore):
    """Smart retrieval based on question type."""
    
    question_lower = question.lower()
    
    # Determine question type and adjust search
    if any(term in question_lower for term in ['what changed', 'new rule', '2024']):
        # Focus on rule changes
        filters = {"type": "changes", "year": 2024}
    elif any(term in question_lower for term in ['interpretation', 'clarification']):
        # Focus on interpretations
        filters = {"type": "interpretations"}
    elif any(term in question_lower for term in ['basic', 'general', 'overview']):
        # Focus on main rules
        filters = {"type": "rules"}
    else:
        filters = None
    
    search_kwargs = {"k": 8}
    if filters:
        search_kwargs["filter"] = filters
    
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    return retriever.get_relevant_documents(question)

# Basketball rule conflict resolution
def resolve_rule_conflicts(documents):
    """Handle conflicts between different rule versions."""
    
    # Group by rule number
    rules_by_number = {}
    for doc in documents:
        rule_num = doc.metadata.get('rule_number')
        if rule_num:
            if rule_num not in rules_by_number:
                rules_by_number[rule_num] = []
            rules_by_number[rule_num].append(doc)
    
    # For each rule, prioritize most recent
    resolved_docs = []
    for rule_num, rule_docs in rules_by_number.items():
        # Sort by year, prioritizing changes over base rules
        rule_docs.sort(key=lambda x: (x.metadata.get('year', 0), 
                                    1 if x.metadata.get('type') == 'changes' else 0), 
                      reverse=True)
        resolved_docs.append(rule_docs[0])  # Take most recent
    
    # Add documents without rule numbers
    for doc in documents:
        if not doc.metadata.get('rule_number'):
            resolved_docs.append(doc)
    
    return resolved_docs
```

This comprehensive guide gives you everything you need to build a sophisticated RAG system for your basketball rules documents. The system can handle multiple document types, resolve conflicts between rule versions, and provide specialized responses based on question types.

Start with the basic implementation and gradually add the advanced features as you become more comfortable with the system. The basketball-specific examples should work well with your current document structure! 