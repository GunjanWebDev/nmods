from langchain.document_loaders import PyPDFLoader
print("langchain installed successfully!")
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
from typing import Dict, List, Optional
import json
import pickle
from pathlib import Path
import tabula
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import faiss
from datetime import datetime
from openai import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI

class PDFReferenceProcessor:
    def __init__(self, pdf_path, openai_api_key):
        self.pdf_path = pdf_path
        self.azure_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2023-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.qa_chain = None
        self.setup_qa_chain()

    def setup_qa_chain(self):
        # Load PDF
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store using Azure OpenAI
        embeddings = OpenAIEmbeddings(
            deployment="your-embeddings-deployment",  # Azure OpenAI embeddings deployment name
            model="text-embedding-ada-002",
            openai_api_version="2023-12-01-preview",
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        # Create QA chain with Azure OpenAI
        llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version="2023-12-01-preview",
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

    async def get_column_mapping(self, column_name):
        if not self.qa_chain:
            return None
            
        query = f"""Analyze the mapping for the RL column '{column_name}'. 
        Return in format: "suggested_name|confidence_score". 
        Use confidence 0.0-1.0 based on how closely it matches the documentation."""
        
        try:
            result = await self.qa_chain.arun(query)
            # Parse the result
            parts = result.strip().split('|')
            mapping = parts[0].strip()
            confidence = float(parts[1]) if len(parts) > 1 else 0.7  # Default confidence for PDF matches
            
            return {
                'mapping': mapping if mapping.lower() != 'none' else None,
                'confidence': confidence
            }
        except Exception as e:
            print(f"Error querying PDF reference: {e}")
            return None

class PDFProcessor:
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.index_path = self.cache_dir / "faiss_index.idx"
        self.chunks_path = self.cache_dir / "chunks.pkl"
        self.metadata_path = self.cache_dir / "metadata.json"
        self.index = None
        self.chunks = []
        self.vectorizer = TfidfVectorizer()
        nltk.download('punkt', quiet=True)

    def process_pdf(self, pdf_path: str):
        """Process PDF and create searchable index"""
        # Check if cached version exists and is newer than PDF
        if self._is_cache_valid(pdf_path):
            return self._load_cached_data()

        # Extract tables
        tables = tabula.read_pdf(pdf_path, pages='all')
        
        # Extract text and split into chunks
        chunks = self._extract_text_chunks(pdf_path)
        
        # Create index
        self._create_index(chunks)
        
        # Cache the processed data
        self._cache_data(chunks)

    def _extract_text_chunks(self, pdf_path: str) -> List[Dict]:
        chunks = []
        # Process tables
        tables = tabula.read_pdf(pdf_path, pages='all')
        for idx, table in enumerate(tables):
            chunk = {
                'type': 'table',
                'content': table.to_dict(),
                'metadata': {'table_id': idx}
            }
            chunks.append(chunk)

        # Process text (using PyPDF2 or your preferred PDF text extractor)
        # ... implementation for text extraction ...

        return chunks

    def _create_index(self, chunks: List[Dict]):
        texts = [json.dumps(chunk['content']) for chunk in chunks]
        vectors = self.vectorizer.fit_transform(texts).toarray()
        
        # Initialize FAISS index
        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(vectors.astype(np.float32))
        
        # Save index
        faiss.write_index(self.index, str(self.index_path))

    def _cache_data(self, chunks: List[Dict]):
        # Save chunks
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(chunks, f)

        # Save metadata
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'chunk_count': len(chunks)
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant chunks"""
        if not self.index:
            self._load_cached_data()

        # Convert query to vector
        query_vector = self.vectorizer.transform([query]).toarray().astype(np.float32)
        
        # Search index
        distances, indices = self.index.search(query_vector, k)
        
        # Return relevant chunks
        results = []
        for idx in indices[0]:
            results.append(self.chunks[idx])
        
        return results

    def get_column_mapping(self, column_name: str) -> Optional[Dict]:
        """Get mapping suggestion for a column"""
        relevant_chunks = self.search(column_name)
        if not relevant_chunks:
            return None

        # Analyze chunks to find best mapping
        best_mapping = self._analyze_chunks(column_name, relevant_chunks)
        return best_mapping

    def _analyze_chunks(self, column_name: str, chunks: List[Dict]) -> Optional[Dict]:
        """Analyze chunks to determine best mapping"""
        best_score = 0
        best_mapping = None

        for chunk in chunks:
            if chunk['type'] == 'table':
                score, mapping = self._analyze_table_chunk(column_name, chunk)
            else:
                score, mapping = self._analyze_text_chunk(column_name, chunk)

            if score > best_score:
                best_score = score
                best_mapping = mapping

        if best_mapping:
            return {
                'mapping': best_mapping,
                'confidence': best_score
            }
        return None

    def _analyze_table_chunk(self, column_name: str, chunk: Dict) -> Tuple[float, Optional[str]]:
        """Analyze table chunk for mapping"""
        # Implementation for table analysis
        pass

    def _analyze_text_chunk(self, column_name: str, chunk: Dict) -> Tuple[float, Optional[str]]:
        """Analyze text chunk for mapping"""
        # Implementation for text analysis
        pass

    def _is_cache_valid(self, pdf_path: str) -> bool:
        """Check if cached data is valid"""
        if not all(p.exists() for p in [self.index_path, self.chunks_path, self.metadata_path]):
            return False

        pdf_mtime = os.path.getmtime(pdf_path)
        cache_mtime = os.path.getmtime(self.metadata_path)
        return cache_mtime > pdf_mtime
