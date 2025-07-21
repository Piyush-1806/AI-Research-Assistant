import re
from typing import Dict, List, Union
import fitz  # PyMuPDF
import PyPDF2
from ctransformers import AutoModelForCausalLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import json

# Set environment variable to avoid tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize models (will be loaded on first use)
_scibert_model = None
_llm_model = None
_faiss_index = None
_embeddings = None

def get_scibert():
    """Get or initialize the SciBERT model."""
    global _scibert_model
    if _scibert_model is None:
        print("Loading SciBERT model (first time only)...")
        _scibert_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
        print("SciBERT model loaded successfully!")
    return _scibert_model

def get_llm():
    """Get or initialize the LLM."""
    global _llm_model
    if _llm_model is None:
        print("Loading LLM model (first time only)...")
        _llm_model = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            model_type="mistral",
            context_length=4096,
            gpu_layers=0  # CPU only
        )
        print("LLM model loaded successfully!")
    return _llm_model

def parse_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF with PyPDF2 as fallback."""
    try:
        # Try PyMuPDF first (better quality)
        text = ""
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    except Exception as e:
        print(f"PyMuPDF failed: {str(e)}, trying PyPDF2...")
        try:
            # Fallback to PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e2:
            print(f"Both PDF readers failed. PyPDF2 error: {str(e2)}")
            return ""

def chunk_text(text: str) -> List[Document]:
    """Split text into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.create_documents([text])
    return chunks

def get_relevant_chunks(query: str, chunks: List[Document], k: int = 3) -> List[Document]:
    """Get the most relevant chunks using FAISS and SciBERT embeddings."""
    global _faiss_index, _embeddings
    
    try:
        # Get SciBERT model
        model = get_scibert()
        
        # Create embeddings if not exists
        if _embeddings is None:
            texts = [chunk.page_content for chunk in chunks]
            _embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
            
            # Create FAISS index
            dimension = _embeddings.shape[1]
            _faiss_index = faiss.IndexFlatL2(dimension)
            _faiss_index.add(_embeddings.astype(np.float32))
        
        # Encode query
        query_embedding = model.encode([query], convert_to_tensor=False, show_progress_bar=False)
        
        # Find similar chunks
        distances, indices = _faiss_index.search(query_embedding.astype(np.float32), k)
        
        # Return relevant chunks
        return [chunks[idx] for idx in indices[0]]
    except Exception as e:
        print(f"Error in get_relevant_chunks: {str(e)}")
        # Fallback to simple keyword matching if embedding fails
        query_words = set(query.lower().split())
        chunk_scores = []
        
        for chunk in chunks:
            chunk_words = set(chunk.page_content.lower().split())
            score = len(query_words.intersection(chunk_words))
            chunk_scores.append((score, chunk))
        
        chunk_scores.sort(reverse=True, key=lambda x: x[0])
        return [chunk for score, chunk in chunk_scores[:k]]

def answer_question(query: str, context_chunks: List[Document]) -> str:
    """Generate an answer using the LLM based on relevant chunks."""
    try:
        # Combine chunks into context
        context = "\n".join([chunk.page_content for chunk in context_chunks])
        
        # Create the prompt
        prompt = f"""You are a helpful AI research assistant. Based on the following context from an academic paper, answer the question accurately and concisely. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {query}

Answer: """

        # Get LLM response
        llm = get_llm()
        response = llm(prompt, max_new_tokens=512, temperature=0.1)
        
        return str(response).strip()
    except Exception as e:
        print(f"Error in answer_question: {str(e)}")
        return "Sorry, I encountered an error while generating the answer."

def process_query(query: str, text: str) -> Dict[str, Union[str, List[str]]]:
    """Process a query about the PDF content."""
    try:
        # Reset embeddings for new document
        global _embeddings, _faiss_index
        _embeddings = None
        _faiss_index = None
        
        # Split text into chunks
        chunks = chunk_text(text)
        
        # Get relevant chunks using semantic search
        relevant_chunks = get_relevant_chunks(query, chunks)
        
        # Generate answer
        answer = answer_question(query, relevant_chunks)
        
        # Return both answer and relevant chunks for transparency
        return {
            "answer": answer,
            "relevant_chunks": [chunk.page_content for chunk in relevant_chunks]
        }
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return {
            "answer": "Sorry, I encountered an error while processing your question.",
            "relevant_chunks": []
        }

def extract_references(text: str) -> Dict[str, List[str]]:
    """
    Extract references and in-text citations from text.
    Returns a dictionary with 'references' and 'intext_citations'.
    """
    try:
        # Initialize results
        results = {
            "references": [],
            "intext_citations": []
        }
        
        if not text:
            print("Warning: Empty text provided to extract_references")
            return results
            
        # Find the references section using various common headers
        ref_patterns = [
            r'(?i)references?\s*$',
            r'(?i)bibliography\s*$',
            r'(?i)works cited\s*$',
            r'(?i)citations?\s*$'
        ]
        
        # Find the start of references section
        ref_start = None
        for pattern in ref_patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE))
            if matches:
                ref_start = matches[-1]  # Take the last match as it's likely the actual references
                break
        
        if ref_start:
            # Get text after references header
            refs_text = text[ref_start.end():]
            
            # Split references by common patterns
            # This handles numbered refs, bullet points, and line breaks
            ref_splits = re.split(r'\n\s*(?:\d+\.|\[\d+\]|•|\*|\-)\s*', refs_text)
            
            # Clean and filter references
            references = []
            for ref in ref_splits:
                ref = ref.strip()
                # Only keep refs that look legitimate (more than 10 chars, contains year)
                if len(ref) > 10 and re.search(r'\d{4}', ref):
                    references.append(ref)
            
            results["references"] = references
        
        # Extract in-text citations
        # Pattern matches [1], [1,2], (Smith et al., 2020), (Smith and Jones, 2020)
        intext_patterns = [
            r'\[\d+(?:,\s*\d+)*\]',  # [1] or [1,2]
            r'\([A-Z][a-z]+(?:\s+(?:et al\.|and)\s+[A-Z][a-z]+)?,\s*\d{4}\)'  # (Smith et al., 2020)
        ]
        
        intext_citations = []
        for pattern in intext_patterns:
            citations = re.findall(pattern, text)
            intext_citations.extend(citations)
        
        results["intext_citations"] = list(set(intext_citations))  # Remove duplicates
        
        # Debug print
        print(f"Found {len(results['references'])} references and {len(results['intext_citations'])} in-text citations")
        
        return results
        
    except Exception as e:
        print(f"Error extracting references: {str(e)}")
        return {"references": [], "intext_citations": []} 

def summarize_text(text: str, max_length: int = 1000) -> str:
    """Generate a concise summary of the text using the LLM."""
    try:
        # Get LLM once at the start
        llm = get_llm()
        
        # Extract the most important parts of the text
        # Look for common section headers in academic papers
        important_sections = []
        section_markers = [
            "abstract", "introduction", "conclusion", "results", "discussion",
            "methodology", "method", "findings", "contribution"
        ]
        
        # Split text into lines and find important sections
        lines = text.lower().split("\n")
        current_section = ""
        section_text = []
        
        for line in lines:
            # Check if this line is a section header
            if any(marker in line for marker in section_markers):
                # Save previous section if it exists
                if current_section and section_text:
                    important_sections.append("\n".join(section_text))
                current_section = line
                section_text = []
            elif current_section:  # If we're in an important section
                section_text.append(line)
                
        # Add the last section if it exists
        if current_section and section_text:
            important_sections.append("\n".join(section_text))
        
        # If no sections found, use the first and last parts of the text
        if not important_sections:
            text_length = len(text)
            start_text = text[:min(2000, text_length // 3)]
            end_text = text[max(0, text_length - 2000):]
            important_sections = [start_text, end_text]
        
        # Combine important sections (limited to ~4000 chars total)
        combined_text = ""
        total_length = 0
        for section in important_sections:
            section_length = len(section)
            if total_length + section_length <= 4000:
                combined_text += section + "\n\n"
                total_length += section_length
            else:
                break
        
        # Create a focused prompt for better summarization
        prompt = f"""You are a helpful AI research assistant. Please provide a concise summary of this academic text. Focus on these key aspects:
1. Main research problem or objective
2. Key methodology or approach
3. Most important findings or contributions
4. Significant conclusions

Text:
{combined_text}

Summary (be concise): """

        # Generate summary with reduced tokens
        summary = llm(prompt, max_new_tokens=max_length, temperature=0.1)
        return str(summary).strip()
            
    except Exception as e:
        print(f"Error in summarize_text: {str(e)}")
        return "Sorry, I encountered an error while generating the summary."

def suggest_papers(text: str, k: int = 3) -> List[Dict[str, str]]:
    """Find similar papers based on semantic similarity."""
    try:
        # Get SciBERT model
        model = get_scibert()
        
        # Create text embedding
        text_embedding = model.encode([text], convert_to_tensor=False, show_progress_bar=False)[0]
        
        # Load arxiv embeddings
        arxiv_data_path = "data/arxiv_embeddings.json"
        if not os.path.exists(arxiv_data_path):
            return [{"title": "Error: arxiv_embeddings.json not found. Please run the build_arxiv_embeddings.py script first.", "url": ""}]
            
        with open(arxiv_data_path, 'r') as f:
            arxiv_data = json.load(f)
            
        # Create FAISS index for arxiv papers
        embeddings = np.array([paper['embedding'] for paper in arxiv_data['papers']], dtype=np.float32)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Find similar papers
        distances, indices = index.search(text_embedding.reshape(1, -1).astype(np.float32), k)
        
        # Return paper info
        suggestions = []
        for idx in indices[0]:
            paper = arxiv_data['papers'][idx]
            suggestions.append({
                "title": paper['title'],
                "url": paper['url'],
                "abstract": paper.get('abstract', '')
            })
            
        return suggestions
    except Exception as e:
        print(f"Error in suggest_papers: {str(e)}")
        return [{"title": f"Error suggesting papers: {str(e)}", "url": ""}] 