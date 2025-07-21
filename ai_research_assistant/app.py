
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
from . import inference
from typing import List


app = FastAPI(title="AI Research Assistant")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskResponse(BaseModel):
    answer: str
    relevant_chunks: List[str]

class SuggestResponse(BaseModel):
    title: str
    url: str
    abstract: str = ""

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
async def ask(query: str = Form(...), file: UploadFile = File(...)):
    """Answer a question from a PDF using LLM."""
    try:
        print(f"Processing query: {query}")
        print(f"Processing file: {file.filename}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            print(f"File size: {len(content)} bytes")
            tmp.write(content)
            tmp_path = tmp.name
            
        print(f"Temporary file created at: {tmp_path}")
        
        # Extract text from PDF
        text = inference.parse_pdf(tmp_path)
        print(f"Extracted text length: {len(text)} characters")
        
        # Process query and get answer
        result = inference.process_query(query, text)
        print(f"Generated answer length: {len(result['answer'])} characters")
        
        os.remove(tmp_path)
        print(f"Temporary file removed: {tmp_path}")
        
        return result
    except Exception as e:
        print(f"Error in ask endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/citations")
async def citations(file: UploadFile = File(...)):
    """Extract citations from a PDF."""
    try:
        print(f"Processing file: {file.filename}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            print(f"File size: {len(content)} bytes")
            tmp.write(content)
            tmp_path = tmp.name
            
        print(f"Temporary file created at: {tmp_path}")
        
        text = inference.parse_pdf(tmp_path)
        print(f"Extracted text length: {len(text)} characters")
        print("First 500 characters of extracted text:")
        print(text[:500])
        
        citations = inference.extract_references(text)
        print("Extracted citations:")
        print(f"Number of references: {len(citations['references'])}")
        print(f"Number of in-text citations: {len(citations['intext_citations'])}")
        
        os.remove(tmp_path)
        print(f"Temporary file removed: {tmp_path}")
        
        return citations
    except Exception as e:
        print(f"Error in citations endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 

@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    """Generate a summary of the PDF content."""
    try:
        print(f"Processing file: {file.filename}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            print(f"File size: {len(content)} bytes")
            tmp.write(content)
            tmp_path = tmp.name
            
        print(f"Temporary file created at: {tmp_path}")
        
        # Extract text from PDF
        text = inference.parse_pdf(tmp_path)
        print(f"Extracted text length: {len(text)} characters")
        
        # Generate summary
        summary = inference.summarize_text(text)
        print(f"Generated summary length: {len(summary)} characters")
        
        os.remove(tmp_path)
        print(f"Temporary file removed: {tmp_path}")
        
        return {"summary": summary}
    except Exception as e:
        print(f"Error in summarize endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggest", response_model=List[SuggestResponse])
async def suggest(file: UploadFile = File(...)):
    """Suggest similar papers based on the PDF content."""
    try:
        print(f"Processing file: {file.filename}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            print(f"File size: {len(content)} bytes")
            tmp.write(content)
            tmp_path = tmp.name
            
        print(f"Temporary file created at: {tmp_path}")
        
        # Extract text from PDF
        text = inference.parse_pdf(tmp_path)
        print(f"Extracted text length: {len(text)} characters")
        
        # Get suggestions
        suggestions = inference.suggest_papers(text)
        print(f"Found {len(suggestions)} similar papers")
        
        os.remove(tmp_path)
        print(f"Temporary file removed: {tmp_path}")
        
        return suggestions
    except Exception as e:
        print(f"Error in suggest endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 