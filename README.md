# AI Research Assistant 🤖📚

An intelligent, locally-running agent designed to help researchers analyze academic papers efficiently. Built with modern NLP techniques and focused on performance, this tool provides semantic search, summarization, citation extraction, and paper recommendations.

## Features 🌟

- **Semantic Question Answering**: Ask questions about any part of a paper and get contextually relevant answers
- **Smart Summarization**: Generate concise summaries focusing on key contributions, methodology, and findings
- **Citation Analysis**: Extract and analyze references and in-text citations
- **Paper Recommendations**: Get relevant paper suggestions based on content similarity
- **Modular Command Processor**: Flexible command system for batch operations

## Tech Stack 🛠️

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **PDF Processing**: PyMuPDF (with PyPDF2 fallback)
- **ML/NLP**:
  - SciBERT (allenai/scibert_scivocab_uncased) for embeddings
  - FLAN-T5-small for summarization
  - FAISS for efficient similarity search
  - Sentence Transformers for semantic processing
- **Other**: NumPy, regex, LangChain

## Installation 🔧

1. Clone the repository:
```bash
git clone https://github.com/Piyush-1806/ai-research-assistant.git
cd ai-research-assistant
```

2. Create and activate a virtual environment (Python 3.10+ recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r ai_research_assistant/requirements.txt
```

4. Download the arxiv embeddings data:
```bash
python build_arxiv_embeddings.py
```

## Usage 🚀

1. Start the FastAPI backend:
```bash
uvicorn ai_research_assistant.app:app --reload --port 8001
```

2. In a new terminal, start the Streamlit frontend:
```bash
streamlit run streamlit_app.py
```

3. Open your browser and navigate to:
- Frontend UI: http://localhost:8501
- API Documentation: http://localhost:8001/docs

## API Endpoints 📡

- `/ask` - Answer questions about PDF content
- `/summarize` - Generate paper summaries
- `/citations` - Extract and analyze citations
- `/suggest` - Get paper recommendations
- `/health` - Check API status
- `/batch_ask` - Process multiple queries
- `/mcp_server` - Modular Command Processor interface

## Example Usage 📝

### Question Answering
```json
POST /ask
{
    "query": "What are the main contributions of this paper?",
    "context": "... paper content ..."
}
```

### Summarization
```json
POST /summarize
{
    "context": "... paper content ..."
}
```

### Paper Suggestions
```json
POST /suggest
{
    "paragraph": "... research content to find similar papers for ..."
}
```

## Project Structure 📁

```
ai_research_assistant/
├── app.py              # FastAPI application
├── inference.py        # Core ML/NLP functionality
├── config.py          # Configuration settings
├── mcp_server.py      # Modular Command Processor
├── requirements.txt   # Project dependencies
└── weights/          # Model weights and embeddings
```

## Performance Optimizations ⚡

- Model caching for faster inference
- FAISS for efficient similarity search
- Intelligent text chunking
- Parallel processing where applicable
- Memory-efficient PDF processing

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- HuggingFace for providing pre-trained models
- Allen AI for SciBERT
- Facebook Research for FAISS
- The arXiv team for the paper dataset

## Contact 📧

Piyush Patra - kunalpatra18@gmail.com

Project Link: https://github.com/Piyush-1806/ai-research-assistant 
