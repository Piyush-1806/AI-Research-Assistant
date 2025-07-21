from typing import Any, Dict
from . import inference
import tempfile
import os


def process_command(command: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatches a command to the appropriate function in inference.py.
    Args:
        command: The command string (e.g., 'summarize_pdf', 'ask_question').
        args: Arguments for the command.
    Returns:
        Structured JSON response from the dispatched function.
    """
    try:
        if command == "summarize_pdf":
            context = args.get("context", "")
            summary = inference.summarize_text(context)
            return {"summary": summary}
        elif command == "ask_question":
            query = args.get("query", "")
            # Accept either a PDF path or raw text
            if "pdf_path" in args:
                text = inference.parse_pdf(args["pdf_path"])
            else:
                text = args.get("text", "")
            chunks = inference.chunk_text(text)
            if not chunks:
                return {"query": query, "results": [], "error": "No text found."}
            embeddings = inference.embed_chunks(chunks)
            index = inference.build_faiss_index(embeddings)
            top_chunks = inference.retrieve_top_k(query, index, chunks)
            return {"query": query, "results": [{"chunk": c, "distance": d} for c, d in top_chunks]}
        elif command == "extract_citations":
            text = args.get("text", "")
            citations = inference.extract_references(text)
            return {"citations": citations}
        elif command == "suggest_papers":
            paragraph = args.get("paragraph", "")
            suggestions = inference.suggest_related_papers(paragraph)
            return {"suggestions": suggestions}
        else:
            return {"error": f"Unknown command: {command}"}
    except Exception as e:
        return {"error": str(e)} 