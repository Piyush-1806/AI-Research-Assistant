import json
import numpy as np
from sentence_transformers import SentenceTransformer
import arxiv
import os

def fetch_papers(max_results=100):
    """Fetch recent papers from arXiv."""
    client = arxiv.Client()
    search = arxiv.Search(
        query="cat:cs.AI OR cat:cs.CL",  # AI and Computational Linguistics categories
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    papers = []
    for result in client.results(search):
        papers.append({
            'title': result.title,
            'abstract': result.summary,
            'url': result.pdf_url,
            'authors': [author.name for author in result.authors],
            'published': result.published.isoformat()
        })
    return papers

def create_embeddings(papers):
    """Create embeddings for paper titles and abstracts."""
    print("Loading SciBERT model...")
    model = SentenceTransformer('allenai/scibert_scivocab_uncased')
    
    print("Creating embeddings...")
    for paper in papers:
        # Combine title and abstract for better semantic matching
        text = f"{paper['title']} {paper['abstract']}"
        embedding = model.encode([text], convert_to_tensor=False, show_progress_bar=False)[0]
        paper['embedding'] = embedding.tolist()  # Convert to list for JSON serialization
    
    return papers

def main():
    output_path = "data/arxiv_embeddings.json"
    
    print("Fetching papers from arXiv...")
    papers = fetch_papers(max_results=100)  # Adjust max_results as needed
    
    print(f"Found {len(papers)} papers")
    papers = create_embeddings(papers)
    
    print(f"Saving embeddings to {output_path}")
    with open(output_path, 'w') as f:
        json.dump({'papers': papers}, f)
    
    print("Done!")

if __name__ == "__main__":
    main() 