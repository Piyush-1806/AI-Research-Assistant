import os 
# Model and embedding configuration
SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"
FLAN_T5_MODEL = "google/flan-t5-small"
ARXIV_EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "weights/arxiv_embeddings.json")


# Chunking and retrieval
CHUNK_SIZE = 300  # characters per chunk
CHUNK_OVERLAP = 50
TOP_K = 5  # Number of chunks to retrieve for QA

# Paper suggestion config
DEFAULT_MIN_SIMILARITY = 0.5
DEFAULT_MAX_RESULTS = 10
SUPPORTED_CATEGORIES = ["cs.AI", "cs.LG", "stat.ML"]  # Add more as needed
SUPPORTED_SORT_OPTIONS = ["similarity", "date", "citations"]

# FAISS index config
FAISS_INDEX_FACTORY = "Flat" 