import streamlit as st
import requests
from typing import List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

API_URL = "http://localhost:8001"  # Updated port to match FastAPI server

# Configure session with retries
session = requests.Session()
retries = Retry(total=5,
                backoff_factor=0.1,
                status_forcelist=[500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("🧠 AI Research Assistant")

# Check API connection
def check_api():
    try:
        response = session.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Wait for API to be ready
def wait_for_api(timeout=30):
    start_time = time.time()
    with st.spinner("Waiting for API server to be ready..."):
        while time.time() - start_time < timeout:
            if check_api():
                return True
            time.sleep(1)
        return False

if not check_api():
    if not wait_for_api():
        st.error("Could not connect to API server. Please make sure it's running.")
        st.stop()

st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", [
    "Summarize Text",
    "Suggest Related Papers",
    "Ask a Question (PDF)",
    "Extract Citations (PDF)",
    "Batch Ask (PDF)"
])

# --- Summarize Text ---
if section == "Summarize Text":
    st.header("📝 Summarize PDF")
    pdf_file = st.file_uploader("Upload PDF to summarize", type=["pdf"])
    if st.button("Summarize"):
        if not pdf_file:
            st.warning("Please upload a PDF file.")
        else:
            try:
                progress_text = st.empty()
                with st.spinner("Generating summary..."):
                    progress_text.text("Uploading PDF...")
                    files = {"file": (pdf_file.name, pdf_file, "application/pdf")}
                    
                    # Increased timeout to 5 minutes
                    resp = session.post(f"{API_URL}/summarize", files=files, timeout=300)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        progress_text.empty()
                        st.success("Summary:")
                        st.write(data["summary"])
                    else:
                        st.error(f"Error: {resp.text}")
            except requests.exceptions.Timeout:
                st.error("Request timed out. For large PDFs, this might take longer than expected. Please try again or try with a smaller PDF.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the server. Please make sure it's running.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# --- Suggest Related Papers ---
elif section == "Suggest Related Papers":
    st.header("🔍 Find Similar Papers")
    pdf_file = st.file_uploader("Upload PDF to find similar papers", type=["pdf"])
    max_results = st.number_input("Number of suggestions", 1, 10, 3)
    
    if st.button("Find Similar Papers"):
        if not pdf_file:
            st.warning("Please upload a PDF file.")
        else:
            try:
                with st.spinner("Finding similar papers..."):
                    files = {"file": (pdf_file.name, pdf_file, "application/pdf")}
                    resp = session.post(f"{API_URL}/suggest", files=files, timeout=60)
                    
                    if resp.status_code == 200:
                        papers = resp.json()
                        if papers:
                            st.success(f"Found {len(papers)} similar papers:")
                            for paper in papers:
                                with st.expander(paper["title"]):
                                    st.write(f"**Abstract:** {paper['abstract']}")
                                    st.write(f"**URL:** {paper['url']}")
                        else:
                            st.info("No similar papers found.")
                    else:
                        st.error(f"Error: {resp.text}")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The server is taking too long to respond.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the server. Please make sure it's running.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# --- Ask a Question (PDF) ---
elif section == "Ask a Question (PDF)":
    st.header("Ask a Question from a PDF")
    query = st.text_input("Your question")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if st.button("Ask"):
        if not query.strip() or not pdf_file:
            st.warning("Please provide both a question and a PDF file.")
        else:
            try:
                with st.spinner("Processing PDF and answering..."):
                    files = {"file": (pdf_file.name, pdf_file, "application/pdf")}
                    data = {"query": query}
                    resp = session.post(f"{API_URL}/ask", data=data, files=files, timeout=300)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        
                        # Display the answer
                        st.success("Answer:")
                        st.write(data["answer"])
                        
                        # Display relevant chunks
                        if data.get("relevant_chunks"):
                            st.write("### Relevant Excerpts from the Paper")
                            for i, chunk in enumerate(data["relevant_chunks"], 1):
                                with st.expander(f"Excerpt {i}"):
                                    st.write(chunk)
                    else:
                        st.error(f"Error: {resp.text}")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The server is taking too long to respond.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the server. Please make sure it's running.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# --- Extract Citations (PDF) ---
elif section == "Extract Citations (PDF)":
    st.header("Extract Citations from PDF")
    pdf_file = st.file_uploader("Upload PDF for citation extraction", type=["pdf"])
    if st.button("Extract Citations"):
        if not pdf_file:
            st.warning("Please upload a PDF file.")
        else:
            try:
                with st.spinner("Extracting citations..."):
                    files = {"file": (pdf_file.name, pdf_file, "application/pdf")}
                    resp = session.post(f"{API_URL}/citations", files=files, timeout=60)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        
                        # Display References
                        if data.get("references"):
                            st.success(f"Found {len(data['references'])} references:")
                            st.write("### References")
                            for i, ref in enumerate(data["references"], 1):
                                st.write(f"{i}. {ref}")
                        else:
                            st.info("No references found in the document.")
                        
                        # Display In-text Citations
                        if data.get("intext_citations"):
                            st.write("### In-text Citations")
                            st.success(f"Found {len(data['intext_citations'])} in-text citations:")
                            citations_text = ", ".join(data["intext_citations"])
                            st.write(citations_text)
                        else:
                            st.info("No in-text citations found in the document.")
                    else:
                        st.error(f"Error: {resp.text}")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The server is taking too long to respond.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the server. Please make sure it's running.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# --- Batch Ask (PDF) ---
elif section == "Batch Ask (PDF)":
    st.header("Batch Ask: Multiple Questions from a PDF")
    queries = st.text_area("Enter one question per line")
    pdf_file = st.file_uploader("Upload PDF for batch ask", type=["pdf"])
    if st.button("Batch Ask"):
        if not queries.strip() or not pdf_file:
            st.warning("Please provide questions and a PDF file.")
        else:
            with st.spinner("Processing batch questions..."):
                files = {"file": (pdf_file.name, pdf_file, "application/pdf")}
                data = [("queries", q) for q in queries.splitlines() if q.strip()]
                resp = requests.post(f"{API_URL}/batch_ask", data=data, files=files)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("results"):
                        st.success("Results:")
                        for res in data["results"]:
                            st.write(f"**Query:** {res['query']}")
                            st.write(f"**Text length:** {res['text_length']}")
                            st.markdown("---")
                    else:
                        st.info("No results found.")
                else:
                    st.error(f"Error: {resp.text}") 