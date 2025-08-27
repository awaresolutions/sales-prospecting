import streamlit as st
import pdfplumber
import tempfile
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# --- LLM Integration ---
def run_llama(prompt, model_name="llama3"):
    """
    Sends a prompt to a Llama model hosted by Ollama and returns the response.
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        st.error(f"Error: Could not connect to Ollama or model not found. Details: {e}")
        return None

# --- PDF Processing Functions ---
def extract_text_from_pdf(uploaded_file):
    """Extracts text from a PDF file."""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def embed_text_chunks(text, chunk_size=500):
    """Chunks text, creates embeddings, and builds a FAISS index."""
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, chunks, index

def retrieve_context(query, model, chunks, index, k=5): # Increased k for better context
    """Retrieves relevant text chunks based on a query."""
    query_embedding = model.encode([query]).astype('float32')
    D, I = index.search(query_embedding, k)
    context = "\n---\n".join([chunks[i] for i in I[0]])
    return context

# --- Streamlit App ---
st.set_page_config(page_title="📄 PDF Plan Analyzer with Llama", layout="wide")
st.title("Sales Prospecting with a Local LLM - Ollama")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting and embedding..."):
        text = extract_text_from_pdf(uploaded_file)
        model, chunks, index = embed_text_chunks(text)
    st.success("PDF processed successfully.")

    # Define the questions to be displayed in the left column
    questions = [
        "Where are the investment priorities to drive top-line growth?",
        "What operational areas are being targeted for cost efficiency?",
        "How are capital expenditures aligned with strategic priorities?",
        "What role does technology play in their investment strategy?",
        "Are there measurable outcomes tied to these investments?"
    ]

    # Create two columns for the layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Select a Question to Analyze")
        selected_question = st.radio("Choose a specific question:", questions, index=None)

    with col2:
        if selected_question:
            st.subheader("💡 Answer from LLM")
            with st.spinner("Generating answer..."):
                context = retrieve_context(selected_question, model, chunks, index)
                
                # Construct the prompt for the LLM
                prompt = f"""You are a helpful assistant. Use the following context to answer the question.
If the answer is not in the context, just say "I can't find the answer in the provided document." Do not try to make up an answer.

Context:
{context}

Question: {selected_question}
Answer:"""

                response = run_llama(prompt)

            if response:
                st.markdown(response)
                with st.expander("📚 Context Used"):
                    st.write(context)
        else:
            st.info("Please upload a PDF and select a question from the left panel.")
