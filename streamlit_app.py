"""
Streamlit UI for RAG system.
Simple interface for asking questions and viewing answers with citations.
"""
import streamlit as st
import sys
from pathlib import Path
import tempfile
import shutil
import yaml

sys.path.append(str(Path(__file__).parent))

from rag.chain import RAGChain
from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.embed_and_upsert import EmbeddingUpserter
from pinecone import Pinecone, ServerlessSpec
from voice.stt import LemonFoxSTT
from voice.tts import LemonFoxTTS
from audio_recorder_streamlit import audio_recorder
import base64
import os
from dotenv import load_dotenv
import time

load_dotenv()

def get_env(key, default=None):
    """Get environment variable from st.secrets or os.environ."""
    try:
        if hasattr(st, 'secrets') and key in st.secrets.get("secrets", {}):
            value = st.secrets["secrets"][key]
            os.environ[key] = value
            return value
    except:
        pass
    return os.getenv(key, default)

def init_secrets():
    """Initialize environment variables from Streamlit secrets."""
    try:
        if hasattr(st, 'secrets') and 'secrets' in st.secrets:
            for key, value in st.secrets["secrets"].items():
                os.environ[key] = str(value)
    except:
        pass

st.set_page_config(
    page_title="RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f2937;
        letter-spacing: -0.5px;
    }

    /* Citation box styling */
    .citation-box {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
    }

    /* Confidence score styling */
    .confidence-score {
        font-size: 1.2rem;
        font-weight: bold;
    }

    /* Button styling */
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        background-color: #f1f5f9;
        border-radius: 8px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2563eb;
        color: white;
    }

    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
    }

    .stTextInput > div > div > input:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #1f2937;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
    }

    /* Info box styling */
    .stAlert {
        border-radius: 8px;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8fafc;
        border-radius: 8px;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

def get_pinecone_client():
    """Get Pinecone client."""
    return Pinecone(api_key=get_env('PINECONE_API_KEY'))

def list_indexes():
    """List all available Pinecone indexes."""
    try:
        pc = get_pinecone_client()
        indexes = pc.list_indexes()
        return [index['name'] for index in indexes]
    except Exception as e:
        st.error(f"Error listing indexes: {e}")
        return []

def create_index(index_name, dimension=768):
    """Create a new Pinecone index."""
    try:
        pc = get_pinecone_client()

        if pc.has_index(index_name):
            return {"success": False, "error": f"Index '{index_name}' already exists"}

        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        return {"success": True, "message": f"Index '{index_name}' created successfully"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def delete_index(index_name):
    """Delete a Pinecone index."""
    try:
        pc = get_pinecone_client()
        pc.delete_index(index_name)
        return {"success": True, "message": f"Index '{index_name}' deleted successfully"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@st.cache_resource
def load_rag_chain(_index_name=None):
    """Load and cache the RAG chain with specific index."""
    try:
        if _index_name:
            os.environ['PINECONE_INDEX_NAME'] = _index_name
        return RAGChain.from_config()
    except Exception as e:
        st.error(f"Error initializing RAG chain: {e}")
        return None

@st.cache_resource
def load_stt_client():
    """Load and cache LemonFox STT client."""
    try:
        stt = LemonFoxSTT()
        return stt
    except Exception as e:
        st.error(f"Error initializing STT client: {e}")
        return None

@st.cache_resource
def load_tts_client():
    """Load and cache LemonFox TTS client."""
    try:
        tts = LemonFoxTTS()
        return tts
    except Exception as e:
        st.error(f"Error initializing TTS client: {e}")
        return None

def format_confidence_color(score: float) -> str:
    """Get color based on confidence score."""
    if score >= 0.8:
        return "green"
    elif score >= 0.6:
        return "orange"
    else:
        return "red"

def run_ingestion(uploaded_files, chunk_size, chunk_overlap, index_name):
    """Run the document ingestion pipeline."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for uploaded_file in uploaded_files:
                file_path = temp_path / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            loader = DocumentLoader()
            documents = loader.load_directory(str(temp_path))

            if not documents:
                return {"success": False, "error": "No documents were successfully loaded"}

            chunker = DocumentChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = chunker.chunk_documents(documents)

            os.environ['PINECONE_INDEX_NAME'] = index_name
            upserter = EmbeddingUpserter.from_config()
            result = upserter.upsert_documents(chunks, show_progress=False)

            index_stats = upserter.get_index_stats()
            chunk_stats = chunker.get_chunk_stats(chunks)

            return {
                "success": True,
                "documents_loaded": len(documents),
                "chunks_created": len(chunks),
                "chunk_stats": chunk_stats,
                "index_stats": index_stats,
                "upsert_result": result
            }

    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    """Main Streamlit application."""

    init_secrets()

    if 'selected_index' not in st.session_state:
        st.session_state.selected_index = get_env('PINECONE_INDEX_NAME', 'rag-knowledge-base')

    if 'processing_audio' not in st.session_state:
        st.session_state.processing_audio = False

    st.markdown('<div class="main-header">RAG Agent</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Ask Questions", "Upload Documents"])

    with tab2:
        st.markdown("### Document Upload")
        st.markdown("Upload documents to add them to the knowledge base")

        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=["pdf", "txt", "md", "html"],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, Markdown, HTML"
        )

        st.markdown("### Chunking Settings")
        col1, col2 = st.columns(2)

        with col1:
            chunk_size = st.number_input(
                "Chunk Size (characters)",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                help="Size of each text chunk in characters"
            )

        with col2:
            chunk_overlap = st.number_input(
                "Chunk Overlap (characters)",
                min_value=0,
                max_value=1000,
                value=200,
                step=50,
                help="Overlap between consecutive chunks"
            )

        st.info(f"Documents will be added to: **{st.session_state.selected_index}**")

        if st.button("Ingest Documents", type="primary", disabled=not uploaded_files):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    result = run_ingestion(
                        uploaded_files,
                        chunk_size,
                        chunk_overlap,
                        st.session_state.selected_index
                    )

                    if result["success"]:
                        st.success("Documents ingested successfully!")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Documents Loaded", result["documents_loaded"])
                        with col2:
                            st.metric("Chunks Created", result["chunks_created"])
                        with col3:
                            st.metric("Total Vectors", result["index_stats"]["total_vectors"])

                        with st.expander("Detailed Statistics"):
                            st.json(result["chunk_stats"])

                        st.cache_resource.clear()
                        st.info("Knowledge base updated! Switch to 'Ask Questions' tab to query.")
                    else:
                        st.error(f"Ingestion failed: {result['error']}")
            else:
                st.warning("Please upload at least one file")

    with st.sidebar:
        st.header("Index Management")

        available_indexes = list_indexes()

        if available_indexes:
            selected_index = st.selectbox(
                "Select Index",
                options=available_indexes,
                index=available_indexes.index(st.session_state.selected_index)
                      if st.session_state.selected_index in available_indexes else 0,
                help="Choose which Pinecone index to use (Free tier: max 5 indexes)"
            )

            if selected_index != st.session_state.selected_index:
                st.session_state.selected_index = selected_index
                os.environ['PINECONE_INDEX_NAME'] = selected_index
                st.cache_resource.clear()
        else:
            st.warning("No indexes found")

        with st.expander("Create New Index"):
            new_index_name = st.text_input(
                "Index Name",
                placeholder="e.g., my-knowledge-base",
                help="Lowercase, alphanumeric and hyphens only"
            )

            index_dimension = st.selectbox(
                "Embedding Dimension",
                options=[768, 1024, 1536, 3072],
                index=0,
                help="Must match your embedding model dimension"
            )

            if st.button("Create Index", type="primary"):
                if new_index_name:
                    with st.spinner("Creating index..."):
                        result = create_index(new_index_name, index_dimension)
                        if result["success"]:
                            st.success(result["message"])
                            st.rerun()
                        else:
                            st.error(result["error"])
                else:
                    st.warning("Please enter an index name")

        with st.expander("Delete Index"):
            st.warning("This action cannot be undone!")
            index_to_delete = st.selectbox(
                "Select index to delete",
                options=available_indexes,
                key="delete_select"
            )

            confirm_delete = st.checkbox("I understand this will delete all data")

            if st.button("Delete Index", type="secondary", disabled=not confirm_delete):
                with st.spinner("Deleting index..."):
                    result = delete_index(index_to_delete)
                    if result["success"]:
                        st.success(result["message"])
                        if index_to_delete == st.session_state.selected_index:
                            remaining = [idx for idx in available_indexes if idx != index_to_delete]
                            st.session_state.selected_index = remaining[0] if remaining else None
                        st.rerun()
                    else:
                        st.error(result["error"])

        st.markdown("---")
        st.header("Query Settings")

        top_k = st.slider(
            "Number of documents to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="More documents provide more context but may slow down response"
        )

        include_sources = st.checkbox(
            "Show full source documents",
            value=False,
            help="Display complete content of retrieved documents"
        )

        st.markdown("---")
        st.markdown("### Current Settings")

        try:
            with open("config.yaml", 'r') as f:
                config = yaml.safe_load(f)

            st.markdown("**Document Processing:**")
            st.text(f"• Chunk Size: {config['document_processing']['chunk_size']}")
            st.text(f"• Chunk Overlap: {config['document_processing']['chunk_overlap']}")

            st.markdown("**Retrieval:**")
            st.text(f"• Top-K: {config['retrieval']['top_k']}")
            st.text(f"• Score Threshold: {config['retrieval']['score_threshold']}")

            st.markdown("**Generation:**")
            st.text(f"• Model: {config['generation']['model']}")
            st.text(f"• Temperature: {config['generation']['temperature']}")
            st.text(f"• Max Tokens: {config['generation']['max_tokens']}")

            st.markdown("**Embeddings:**")
            st.text(f"• Model: {config['embeddings']['model']}")
            st.text(f"• Dimension: {config['embeddings']['dimension']}")
        except Exception as e:
            st.warning(f"Cannot load config: {str(e)[:30]}")

        st.markdown("---")
        st.markdown("### System Status")

        st.info(f"Current Index: **{st.session_state.selected_index}**")

        rag_chain = load_rag_chain(st.session_state.selected_index)
        if rag_chain:
            st.success("RAG Chain: Ready")
            try:
                stats = rag_chain.retriever.get_index_stats()
                st.info(f"Vectors: {stats.get('total_vectors', 0)} | Dimension: {stats.get('dimension', 0)}")
            except Exception as e:
                st.warning(f"Cannot fetch index stats: {str(e)[:50]}")
        else:
            st.error("RAG Chain: Not initialized")

    with tab1:
        st.markdown("### Ask Questions")

        rag_chain = load_rag_chain(st.session_state.selected_index)

        if rag_chain is None:
            st.error("RAG system is not initialized. Please check your configuration.")
            st.info("Try creating a new index in the sidebar or check your API keys.")
            return

        input_tab1, input_tab2 = st.tabs(["Text Input", "Voice Input"])

        question = ""

        with input_tab1:
            question = st.text_input(
                "Enter your question:",
                placeholder="e.g., What is artificial intelligence?",
                key="question_input"
            )

            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                ask_button = st.button("Ask", type="primary", use_container_width=True)

            if ask_button and question:
                with st.spinner("Searching knowledge base and generating answer..."):
                    try:
                        response = rag_chain.ask(
                            question=question,
                            top_k=top_k,
                            include_sources=include_sources
                        )

                        st.markdown("---")
                        st.markdown("### Answer")
                        st.markdown(response['answer'])

                        confidence = response['confidence_score']
                        color = format_confidence_color(confidence)

                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Confidence Score", f"{confidence:.2f}", delta=None)

                        with col2:
                            st.metric("Documents Retrieved", response['retrieved_docs_count'])

                        with col3:
                            avg_score = response.get('avg_retrieval_score', 0)
                            st.metric("Avg Retrieval Score", f"{avg_score:.3f}")

                        if response['citations']:
                            st.markdown("---")
                            st.markdown("### Citations")

                            for citation in response['citations']:
                                with st.expander(
                                    f"[{citation['id']}] {citation['source']} (Chunk {citation['chunk_id']})",
                                    expanded=False
                                ):
                                    st.markdown(f"**Snippet:**")
                                    st.markdown(citation['snippet'])

                        if include_sources and response.get('sources'):
                            st.markdown("---")
                            st.markdown("### Source Documents")

                            for i, source in enumerate(response['sources'], 1):
                                with st.expander(
                                    f"Source {i}: {source['filename']} (Score: {source['score']:.3f})",
                                    expanded=False
                                ):
                                    st.markdown(f"**Chunk ID:** {source['chunk_id']}")
                                    st.markdown(f"**Content:**")
                                    st.text_area(
                                        "Content",
                                        value=source['content'],
                                        height=200,
                                        key=f"source_{i}",
                                        label_visibility="collapsed"
                                    )

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Make sure the knowledge base is populated. Upload documents first!")

            elif ask_button and not question:
                st.warning("Please enter a question")

        with input_tab2:
            st.info("Click the microphone button below and speak your question")

            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#2563eb",
                neutral_color="#6b7280",
                icon_name="microphone",
                icon_size="2x"
            )

            if audio_bytes:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                    temp_audio.write(audio_bytes)
                    temp_audio_path = temp_audio.name

                with st.spinner("Transcribing audio..."):
                    try:
                        stt = load_stt_client()

                        if stt:
                            result = stt.transcribe(temp_audio_path)

                            if result["status"] == "success":
                                question = result["text"]
                                st.success(f"Transcribed: {question}")

                                with st.spinner("Generating answer..."):
                                    response = rag_chain.ask(
                                        question=question,
                                        top_k=min(top_k, 3),
                                        include_sources=include_sources
                                    )

                                    st.markdown("---")
                                    st.markdown("### Answer")
                                    st.markdown(response['answer'])

                                    with st.spinner("Generating voice response..."):
                                        tts = load_tts_client()
                                        if tts:
                                            tts_result = tts.synthesize(
                                                text=response['answer'],
                                                voice="sarah",
                                                response_format="opus",
                                                stream=True
                                            )

                                            if tts_result["status"] == "success":
                                                st.audio(tts_result['audio_content'], format='audio/ogg', autoplay=True)
                                            else:
                                                st.warning(f"Could not generate voice: {tts_result.get('error', 'Unknown error')}")
                                        else:
                                            st.warning("TTS client not initialized")

                                    confidence = response['confidence_score']
                                    color = format_confidence_color(confidence)

                                    st.markdown("---")
                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        st.metric("Confidence Score", f"{confidence:.2f}", delta=None)

                                    with col2:
                                        st.metric("Documents Retrieved", response['retrieved_docs_count'])

                                    with col3:
                                        avg_score = response.get('avg_retrieval_score', 0)
                                        st.metric("Avg Retrieval Score", f"{avg_score:.3f}")

                                    if response['citations']:
                                        st.markdown("---")
                                        st.markdown("### Citations")

                                        for citation in response['citations']:
                                            with st.expander(
                                                f"[{citation['id']}] {citation['source']} (Chunk {citation['chunk_id']})",
                                                expanded=False
                                            ):
                                                st.markdown(f"**Snippet:**")
                                                st.markdown(citation['snippet'])

                                    if include_sources and response.get('sources'):
                                        st.markdown("---")
                                        st.markdown("### Source Documents")

                                        for i, source in enumerate(response['sources'], 1):
                                            with st.expander(
                                                f"Source {i}: {source['filename']} (Score: {source['score']:.3f})",
                                                expanded=False
                                            ):
                                                st.markdown(f"**Chunk ID:** {source['chunk_id']}")
                                                st.markdown(f"**Content:**")
                                                st.text_area(
                                                    "Content",
                                                    value=source['content'],
                                                    height=200,
                                                    key=f"source_voice_{i}",
                                                    label_visibility="collapsed"
                                                )
                            else:
                                st.error(f"Transcription failed: {result.get('error', 'Unknown error')}")
                        else:
                            st.error("Failed to initialize STT client. Check LEMONFOX_API_KEY in .env")
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        try:
                            os.unlink(temp_audio_path)
                        except:
                            pass

if __name__ == "__main__":
    main()
