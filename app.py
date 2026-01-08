"""
Image Retrieval System - Streamlit UI
CS336: Multimedia Information Retrieval 2025
"""
import streamlit as st
import os
import sys
from PIL import Image
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

# Page config
st.set_page_config(
    page_title="Image Retrieval System",
    page_icon="🖼️",
    layout="wide"
)

# Title
st.title("🖼️ Image Retrieval System")
st.markdown("**CS336: Multimedia Information Retrieval 2025**")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Settings")

# Track selection
track = st.sidebar.radio(
    "Select Track",
    ["Track A: Image → Image", "Track B: Text → Image"]
)

# Method selection based on track
if track == "Track A: Image → Image":
    method = st.sidebar.selectbox(
        "Select Method",
        ["CLIP (Main)", "ResNet50", "Histogram (HSV)"]
    )
    method_key = {
        "CLIP (Main)": "clip",
        "ResNet50": "resnet",
        "Histogram (HSV)": "histogram"
    }[method]
else:
    method = st.sidebar.selectbox(
        "Select Method",
        ["CLIP (Main)", "SBERT", "BM25"]
    )
    method_key = {
        "CLIP (Main)": "clip",
        "SBERT": "sbert",
        "BM25": "bm25"
    }[method]

# Index type
index_type = st.sidebar.selectbox(
    "Index Type",
    ["Flat (Exact)", "IVF (Approximate)", "HNSW (Fast)"]
)

# Top-K
top_k = st.sidebar.slider("Top-K Results", 5, 50, 20)

# ====================
# MAIN CONTENT
# ====================

# Initialize session state
if 'search_engine' not in st.session_state:
    st.session_state.search_engine = None
if 'text_search_engine' not in st.session_state:
    st.session_state.text_search_engine = None


@st.cache_resource
def load_image_search_engine():
    """Load Image Search Engine (Track A)"""
    from src.search.search import ImageSearchEngine
    
    engine = ImageSearchEngine()
    
    # Load features
    methods_to_load = [
        ("histogram", os.path.join(FEATURES_TRACK_A, "histogram.npy")),
        ("resnet", os.path.join(FEATURES_TRACK_A, "resnet.npy")),
        ("clip", os.path.join(FEATURES_TRACK_A, "clip.npy")),
    ]
    
    for method_name, feat_path in methods_to_load:
        if os.path.exists(feat_path):
            paths_file = feat_path.replace('.npy', '_paths.txt')
            if os.path.exists(paths_file):
                engine.load_features(method_name, feat_path, paths_file)
                engine.build_index(method_name, index_type="Flat")
    
    return engine


@st.cache_resource
def load_text_search_engine():
    """Load Text Search Engine (Track B)"""
    from src.search.search import TextImageSearchEngine
    
    engine = TextImageSearchEngine()
    
    feat_path = os.path.join(FEATURES_TRACK_B, "clip.npy")
    paths_file = os.path.join(FEATURES_TRACK_B, "clip_paths.txt")
    
    if os.path.exists(feat_path) and os.path.exists(paths_file):
        engine.load_features(feat_path, paths_file)
        engine.build_index(index_type="Flat")
    
    return engine


def display_results(results: list, scores: list, latency: float):
    """Display search results in a grid"""
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("🔍 Results Found", len(results))
    col2.metric("⏱️ Latency", f"{latency:.2f} ms")
    col3.metric("📊 Method", method_key)
    
    st.markdown("---")
    st.subheader("🎯 Search Results")
    
    # Display in grid (5 columns)
    cols_per_row = 5
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(results):
                img_path = results[idx]
                score = scores[idx]
                
                with col:
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True)
                    else:
                        st.warning("Image not found")
                    
                    st.caption(f"Score: {score:.4f}")
                    st.caption(f"#{idx+1}")


# ====================
# TRACK A: Image → Image
# ====================
if track == "Track A: Image → Image":
    st.header("🖼️ Image → Image Search")
    st.markdown("Upload an image to find similar images in the database.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Query Image",
        type=['jpg', 'jpeg', 'png', 'webp']
    )
    
    if uploaded_file is not None:
        # Display query image
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Query Image")
            query_img = Image.open(uploaded_file)
            st.image(query_img, use_container_width=True)
        
        with col2:
            # Save temp file
            temp_path = "/tmp/query_image.jpg"
            query_img.save(temp_path)
            
            # Search button
            if st.button("🔍 Search", type="primary"):
                with st.spinner("Searching..."):
                    try:
                        # Load engine
                        engine = load_image_search_engine()
                        
                        if method_key in engine.features:
                            results, scores, latency = engine.search(
                                temp_path, 
                                method=method_key,
                                top_k=top_k
                            )
                            
                            display_results(results, scores, latency)
                        else:
                            st.error(f"Features for {method_key} not loaded. Please extract features first.")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Make sure you have extracted features and built the index.")

# ====================
# TRACK B: Text → Image
# ====================
else:
    st.header("📝 Text → Image Search")
    st.markdown("Enter a text description to find matching images.")
    
    # Text input
    query_text = st.text_input(
        "Enter your query",
        placeholder="e.g., a red dress, a person wearing sunglasses"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        search_by_text = st.button("🔍 Search by Text", type="primary")
    
    with col2:
        # Option to also search by image
        uploaded_file = st.file_uploader(
            "Or upload an image",
            type=['jpg', 'jpeg', 'png', 'webp'],
            key="track_b_upload"
        )
    
    # Search by text
    if search_by_text and query_text:
        with st.spinner("Searching..."):
            try:
                engine = load_text_search_engine()
                
                if engine.image_features is not None:
                    results, scores, latency = engine.search_by_text(
                        query_text,
                        top_k=top_k
                    )
                    
                    st.markdown(f"**Query:** *{query_text}*")
                    display_results(results, scores, latency)
                else:
                    st.error("Features not loaded. Please extract features first.")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Search by image (Track B)
    if uploaded_file is not None:
        query_img = Image.open(uploaded_file)
        st.image(query_img, width=200, caption="Query Image")
        
        if st.button("🔍 Search by Image", key="search_by_img"):
            with st.spinner("Searching..."):
                try:
                    temp_path = "/tmp/query_image_b.jpg"
                    query_img.save(temp_path)
                    
                    engine = load_text_search_engine()
                    
                    if engine.image_features is not None:
                        results, scores, latency = engine.search_by_image(
                            temp_path,
                            top_k=top_k
                        )
                        
                        display_results(results, scores, latency)
                    else:
                        st.error("Features not loaded.")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ====================
# FOOTER
# ====================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Image Retrieval System | CS336 MIR 2025 | 
        Methods: Histogram, ResNet50, CLIP, BM25, SBERT
    </div>
    """,
    unsafe_allow_html=True
)
