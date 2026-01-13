"""
Image Retrieval System - Polished UI v2
CS336: Multimedia Information Retrieval 2025
Fixed: Contrast, Header size, Upload area, Sidebar organization
"""
import streamlit as st
import os
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

# ====================
# PAGE CONFIG
# ====================
st.set_page_config(
    page_title="Image Retrieval System",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================
# CUSTOM CSS - Fixed contrast & layout
# ====================
st.markdown("""
<style>
    /* Better contrast for dark theme */
    .stMarkdown p, .stMarkdown li {
        color: #E0E0E0 !important;
    }
    
    /* Compact header */
    .compact-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .compact-header h1 {
        margin: 0;
        font-size: 1.5rem;
        color: white;
    }
    
    .compact-header p {
        margin: 0;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.85);
    }
    
    /* Upload area - more prominent */
    .upload-container {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.1);
        transition: all 0.3s;
        margin: 1rem 0;
    }
    
    .upload-container:hover {
        border-color: #764ba2;
        background: rgba(118, 75, 162, 0.15);
    }
    
    .upload-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .upload-text {
        color: #B0B0B0;
        font-size: 1rem;
    }
    
    .upload-hint {
        color: #888;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    
    /* Metric cards - better visibility */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
        border: 1px solid #444;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #B0B0B0;
        margin-top: 0.3rem;
    }
    
    /* Result card with hover */
    .result-item {
        background: #2d2d2d;
        border-radius: 10px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid #444;
    }
    
    .result-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        border-color: #667eea;
    }
    
    .result-info {
        text-align: center;
        padding: 0.5rem;
        background: #1a1a1a;
        border-radius: 5px;
        margin-top: 0.5rem;
    }
    
    .result-rank {
        font-weight: bold;
        color: #667eea;
        font-size: 1rem;
    }
    
    .result-score {
        font-size: 0.8rem;
        color: #888;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem;
        color: #666;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    .empty-state-text {
        font-size: 1.1rem;
        color: #888;
    }
    
    .empty-state-hint {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Sidebar improvements */
    .sidebar-section {
        background: #2d2d2d;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #444;
    }
    
    .sidebar-title {
        font-size: 0.9rem;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Track badge */
    .track-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
        margin: 0.5rem 0;
    }
    
    .track-a {
        background: rgba(102, 126, 234, 0.2);
        color: #667eea;
        border: 1px solid #667eea;
    }
    
    .track-b {
        background: rgba(118, 75, 162, 0.2);
        color: #a855f7;
        border: 1px solid #a855f7;
    }
    
    /* Better slider display */
    .slider-value {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    
    /* Footer - better visibility */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid #444;
        color: #888;
    }
    
    .footer p {
        margin: 0.3rem 0;
        color: #888 !important;
    }
    
    .footer strong {
        color: #B0B0B0;
    }
    
    /* Tips box */
    .tips-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .tips-title {
        color: #667eea;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Query input styling */
    .query-input {
        background: #2d2d2d;
        border: 2px solid #444;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Loading animation */
    .loading-text {
        color: #667eea;
        font-size: 1.1rem;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)

# ====================
# CACHE FUNCTIONS
# ====================
@st.cache_resource
def load_track_a_engine():
    from src.search.search import ImageSearchEngine
    engine = ImageSearchEngine()
    
    for method in ["histogram", "resnet", "clip"]:
        feat_path = os.path.join(FEATURES_TRACK_A, f"{method}.npy")
        paths_file = os.path.join(FEATURES_TRACK_A, f"{method}_paths.txt")
        if os.path.exists(feat_path) and os.path.exists(paths_file):
            engine.load_features(method, feat_path, paths_file)
            engine.build_index(method, index_type="Flat")
    
    return engine

@st.cache_resource
def load_track_b_meta(features_dir: str, embed: str):
    import json
    meta_path = Path(features_dir) / f"{embed}_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_resource
def load_track_b_embeddings(features_dir: str, embed: str):
    npy_path = Path(features_dir) / f"{embed}.npy"
    if npy_path.exists():
        return np.load(str(npy_path), mmap_mode="r")
    return None

@st.cache_resource
def load_track_b_index(index_path: str):
    import faiss
    if Path(index_path).exists():
        return faiss.read_index(index_path)
    return None

@st.cache_resource
def load_track_b_img2path(data_dir: str):
    img_dir = Path(data_dir) / "images"
    if img_dir.exists():
        return {p.name: str(p) for p in img_dir.glob("*.jpg")}
    return {}

# ====================
# SEARCH FUNCTIONS
# ====================
def encode_query_track_b(query: str, embed: str, meta: dict, device: str = "cpu"):
    if embed == "sbert":
        from sentence_transformers import SentenceTransformer
        model_name = meta.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name, device=device)
        v = model.encode([query], normalize_embeddings=True)[0]
        return np.asarray(v, dtype=np.float32)
    elif embed == "clip_text":
        import torch
        import open_clip
        model_name = meta.get("model", "ViT-B-32")
        pretrained = meta.get("pretrained", "openai")
        dev = torch.device(device)
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model.eval().to(dev)
        tokenizer = open_clip.get_tokenizer(model_name)
        with torch.no_grad():
            tokens = tokenizer([query]).to(dev)
            feat = model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.detach().cpu().numpy()[0].astype(np.float32)
    return None

def search_track_b(query, embed, index_type, top_k, features_dir, index_dir, data_dir, nprobe=16):
    import faiss
    
    meta = load_track_b_meta(features_dir, embed)
    if meta is None:
        return [], [], 0, "Meta file not found"
    
    doc2img = meta.get("doc2img", [])
    img2path = load_track_b_img2path(data_dir)
    
    t0 = time.perf_counter()
    q = encode_query_track_b(query, embed, meta)
    t_encode = time.perf_counter() - t0
    
    if q is None:
        return [], [], 0, "Failed to encode query"
    
    t1 = time.perf_counter()
    
    if index_type == "brute":
        X = load_track_b_embeddings(features_dir, embed)
        if X is None:
            return [], [], 0, "Embeddings not found"
        scores = (X @ q).astype(np.float32)
        topk_cap = min(200, len(scores))
        cap_idx = np.argsort(scores)[::-1][:topk_cap]
        cap_sc = scores[cap_idx]
    else:
        index_path = str(Path(index_dir) / f"{embed}_{index_type}.faiss")
        index = load_track_b_index(index_path)
        if index is None:
            return [], [], 0, f"Index not found"
        if index_type == "ivf" and hasattr(index, "nprobe"):
            index.nprobe = nprobe
        q_2d = q.reshape(1, -1).astype(np.float32)
        D, I = index.search(q_2d, 200)
        cap_idx, cap_sc = I[0], D[0]
    
    t_search = time.perf_counter() - t1
    
    img_scores = {}
    for idx, sc in zip(cap_idx, cap_sc):
        if idx < 0 or idx >= len(doc2img):
            continue
        img = doc2img[int(idx)]
        if img not in img_scores or sc > img_scores[img]:
            img_scores[img] = float(sc)
    
    ranked = sorted(img_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    total_time = (t_encode + t_search) * 1000
    
    results = [img2path.get(img, "") for img, _ in ranked]
    scores = [sc for _, sc in ranked]
    
    return results, scores, total_time, None

# ====================
# DISPLAY FUNCTIONS
# ====================
def display_metrics(num_results, latency, method):
    cols = st.columns(4)
    metrics = [
        ("🔍", num_results, "Results"),
        ("⏱️", f"{latency:.1f}ms", "Latency"),
        ("📊", method, "Method"),
        ("🎯", f"Top-{num_results}", "Retrieved")
    ]
    for col, (icon, value, label) in zip(cols, metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{icon} {value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

def display_results_grid(results, scores, cols_per_row=5):
    if not results:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🔍</div>
            <div class="empty-state-text">No results found</div>
            <div class="empty-state-hint">Try a different query or method</div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(results):
                with col:
                    img_path = results[idx]
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                        st.markdown(f"""
                        <div class="result-info">
                            <span class="result-rank">#{idx+1}</span><br>
                            <span class="result-score">Score: {scores[idx]:.4f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning(f"#{idx+1} - Not found")

def show_empty_state():
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">📤</div>
        <div class="empty-state-text">Upload an image to see results here</div>
        <div class="empty-state-hint">Supported formats: JPG, PNG, WebP</div>
    </div>
    """, unsafe_allow_html=True)

# ====================
# HEADER - Compact
# ====================
st.markdown("""
<div class="compact-header">
    <div>
        <h1>🖼️ Image Retrieval System</h1>
        <p>CS336: Multimedia Information Retrieval 2025 | UIT</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ====================
# SIDEBAR
# ====================
with st.sidebar:
    # Track Selection
    st.markdown('<div class="sidebar-title">🎯 Select Track</div>', unsafe_allow_html=True)
    track = st.radio(
        "",
        ["Track A: Image → Image", "Track B: Text → Image"],
        label_visibility="collapsed"
    )
    
    if "Track A" in track:
        st.markdown('<div class="track-badge track-a">🖼️ Image Query</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="track-badge track-b">📝 Text Query</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Settings Section
    if "Track A" in track:
        st.markdown('<div class="sidebar-title">⚙️ Track A Settings</div>', unsafe_allow_html=True)
        
        method = st.selectbox(
            "Feature Method",
            ["Histogram + Chi-square ⭐", "CLIP", "ResNet50", "Histogram + Cosine"]
        )
        
        method_map = {
            "Histogram + Chi-square ⭐": ("histogram", "chi_square"),
            "Histogram + Cosine": ("histogram", "cosine"),
            "ResNet50": ("resnet", "cosine"),
            "CLIP": ("clip", "cosine")
        }
        method_key, similarity = method_map[method]
        
        # Results info in expander
        with st.expander("📊 Evaluation Results"):
            st.markdown("""
            | Method | mAP | P@1 |
            |--------|-----|-----|
            | **Hist+Chi²** | **0.355** | **0.604** |
            | CLIP | 0.267 | 0.455 |
            | ResNet | 0.136 | 0.277 |
            
            *Dataset: DeepFashion (52K images)*
            """)
    
    else:
        st.markdown('<div class="sidebar-title">⚙️ Track B Settings</div>', unsafe_allow_html=True)
        
        method = st.selectbox("Embedding", ["CLIP Text", "SBERT"])
        method_map = {"CLIP Text": "clip_text", "SBERT": "sbert"}
        method_key = method_map[method]
        
        index_type = st.selectbox("Index Type", ["Flat", "IVF", "Brute Force"])
        index_map = {"Flat": "flat", "IVF": "ivf", "Brute Force": "brute"}
        index_type_key = index_map[index_type]
        
        nprobe = 16
        if index_type == "IVF":
            nprobe = st.slider("IVF nprobe", 1, 64, 16)
        
        with st.expander("📊 Dataset Info"):
            st.markdown("**Flickr30k**: ~31K images, 155K captions")
    
    st.markdown("---")
    
    # Common Settings
    st.markdown('<div class="sidebar-title">🎛️ Display Settings</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        top_k = st.slider("Top-K Results", 5, 50, 20, label_visibility="collapsed")
    with col2:
        st.markdown(f'<span class="slider-value">{top_k}</span>', unsafe_allow_html=True)
    st.caption("Top-K Results")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        cols_per_row = st.slider("Grid Columns", 3, 6, 5, label_visibility="collapsed")
    with col2:
        st.markdown(f'<span class="slider-value">{cols_per_row}</span>', unsafe_allow_html=True)
    st.caption("Grid Columns")
    
    st.markdown("---")
    
    # Team - Compact
    with st.expander("👥 Team"):
        st.markdown("""
        **Nguyễn Trọng Tất Thành**  
        *Track A + UI*
        
        **Trần Vạn Tấn**  
        *Track B + Report*
        """)

# ====================
# MAIN CONTENT
# ====================
if "Track A" in track:
    st.markdown("## 🖼️ Image → Image Search")
    st.caption("Upload a fashion image to find similar items in DeepFashion dataset")
    
    # Tips
    with st.expander("💡 Tips & Best Practices", expanded=False):
        st.markdown("""
        - **Best method**: Histogram + Chi-square (mAP = 0.355)
        - **Fastest**: HNSW index (54x faster, 98% recall)
        - Works best for: color matching, similar patterns
        """)
    
    # Upload Area - More prominent
    uploaded_file = st.file_uploader(
        "📤 **Drag and drop or click to upload**",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Supported: JPG, PNG, WebP (max 200MB)"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 📷 Query")
            query_img = Image.open(uploaded_file)
            st.image(query_img, use_container_width=True)
            search_btn = st.button("🔍 Search", type="primary", use_container_width=True)
        
        with col2:
            if search_btn:
                with st.spinner("🔍 Searching similar images..."):
                    try:
                        temp_path = "/tmp/query_image.jpg"
                        query_img.save(temp_path)
                        
                        engine = load_track_a_engine()
                        
                        if method_key in engine.features:
                            results, scores, latency = engine.search(
                                temp_path, method=method_key, top_k=top_k
                            )
                            st.success(f"✅ Found {len(results)} results!")
                            display_metrics(len(results), latency, method.split()[0])
                        else:
                            st.error(f"❌ Features not loaded for {method_key}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.markdown("""
                <div class="tips-box">
                    <div class="tips-title">👈 Click "Search" to find similar images</div>
                    <p style="color: #888; margin: 0;">Using: {}</p>
                </div>
                """.format(method), unsafe_allow_html=True)
        
        # Results
        if uploaded_file and 'results' in dir() and search_btn:
            st.markdown("---")
            st.markdown("### 🎯 Search Results")
            display_results_grid(results, scores, cols_per_row)
    else:
        show_empty_state()

else:
    # TRACK B
    st.markdown("## 📝 Text → Image Search")
    st.caption("Enter a text description to find matching images in Flickr30k dataset")
    
    # Example queries
    examples = ["Custom...", "A dog running on the grass", 
                "People standing in front of a building",
                "A man riding a bicycle", "Children playing in water"]
    
    col1, col2 = st.columns([4, 1])
    with col2:
        example = st.selectbox("Examples", examples, label_visibility="collapsed")
    with col1:
        default = "" if example == "Custom..." else example
        query_text = st.text_input(
            "🔤 Enter your query",
            value=default,
            placeholder="e.g., A dog running on the grass"
        )
    
    if st.button("🔍 Search Images", type="primary", use_container_width=True):
        if query_text:
            with st.spinner("🔍 Searching..."):
                try:
                    results, scores, latency, error = search_track_b(
                        query=query_text, embed=method_key,
                        index_type=index_type_key, top_k=top_k,
                        features_dir="features/track_b",
                        index_dir="indexes/track_b",
                        data_dir="data/flickr30k", nprobe=nprobe
                    )
                    
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        st.success(f"✅ Found {len(results)} results!")
                        st.markdown(f"**Query:** *{query_text}*")
                        display_metrics(len(results), latency, method)
                        st.markdown("---")
                        display_results_grid(results, scores, cols_per_row)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a query")
    else:
        show_empty_state()

# ====================
# FOOTER
# ====================
st.markdown("""
<div class="footer">
    <p><strong>Image Retrieval System</strong></p>
    <p>CS336: Multimedia Information Retrieval 2025 | UIT</p>
    <p>Methods: Histogram • ResNet50 • CLIP • BM25 • SBERT</p>
</div>
""", unsafe_allow_html=True)
