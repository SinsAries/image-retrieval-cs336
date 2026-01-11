"""
Image Retrieval System - Unified Streamlit UI
CS336: Multimedia Information Retrieval 2025
Track A: Image → Image (DeepFashion)
Track B: Text → Image (Flickr30k)
"""
import streamlit as st
import os
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

# ====================
# PAGE CONFIG
# ====================
st.set_page_config(
    page_title="Image Retrieval System",
    page_icon="🖼️",
    layout="wide"
)

# ====================
# CACHE FUNCTIONS
# ====================
@st.cache_resource
def load_track_a_engine():
    """Load Image Search Engine (Track A)"""
    from src.search.search import ImageSearchEngine
    
    engine = ImageSearchEngine()
    
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
def load_track_b_meta(features_dir: str, embed: str):
    """Load Track B metadata"""
    import json
    meta_path = Path(features_dir) / f"{embed}_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_resource
def load_track_b_embeddings(features_dir: str, embed: str):
    """Load Track B embeddings (memmap)"""
    npy_path = Path(features_dir) / f"{embed}.npy"
    if npy_path.exists():
        return np.load(str(npy_path), mmap_mode="r")
    return None

@st.cache_resource
def load_track_b_index(index_path: str):
    """Load Track B FAISS index"""
    import faiss
    if Path(index_path).exists():
        return faiss.read_index(index_path)
    return None

@st.cache_resource
def load_track_b_img2path(data_dir: str):
    """Build image name to path mapping"""
    img_dir = Path(data_dir) / "images"
    if img_dir.exists():
        return {p.name: str(p) for p in img_dir.glob("*.jpg")}
    return {}

# ====================
# TRACK B SEARCH FUNCTIONS
# ====================
def encode_query_track_b(query: str, embed: str, meta: dict, device: str = "cpu"):
    """Encode text query for Track B"""
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

def search_track_b(query: str, embed: str, index_type: str, top_k: int, 
                   features_dir: str, index_dir: str, data_dir: str, nprobe: int = 16):
    """Search Track B"""
    import faiss
    
    meta = load_track_b_meta(features_dir, embed)
    if meta is None:
        return [], [], 0, "Meta file not found"
    
    doc2img = meta.get("doc2img", [])
    img2path = load_track_b_img2path(data_dir)
    
    # Encode query
    t0 = time.perf_counter()
    q = encode_query_track_b(query, embed, meta)
    t_encode = time.perf_counter() - t0
    
    if q is None:
        return [], [], 0, "Failed to encode query"
    
    # Search
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
            return [], [], 0, f"Index not found: {index_path}"
        
        if index_type == "ivf" and hasattr(index, "nprobe"):
            index.nprobe = nprobe
        
        q_2d = q.reshape(1, -1).astype(np.float32)
        D, I = index.search(q_2d, 200)
        cap_idx = I[0]
        cap_sc = D[0]
    
    t_search = time.perf_counter() - t1
    
    # Aggregate caption -> image (max score per image)
    t2 = time.perf_counter()
    img_scores = {}
    for idx, sc in zip(cap_idx, cap_sc):
        if idx < 0 or idx >= len(doc2img):
            continue
        img = doc2img[int(idx)]
        if img not in img_scores or sc > img_scores[img]:
            img_scores[img] = float(sc)
    
    ranked = sorted(img_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    t_agg = time.perf_counter() - t2
    
    total_time = (t_encode + t_search + t_agg) * 1000  # ms
    
    results = [img2path.get(img, "") for img, _ in ranked]
    scores = [sc for _, sc in ranked]
    
    return results, scores, total_time, None

# ====================
# DISPLAY FUNCTIONS
# ====================
def display_results(results: list, scores: list, latency: float, method: str):
    """Display search results in a grid"""
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🔍 Results Found", len(results))
    col2.metric("⏱️ Latency", f"{latency:.2f} ms")
    col3.metric("📊 Method", method)
    
    st.markdown("---")
    st.subheader("🎯 Search Results")
    
    cols_per_row = 5
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(results):
                img_path = results[idx]
                score = scores[idx]
                
                with col:
                    if img_path and os.path.exists(img_path):
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True)
                        st.caption(f"#{idx+1} | Score: {score:.4f}")
                    else:
                        st.warning(f"Image not found")
                        st.caption(f"#{idx+1}")

# ====================
# SIDEBAR
# ====================
st.sidebar.title("⚙️ Settings")

# Track selection
track = st.sidebar.radio(
    "🎯 Select Track",
    ["Track A: Image → Image", "Track B: Text → Image"]
)

st.sidebar.markdown("---")

# Method selection based on track
if track == "Track A: Image → Image":
    st.sidebar.subheader("🖼️ Track A Settings")
    
    method = st.sidebar.selectbox(
        "Feature Method",
        ["Histogram (Chi-square)", "Histogram (Cosine)", "ResNet50", "CLIP"]
    )
    
    method_map = {
        "Histogram (Chi-square)": ("histogram", "chi_square"),
        "Histogram (Cosine)": ("histogram", "cosine"),
        "ResNet50": ("resnet", "cosine"),
        "CLIP": ("clip", "cosine")
    }
    method_key, similarity = method_map[method]
    
    index_type = st.sidebar.selectbox(
        "Index Type",
        ["Flat (Exact)", "IVF (Approximate)", "HNSW (Fast)"]
    )
    
    dataset_info = st.sidebar.expander("📂 Dataset Info")
    with dataset_info:
        st.write("**DeepFashion In-shop**")
        st.write("- 52,712 images")
        st.write("- 14,218 queries")
        st.write("- 12,612 gallery")

else:
    st.sidebar.subheader("📝 Track B Settings")
    
    method = st.sidebar.selectbox(
        "Embedding Method",
        ["CLIP Text", "SBERT"]
    )
    
    method_map = {
        "CLIP Text": "clip_text",
        "SBERT": "sbert"
    }
    method_key = method_map[method]
    
    index_type = st.sidebar.selectbox(
        "Index Type",
        ["Flat (Exact)", "IVF (Approximate)", "Brute Force"]
    )
    
    index_map = {
        "Flat (Exact)": "flat",
        "IVF (Approximate)": "ivf",
        "Brute Force": "brute"
    }
    index_type_key = index_map[index_type]
    
    if index_type == "IVF (Approximate)":
        nprobe = st.sidebar.slider("IVF nprobe", 1, 64, 16)
    else:
        nprobe = 16
    
    dataset_info = st.sidebar.expander("📂 Dataset Info")
    with dataset_info:
        st.write("**Flickr30k**")
        st.write("- ~31K images")
        st.write("- 5 captions/image")
        st.write("- ~155K captions")

# Top-K
st.sidebar.markdown("---")
top_k = st.sidebar.slider("🔝 Top-K Results", 5, 50, 20)

# ====================
# HEADER
# ====================
st.title("🖼️ Image Retrieval System")
st.markdown("**CS336: Multimedia Information Retrieval 2025**")

# ====================
# TRACK A: Image → Image
# ====================
if track == "Track A: Image → Image":
    st.header("🖼️ Image → Image Search")
    st.markdown("Upload an image to find similar fashion items in DeepFashion dataset.")
    
    # Example queries
    with st.expander("💡 Tips"):
        st.markdown("""
        - **Histogram + Chi-square**: Best for color-based matching (mAP=0.35)
        - **CLIP**: Best for semantic similarity (mAP=0.27)
        - **ResNet50**: Deep features baseline (mAP=0.14)
        """)
    
    uploaded_file = st.file_uploader(
        "Upload Query Image",
        type=['jpg', 'jpeg', 'png', 'webp']
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Query Image")
            query_img = Image.open(uploaded_file)
            st.image(query_img, use_container_width=True)
        
        with col2:
            if st.button("🔍 Search", type="primary", use_container_width=True):
                with st.spinner("Searching..."):
                    try:
                        # Save temp file
                        temp_path = "/tmp/query_image.jpg"
                        query_img.save(temp_path)
                        
                        engine = load_track_a_engine()
                        
                        if method_key in engine.features:
                            results, scores, latency = engine.search(
                                temp_path,
                                method=method_key,
                                top_k=top_k
                            )
                            display_results(results, scores, latency, method)
                        else:
                            st.error(f"Features for {method_key} not loaded.")
                            st.info("Run feature extraction first: `python src/extract/{method_key}.py`")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# ====================
# TRACK B: Text → Image
# ====================
else:
    st.header("📝 Text → Image Search")
    st.markdown("Enter a text description to find matching images in Flickr30k dataset.")
    
    # Example queries
    example_queries = [
        "a dog running on the grass",
        "a group of people standing in front of a building",
        "a man riding a bicycle on the street",
        "two children playing in the water",
        "a woman holding a camera",
    ]
    
    col1, col2 = st.columns([3, 1])
    with col2:
        example = st.selectbox("📌 Examples", ["(Custom)"] + example_queries)
    with col1:
        default_query = "" if example == "(Custom)" else example
        query_text = st.text_input(
            "Enter your query",
            value=default_query,
            placeholder="e.g., a dog running on the grass"
        )
    
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if query_text:
            with st.spinner("Searching..."):
                try:
                    features_dir = FEATURES_TRACK_B
                    index_dir = INDEXES_DIR.replace("indexes", "indexes/track_b") if "track_b" not in INDEXES_DIR else INDEXES_DIR
                    data_dir = FLICKR30K_DIR
                    
                    # Fix paths
                    if not os.path.exists(features_dir):
                        features_dir = "features/track_b"
                    if not os.path.exists(data_dir):
                        data_dir = "data/flickr30k"
                    index_dir = "indexes/track_b"
                    
                    results, scores, latency, error = search_track_b(
                        query=query_text,
                        embed=method_key,
                        index_type=index_type_key,
                        top_k=top_k,
                        features_dir=features_dir,
                        index_dir=index_dir,
                        data_dir=data_dir,
                        nprobe=nprobe
                    )
                    
                    if error:
                        st.error(f"Error: {error}")
                        st.info("Make sure you have extracted features and built the index.")
                    else:
                        st.markdown(f"**Query:** *{query_text}*")
                        display_results(results, scores, latency, method)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.warning("Please enter a query text.")

# ====================
# FOOTER
# ====================
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Track A Methods:**")
    st.markdown("Histogram, ResNet50, CLIP")

with col2:
    st.markdown("**Track B Methods:**")
    st.markdown("BM25, SBERT, CLIP")

with col3:
    st.markdown("**Team:**")
    st.markdown("Nguyễn Trọng Tất Thành & Trần Vạn Tấn")

st.markdown(
    """
    <div style='text-align: center; color: gray; margin-top: 20px;'>
        Image Retrieval System | CS336 MIR 2025 | UIT
    </div>
    """,
    unsafe_allow_html=True
)
