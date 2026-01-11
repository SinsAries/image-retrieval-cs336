from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

import src.track_b.faiss_index as fi


# =========================
# Utils
# =========================
def human_bytes(n: int) -> str:
    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < step:
            return f"{x:.2f} {u}"
        x /= step
    return f"{x:.2f} PB"


def safe_exists(p: str) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False


def file_size(p: str) -> int:
    try:
        return Path(p).stat().st_size
    except Exception:
        return 0


# =========================
# Cache (để UI chạy mượt)
# =========================
@st.cache_resource
def cached_meta(features_dir: str, embed: str) -> dict:
    _, meta_path = fi.resolve_feature_paths(features_dir, embed)
    return fi.load_meta(meta_path)


@st.cache_resource
def cached_img2path(data_dir: str) -> dict:
    return fi.build_img2path(data_dir)


@st.cache_resource
def cached_memmap(features_dir: str, embed: str):
    npy_path, _ = fi.resolve_feature_paths(features_dir, embed)
    return fi.load_embeddings_memmap(npy_path)


@st.cache_resource
def cached_faiss_index(index_path_str: str):
    return fi.load_index(index_path_str)


def run_search(
    *,
    data_dir: str,
    features_dir: str,
    index_dir: str,
    embed: str,
    index_type: str,
    device: str,
    query: str,
    topk_images: int,
    topk_caption: int,
    nprobe: int,
    brute_limit: int,
    brute_chunk: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Return:
      rows: list dict {rank, image_name, score, path}
      timings: {encode, search, agg, total}
    """
    meta = cached_meta(features_dir, embed)
    doc2img = meta["doc2img"]
    img2path = cached_img2path(data_dir)

    # encode query
    t0 = time.perf_counter()
    q = fi.encode_query(embed, meta, query, device=device)
    t_encode = time.perf_counter() - t0

    # caption-level search
    t1 = time.perf_counter()

    if index_type == "brute":
        X = cached_memmap(features_dir, embed)
        cap_idx, cap_sc = fi.topk_dot_chunked(
            X=X,
            q=q,
            k=int(topk_caption),
            limit=(None if int(brute_limit) <= 0 else int(brute_limit)),
            chunk_size=int(brute_chunk),
        )
    else:
        index_path = Path(index_dir) / f"{embed}_{index_type}.faiss"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Missing index file: {index_path}. Build it first (UI Build button or CMD build)."
            )

        index = cached_faiss_index(str(index_path))
        D, I = fi.search_index_ip(
            index=index,
            q=q,
            topk=int(topk_caption),
            nprobe=(int(nprobe) if index_type == "ivf" else None),
        )
        cap_idx = I[0].astype(np.int64)
        cap_sc = D[0].astype(np.float32)

    t_search = time.perf_counter() - t1

    # aggregate caption -> image
    t2 = time.perf_counter()
    img_items = fi.aggregate_max_per_image(
        cap_idx,
        cap_sc,
        doc2img,
        topk_images=int(topk_images),
    )
    t_agg = time.perf_counter() - t2

    total = t_encode + t_search + t_agg

    rows: List[Dict[str, Any]] = []
    for rank, (img, score) in enumerate(img_items, start=1):
        rows.append(
            {
                "rank": rank,
                "image_name": img,
                "score": float(score),
                "path": img2path.get(img, ""),
            }
        )

    timings = {"encode": t_encode, "search": t_search, "agg": t_agg, "total": total}
    return rows, timings


# =========================
# UI
# =========================
st.set_page_config(page_title="Track B - Text→Image", layout="wide")
st.title("Track B — Text → Image Retrieval (SBERT / CLIP-text)")
st.caption("Track B only. Không đụng Track A. Chỉ dùng caption embeddings + FAISS (flat/ivf) hoặc brute-force.")

# defaults
DEFAULT_DATA_DIR = r"data\flickr30k"
DEFAULT_FEATURES_DIR = r"features\track_b"
DEFAULT_INDEX_DIR = r"indexes\track_b"

with st.sidebar:
    st.header("Dataset / Track")

    track = st.selectbox("Track", ["B (Text→Image)"], disabled=True)
    dataset_name = st.selectbox("Dataset", ["flickr30k"])

    data_dir = st.text_input("data_dir", value=DEFAULT_DATA_DIR)
    features_dir = st.text_input("features_dir", value=DEFAULT_FEATURES_DIR)
    index_dir = st.text_input("index_dir", value=DEFAULT_INDEX_DIR)

    st.divider()
    st.header("Method")

    embed = st.selectbox("Embedding (caption vectors)", ["clip_text", "sbert"])
    index_type = st.selectbox("Index type", ["ivf", "flat", "brute"])
    device = st.selectbox("Device", ["cpu"])

    st.divider()
    st.header("Search params")

    topk_images = st.slider("TopK images", 1, 50, 20)
    topk_caption = st.slider("TopK captions (before aggregation)", 10, 1000, 200)

    nprobe = 16
    if index_type == "ivf":
        nprobe = st.slider("IVF nprobe", 1, 64, 16)

    brute_limit = 0
    brute_chunk = 50000
    if index_type == "brute":
        brute_limit = st.number_input("Brute limit (0=full captions)", min_value=0, value=0, step=10000)
        brute_chunk = st.number_input("Brute chunk_size", min_value=10000, value=50000, step=10000)

    st.divider()
    st.header("Build index (Flat/IVF)")

    with st.expander("Advanced build params", expanded=False):
        build_nlist = st.number_input("IVF nlist", min_value=128, value=2048, step=128)
        build_train_size = st.number_input("IVF train_size", min_value=10000, value=100000, step=10000)
        build_chunk = st.number_input("Add chunk_size", min_value=10000, value=50000, step=10000)

    build_index_btn = st.button("Build index now", help="Tạo file .faiss trong indexes_dir (đã gitignore)")

    st.divider()
    st.header("Display")
    grid_cols = st.slider("Grid columns", 3, 8, 5)
    show_table = st.checkbox("Show result table", value=True)
    show_debug = st.checkbox("Show debug info", value=False)

tabs = st.tabs(["🔎 Search", "🧪 E2E self-test", "🧰 Diagnostics"])

# =========================
# TAB 1: SEARCH
# =========================
with tabs[0]:
    st.subheader("Query")

    example_queries = [
        "a dog running on the grass",
        "a group of people standing in front of a building",
        "a man riding a bicycle on the street",
        "two children playing in the water",
        "a woman holding a camera",
    ]
    colq1, colq2 = st.columns([3, 1])
    with colq2:
        picked = st.selectbox("Examples", ["(custom)"] + example_queries)
    with colq1:
        query = st.text_input("Nhập text query", value=("a dog running on the grass" if picked == "(custom)" else picked))

    # CMD equivalent
    if index_type == "brute":
        cmd_equiv = (
            f'python -m src.track_b.faiss_index search --embed {embed} --index_type brute --data_dir "{data_dir}" '
            f'--query "{query}" --topk {topk_images} --topk_caption {topk_caption} '
            f'--limit {int(brute_limit)} --chunk_size {int(brute_chunk)}'
        )
    else:
        cmd_equiv = (
            f'python -m src.track_b.faiss_index search --embed {embed} --index_type {index_type} --data_dir "{data_dir}" '
            f'--query "{query}" --topk {topk_images} --topk_caption {topk_caption}'
        )
        if index_type == "ivf":
            cmd_equiv += f" --nprobe {nprobe}"

    st.caption("CMD equivalent (để bạn hiểu UI đang làm đúng lệnh nào):")
    st.code(cmd_equiv, language="bat")

    # actions
    colA, colB, colC = st.columns([1, 1, 3])
    with colA:
        run_btn = st.button("Search", type="primary")
    with colB:
        clear_btn = st.button("Clear caches")
    with colC:
        st.info(
            "Encode (SBERT/CLIP) thường chậm ở lần đầu vì load model. "
            "Sau đó nhanh hơn vì model được giữ trong RAM (trong process Streamlit)."
        )

    if clear_btn:
        cached_meta.clear()
        cached_img2path.clear()
        cached_memmap.clear()
        cached_faiss_index.clear()
        st.success("Cleared Streamlit caches.")

    # build index action (global)
    if build_index_btn:
        if index_type not in ("flat", "ivf"):
            st.warning("Bạn đang chọn brute. Build index chỉ áp dụng cho flat/ivf.")
        else:
            with st.spinner("Đang build FAISS index..."):
                X = cached_memmap(features_dir, embed)
                cfg = fi.FaissBuildConfig(
                    index_type=index_type,
                    nlist=int(build_nlist),
                    train_size=int(build_train_size),
                    chunk_size=int(build_chunk),
                )
                index = fi.build_index(X, cfg)
                out_path = Path(index_dir) / f"{embed}_{index_type}.faiss"
                fi.save_index(index, out_path)
            st.success(f"Build xong: {out_path} (ntotal={index.ntotal}, dim={X.shape[1]})")
            cached_faiss_index.clear()

    # run search
    if run_btn:
        try:
            with st.spinner("Searching..."):
                rows, timings = run_search(
                    data_dir=data_dir,
                    features_dir=features_dir,
                    index_dir=index_dir,
                    embed=embed,
                    index_type=index_type,
                    device=device,
                    query=query,
                    topk_images=int(topk_images),
                    topk_caption=int(topk_caption),
                    nprobe=int(nprobe),
                    brute_limit=int(brute_limit),
                    brute_chunk=int(brute_chunk),
                )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Encode (s)", f"{timings['encode']:.4f}")
            m2.metric("Search (s)", f"{timings['search']:.4f}")
            m3.metric("Aggregate (s)", f"{timings['agg']:.4f}")
            m4.metric("Total (s)", f"{timings['total']:.4f}")

            # download results
            import csv
            import io as _io

            buf = _io.StringIO()
            w = csv.DictWriter(buf, fieldnames=["rank", "image_name", "score", "path"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
            st.download_button(
                "Download results CSV",
                data=buf.getvalue().encode("utf-8"),
                file_name=f"track_b_{embed}_{index_type}_results.csv",
                mime="text/csv",
            )

            if show_table:
                st.subheader("Top results (table)")
                st.dataframe(rows, use_container_width=True)

            st.subheader("Image grid")
            cols = st.columns(int(grid_cols))
            for i, r in enumerate(rows):
                c = cols[i % int(grid_cols)]
                with c:
                    if r["path"] and safe_exists(r["path"]):
                        st.image(
                            r["path"],
                            caption=f'{r["image_name"]}\nscore={r["score"]:.4f}',
                            use_container_width=True,
                        )
                    else:
                        st.warning(f"Missing image path:\n{r['image_name']}")

            if show_debug:
                st.subheader("Debug")
                st.write("First row:", rows[0] if rows else None)

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)


# =========================
# TAB 2: E2E SELF-TEST
# =========================
with tabs[1]:
    st.subheader("E2E self-test (tick 'Test UI end-to-end')")
    st.write(
        "Nút dưới đây sẽ chạy nhanh 3 mode để chứng minh UI hoạt động end-to-end:\n"
        "- IVF (fast)\n"
        "- Flat (exact)\n"
        "- Brute (baseline, limit nhỏ)\n"
    )

    test_query = st.text_input("Self-test query", value="a dog running on the grass", key="e2e_query")
    run_test = st.button("Run self-test")

    if run_test:
        results = []
        modes = [
            ("ivf", {"nprobe": 16, "limit": 0}),
            ("flat", {"nprobe": 0, "limit": 0}),
            ("brute", {"nprobe": 0, "limit": 30000}),  # limit nhỏ cho nhanh
        ]

        for mode, extra in modes:
            try:
                rows, timings = run_search(
                    data_dir=data_dir,
                    features_dir=features_dir,
                    index_dir=index_dir,
                    embed=embed,
                    index_type=mode,
                    device=device,
                    query=test_query,
                    topk_images=5,
                    topk_caption=200,
                    nprobe=int(extra["nprobe"]) if mode == "ivf" else 0,
                    brute_limit=int(extra["limit"]) if mode == "brute" else 0,
                    brute_chunk=50000,
                )
                top1 = rows[0]["image_name"] if rows else "(none)"
                results.append(
                    {
                        "mode": mode,
                        "top1": top1,
                        "encode_s": timings["encode"],
                        "search_s": timings["search"],
                        "total_s": timings["total"],
                        "status": "OK",
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "mode": mode,
                        "top1": "(error)",
                        "encode_s": 0.0,
                        "search_s": 0.0,
                        "total_s": 0.0,
                        "status": f"ERROR: {e}",
                    }
                )

        st.subheader("Self-test summary")
        st.dataframe(results, use_container_width=True)
        st.success("Nếu 3 dòng đều OK và có top1 => bạn tick được 'Test UI end-to-end'.")


# =========================
# TAB 3: DIAGNOSTICS
# =========================
with tabs[2]:
    st.subheader("Diagnostics (tick 'Dataset selection' + dễ debug đường dẫn)")

    # Paths check
    st.markdown("### Paths check")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.write("data_dir exists:", safe_exists(data_dir))
        img_dir = str(Path(data_dir) / "images")
        st.write("images dir exists:", safe_exists(img_dir))
    with p2:
        npy_path, meta_path = fi.resolve_feature_paths(features_dir, embed)
        st.write("features npy:", str(npy_path))
        st.write("exists:", npy_path.exists())
        st.write("size:", human_bytes(file_size(str(npy_path))))
    with p3:
        st.write("meta json:", str(meta_path))
        st.write("exists:", meta_path.exists())
        st.write("size:", human_bytes(file_size(str(meta_path))))

    # Meta info
    st.markdown("### Meta info")
    try:
        meta = cached_meta(features_dir, embed)
        st.write("meta keys:", list(meta.keys()))
        if "model" in meta:
            st.write("model:", meta["model"])
        if "pretrained" in meta:
            st.write("pretrained:", meta["pretrained"])
        if "doc2img" in meta:
            st.write("doc2img length:", len(meta["doc2img"]))
    except Exception as e:
        st.error(f"Cannot load meta: {e}")

    # Embeddings shape
    st.markdown("### Embeddings shape")
    try:
        X = cached_memmap(features_dir, embed)
        st.write("X.shape:", tuple(X.shape))
        st.write("dtype:", str(X.dtype))
    except Exception as e:
        st.error(f"Cannot load embeddings memmap: {e}")

    # Index files
    st.markdown("### Index files")
    for t in ["flat", "ivf"]:
        ip = Path(index_dir) / f"{embed}_{t}.faiss"
        st.write(f"{ip} | exists={ip.exists()} | size={human_bytes(file_size(str(ip)))}")

    st.info(
        "Nếu index file chưa có: vào tab Search và bấm 'Build index now' (chọn flat/ivf) "
        "hoặc build bằng CMD: python -m src.track_b.faiss_index build ..."
    )
