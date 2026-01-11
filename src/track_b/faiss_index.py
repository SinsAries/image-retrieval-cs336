from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import faiss

# SBERT
from sentence_transformers import SentenceTransformer

# CLIP text
import torch
import open_clip


PathLike = Union[str, Path]


# =========================
# 0) IO helpers (features/meta/images)
# =========================

def resolve_feature_paths(features_dir: str, embed: str) -> Tuple[Path, Path]:
    feat_dir = Path(features_dir)
    npy_path = feat_dir / f"{embed}.npy"
    meta_path = feat_dir / f"{embed}_meta.json"
    return npy_path, meta_path


def load_meta(meta_path: Path) -> dict:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_embeddings_memmap(npy_path: Path) -> np.memmap:
    # memmap: không load hết vào RAM
    return np.load(str(npy_path), mmap_mode="r")


def build_img2path(data_dir: str) -> Dict[str, str]:
    img_dir = Path(data_dir) / "images"
    return {p.name: str(p) for p in img_dir.glob("*.jpg")}


# =========================
# 1) Caption -> Image aggregation
# =========================

def aggregate_max_per_image(
    caption_indices: np.ndarray,
    caption_scores: np.ndarray,
    doc2img: List[str],
    topk_images: int,
) -> List[Tuple[str, float]]:
    """
    Vì 1 ảnh có nhiều captions, ta gộp theo ảnh bằng max score.
    """
    best: Dict[str, float] = {}
    for idx, sc in zip(caption_indices.tolist(), caption_scores.tolist()):
        img = doc2img[int(idx)]
        s = float(sc)
        if (img not in best) or (s > best[img]):
            best[img] = s
    items = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return items[:topk_images]


# =========================
# 2) Query encoders (SBERT / CLIP-text)
# =========================

_SBER_CACHE: Dict[Tuple[str, str], SentenceTransformer] = {}
_CLIP_CACHE: Dict[Tuple[str, str, str], Tuple[torch.nn.Module, any]] = {}


def encode_query_sbert(query: str, model_name: str, device: str = "cpu") -> np.ndarray:
    key = (model_name, device)
    if key not in _SBER_CACHE:
        _SBER_CACHE[key] = SentenceTransformer(model_name, device=device)
    model = _SBER_CACHE[key]
    v = model.encode([query], normalize_embeddings=True)[0]
    return np.asarray(v, dtype=np.float32)


def encode_query_clip_text(
    query: str,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cpu",
) -> np.ndarray:
    dev = torch.device(device)
    key = (model_name, pretrained, device)

    if key not in _CLIP_CACHE:
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model.eval().to(dev)
        tokenizer = open_clip.get_tokenizer(model_name)
        _CLIP_CACHE[key] = (model, tokenizer)

    model, tokenizer = _CLIP_CACHE[key]
    with torch.no_grad():
        tokens = tokenizer([query]).to(dev)
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)  # normalize
    return feat.detach().cpu().numpy()[0].astype(np.float32)


def encode_query(embed: str, meta: dict, query: str, device: str) -> np.ndarray:
    """
    Encode query -> vector (float32, normalized) đúng model đã dùng khi extract embeddings.
    """
    if embed == "sbert":
        model_name = meta.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        return encode_query_sbert(query, model_name=model_name, device=device)

    if embed == "clip_text":
        model_name = meta.get("model", "ViT-B-32")
        pretrained = meta.get("pretrained", "openai")
        return encode_query_clip_text(query, model_name=model_name, pretrained=pretrained, device=device)

    raise ValueError(f"Unknown embed: {embed}")


# =========================
# 3) Brute-force search (exact, no FAISS index)
# =========================

def topk_dot_chunked(
    X: np.memmap,
    q: np.ndarray,
    k: int,
    limit: Optional[int],
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Brute-force: score[i] = X[i] dot q, chạy theo chunk để nhẹ RAM.
    Return: (caption_indices, caption_scores) sorted desc.
    """
    n_total = int(X.shape[0])
    n = n_total if (limit is None or limit <= 0) else min(int(limit), n_total)

    best_scores = np.empty((0,), dtype=np.float32)
    best_idx = np.empty((0,), dtype=np.int64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        scores = (X[start:end] @ q).astype(np.float32)

        kk = min(k, int(scores.shape[0]))
        part = np.argpartition(scores, -kk)[-kk:]
        cand_scores = scores[part]
        cand_idx = (part + start).astype(np.int64)

        best_scores = np.concatenate([best_scores, cand_scores])
        best_idx = np.concatenate([best_idx, cand_idx])

        if best_scores.shape[0] > k:
            keep = np.argpartition(best_scores, -k)[-k:]
            best_scores = best_scores[keep]
            best_idx = best_idx[keep]

    order = np.argsort(-best_scores)
    return best_idx[order], best_scores[order]


# =========================
# 4) FAISS core (flat / ivf)
# =========================

@dataclass(frozen=True)
class FaissBuildConfig:
    index_type: str            # "flat" | "ivf"
    nlist: int = 2048
    train_size: int = 100000
    chunk_size: int = 50000


def _as_float32_contiguous(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x.astype(np.float32, copy=False))


def add_in_chunks(index, X: np.memmap, chunk_size: int = 50000) -> None:
    n = int(X.shape[0])
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        xb = _as_float32_contiguous(np.asarray(X[start:end]))
        index.add(xb)


def sample_train_vectors(X: np.memmap, train_size: int) -> np.ndarray:
    n = int(X.shape[0])
    m = min(int(train_size), n)
    idx = np.linspace(0, n - 1, num=m, dtype=np.int64)
    train_x = np.asarray(X[idx])
    return _as_float32_contiguous(train_x)


def build_flat_ip(X: np.memmap, chunk_size: int) -> faiss.Index:
    d = int(X.shape[1])
    index = faiss.IndexFlatIP(d)
    add_in_chunks(index, X, chunk_size=chunk_size)
    return index


def build_ivf_flat_ip(X: np.memmap, nlist: int, train_size: int, chunk_size: int) -> faiss.Index:
    d = int(X.shape[1])
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, int(nlist), faiss.METRIC_INNER_PRODUCT)

    train_x = sample_train_vectors(X, train_size=train_size)
    index.train(train_x)
    add_in_chunks(index, X, chunk_size=chunk_size)
    return index


def build_index(X: np.memmap, cfg: FaissBuildConfig) -> faiss.Index:
    if cfg.index_type == "flat":
        return build_flat_ip(X, chunk_size=cfg.chunk_size)
    if cfg.index_type == "ivf":
        return build_ivf_flat_ip(X, nlist=cfg.nlist, train_size=cfg.train_size, chunk_size=cfg.chunk_size)
    raise ValueError(f"Unsupported index_type: {cfg.index_type}")


def save_index(index: faiss.Index, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: PathLike) -> faiss.Index:
    return faiss.read_index(str(path))


def search_index_ip(index: faiss.Index, q: np.ndarray, topk: int, nprobe: Optional[int] = None):
    if q.ndim == 1:
        q = q.reshape(1, -1)
    q = _as_float32_contiguous(q)

    if nprobe is not None and hasattr(index, "nprobe"):
        index.nprobe = int(nprobe)

    D, I = index.search(q, int(topk))
    return D, I


# =========================
# 5) CLI: build / search (brute|flat|ivf)
# =========================

def cmd_build(args) -> None:
    """
    Build index (flat/ivf) và save ra indexes/track_b/*.faiss
    """
    npy_path, _ = resolve_feature_paths(args.features_dir, args.embed)
    X = load_embeddings_memmap(npy_path)

    cfg = FaissBuildConfig(
        index_type=args.index_type,
        nlist=args.nlist,
        train_size=args.train_size,
        chunk_size=args.chunk_size,
    )
    index = build_index(X, cfg)

    out_path = Path(args.index_dir) / f"{args.embed}_{args.index_type}.faiss"
    save_index(index, out_path)
    print(f"[OK] saved index: {out_path} | ntotal={index.ntotal} | dim={X.shape[1]}")


def cmd_search(args) -> None:
    """
    Search ảnh theo 3 chế độ:
      - brute: không cần .faiss
      - flat/ivf: dùng FAISS index đã build
    """
    # Load meta để lấy doc2img (caption -> image) + info model encode query
    _, meta_path = resolve_feature_paths(args.features_dir, args.embed)
    meta = load_meta(meta_path)
    doc2img = meta["doc2img"]
    img2path = build_img2path(args.data_dir)

    # Encode query
    t0 = time.perf_counter()
    q = encode_query(args.embed, meta, args.query, device=args.device)
    t_encode = time.perf_counter() - t0

    # Search caption-level
    if args.index_type == "brute":
        npy_path, _ = resolve_feature_paths(args.features_dir, args.embed)
        X = load_embeddings_memmap(npy_path)

        t1 = time.perf_counter()
        cap_idx, cap_sc = topk_dot_chunked(
            X=X,
            q=q,
            k=args.topk_caption,
            limit=args.limit,
            chunk_size=args.chunk_size,
        )
        t_search = time.perf_counter() - t1

    else:
        index_path = Path(args.index_dir) / f"{args.embed}_{args.index_type}.faiss"
        index = load_index(index_path)
        if args.index_type == "ivf":
            index.nprobe = int(args.nprobe)

        t1 = time.perf_counter()
        D, I = search_index_ip(
            index=index,
            q=q,
            topk=args.topk_caption,
            nprobe=args.nprobe if args.index_type == "ivf" else None,
        )
        t_search = time.perf_counter() - t1
        cap_idx = I[0].astype(np.int64)
        cap_sc = D[0].astype(np.float32)

    # Aggregate caption -> image
    t2 = time.perf_counter()
    img_items = aggregate_max_per_image(cap_idx, cap_sc, doc2img, topk_images=args.topk)
    t_agg = time.perf_counter() - t2

    print("=" * 80)
    print(f"[SEARCH] embed={args.embed} index={args.index_type} query={args.query!r}")
    if args.index_type == "ivf":
        # nlist in IVF index; brute/flat thì không có
        try:
            index_path = Path(args.index_dir) / f"{args.embed}_{args.index_type}.faiss"
            index = load_index(index_path)
            print(f"nprobe={args.nprobe} nlist={getattr(index, 'nlist', 'NA')}")
        except Exception:
            pass

    print(f"encode={t_encode:.4f}s search={t_search:.4f}s agg={t_agg:.4f}s total={(t_encode+t_search+t_agg):.4f}s")
    print("-" * 80)
    for r, (img, score) in enumerate(img_items, start=1):
        print(f"{r:02d}. {img} score={score:.4f} path={img2path.get(img,'')}")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser(prog="faiss_index", description="Track B: single-file faiss_index (brute/flat/ivf)")
    sub = ap.add_subparsers(dest="command", required=True)

    # build
    ap_b = sub.add_parser("build", help="build & save FAISS index (flat/ivf)")
    ap_b.add_argument("--embed", choices=["sbert", "clip_text"], required=True)
    ap_b.add_argument("--features_dir", default=r"features\track_b")
    ap_b.add_argument("--index_dir", default=r"indexes\track_b")
    ap_b.add_argument("--index_type", choices=["flat", "ivf"], required=True)
    ap_b.add_argument("--nlist", type=int, default=2048)
    ap_b.add_argument("--train_size", type=int, default=100000)
    ap_b.add_argument("--chunk_size", type=int, default=50000)
    ap_b.set_defaults(func=cmd_build)

    # search
    ap_s = sub.add_parser("search", help="search images (brute/flat/ivf)")
    ap_s.add_argument("--embed", choices=["sbert", "clip_text"], required=True)
    ap_s.add_argument("--index_type", choices=["brute", "flat", "ivf"], required=True)
    ap_s.add_argument("--features_dir", default=r"features\track_b")
    ap_s.add_argument("--index_dir", default=r"indexes\track_b")
    ap_s.add_argument("--data_dir", required=True)
    ap_s.add_argument("--query", required=True)
    ap_s.add_argument("--topk", type=int, default=5)
    ap_s.add_argument("--topk_caption", type=int, default=200)
    ap_s.add_argument("--nprobe", type=int, default=16)
    ap_s.add_argument("--device", default="cpu")

    # brute-only options (ignored for flat/ivf)
    ap_s.add_argument("--limit", type=int, default=0, help="brute only: 0=full captions")
    ap_s.add_argument("--chunk_size", type=int, default=50000, help="brute only: dot-product chunk size")

    ap_s.set_defaults(func=cmd_search)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
