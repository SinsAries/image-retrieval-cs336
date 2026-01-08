import argparse
import json
import os
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from .flickr30k import load_flickr30k


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/flickr30k")
    ap.add_argument("--captions_file", default="captions.txt")
    ap.add_argument("--out_npy", default="features/track_b/sbert.npy")
    ap.add_argument("--out_meta", default="features/track_b/sbert_meta.json")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--chunk_size", type=int, default=10000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="0 means all captions; >0 means only first N captions")
    args = ap.parse_args()

    docs, doc2img, _ = load_flickr30k(args.data_dir, captions_file=args.captions_file)
    if args.limit and args.limit > 0:
        docs = docs[:args.limit]
        doc2img = doc2img[:args.limit]
    os.makedirs(os.path.dirname(args.out_npy), exist_ok=True)

    print(f"[SBERT] docs={len(docs)} model={args.model} device={args.device}")
    model = SentenceTransformer(args.model, device=args.device)

    # Find embedding dim
    dim = model.encode(["test"], convert_to_numpy=True).shape[1]
    n = len(docs)

    tmp_path = args.out_npy + ".memmap"
    embs_mm = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(n, dim))

    t0 = time.time()
    for start in range(0, n, args.chunk_size):
        end = min(start + args.chunk_size, n)
        chunk = docs[start:end]

        chunk_emb = model.encode(
            chunk,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)

        chunk_emb = l2_normalize(chunk_emb).astype(np.float32)
        embs_mm[start:end] = chunk_emb
        embs_mm.flush()
        print(f"[SBERT] chunk {start}:{end} done")

    t1 = time.time()

    np.save(args.out_npy, np.asarray(embs_mm))
    try:
        del embs_mm
        os.remove(tmp_path)
    except Exception:
        pass

    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump({"doc2img": doc2img, "model": args.model}, f, ensure_ascii=False)

    print(f"[SBERT] saved {args.out_npy} shape={(n, dim)} time={t1-t0:.2f}s")
    print(f"[SBERT] saved {args.out_meta}")


if __name__ == "__main__":
    main()
