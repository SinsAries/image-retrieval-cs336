from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import src.track_b.faiss_index as fi

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def simple_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def load_captions_csv(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    Read data_dir/captions.txt (CSV with header: image_name,comment_number,comment)
    Return:
      docs: list of caption text
      doc2img: list mapping caption_idx -> image_name
    """
    cap_path = Path(data_dir) / "captions.txt"
    docs: List[str] = []
    doc2img: List[str] = []
    with cap_path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            doc2img.append(row["image_name"])
            docs.append(row["comment"])
    return docs, doc2img


def pick_queries_unique_images(
    docs: List[str],
    doc2img: List[str],
    n: int,
    seed: int,
) -> List[Dict]:
    """
    Pick n captions, but try to avoid sampling multiple captions from the same image
    so evaluation covers more distinct images.
    """
    rng = random.Random(seed)
    idxs = list(range(len(docs)))
    rng.shuffle(idxs)

    out: List[Dict] = []
    seen_imgs = set()

    for i in idxs:
        img = doc2img[i]
        if img in seen_imgs:
            continue
        q = docs[i].strip()
        if not q:
            continue
        out.append({"caption_idx": i, "query": q, "gt_image": img})
        seen_imgs.add(img)
        if len(out) >= n:
            break

    return out


def build_bm25(docs: List[str]) -> "BM25Okapi":
    if BM25Okapi is None:
        raise RuntimeError("Missing rank-bm25. Run: pip install rank-bm25")
    tok = [simple_tokenize(x) for x in docs]
    return BM25Okapi(tok)


def aggregate_image_ranking(
    cap_idx: np.ndarray,
    cap_sc: np.ndarray,
    doc2img: List[str],
    topk_images: int,
):
    """
    Caption-level -> Image-level:
      For each image, take max caption score among retrieved captions.
    """
    return fi.aggregate_max_per_image(
        cap_idx.astype(np.int64),
        cap_sc.astype(np.float32),
        doc2img,
        topk_images=int(topk_images),
    )


def rank_metrics(
    ranked_imgs: List[str],
    gt: str,
    ks: List[int],
) -> Tuple[Dict[int, int], float, int]:
    """
    1 relevant image per query.
    hits[k] = 1 if gt in top-k
    MRR = 1/rank if found else 0
    MedR uses rank; if not found in list, use len(list)+1
    """
    hits = {k: int(gt in ranked_imgs[:k]) for k in ks}

    rank = None
    for i, img in enumerate(ranked_imgs, start=1):
        if img == gt:
            rank = i
            break

    mrr = 0.0 if rank is None else 1.0 / rank
    used_rank = rank if rank is not None else (len(ranked_imgs) + 1)
    return hits, mrr, used_rank


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Day 5 evaluation for Track B (caption-level -> image aggregation)")

    ap.add_argument("--data_dir", default=r"data\flickr30k")
    ap.add_argument("--features_dir", default=r"features\track_b")
    ap.add_argument("--index_dir", default=r"indexes\track_b")
    ap.add_argument("--out_dir", default=r"results\track_b")

    ap.add_argument("--n_queries", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--methods", default="bm25,sbert,clip_text")
    ap.add_argument("--k_list", default="1,5,10,20")
    ap.add_argument("--topk_images", type=int, default=20)
    ap.add_argument("--topk_caption", type=int, default=200)

    ap.add_argument("--sbert_index", choices=["flat", "ivf"], default="flat")
    ap.add_argument("--clip_index", choices=["flat", "ivf"], default="ivf")
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--device", default="cpu")

    # avoid trivial self-match (query caption is literally in the caption index)
    ap.add_argument("--exclude_query_caption", type=int, default=1, help="1=exclude the exact query caption_idx from retrieved captions")
    # retrieve extra captions to compensate after filtering query caption out
    ap.add_argument("--extra_caption", type=int, default=50)

    args = ap.parse_args()

    ks = parse_int_list(args.k_list)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load caption texts
    docs, _doc2img_csv = load_captions_csv(args.data_dir)

    # load doc2img from meta to ensure same ordering as embeddings/index
    # (use clip_text_meta as "canonical" since it must exist if clip_text is used)
    _, meta_path_any = fi.resolve_feature_paths(args.features_dir, "clip_text")
    meta_any = fi.load_meta(meta_path_any)
    doc2img = meta_any["doc2img"]

    if len(doc2img) != len(docs):
        raise RuntimeError(f"Mismatch: len(docs)={len(docs)} vs len(doc2img)={len(doc2img)}")

    queries = pick_queries_unique_images(docs, doc2img, args.n_queries, args.seed)
    (out_dir / "queries_day5.json").write_text(
        json.dumps(queries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows = []

    for method in methods:
        hits_sum = {k: 0 for k in ks}
        mrr_sum = 0.0
        ranks: List[int] = []

        # latency buckets (ms)
        encode_ms: List[float] = []
        search_ms: List[float] = []
        agg_ms: List[float] = []
        total_ms: List[float] = []

        bm25 = None
        index = None
        meta = None
        embed = None
        index_type = None

        # init per-method resources
        if method == "bm25":
            bm25 = build_bm25(docs)
        elif method == "sbert":
            embed = "sbert"
            index_type = args.sbert_index
            _, mp = fi.resolve_feature_paths(args.features_dir, embed)
            meta = fi.load_meta(mp)
            idx_path = Path(args.index_dir) / f"{embed}_{index_type}.faiss"
            if not idx_path.exists():
                raise FileNotFoundError(f"Missing index: {idx_path}")
            index = fi.load_index(str(idx_path))
        elif method == "clip_text":
            embed = "clip_text"
            index_type = args.clip_index
            _, mp = fi.resolve_feature_paths(args.features_dir, embed)
            meta = fi.load_meta(mp)
            idx_path = Path(args.index_dir) / f"{embed}_{index_type}.faiss"
            if not idx_path.exists():
                raise FileNotFoundError(f"Missing index: {idx_path}")
            index = fi.load_index(str(idx_path))
        else:
            raise ValueError(f"Unknown method={method}")

        for q in queries:
            t_total0 = time.perf_counter()

            gt = q["gt_image"]
            qtext = q["query"]
            qcap = int(q["caption_idx"])

            # default timings
            t_enc = 0.0
            t_search = 0.0
            t_agg = 0.0

            if method == "bm25":
                # encode = 0
                t_search0 = time.perf_counter()

                scores = np.asarray(bm25.get_scores(simple_tokenize(qtext)), dtype=np.float32)
                if args.exclude_query_caption:
                    scores[qcap] = -1e9

                kcap = min(int(args.topk_caption) + int(args.extra_caption), scores.shape[0])
                idx = np.argpartition(scores, -kcap)[-kcap:]
                idx = idx[np.argsort(-scores[idx])]
                idx = idx[: int(args.topk_caption)]

                cap_idx = idx.astype(np.int64)
                cap_sc = scores[idx].astype(np.float32)

                t_search = time.perf_counter() - t_search0

                t_agg0 = time.perf_counter()
                img_items = aggregate_image_ranking(cap_idx, cap_sc, doc2img, args.topk_images)
                t_agg = time.perf_counter() - t_agg0

            else:
                # encode query
                t_enc0 = time.perf_counter()
                qv = fi.encode_query(embed, meta, qtext, device=args.device)
                t_enc = time.perf_counter() - t_enc0

                # faiss search
                t_search0 = time.perf_counter()
                D, I = fi.search_index_ip(
                    index=index,
                    q=qv,
                    topk=int(args.topk_caption) + int(args.extra_caption),
                    nprobe=(int(args.nprobe) if index_type == "ivf" else None),
                )
                t_search = time.perf_counter() - t_search0

                cap_idx = I[0].astype(np.int64)
                cap_sc = D[0].astype(np.float32)

                if args.exclude_query_caption:
                    mask = cap_idx != qcap
                    cap_idx = cap_idx[mask]
                    cap_sc = cap_sc[mask]

                cap_idx = cap_idx[: int(args.topk_caption)]
                cap_sc = cap_sc[: int(args.topk_caption)]

                # aggregate
                t_agg0 = time.perf_counter()
                img_items = aggregate_image_ranking(cap_idx, cap_sc, doc2img, args.topk_images)
                t_agg = time.perf_counter() - t_agg0

            ranked_imgs = [img for img, _ in img_items]
            hits, mrr, used_rank = rank_metrics(ranked_imgs, gt, ks=ks)

            for k in ks:
                hits_sum[k] += hits[k]
            mrr_sum += mrr
            ranks.append(used_rank)

            # save latency (ms)
            encode_ms.append(1000.0 * t_enc)
            search_ms.append(1000.0 * t_search)
            agg_ms.append(1000.0 * t_agg)
            total_ms.append(1000.0 * (time.perf_counter() - t_total0))

        n = len(queries)
        row = {
            "method": method,
            **{f"R@{k}": hits_sum[k] / n for k in ks},
            "MRR": mrr_sum / n,
            "MedR": float(np.median(np.asarray(ranks, dtype=np.float32))),
            "avg_encode_ms": float(np.mean(np.asarray(encode_ms, dtype=np.float32))),
            "avg_search_ms": float(np.mean(np.asarray(search_ms, dtype=np.float32))),
            "avg_agg_ms": float(np.mean(np.asarray(agg_ms, dtype=np.float32))),
            "avg_total_ms": float(np.mean(np.asarray(total_ms, dtype=np.float32))),
            "n_queries": n,
            "exclude_query_caption": int(bool(args.exclude_query_caption)),
            "sbert_index": args.sbert_index,
            "clip_index": args.clip_index,
            "nprobe": args.nprobe,
            "topk_caption": int(args.topk_caption),
            "topk_images": int(args.topk_images),
        }
        rows.append(row)

    # print summary
    print("\n=== SUMMARY (copy screenshot this) ===")
    headers = ["method"] + [f"R@{k}" for k in ks] + ["MRR", "MedR", "avg_encode_ms", "avg_search_ms", "avg_agg_ms", "avg_total_ms"]
    print(" | ".join(h.ljust(12) for h in headers))
    print("-" * 110)
    for r in rows:
        line = [
            str(r["method"]).ljust(12),
            *[f"{r[f'R@{k}']:.4f}".ljust(12) for k in ks],
            f"{r['MRR']:.4f}".ljust(12),
            f"{r['MedR']:.1f}".ljust(12),
            f"{r['avg_encode_ms']:.3f}".ljust(12),
            f"{r['avg_search_ms']:.3f}".ljust(12),
            f"{r['avg_agg_ms']:.3f}".ljust(12),
            f"{r['avg_total_ms']:.3f}".ljust(12),
        ]
        print(" | ".join(line))

    out_csv = Path(args.out_dir) / "eval_day5_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n[OK] saved: {out_csv}")
    print(f"[OK] saved: {out_dir / 'queries_day5.json'}")


if __name__ == "__main__":
    main()
