import argparse
import json
import os
import time
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

from .flickr30k import load_flickr30k, simple_tokenize


def build_bm25_corpus(
    data_dir: str,
    captions_file: str | None,
    cache_path: str,
) -> Tuple[List[str], List[str]]:
    """
    Build and cache corpus metadata (docs, doc2img).
    We cache docs + doc2img as JSON so later runs are faster to start.
    """
    docs, doc2img, _img2path = load_flickr30k(data_dir, captions_file=captions_file)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"docs": docs, "doc2img": doc2img}, f, ensure_ascii=False)

    return docs, doc2img


def load_or_build_corpus(
    data_dir: str,
    captions_file: str | None,
    cache_path: str,
) -> Tuple[List[str], List[str]]:
    if os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj["docs"], obj["doc2img"]
    return build_bm25_corpus(data_dir, captions_file, cache_path)


def bm25_search_images(
    query: str,
    docs: List[str],
    doc2img: List[str],
    topk: int = 20,
) -> List[Tuple[str, float]]:
    """
    Returns list of (image_filename, score) sorted descending by score.
    Score is max over captions for the same image.
    """
    tokenized_docs = [simple_tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized_docs)

    q_tokens = simple_tokenize(query)
    scores = bm25.get_scores(q_tokens)  # score per caption/doc

    img_score: Dict[str, float] = {}
    for s, img in zip(scores, doc2img):
        prev = img_score.get(img)
        if prev is None or s > prev:
            img_score[img] = float(s)

    ranked = sorted(img_score.items(), key=lambda x: x[1], reverse=True)
    return ranked[:topk]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/flickr30k", help="Flickr30k folder, contains images/ and captions file")
    ap.add_argument("--captions_file", default=None, help="Optional captions filename inside data_dir (e.g., captions.txt)")
    ap.add_argument("--query", required=True, help="Text query")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--cache", default="features/track_b/flickr30k_corpus.json")
    args = ap.parse_args()

    t0 = time.time()
    docs, doc2img = load_or_build_corpus(args.data_dir, args.captions_file, args.cache)
    t1 = time.time()

    ranked = bm25_search_images(args.query, docs, doc2img, topk=args.topk)
    t2 = time.time()

    # Try to print file paths if images exist
    _docs, _doc2img, img2path = load_flickr30k(args.data_dir, captions_file=args.captions_file)

    print(f"[BM25] corpus_docs={len(docs)} load_corpus={t1-t0:.3f}s search={t2-t1:.3f}s")
    for i, (img, score) in enumerate(ranked, start=1):
        path = img2path.get(img, img)
        print(f"{i:02d}. score={score:.4f}  img={img}  path={path}")


if __name__ == "__main__":
    main()
